"""Data coordinator for llama-swap."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import (
    LlamaSwapAuthError,
    LlamaSwapClient,
    LlamaSwapError,
    LlamaSwapNotFoundError,
)
from .const import ACTIVE_STATES, DOMAIN, STATE_READY, STATE_STOPPED

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelInfo:
    """Everything known about one configured model."""

    id: str
    name: str = ""
    description: str = ""
    state: str = STATE_STOPPED
    kind: str = "model"
    aliases: list[str] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    architecture: dict[str, Any] = field(default_factory=dict)
    supported_parameters: list[str] = field(default_factory=list)
    context_length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cmd: str | None = None
    proxy: str | None = None
    ttl: int | None = None
    unlisted: bool = False

    @property
    def is_loaded(self) -> bool:
        """Return True while the model holds a process."""
        return self.state in ACTIVE_STATES

    @property
    def display_name(self) -> str:
        """Return the friendly name, falling back to the model ID."""
        return self.name or self.id


@dataclass(slots=True)
class LlamaSwapData:
    """One poll's worth of server state."""

    models: dict[str, ModelInfo] = field(default_factory=dict)
    version: dict[str, Any] = field(default_factory=dict)
    profiles: list[dict[str, Any]] = field(default_factory=list)
    active_profile: str | None = None
    sys_stats: dict[str, Any] = field(default_factory=dict)
    gpu_stats: list[dict[str, Any]] = field(default_factory=list)
    performance_enabled: bool = False

    @property
    def loaded_models(self) -> list[ModelInfo]:
        """Return the models currently holding a process, ID-sorted."""
        return sorted(
            (model for model in self.models.values() if model.is_loaded),
            key=lambda model: model.id,
        )


type LlamaSwapConfigEntry = ConfigEntry[LlamaSwapCoordinator]


class LlamaSwapCoordinator(DataUpdateCoordinator[LlamaSwapData]):
    """Polls a llama-swap instance and normalises its API responses.

    Optional endpoints (/api/version, /api/profiles, /api/performance) only
    exist on newer builds. Each is probed once and permanently skipped if the
    server 404s, so older servers still get the core model entities.
    """

    config_entry: LlamaSwapConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        entry: LlamaSwapConfigEntry,
        client: LlamaSwapClient,
        scan_interval: int,
    ) -> None:
        """Initialise the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            config_entry=entry,
            update_interval=timedelta(seconds=scan_interval),
        )
        self.client = client
        self._has_version = True
        self._has_profiles = True
        self._has_performance = True

    @property
    def supports_profiles(self) -> bool:
        """Return True when the server exposes the profiles API."""
        return self._has_profiles

    async def _async_update_data(self) -> LlamaSwapData:
        """Fetch the current server state."""
        try:
            models_raw, running_raw = await asyncio.gather(
                self.client.async_get_models(),
                self.client.async_get_running(),
            )
        except LlamaSwapAuthError as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except LlamaSwapError as err:
            raise UpdateFailed(str(err)) from err

        data = LlamaSwapData(models=_build_models(models_raw, running_raw))

        if self._has_version:
            try:
                data.version = await self.client.async_get_version()
            except LlamaSwapNotFoundError:
                self._has_version = False
            except LlamaSwapError as err:
                _LOGGER.debug("Could not read version: %s", err)

        if self._has_profiles:
            try:
                payload = await self.client.async_get_profiles()
            except LlamaSwapNotFoundError:
                self._has_profiles = False
            except LlamaSwapError as err:
                _LOGGER.debug("Could not read profiles: %s", err)
            else:
                raw_profiles = payload.get("profiles")
                data.profiles = raw_profiles if isinstance(raw_profiles, list) else []
                active = payload.get("active")
                data.active_profile = active if isinstance(active, str) else None

        if self._has_performance:
            try:
                payload = await self.client.async_get_performance()
            except LlamaSwapNotFoundError:
                self._has_performance = False
            except LlamaSwapError as err:
                _LOGGER.debug("Could not read performance stats: %s", err)
            else:
                data.performance_enabled = bool(payload.get("enabled"))
                sys_stats = payload.get("sys_stats")
                if isinstance(sys_stats, list) and sys_stats:
                    last = sys_stats[-1]
                    data.sys_stats = last if isinstance(last, dict) else {}
                data.gpu_stats = _latest_gpu_stats(payload.get("gpu_stats"))

        return data


def _build_models(
    models_raw: list[dict[str, Any]], running_raw: list[dict[str, Any]]
) -> dict[str, ModelInfo]:
    """Merge /v1/models metadata with the /running process states."""
    running: dict[str, dict[str, Any]] = {
        entry["model"]: entry
        for entry in running_raw
        if isinstance(entry, dict) and isinstance(entry.get("model"), str)
    }

    models: dict[str, ModelInfo] = {}
    for record in models_raw:
        if not isinstance(record, dict):
            continue
        model_id = record.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue

        swap_meta = _llamaswap_meta(record)
        info = ModelInfo(
            id=model_id,
            name=_as_str(record.get("name")),
            description=_as_str(record.get("description")),
            kind=_as_str(swap_meta.get("type")) or "model",
            aliases=_as_str_list(swap_meta.get("aliases")),
            capabilities=_as_dict(record.get("capabilities")),
            architecture=_as_dict(record.get("architecture")),
            supported_parameters=_as_str_list(record.get("supported_parameters")),
            context_length=_as_int(record.get("context_length")),
            metadata=swap_meta,
        )

        proc = running.get(model_id)
        if proc is not None:
            info.state = _as_str(proc.get("state")) or STATE_STOPPED
            info.cmd = _as_str(proc.get("cmd")) or None
            info.proxy = _as_str(proc.get("proxy")) or None
            info.ttl = _as_int(proc.get("ttl"))
        elif _as_str(_as_dict(record.get("status")).get("value")) == "loaded":
            # Selectors and aliases report "loaded" without a /running row of
            # their own, because the process belongs to the target model.
            info.state = STATE_READY

        models[model_id] = info

    # Unlisted models are hidden from /v1/models but still show up in /running.
    for model_id, proc in running.items():
        if model_id in models:
            continue
        models[model_id] = ModelInfo(
            id=model_id,
            name=_as_str(proc.get("name")),
            description=_as_str(proc.get("description")),
            state=_as_str(proc.get("state")) or STATE_STOPPED,
            cmd=_as_str(proc.get("cmd")) or None,
            proxy=_as_str(proc.get("proxy")) or None,
            ttl=_as_int(proc.get("ttl")),
            unlisted=True,
        )

    return models


def _latest_gpu_stats(raw: Any) -> list[dict[str, Any]]:
    """Return the newest buffered sample for each GPU."""
    if not isinstance(raw, list):
        return []
    latest: dict[int, dict[str, Any]] = {}
    for sample in raw:
        if not isinstance(sample, dict):
            continue
        gpu_id = sample.get("id")
        if not isinstance(gpu_id, int) or isinstance(gpu_id, bool):
            continue
        # Samples arrive oldest first, so the last write per ID is the newest.
        latest[gpu_id] = sample
    return [latest[key] for key in sorted(latest)]


def _llamaswap_meta(record: dict[str, Any]) -> dict[str, Any]:
    """Return the meta.llamaswap block holding llama-swap's own fields."""
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return {}
    return _as_dict(meta.get("llamaswap"))


def _as_str(value: Any) -> str:
    """Coerce a JSON value to a string, treating anything else as empty."""
    return value.strip() if isinstance(value, str) else ""


def _as_int(value: Any) -> int | None:
    """Coerce a JSON value to an int, or None when it is not a whole number."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a JSON value to a dict, or an empty one."""
    return dict(value) if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    """Coerce a JSON value to a list of strings."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
