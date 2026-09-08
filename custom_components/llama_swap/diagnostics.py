"""Diagnostics support for llama-swap."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from .coordinator import LlamaSwapConfigEntry

# The upstream command line can embed local paths and secrets.
TO_REDACT = {CONF_API_KEY, "cmd"}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: LlamaSwapConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator = entry.runtime_data
    data = coordinator.data
    return {
        "entry": {
            "data": async_redact_data(dict(entry.data), TO_REDACT),
            "options": dict(entry.options),
        },
        "version": data.version,
        "active_profile": data.active_profile,
        "profiles": data.profiles,
        "performance_enabled": data.performance_enabled,
        "sys_stats": data.sys_stats,
        "gpu_stats": data.gpu_stats,
        "models": async_redact_data(
            {model_id: asdict(model) for model_id, model in data.models.items()},
            TO_REDACT,
        ),
    }
