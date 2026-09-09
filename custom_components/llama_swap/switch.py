"""Switch platform for llama-swap."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from homeassistant.components.switch import SwitchDeviceClass, SwitchEntity
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import LlamaSwapError
from .const import STATE_READY, STATE_STARTING
from .coordinator import LlamaSwapConfigEntry, LlamaSwapCoordinator
from .entity import LlamaSwapModelEntity, async_setup_model_entities

PARALLEL_UPDATES = 1


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LlamaSwapConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the llama-swap model switches."""
    coordinator = entry.runtime_data

    def _build(model_id: str) -> Iterable[Entity]:
        return (LlamaSwapModelSwitch(coordinator, model_id),)

    async_setup_model_entities(coordinator, async_add_entities, _build)


class LlamaSwapModelSwitch(LlamaSwapModelEntity, SwitchEntity):
    """Loads and unloads one model.

    Turning the switch on issues a request routed to the model, which is what
    makes llama-swap swap it in. Large models take a while to load, so the call
    only returns once the upstream server is up.
    """

    _attr_translation_key = "model_switch"
    _attr_device_class = SwitchDeviceClass.SWITCH

    def __init__(self, coordinator: LlamaSwapCoordinator, model_id: str) -> None:
        """Initialise the switch."""
        super().__init__(coordinator, model_id, "switch")

    @property
    def is_on(self) -> bool | None:
        """Return True while the model holds a process."""
        model = self.model
        return model.is_loaded if model else None

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Load the model."""
        try:
            await self.coordinator.client.async_load_model(self.model_id)
        except LlamaSwapError as err:
            raise HomeAssistantError(f"Failed to load {self.model_id}: {err}") from err
        await self.coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Unload the model."""
        try:
            await self.coordinator.client.async_unload_model(self.model_id)
        except LlamaSwapError as err:
            raise HomeAssistantError(
                f"Failed to unload {self.model_id}: {err}"
            ) from err
        await self.coordinator.async_request_refresh()

    @property
    def icon(self) -> str:
        """Return an icon reflecting the load state."""
        model = self.model
        if model is None:
            return "mdi:help-circle-outline"
        if model.state == STATE_STARTING:
            return "mdi:progress-upload"
        if model.state == STATE_READY:
            return "mdi:brain"
        return "mdi:sleep"
