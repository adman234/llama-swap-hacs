"""Binary sensor platform for llama-swap."""

from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import LlamaSwapConfigEntry, LlamaSwapCoordinator
from .entity import LlamaSwapEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LlamaSwapConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the llama-swap binary sensors."""
    async_add_entities([LlamaSwapAnyModelLoaded(entry.runtime_data)])


class LlamaSwapAnyModelLoaded(LlamaSwapEntity, BinarySensorEntity):
    """Reports whether the server currently has any model loaded."""

    _attr_translation_key = "any_model_loaded"
    _attr_icon = "mdi:chip"

    def __init__(self, coordinator: LlamaSwapCoordinator) -> None:
        """Initialise the binary sensor."""
        super().__init__(coordinator, "any_model_loaded")

    @property
    def is_on(self) -> bool:
        """Return True when at least one model holds a process."""
        return bool(self.coordinator.data.loaded_models)
