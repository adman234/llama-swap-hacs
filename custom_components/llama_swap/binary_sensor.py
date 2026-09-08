"""Binary sensor platform for llama-swap."""

from __future__ import annotations

from collections.abc import Iterable

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import LlamaSwapConfigEntry, LlamaSwapCoordinator
from .entity import (
    LlamaSwapEntity,
    LlamaSwapModelEntity,
    async_setup_model_entities,
)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LlamaSwapConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the llama-swap binary sensors."""
    coordinator = entry.runtime_data
    async_add_entities([LlamaSwapAnyModelLoaded(coordinator)])

    def _build(model_id: str) -> Iterable[Entity]:
        return (LlamaSwapModelLoaded(coordinator, model_id),)

    async_setup_model_entities(coordinator, async_add_entities, _build)


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


class LlamaSwapModelLoaded(LlamaSwapModelEntity, BinarySensorEntity):
    """Reports whether one model is loaded."""

    _attr_translation_key = "model_loaded"
    _attr_device_class = BinarySensorDeviceClass.RUNNING

    def __init__(self, coordinator: LlamaSwapCoordinator, model_id: str) -> None:
        """Initialise the binary sensor."""
        super().__init__(coordinator, model_id, "loaded")

    @property
    def is_on(self) -> bool | None:
        """Return True while the model holds a process."""
        model = self.model
        return model.is_loaded if model else None
