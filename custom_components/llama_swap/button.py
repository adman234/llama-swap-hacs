"""Button platform for llama-swap."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import LlamaSwapError
from .coordinator import LlamaSwapConfigEntry, LlamaSwapCoordinator
from .entity import LlamaSwapEntity

PARALLEL_UPDATES = 1


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LlamaSwapConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the llama-swap buttons."""
    async_add_entities([LlamaSwapUnloadAllButton(entry.runtime_data)])


class LlamaSwapUnloadAllButton(LlamaSwapEntity, ButtonEntity):
    """Stops every running model."""

    _attr_translation_key = "unload_all"
    _attr_icon = "mdi:eject"

    def __init__(self, coordinator: LlamaSwapCoordinator) -> None:
        """Initialise the button."""
        super().__init__(coordinator, "unload_all")

    async def async_press(self) -> None:
        """Unload every model."""
        try:
            await self.coordinator.client.async_unload_all()
        except LlamaSwapError as err:
            raise HomeAssistantError(f"Failed to unload models: {err}") from err
        await self.coordinator.async_request_refresh()
