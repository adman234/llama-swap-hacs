"""Select platform for llama-swap."""

from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import LlamaSwapError
from .coordinator import LlamaSwapConfigEntry, LlamaSwapCoordinator
from .entity import LlamaSwapEntity, async_setup_dynamic_entities

PARALLEL_UPDATES = 1

# Shown when no profile is active. llama-swap models this as a null profile.
PROFILE_NONE = "none"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LlamaSwapConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the profile selector once the server reports any profiles."""
    coordinator = entry.runtime_data
    async_setup_dynamic_entities(
        coordinator,
        async_add_entities,
        lambda data: ["profile"] if data.profiles else [],
        lambda _key: (LlamaSwapProfileSelect(coordinator),),
    )


class LlamaSwapProfileSelect(LlamaSwapEntity, SelectEntity):
    """Selects the active llama-swap profile."""

    _attr_translation_key = "profile"
    _attr_icon = "mdi:account-switch"

    def __init__(self, coordinator: LlamaSwapCoordinator) -> None:
        """Initialise the select."""
        super().__init__(coordinator, "profile")

    @property
    def options(self) -> list[str]:
        """Return the configured profiles, plus the no-profile option."""
        profiles = [
            profile["id"]
            for profile in self.coordinator.data.profiles
            if isinstance(profile, dict) and isinstance(profile.get("id"), str)
        ]
        return [PROFILE_NONE, *profiles]

    @property
    def current_option(self) -> str:
        """Return the active profile."""
        return self.coordinator.data.active_profile or PROFILE_NONE

    async def async_select_option(self, option: str) -> None:
        """Activate a profile, or deactivate the current one."""
        profile = None if option == PROFILE_NONE else option
        try:
            await self.coordinator.client.async_set_profile(profile)
        except LlamaSwapError as err:
            raise HomeAssistantError(f"Failed to set profile: {err}") from err
        await self.coordinator.async_request_refresh()
