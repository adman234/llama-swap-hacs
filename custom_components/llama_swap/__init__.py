"""The llama-swap integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    CONF_HOST,
    CONF_PORT,
    CONF_SCAN_INTERVAL,
    CONF_SSL,
    Platform,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import LlamaSwapClient
from .const import (
    CONF_VERIFY_SSL,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_SSL,
    DEFAULT_VERIFY_SSL,
    DOMAIN,
)
from .coordinator import LlamaSwapConfigEntry, LlamaSwapCoordinator
from .entity import current_device_identifiers
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.BUTTON,
    Platform.SELECT,
    Platform.SENSOR,
    Platform.SWITCH,
]


async def async_setup_entry(hass: HomeAssistant, entry: LlamaSwapConfigEntry) -> bool:
    """Set up llama-swap from a config entry."""
    client = LlamaSwapClient(
        async_get_clientsession(hass),
        entry.data[CONF_HOST],
        entry.data[CONF_PORT],
        use_ssl=entry.data.get(CONF_SSL, DEFAULT_SSL),
        verify_ssl=entry.data.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL),
        api_key=entry.data.get(CONF_API_KEY),
    )

    coordinator = LlamaSwapCoordinator(
        hass,
        entry,
        client,
        entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
    )
    await coordinator.async_config_entry_first_refresh()

    entry.runtime_data = coordinator
    async_setup_services(hass)
    _async_remove_stale_devices(hass, coordinator)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


@callback
def _async_remove_stale_devices(
    hass: HomeAssistant, coordinator: LlamaSwapCoordinator
) -> None:
    """Drop devices for models and GPUs the server no longer has.

    Only done at setup. Pruning on every poll would delete a model's device,
    and its history, during the moments a llama-swap config reload leaves it
    out of the listing.
    """
    device_registry = dr.async_get(hass)
    current = current_device_identifiers(coordinator)
    entry_id = coordinator.config_entry.entry_id

    for device in dr.async_entries_for_config_entry(device_registry, entry_id):
        if any(
            domain == DOMAIN and identifier in current
            for domain, identifier in device.identifiers
        ):
            continue
        _LOGGER.debug(
            "Removing device for a model llama-swap no longer has: %s", device.name
        )
        device_registry.async_update_device(device.id, remove_config_entry_id=entry_id)


async def async_remove_config_entry_device(
    hass: HomeAssistant, entry: LlamaSwapConfigEntry, device: dr.DeviceEntry
) -> bool:
    """Allow deleting a device the server no longer reports.

    A device that is still current cannot be deleted, since the next poll
    would only recreate it.
    """
    current = current_device_identifiers(entry.runtime_data)
    return not any(
        domain == DOMAIN and identifier in current
        for domain, identifier in device.identifiers
    )


async def async_unload_entry(hass: HomeAssistant, entry: LlamaSwapConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)
