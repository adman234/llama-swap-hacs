"""Services for the llama-swap integration."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv

from .api import LlamaSwapError, LlamaSwapNotFoundError
from .const import (
    ATTR_MODEL,
    ATTR_PROFILE,
    DOMAIN,
    SERVICE_LOAD_MODEL,
    SERVICE_SET_PROFILE,
    SERVICE_UNLOAD_ALL,
    SERVICE_UNLOAD_MODEL,
)
from .coordinator import LlamaSwapConfigEntry, LlamaSwapCoordinator

CONF_ENTRY_ID = "config_entry_id"

_BASE_SCHEMA = {vol.Optional(CONF_ENTRY_ID): cv.string}

LOAD_MODEL_SCHEMA = vol.Schema({**_BASE_SCHEMA, vol.Required(ATTR_MODEL): cv.string})
UNLOAD_MODEL_SCHEMA = LOAD_MODEL_SCHEMA
UNLOAD_ALL_SCHEMA = vol.Schema(_BASE_SCHEMA)
SET_PROFILE_SCHEMA = vol.Schema(
    {**_BASE_SCHEMA, vol.Optional(ATTR_PROFILE): vol.Any(cv.string, None)}
)


def _coordinator(hass: HomeAssistant, call: ServiceCall) -> LlamaSwapCoordinator:
    """Resolve which llama-swap instance a service call targets.

    With a single configured server the entry ID may be omitted; with several,
    the call has to say which one.
    """
    entries: list[LlamaSwapConfigEntry] = [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.state is ConfigEntryState.LOADED
    ]

    entry_id = call.data.get(CONF_ENTRY_ID)
    if entry_id is not None:
        for entry in entries:
            if entry.entry_id == entry_id:
                return entry.runtime_data
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entry_not_found",
            translation_placeholders={"entry_id": entry_id},
        )

    if not entries:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="no_entries"
        )
    if len(entries) > 1:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="entry_id_required"
        )
    return entries[0].runtime_data


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register the integration's services, once per Home Assistant run."""
    if hass.services.has_service(DOMAIN, SERVICE_UNLOAD_ALL):
        return

    async def _load_model(call: ServiceCall) -> None:
        coordinator = _coordinator(hass, call)
        model = call.data[ATTR_MODEL]
        try:
            await coordinator.client.async_load_model(model)
        except LlamaSwapNotFoundError as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="unknown_model",
                translation_placeholders={"model": model},
            ) from err
        except LlamaSwapError as err:
            raise HomeAssistantError(str(err)) from err
        await coordinator.async_request_refresh()

    async def _unload_model(call: ServiceCall) -> None:
        coordinator = _coordinator(hass, call)
        try:
            await coordinator.client.async_unload_model(call.data[ATTR_MODEL])
        except LlamaSwapError as err:
            raise HomeAssistantError(str(err)) from err
        await coordinator.async_request_refresh()

    async def _unload_all(call: ServiceCall) -> None:
        coordinator = _coordinator(hass, call)
        try:
            await coordinator.client.async_unload_all()
        except LlamaSwapError as err:
            raise HomeAssistantError(str(err)) from err
        await coordinator.async_request_refresh()

    async def _set_profile(call: ServiceCall) -> None:
        coordinator = _coordinator(hass, call)
        profile = call.data.get(ATTR_PROFILE) or None
        try:
            await coordinator.client.async_set_profile(profile)
        except LlamaSwapNotFoundError as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="unknown_profile",
                translation_placeholders={"profile": profile or ""},
            ) from err
        except LlamaSwapError as err:
            raise HomeAssistantError(str(err)) from err
        await coordinator.async_request_refresh()

    hass.services.async_register(
        DOMAIN, SERVICE_LOAD_MODEL, _load_model, schema=LOAD_MODEL_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_UNLOAD_MODEL, _unload_model, schema=UNLOAD_MODEL_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_UNLOAD_ALL, _unload_all, schema=UNLOAD_ALL_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_SET_PROFILE, _set_profile, schema=SET_PROFILE_SCHEMA
    )
