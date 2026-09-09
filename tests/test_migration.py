"""Tests for removing the duplicate per-model entities on existing installs."""

from __future__ import annotations

from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap.const import CONF_VERIFY_SSL, DOMAIN
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .conftest import setup_integration


def _old_entry(minor_version: int = 1) -> MockConfigEntry:
    """Return a config entry as an earlier version would have written it."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="llama.local:8080",
        minor_version=minor_version,
        data={
            CONF_HOST: "llama.local",
            CONF_PORT: 8080,
            CONF_SSL: False,
            CONF_VERIFY_SSL: True,
        },
    )


def _register_legacy_entities(
    hass: HomeAssistant, entry: MockConfigEntry
) -> dict[str, str]:
    """Register the entities an older version created, all enabled."""
    entity_registry = er.async_get(hass)
    created: dict[str, str] = {}
    for domain, key, object_id in (
        ("binary_sensor", "loaded", "qwen3_coder_30b_loaded"),
        ("button", "unload", "qwen3_coder_30b_unload"),
    ):
        registry_entry = entity_registry.async_get_or_create(
            domain,
            DOMAIN,
            f"{entry.entry_id}_model_qwen3-coder_{key}",
            suggested_object_id=object_id,
            config_entry=entry,
        )
        created[key] = registry_entry.entity_id
    return created


async def test_existing_duplicates_are_removed(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker
) -> None:
    """An install that already had them loses them outright.

    Nothing creates these entities any more, so leaving the registry entries
    behind would strand them in the UI as permanently unavailable.
    """
    entry = _old_entry()
    entry.add_to_hass(hass)
    legacy = _register_legacy_entities(hass, entry)

    entity_registry = er.async_get(hass)
    for entity_id in legacy.values():
        assert entity_registry.async_get(entity_id) is not None

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    for entity_id in legacy.values():
        assert entity_registry.async_get(entity_id) is None
        assert hass.states.get(entity_id) is None

    assert entry.minor_version == 3

    # What replaces them is still there.
    assert hass.states.get("switch.qwen3_coder_30b_loaded").state == STATE_ON
    assert hass.states.get("sensor.qwen3_coder_30b_state").state == "ready"


async def test_already_disabled_entries_are_removed(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker
) -> None:
    """Installs that stopped at the disabling step get cleaned up too."""
    entry = _old_entry(minor_version=2)
    entry.add_to_hass(hass)
    legacy = _register_legacy_entities(hass, entry)

    entity_registry = er.async_get(hass)
    for entity_id in legacy.values():
        entity_registry.async_update_entity(
            entity_id, disabled_by=er.RegistryEntryDisabler.INTEGRATION
        )

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    for entity_id in legacy.values():
        assert entity_registry.async_get(entity_id) is None
    assert entry.minor_version == 3


async def test_server_equivalents_survive(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker
) -> None:
    """The server-wide binary sensor and unload button are not caught.

    Their unique IDs end the same way, so the match has to be anchored on the
    per-model prefix rather than the suffix alone.
    """
    entry = _old_entry()
    entry.add_to_hass(hass)
    entity_registry = er.async_get(hass)
    server_loaded = entity_registry.async_get_or_create(
        "binary_sensor",
        DOMAIN,
        f"{entry.entry_id}_any_model_loaded",
        suggested_object_id="llama_local_8080_model_loaded",
        config_entry=entry,
    )
    server_unload = entity_registry.async_get_or_create(
        "button",
        DOMAIN,
        f"{entry.entry_id}_unload_all",
        suggested_object_id="llama_local_8080_unload_all_models",
        config_entry=entry,
    )

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    for entity_id in (server_loaded.entity_id, server_unload.entity_id):
        assert entity_registry.async_get(entity_id) is not None
        assert hass.states.get(entity_id) is not None


async def test_removal_does_not_come_back(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker
) -> None:
    """Nothing recreates them on a later reload."""
    entry = _old_entry()
    entry.add_to_hass(hass)
    legacy = _register_legacy_entities(hass, entry)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entity_registry = er.async_get(hass)
    for entity_id in legacy.values():
        assert entity_registry.async_get(entity_id) is None


async def test_new_install_never_creates_them(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """A fresh entry has no per-model binary sensor or unload button at all."""
    await setup_integration(hass, config_entry)

    assert config_entry.minor_version == 3
    entity_registry = er.async_get(hass)
    assert entity_registry.async_get("binary_sensor.qwen3_coder_30b_loaded") is None
    assert entity_registry.async_get("button.qwen3_coder_30b_unload") is None

    # The per-model switch and state sensor remain.
    assert entity_registry.async_get("switch.qwen3_coder_30b_loaded") is not None
    assert entity_registry.async_get("sensor.qwen3_coder_30b_state") is not None
