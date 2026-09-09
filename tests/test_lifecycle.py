"""Tests for the 404 handling, device cleanup and late-appearing entities."""

from __future__ import annotations

from typing import Any

from freezegun.api import FrozenDateTimeFactory
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap import async_remove_config_entry_device
from custom_components.llama_swap.const import (
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    NO_MODEL,
)
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr

from .conftest import BASE_URL, async_advance, setup_integration
from .test_coordinator import _mock_legacy_server

# What llama-swap's own error envelope looks like for a refused model, versus
# what Go's mux writes for a route that does not exist at all.
PEER_REFUSED = '{"error":{"message":"no local server found for requested model"}}'
MODEL_REFUSED = '{"error":{"message":"model not found"}}'
ROUTE_ABSENT = "404 page not found\n"


def _paths(mock: AiohttpClientMocker) -> list[str]:
    """Return the path of every request made."""
    return [str(call[1].path) for call in mock.mock_calls]


@pytest.mark.parametrize("body", [PEER_REFUSED, MODEL_REFUSED])
async def test_refused_unload_does_not_unload_everything(
    hass: HomeAssistant,
    mock_llama_swap: AiohttpClientMocker,
    config_entry,
    body: str,
) -> None:
    """A model the server refuses must never trigger a server-wide unload.

    llama-swap answers 404 both for an unknown model and for any peer model,
    and falling back to /unload there would evict every loaded model.
    """
    await setup_integration(hass, config_entry)
    mock_llama_swap.post(
        f"{BASE_URL}/api/models/unload/qwen3-coder", status=404, text=body
    )
    mock_llama_swap.get(f"{BASE_URL}/unload", text="OK")

    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            DOMAIN, "unload_model", {"model": "qwen3-coder"}, blocking=True
        )

    assert "/unload" not in _paths(mock_llama_swap)


async def test_absent_endpoint_still_falls_back(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """A build without the per-model endpoint still unloads everything."""
    await setup_integration(hass, config_entry)
    mock_llama_swap.post(
        f"{BASE_URL}/api/models/unload/qwen3-coder", status=404, text=ROUTE_ABSENT
    )
    mock_llama_swap.get(f"{BASE_URL}/unload", text="OK")

    await hass.services.async_call(
        DOMAIN, "unload_model", {"model": "qwen3-coder"}, blocking=True
    )
    assert "/unload" in _paths(mock_llama_swap)


async def test_version_fetched_once(
    hass: HomeAssistant,
    mock_llama_swap: AiohttpClientMocker,
    config_entry,
    freezer: FrozenDateTimeFactory,
) -> None:
    """The build only changes on restart, so it is not polled every cycle."""
    await setup_integration(hass, config_entry)

    def _version_calls() -> int:
        return _paths(mock_llama_swap).count("/api/version")

    assert _version_calls() == 1
    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)
    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)
    assert _version_calls() == 1
    assert hass.states.get("sensor.llama_local_8080_active_model") is not None


async def test_gpu_entities_appear_later(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    performance_payload: dict[str, Any],
    profiles_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """Turning on performance monitoring does not need a reload.

    llama-swap answers 503 {"enabled": false} while monitoring is off and 200
    with an empty list when no profiles exist, so the endpoints keep being
    polled and the entities can appear the moment they have something to show.
    """
    aioclient_mock.get(f"{BASE_URL}/v1/models", json=models_payload)
    aioclient_mock.get(f"{BASE_URL}/running", json=running_payload)
    aioclient_mock.get(f"{BASE_URL}/api/version", status=404, text=ROUTE_ABSENT)
    aioclient_mock.get(
        f"{BASE_URL}/api/performance", status=503, json={"enabled": False}
    )
    aioclient_mock.get(
        f"{BASE_URL}/api/profiles", json={"active": None, "profiles": []}
    )
    await setup_integration(hass, config_entry)

    assert hass.states.get("sensor.nvidia_geforce_rtx_4090_temperature") is None
    assert hass.states.get("sensor.llama_local_8080_cpu_utilization") is None
    assert hass.states.get("select.llama_local_8080_profile") is None

    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE_URL}/v1/models", json=models_payload)
    aioclient_mock.get(f"{BASE_URL}/running", json=running_payload)
    aioclient_mock.get(f"{BASE_URL}/api/version", status=404, text=ROUTE_ABSENT)
    aioclient_mock.get(f"{BASE_URL}/api/performance", json=performance_payload)
    aioclient_mock.get(f"{BASE_URL}/api/profiles", json=profiles_payload)

    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert hass.states.get("sensor.nvidia_geforce_rtx_4090_temperature").state == "66"
    assert hass.states.get("sensor.llama_local_8080_cpu_utilization").state == "40.0"
    assert hass.states.get("select.llama_local_8080_profile").state == "coding"


async def test_stale_device_removed_on_reload(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
) -> None:
    """A model dropped from llama-swap's config loses its device on reload."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)

    device_registry = dr.async_get(hass)
    gone = (DOMAIN, f"{config_entry.entry_id}_model_gemma3-vision")
    assert device_registry.async_get_device(identifiers={gone}) is not None

    models_payload["data"] = [
        record for record in models_payload["data"] if record["id"] != "gemma3-vision"
    ]
    aioclient_mock.clear_requests()
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)

    await hass.config_entries.async_reload(config_entry.entry_id)
    await hass.async_block_till_done()

    assert device_registry.async_get_device(identifiers={gone}) is None
    # The models that remain keep their devices.
    assert device_registry.async_get_device(
        identifiers={(DOMAIN, f"{config_entry.entry_id}_model_qwen3-coder")}
    )


async def test_device_removal_permission(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """A live device cannot be deleted; an orphan can."""
    await setup_integration(hass, config_entry)
    device_registry = dr.async_get(hass)

    live = device_registry.async_get_device(
        identifiers={(DOMAIN, f"{config_entry.entry_id}_model_qwen3-coder")}
    )
    assert live is not None
    assert not await async_remove_config_entry_device(hass, config_entry, live)

    orphan = device_registry.async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={(DOMAIN, f"{config_entry.entry_id}_model_long-gone")},
        name="Long Gone",
    )
    assert await async_remove_config_entry_device(hass, config_entry, orphan)


async def test_model_facts_are_bounded(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """Remembered command details are dropped once a model is gone for good."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)
    coordinator = config_entry.runtime_data
    assert "qwen3-coder" in coordinator._model_facts

    aioclient_mock.clear_requests()
    models_payload["data"] = [
        record for record in models_payload["data"] if record["id"] != "qwen3-coder"
    ]
    _mock_legacy_server(aioclient_mock, models_payload, {"running": []})
    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert "qwen3-coder" not in coordinator._model_facts


async def test_empty_listing_keeps_facts(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """A momentarily empty listing must not throw the cache away.

    llama-swap reports nothing for a beat while it reloads its config, and
    losing the cache there would blank the context length sensors.
    """
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)
    coordinator = config_entry.runtime_data

    aioclient_mock.clear_requests()
    _mock_legacy_server(aioclient_mock, {"object": "list", "data": []}, {"running": []})
    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert "qwen3-coder" in coordinator._model_facts


async def test_large_attributes_are_not_recorded(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The per-model dump stays out of the recorder database.

    It can run to kilobytes and changes whenever any model does; past 16KB the
    recorder discards the attributes altogether.
    """
    await setup_integration(hass, config_entry)

    for entity_id in (
        "sensor.llama_local_8080_active_model",
        "sensor.llama_local_8080_configured_models",
    ):
        state = hass.states.get(entity_id)
        assert "models" in state.attributes
        assert state.state_info is not None
        assert "models" in state.state_info["unrecorded_attributes"]

    # The count sensor no longer repeats the whole payload.
    loaded = hass.states.get("sensor.llama_local_8080_loaded_models")
    assert "models" not in loaded.attributes
    assert loaded.attributes["loaded_models"] == ["auto", "qwen3-coder"]


async def test_idle_server_is_not_unknown(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
) -> None:
    """Nothing loaded reads "none"; an unreachable server reads unavailable."""
    for record in models_payload["data"]:
        record["status"] = {"value": "unloaded"}
    _mock_legacy_server(aioclient_mock, models_payload, {"running": []})
    await setup_integration(hass, config_entry)

    state = hass.states.get("sensor.llama_local_8080_active_model")
    assert state.state == NO_MODEL
    assert state.state != STATE_UNKNOWN
    assert state.attributes["loaded_models"] == []
    assert state.attributes["loaded_count"] == 0
