"""Tests for setup, entities and services."""

from __future__ import annotations

from typing import Any

import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap.const import DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import device_registry as dr, entity_registry as er

from .conftest import BASE_URL, setup_integration


def _put_bodies(mock: AiohttpClientMocker, path: str) -> list[Any]:
    """Return the JSON bodies of every PUT made to a path."""
    return [
        call[2]
        for call in mock.mock_calls
        if call[0].lower() == "put" and str(call[1]).endswith(path)
    ]


async def test_setup_and_unload(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The entry loads and unloads cleanly."""
    await setup_integration(hass, config_entry)
    assert config_entry.state is ConfigEntryState.LOADED

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.NOT_LOADED


async def test_server_sensors(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """Whole-server sensors report the merged model state."""
    await setup_integration(hass, config_entry)

    active = hass.states.get("sensor.llama_local_8080_active_model")
    assert active is not None
    # "auto" sorts before "qwen3-coder" and is loaded via its pinned target.
    assert active.state == "auto"
    assert active.attributes["loaded_models"] == ["auto", "qwen3-coder"]
    assert active.attributes["loaded_count"] == 2

    assert hass.states.get("sensor.llama_local_8080_loaded_models").state == "2"
    assert hass.states.get("sensor.llama_local_8080_configured_models").state == "3"
    assert (
        hass.states.get("binary_sensor.llama_local_8080_model_loaded").state == STATE_ON
    )


async def test_model_entities(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """Each model gets its own device with state, details and controls."""
    await setup_integration(hass, config_entry)

    state = hass.states.get("sensor.qwen3_coder_30b_state")
    assert state is not None
    assert state.state == "ready"
    attrs = state.attributes
    assert attrs["model_id"] == "qwen3-coder"
    assert attrs["description"] == "Coding model"
    assert attrs["aliases"] == ["coder"]
    assert attrs["capabilities"] == {"function_calling": True}
    assert attrs["architecture"]["modality"] == "text->text"
    assert attrs["supported_parameters"] == ["tools", "tool_choice"]
    assert attrs["context_length"] == 65536
    assert attrs["ttl"] == 300
    assert attrs["proxy"] == "http://127.0.0.1:9001"
    assert attrs["metadata"] == {"tier": "primary"}
    assert attrs["loaded"] is True

    assert hass.states.get("binary_sensor.qwen3_coder_30b_loaded").state == STATE_ON
    assert hass.states.get("switch.qwen3_coder_30b_loaded").state == STATE_ON
    assert hass.states.get("sensor.qwen3_coder_30b_context_length").state == "65536"

    # An unloaded model reports stopped.
    assert hass.states.get("sensor.gemma_3_vision_state").state == "stopped"
    assert hass.states.get("switch.gemma_3_vision_loaded").state == STATE_OFF

    # A selector inherits its target's loaded status without a /running row.
    assert hass.states.get("sensor.auto_state").state == "ready"

    device_registry = dr.async_get(hass)
    device = device_registry.async_get_device(
        identifiers={(DOMAIN, f"{config_entry.entry_id}_model_qwen3-coder")}
    )
    assert device is not None
    assert device.name == "Qwen3 Coder 30B"


async def test_context_length_missing_is_unknown(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """A model without a declared context length reports unknown, not zero."""
    await setup_integration(hass, config_entry)
    assert hass.states.get("sensor.auto_context_length").state == STATE_UNKNOWN


async def test_performance_sensors(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """System and GPU sensors use the newest buffered sample."""
    await setup_integration(hass, config_entry)

    assert hass.states.get("sensor.llama_local_8080_cpu_utilization").state == "40.0"
    assert hass.states.get("sensor.llama_local_8080_memory_utilization").state == "50.0"
    assert hass.states.get("sensor.llama_local_8080_load_average_1m").state == "1.5"

    gpu_temp = hass.states.get("sensor.nvidia_geforce_rtx_4090_temperature")
    assert gpu_temp is not None
    assert gpu_temp.state == "66"
    assert hass.states.get("sensor.nvidia_geforce_rtx_4090_utilization").state == "92.0"
    assert hass.states.get("sensor.nvidia_geforce_rtx_4090_power_draw").state == "310.5"

    device_registry = dr.async_get(hass)
    assert device_registry.async_get_device(
        identifiers={(DOMAIN, f"{config_entry.entry_id}_gpu_0")}
    )


async def test_memory_used_converts_to_gigabytes(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """Memory is reported in MB and displayed in GB."""
    await setup_integration(hass, config_entry)
    state = hass.states.get("sensor.llama_local_8080_memory_used")
    assert state is not None
    assert float(state.state) == pytest.approx(16.0, abs=0.1)


async def test_profile_select(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The profile select mirrors and changes the active profile."""
    await setup_integration(hass, config_entry)

    state = hass.states.get("select.llama_local_8080_profile")
    assert state is not None
    assert state.state == "coding"
    assert state.attributes["options"] == ["none", "coding", "low-power"]

    mock_llama_swap.put(f"{BASE_URL}/api/profiles/active", json={"active": "low-power"})
    await hass.services.async_call(
        "select",
        "select_option",
        {ATTR_ENTITY_ID: "select.llama_local_8080_profile", "option": "low-power"},
        blocking=True,
    )
    assert _put_bodies(mock_llama_swap, "/api/profiles/active") == [
        {"name": "low-power"}
    ]


async def test_switch_loads_and_unloads(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The model switch drives the upstream load and the unload endpoint."""
    await setup_integration(hass, config_entry)

    mock_llama_swap.get(f"{BASE_URL}/upstream/gemma3-vision/health", text="OK")
    await hass.services.async_call(
        "switch",
        "turn_on",
        {ATTR_ENTITY_ID: "switch.gemma_3_vision_loaded"},
        blocking=True,
    )
    assert any(
        str(call[1]).endswith("/upstream/gemma3-vision/health")
        for call in mock_llama_swap.mock_calls
    )

    mock_llama_swap.post(f"{BASE_URL}/api/models/unload/qwen3-coder", text="OK")
    await hass.services.async_call(
        "switch",
        "turn_off",
        {ATTR_ENTITY_ID: "switch.qwen3_coder_30b_loaded"},
        blocking=True,
    )
    assert any(
        str(call[1]).endswith("/api/models/unload/qwen3-coder")
        for call in mock_llama_swap.mock_calls
    )


async def test_unload_all_button(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The unload-all button hits /unload."""
    await setup_integration(hass, config_entry)
    mock_llama_swap.get(f"{BASE_URL}/unload", text="OK")

    await hass.services.async_call(
        "button",
        "press",
        {ATTR_ENTITY_ID: "button.llama_local_8080_unload_all_models"},
        blocking=True,
    )
    assert any(str(call[1]).endswith("/unload") for call in mock_llama_swap.mock_calls)


async def test_services(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The integration's services reach the right endpoints."""
    await setup_integration(hass, config_entry)

    mock_llama_swap.get(f"{BASE_URL}/upstream/gemma3-vision/health", text="OK")
    await hass.services.async_call(
        DOMAIN, "load_model", {"model": "gemma3-vision"}, blocking=True
    )

    mock_llama_swap.post(f"{BASE_URL}/api/models/unload/qwen3-coder", text="OK")
    await hass.services.async_call(
        DOMAIN, "unload_model", {"model": "qwen3-coder"}, blocking=True
    )

    mock_llama_swap.get(f"{BASE_URL}/unload", text="OK")
    await hass.services.async_call(DOMAIN, "unload_all", {}, blocking=True)

    mock_llama_swap.put(f"{BASE_URL}/api/profiles/active", json={"active": None})
    await hass.services.async_call(DOMAIN, "set_profile", {}, blocking=True)
    assert _put_bodies(mock_llama_swap, "/api/profiles/active") == [{"name": None}]


async def test_load_unknown_model_raises(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """Loading a model llama-swap does not know is a validation error."""
    await setup_integration(hass, config_entry)
    mock_llama_swap.get(
        f"{BASE_URL}/upstream/nope/health", status=404, text="model not found"
    )

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "load_model", {"model": "nope"}, blocking=True
        )


async def test_upstream_404_still_counts_as_loaded(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """A 404 from the upstream server itself is not a load failure."""
    await setup_integration(hass, config_entry)
    mock_llama_swap.get(
        f"{BASE_URL}/upstream/gemma3-vision/health",
        status=404,
        text="404 page not found",
    )

    await hass.services.async_call(
        DOMAIN, "load_model", {"model": "gemma3-vision"}, blocking=True
    )


async def test_service_error_raises(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """A server-side failure surfaces as a Home Assistant error."""
    await setup_integration(hass, config_entry)
    mock_llama_swap.get(f"{BASE_URL}/unload", status=500, text="boom")

    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(DOMAIN, "unload_all", {}, blocking=True)


async def test_entity_ids_are_stable(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """Unique IDs are namespaced per entry, model and key."""
    await setup_integration(hass, config_entry)
    entity_registry = er.async_get(hass)

    entry = entity_registry.async_get("sensor.qwen3_coder_30b_state")
    assert entry is not None
    assert entry.unique_id == f"{config_entry.entry_id}_model_qwen3-coder_state"

    entry = entity_registry.async_get("sensor.llama_local_8080_active_model")
    assert entry is not None
    assert entry.unique_id == f"{config_entry.entry_id}_active_model"
