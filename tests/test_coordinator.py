"""Tests for polling behaviour and older llama-swap builds."""

from __future__ import annotations

from typing import Any

import aiohttp
from freezegun.api import FrozenDateTimeFactory
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap.const import DEFAULT_SCAN_INTERVAL
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant

from .conftest import BASE_URL, async_advance, setup_integration


def _mock_legacy_server(
    mock: AiohttpClientMocker,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
) -> None:
    """Mock a llama-swap build without the /api endpoints."""
    mock.get(f"{BASE_URL}/v1/models", json=models_payload)
    mock.get(f"{BASE_URL}/running", json=running_payload)
    mock.get(f"{BASE_URL}/api/version", status=404, text="404 page not found")
    mock.get(f"{BASE_URL}/api/performance", status=404, text="404 page not found")
    mock.get(f"{BASE_URL}/api/profiles", status=404, text="404 page not found")


async def test_legacy_server_still_works(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
) -> None:
    """Servers without /api/* still get the model entities."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)

    assert config_entry.state is ConfigEntryState.LOADED
    assert hass.states.get("sensor.qwen3_coder_30b_state").state == "ready"
    assert hass.states.get("sensor.llama_local_8080_loaded_models").state == "2"

    # Optional entities are simply not created.
    assert hass.states.get("select.llama_local_8080_profile") is None
    assert hass.states.get("sensor.llama_local_8080_cpu_utilization") is None
    assert hass.states.get("sensor.nvidia_geforce_rtx_4090_temperature") is None


async def test_missing_endpoints_probed_once(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """A 404 on an optional endpoint stops it being polled again."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)

    def _version_calls() -> int:
        return sum(
            1
            for call in aioclient_mock.mock_calls
            if str(call[1]).endswith("/api/version")
        )

    assert _version_calls() == 1

    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert _version_calls() == 1
    # The core endpoints are still polled.
    assert hass.states.get("sensor.qwen3_coder_30b_state").state == "ready"


async def test_state_updates_on_poll(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """A model unloading upstream is reflected on the next poll."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)
    assert hass.states.get("binary_sensor.qwen3_coder_30b_loaded").state == STATE_ON

    aioclient_mock.clear_requests()
    for record in models_payload["data"]:
        record["status"] = {"value": "unloaded"}
    _mock_legacy_server(aioclient_mock, models_payload, {"running": []})

    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert hass.states.get("binary_sensor.qwen3_coder_30b_loaded").state == STATE_OFF
    assert hass.states.get("sensor.qwen3_coder_30b_state").state == "stopped"
    assert hass.states.get("sensor.llama_local_8080_loaded_models").state == "0"
    assert (
        hass.states.get("binary_sensor.llama_local_8080_model_loaded").state
        == STATE_OFF
    )


async def test_new_model_gets_entities(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """A model added to llama-swap's config shows up without a reload."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)
    assert hass.states.get("sensor.whisper_state") is None

    models_payload["data"].append(
        {
            "id": "whisper",
            "object": "model",
            "name": "Whisper",
            "meta": {"llamaswap": {"type": "model"}},
            "status": {"value": "unloaded"},
        }
    )
    aioclient_mock.clear_requests()
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)

    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert hass.states.get("sensor.whisper_state").state == "stopped"
    assert hass.states.get("switch.whisper_loaded").state == STATE_OFF


async def test_unlisted_running_model_appears(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    running_payload: dict[str, Any],
) -> None:
    """A model hidden from /v1/models is still picked up from /running."""
    _mock_legacy_server(aioclient_mock, {"object": "list", "data": []}, running_payload)
    await setup_integration(hass, config_entry)

    state = hass.states.get("sensor.qwen3_coder_30b_state")
    assert state is not None
    assert state.state == "ready"
    assert state.attributes["unlisted"] is True


async def test_entities_go_unavailable_on_failure(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """Losing the server marks entities unavailable rather than stale."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)
    assert hass.states.get("sensor.qwen3_coder_30b_state").state == "ready"

    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE_URL}/v1/models", exc=aiohttp.ClientError)
    aioclient_mock.get(f"{BASE_URL}/running", exc=aiohttp.ClientError)

    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert hass.states.get("sensor.qwen3_coder_30b_state").state == STATE_UNAVAILABLE


async def test_auth_failure_starts_reauth(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """The API key being revoked starts a reauth flow."""
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)

    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE_URL}/v1/models", status=401, text="unauthorized")
    aioclient_mock.get(f"{BASE_URL}/running", status=401, text="unauthorized")

    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    flows = hass.config_entries.flow.async_progress_by_handler("llama_swap")
    assert [flow["context"]["source"] for flow in flows] == ["reauth"]


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [(None, False), ("secret", True)],
)
async def test_api_key_is_sent(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    api_key: str | None,
    expected: bool,
) -> None:
    """The API key is sent as a bearer token only when configured."""
    from custom_components.llama_swap.const import DOMAIN as LLAMA_DOMAIN
    from homeassistant.const import CONF_API_KEY, CONF_HOST, CONF_PORT

    data = {CONF_HOST: "llama.local", CONF_PORT: 8080}
    if api_key:
        data[CONF_API_KEY] = api_key

    entry = MockConfigEntry(domain=LLAMA_DOMAIN, title="llama.local:8080", data=data)
    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, entry)

    headers = aioclient_mock.mock_calls[0][3] or {}
    assert ("Authorization" in headers) is expected
    if expected:
        assert headers["Authorization"] == "Bearer secret"
