"""Tests for service targeting, diagnostics and legacy fallbacks."""

from __future__ import annotations

from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap.const import DOMAIN
from custom_components.llama_swap.diagnostics import (
    async_get_config_entry_diagnostics,
)
from homeassistant.const import CONF_API_KEY, CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError

from .conftest import BASE_URL, setup_integration

SECOND_URL = "http://other.local:8080"


def _paths(mock: AiohttpClientMocker) -> list[str]:
    """Return the path of every request made."""
    return [str(call[1].path) for call in mock.mock_calls]


async def test_unload_model_falls_back_on_old_server(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """Without the per-model endpoint, unloading falls back to unload-all."""
    await setup_integration(hass, config_entry)

    mock_llama_swap.post(
        f"{BASE_URL}/api/models/unload/qwen3-coder",
        status=404,
        text="404 page not found",
    )
    mock_llama_swap.get(f"{BASE_URL}/unload", text="OK")

    await hass.services.async_call(
        DOMAIN, "unload_model", {"model": "qwen3-coder"}, blocking=True
    )
    assert "/unload" in _paths(mock_llama_swap)


async def test_entry_id_required_with_two_servers(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
) -> None:
    """With two servers configured, a service call has to name one."""
    for base, host in ((BASE_URL, "llama.local"), (SECOND_URL, "other.local")):
        aioclient_mock.get(f"{base}/v1/models", json=models_payload)
        aioclient_mock.get(f"{base}/running", json=running_payload)
        for path in ("version", "performance", "profiles"):
            aioclient_mock.get(f"{base}/api/{path}", status=404, text="not found")
        await setup_integration(
            hass,
            MockConfigEntry(
                domain=DOMAIN,
                title=f"{host}:8080",
                data={CONF_HOST: host, CONF_PORT: 8080},
            ),
        )

    entries = hass.config_entries.async_entries(DOMAIN)
    assert len(entries) == 2

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(DOMAIN, "unload_all", {}, blocking=True)

    aioclient_mock.get(f"{SECOND_URL}/unload", text="OK")
    await hass.services.async_call(
        DOMAIN,
        "unload_all",
        {"config_entry_id": entries[1].entry_id},
        blocking=True,
    )
    assert f"{SECOND_URL}/unload" in [
        str(call[1]) for call in aioclient_mock.mock_calls
    ]


async def test_unknown_entry_id_raises(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """Naming a config entry that does not exist is a validation error."""
    await setup_integration(hass, config_entry)

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "unload_all", {"config_entry_id": "nope"}, blocking=True
        )


async def test_diagnostics_redacts_secrets(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker
) -> None:
    """Diagnostics keep the API key and upstream command lines out."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="llama.local:8080",
        data={CONF_HOST: "llama.local", CONF_PORT: 8080, CONF_API_KEY: "secret"},
    )
    await setup_integration(hass, entry)

    result = await async_get_config_entry_diagnostics(hass, entry)

    assert result["entry"]["data"][CONF_API_KEY] == "**REDACTED**"
    assert result["models"]["qwen3-coder"]["cmd"] == "**REDACTED**"
    assert result["models"]["qwen3-coder"]["state"] == "ready"
    assert result["version"]["version"] == "v180"
    assert result["active_profile"] == "coding"
    assert result["gpu_stats"][0]["temp_c"] == 66
