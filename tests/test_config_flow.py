"""Tests for the config and options flows."""

from __future__ import annotations

from typing import Any

import aiohttp
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap.const import CONF_VERIFY_SSL, DOMAIN
from homeassistant import config_entries
from homeassistant.const import (
    CONF_API_KEY,
    CONF_HOST,
    CONF_PORT,
    CONF_SCAN_INTERVAL,
    CONF_SSL,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from .conftest import BASE_URL, setup_integration

USER_INPUT: dict[str, Any] = {
    CONF_HOST: "llama.local",
    CONF_PORT: 8080,
    CONF_SSL: False,
    CONF_VERIFY_SSL: True,
}


async def test_user_flow(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, mock_setup_entry
) -> None:
    """A valid server creates an entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "llama.local:8080"
    assert result["data"] == USER_INPUT


async def test_cannot_connect(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, mock_setup_entry
) -> None:
    """A refused connection is reported on the form, and can be retried."""
    aioclient_mock.get(f"{BASE_URL}/v1/models", exc=aiohttp.ClientError)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}

    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE_URL}/v1/models", json={"object": "list", "data": []})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_invalid_auth(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, mock_setup_entry
) -> None:
    """A rejected API key is reported on the form."""
    aioclient_mock.get(f"{BASE_URL}/v1/models", status=401, text="unauthorized")

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {**USER_INPUT, CONF_API_KEY: "wrong"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}


async def test_duplicate_aborts(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The same host and port cannot be added twice."""
    config_entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth_flow(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, models_payload
) -> None:
    """A rejected key triggers reauth, and a new key is stored."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="llama.local:8080",
        data={**USER_INPUT, CONF_API_KEY: "stale"},
    )
    entry.add_to_hass(hass)

    result = await entry.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    aioclient_mock.get(f"{BASE_URL}/v1/models", json=models_payload)
    aioclient_mock.get(f"{BASE_URL}/running", json={"running": []})
    aioclient_mock.get(f"{BASE_URL}/api/version", status=404, text="not found")
    aioclient_mock.get(f"{BASE_URL}/api/performance", status=404, text="not found")
    aioclient_mock.get(f"{BASE_URL}/api/profiles", status=404, text="not found")

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "fresh"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data[CONF_API_KEY] == "fresh"


async def test_options_flow(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """The scan interval can be changed."""
    await setup_integration(hass, config_entry)

    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_SCAN_INTERVAL: 60}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert config_entry.options == {CONF_SCAN_INTERVAL: 60}
    assert config_entry.runtime_data.update_interval.total_seconds() == 60
