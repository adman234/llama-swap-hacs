"""Fixtures for the llama-swap tests."""

from __future__ import annotations

from collections.abc import Generator
from datetime import timedelta
import sys
from typing import Any
from unittest.mock import patch

import pytest
import pytest_socket

if sys.platform == "win32":
    # Two Windows-only accommodations so the suite is runnable on a Windows dev
    # machine. Home Assistant's test plugin blocks sockets, allowing AF_UNIX
    # through so asyncio can build its self-pipe; Windows has no AF_UNIX
    # socketpair, so no event loop can start under that block. aiodns, in turn,
    # refuses to run on the default proactor loop.
    import asyncio

    from homeassistant import runner

    pytest_socket.disable_socket = lambda *args, **kwargs: None
    runner.HassEventLoopPolicy._loop_factory = asyncio.SelectorEventLoop

from freezegun.api import FrozenDateTimeFactory
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap.const import CONF_VERIFY_SSL, DOMAIN
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant

BASE_URL = "http://llama.local:8080"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None]:
    """Load the custom integration in every test."""
    yield


@pytest.fixture
def models_payload() -> dict[str, Any]:
    """Return a /v1/models response covering a model, alias and selector."""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-coder",
                "object": "model",
                "created": 1757000000,
                "owned_by": "llama-swap",
                "name": "Qwen3 Coder 30B",
                "description": "Coding model",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "modality": "text->text",
                },
                "capabilities": {"function_calling": True},
                "supported_parameters": ["tools", "tool_choice"],
                "context_length": 65536,
                "context_window": 65536,
                "meta": {
                    "llamaswap": {
                        "type": "model",
                        "aliases": ["coder"],
                        "tier": "primary",
                    },
                    "n_ctx": 65536,
                },
                "status": {"value": "loaded"},
            },
            {
                "id": "gemma3-vision",
                "object": "model",
                "created": 1757000000,
                "owned_by": "llama-swap",
                "name": "Gemma 3 Vision",
                "description": "",
                "architecture": {"input_modalities": ["text", "image"]},
                "capabilities": {"vision": True},
                "context_length": 8192,
                "meta": {"llamaswap": {"type": "model"}},
                "status": {"value": "unloaded"},
            },
            {
                "id": "auto",
                "object": "model",
                "created": 1757000000,
                "owned_by": "llama-swap",
                "meta": {
                    "llamaswap": {
                        "type": "selector",
                        "strategy": "pin",
                        "targets": ["qwen3-coder"],
                    }
                },
                "status": {"value": "loaded"},
            },
        ],
    }


@pytest.fixture
def running_payload() -> dict[str, Any]:
    """Return a /running response with one ready model."""
    return {
        "running": [
            {
                "model": "qwen3-coder",
                "state": "ready",
                "cmd": "llama-server -m /models/qwen3.gguf -ngl 99",
                "proxy": "http://127.0.0.1:9001",
                "ttl": 300,
                "name": "Qwen3 Coder 30B",
                "description": "Coding model",
            }
        ]
    }


@pytest.fixture
def performance_payload() -> dict[str, Any]:
    """Return an /api/performance response with two samples per source."""
    return {
        "enabled": True,
        "sys_stats": [
            {
                "timestamp": "2026-09-08T10:00:00Z",
                "cpu_util_per_core": [10.0, 20.0],
                "mem_total_mb": 32000,
                "mem_used_mb": 8000,
                "load_avg_1": 0.5,
                "load_avg_5": 0.4,
                "load_avg_15": 0.3,
                "swap_used_mb": 0,
            },
            {
                "timestamp": "2026-09-08T10:00:10Z",
                "cpu_util_per_core": [30.0, 50.0],
                "mem_total_mb": 32000,
                "mem_used_mb": 16000,
                "load_avg_1": 1.5,
                "load_avg_5": 1.4,
                "load_avg_15": 1.3,
                "swap_used_mb": 128,
            },
        ],
        "gpu_stats": [
            {
                "timestamp": "2026-09-08T10:00:00Z",
                "id": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": "GPU-abc",
                "temp_c": 40,
                "gpu_util_pct": 5.0,
                "mem_util_pct": 10.0,
                "mem_used_mb": 2000,
                "mem_total_mb": 24000,
                "power_draw_w": 40.0,
                "fan_speed_pct": 30.0,
            },
            {
                "timestamp": "2026-09-08T10:00:10Z",
                "id": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": "GPU-abc",
                "temp_c": 66,
                "gpu_util_pct": 92.0,
                "mem_util_pct": 80.0,
                "mem_used_mb": 19200,
                "mem_total_mb": 24000,
                "power_draw_w": 310.5,
                "fan_speed_pct": 62.0,
            },
        ],
    }


@pytest.fixture
def profiles_payload() -> dict[str, Any]:
    """Return an /api/profiles response."""
    return {
        "active": "coding",
        "profiles": [
            {"id": "coding", "description": "Dev work", "pins": {}},
            {"id": "low-power", "description": "Quiet", "pins": {}},
        ],
    }


@pytest.fixture
def mock_llama_swap(
    aioclient_mock: AiohttpClientMocker,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    performance_payload: dict[str, Any],
    profiles_payload: dict[str, Any],
) -> AiohttpClientMocker:
    """Mock a fully featured llama-swap server."""
    aioclient_mock.get(f"{BASE_URL}/v1/models", json=models_payload)
    aioclient_mock.get(f"{BASE_URL}/running", json=running_payload)
    aioclient_mock.get(
        f"{BASE_URL}/api/version",
        json={"version": "v180", "commit": "abc1234", "build_date": "2026-09-01"},
    )
    aioclient_mock.get(f"{BASE_URL}/api/performance", json=performance_payload)
    aioclient_mock.get(f"{BASE_URL}/api/profiles", json=profiles_payload)
    return aioclient_mock


@pytest.fixture
def config_entry() -> MockConfigEntry:
    """Return a config entry for the mocked server."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="llama.local:8080",
        data={
            CONF_HOST: "llama.local",
            CONF_PORT: 8080,
            CONF_SSL: False,
            CONF_VERIFY_SSL: True,
        },
    )


async def async_advance(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, seconds: int
) -> None:
    """Move time forward and let the coordinator's next poll finish.

    The scheduled refresh runs as a background task, so the wait has to include
    those or the assertions race the poll.
    """
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)


async def setup_integration(
    hass: HomeAssistant, entry: MockConfigEntry
) -> MockConfigEntry:
    """Add and set up a config entry."""
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


@pytest.fixture
def mock_setup_entry() -> Generator[Any]:
    """Stop the config flow from actually setting the integration up."""
    with patch(
        "custom_components.llama_swap.async_setup_entry", return_value=True
    ) as mock:
        yield mock
