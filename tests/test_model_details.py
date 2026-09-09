"""Tests for context length, model file and the remembered command details."""

from __future__ import annotations

from typing import Any

from freezegun.api import FrozenDateTimeFactory
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.llama_swap.const import DEFAULT_SCAN_INTERVAL
from custom_components.llama_swap.coordinator import (
    model_file_name,
    parse_context_length,
    parse_model_path,
)
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant

from .conftest import async_advance, setup_integration
from .test_coordinator import _mock_legacy_server


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("llama-server -m /models/a.gguf --ctx-size 65536", 65536),
        ("llama-server -m /models/a.gguf -c 8192", 8192),
        ("llama-server --ctx-size=32768 -m /models/a.gguf", 32768),
        ("llama-server --ctx_size 4096", 4096),
        ("llama-server --n-ctx 2048", 2048),
        # No context flag at all.
        ("llama-server -m /models/a.gguf -ngl 99", None),
        # A -c that is not a number belongs to some other tool.
        ("some-server -c config.yaml", None),
        # Zero and negatives are not usable context sizes.
        ("llama-server -c 0", None),
        ("llama-server -c -1", None),
        # A trailing flag with no value must not blow up.
        ("llama-server --ctx-size", None),
        (None, None),
        ("", None),
    ],
)
def test_parse_context_length(cmd: str | None, expected: int | None) -> None:
    """Context size is read from the command line in each accepted spelling."""
    assert parse_context_length(cmd) == expected


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        (
            "llama-server -m /models/Qwen3-Coder-30B-Q4_K_M.gguf -c 65536",
            "/models/Qwen3-Coder-30B-Q4_K_M.gguf",
        ),
        (
            "llama-server --model /srv/models/gemma.gguf",
            "/srv/models/gemma.gguf",
        ),
        (
            'llama-server -m "/models/with space/model.gguf"',
            "/models/with space/model.gguf",
        ),
        (
            "llama-server --hf-repo unsloth/Qwen3-30B-GGUF --hf-file Q4_K_M.gguf",
            "unsloth/Qwen3-30B-GGUF/Q4_K_M.gguf",
        ),
        ("llama-server -hfr bartowski/foo", "bartowski/foo"),
        ("llama-server -ngl 99", None),
        (None, None),
    ],
)
def test_parse_model_path(cmd: str | None, expected: str | None) -> None:
    """The weights are read from -m, --model or the Hugging Face flags."""
    assert parse_model_path(cmd) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/models/a.gguf", "a.gguf"),
        ("C:\\models\\a.gguf", "a.gguf"),
        ("a.gguf", "a.gguf"),
        ("unsloth/Qwen3-30B-GGUF/Q4_K_M.gguf", "Q4_K_M.gguf"),
        ("/models/", "models"),
        (None, None),
    ],
)
def test_model_file_name(path: str | None, expected: str | None) -> None:
    """Only the file name is reported, on either path separator."""
    assert model_file_name(path) == expected


async def test_context_length_from_command_line(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
) -> None:
    """A --ctx-size in the command line populates the context sensor.

    The fixture's /v1/models record for qwen3-coder declares context_length,
    so this uses gemma3-vision, which does not.
    """
    running_payload["running"].append(
        {
            "model": "gemma3-vision",
            "state": "ready",
            "cmd": (
                "llama-server -m /models/gemma-3-12b-it-Q4_K_M.gguf --ctx-size 16384"
            ),
            "proxy": "http://127.0.0.1:9002",
            "ttl": 120,
            "name": "Gemma 3 Vision",
            "description": "",
        }
    )
    for record in models_payload["data"]:
        record.pop("context_length", None)

    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)

    assert hass.states.get("sensor.gemma_3_vision_context_length").state == "16384"
    assert hass.states.get("sensor.gemma_3_vision_model_file").state == (
        "gemma-3-12b-it-Q4_K_M.gguf"
    )
    assert hass.states.get("sensor.gemma_3_vision_unload_after").state == "120"

    attrs = hass.states.get("sensor.gemma_3_vision_state").attributes
    assert attrs["context_length"] == 16384
    assert attrs["context_source"] == "command"
    assert attrs["model_path"] == "/models/gemma-3-12b-it-Q4_K_M.gguf"


async def test_declared_context_length_wins(
    hass: HomeAssistant, mock_llama_swap: AiohttpClientMocker, config_entry
) -> None:
    """capabilities.context beats whatever the command line says."""
    await setup_integration(hass, config_entry)

    assert hass.states.get("sensor.qwen3_coder_30b_context_length").state == "65536"
    attrs = hass.states.get("sensor.qwen3_coder_30b_state").attributes
    assert attrs["context_source"] == "capabilities"


async def test_details_survive_unload(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
    running_payload: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    """Command-line details stay put once the model stops running."""
    running_payload["running"][0]["cmd"] = (
        "llama-server -m /models/qwen3.gguf --ctx-size 65536"
    )
    for record in models_payload["data"]:
        record.pop("context_length", None)

    _mock_legacy_server(aioclient_mock, models_payload, running_payload)
    await setup_integration(hass, config_entry)

    assert hass.states.get("sensor.qwen3_coder_30b_context_length").state == "65536"
    assert (
        "details_cached"
        not in hass.states.get("sensor.qwen3_coder_30b_state").attributes
    )

    aioclient_mock.clear_requests()
    for record in models_payload["data"]:
        record["status"] = {"value": "unloaded"}
    _mock_legacy_server(aioclient_mock, models_payload, {"running": []})
    await async_advance(hass, freezer, DEFAULT_SCAN_INTERVAL + 1)

    assert hass.states.get("sensor.qwen3_coder_30b_state").state == "stopped"
    # The values describe llama-swap's config, so they outlive the process.
    assert hass.states.get("sensor.qwen3_coder_30b_context_length").state == "65536"
    assert hass.states.get("sensor.qwen3_coder_30b_model_file").state == "qwen3.gguf"
    assert hass.states.get("sensor.qwen3_coder_30b_unload_after").state == "300"

    attrs = hass.states.get("sensor.qwen3_coder_30b_state").attributes
    assert attrs["details_cached"] is True


async def test_unknown_until_first_run(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    config_entry: MockConfigEntry,
    models_payload: dict[str, Any],
) -> None:
    """A model that has never run has no command line to read."""
    for record in models_payload["data"]:
        record.pop("context_length", None)
        record["status"] = {"value": "unloaded"}

    _mock_legacy_server(aioclient_mock, models_payload, {"running": []})
    await setup_integration(hass, config_entry)

    assert (
        hass.states.get("sensor.qwen3_coder_30b_context_length").state == STATE_UNKNOWN
    )
    assert hass.states.get("sensor.qwen3_coder_30b_model_file").state == STATE_UNKNOWN
