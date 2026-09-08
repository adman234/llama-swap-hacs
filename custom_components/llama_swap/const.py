"""Constants for the llama-swap integration."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "llama_swap"

CONF_VERIFY_SSL: Final = "verify_ssl"

DEFAULT_PORT: Final = 8080
DEFAULT_SSL: Final = False
DEFAULT_VERIFY_SSL: Final = True
DEFAULT_SCAN_INTERVAL: Final = 15
MIN_SCAN_INTERVAL: Final = 5
MAX_SCAN_INTERVAL: Final = 3600

# Seconds allowed for a model load to complete. Loading pulls a model into
# VRAM and can be slow on large weights, so this is deliberately generous.
LOAD_TIMEOUT: Final = 600

MANUFACTURER: Final = "llama-swap"

# Process states reported by /running. Models with no entry are "stopped".
STATE_STOPPED: Final = "stopped"
STATE_STARTING: Final = "starting"
STATE_READY: Final = "ready"
STATE_STOPPING: Final = "stopping"
STATE_SHUTDOWN: Final = "shutdown"

MODEL_STATES: Final = [
    STATE_STOPPED,
    STATE_STARTING,
    STATE_READY,
    STATE_STOPPING,
    STATE_SHUTDOWN,
]

# States in which a model counts as loaded / occupying resources.
ACTIVE_STATES: Final = frozenset({STATE_STARTING, STATE_READY})

SERVICE_LOAD_MODEL: Final = "load_model"
SERVICE_UNLOAD_MODEL: Final = "unload_model"
SERVICE_UNLOAD_ALL: Final = "unload_all"
SERVICE_SET_PROFILE: Final = "set_profile"

ATTR_MODEL: Final = "model"
ATTR_PROFILE: Final = "profile"
