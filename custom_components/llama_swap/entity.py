"""Base entities for llama-swap."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable

from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, MANUFACTURER
from .coordinator import LlamaSwapCoordinator, LlamaSwapData, ModelInfo


class LlamaSwapEntity(CoordinatorEntity[LlamaSwapCoordinator]):
    """An entity belonging to the llama-swap server itself."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: LlamaSwapCoordinator, key: str) -> None:
        """Initialise the entity."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{key}"
        self._attr_device_info = server_device_info(coordinator)


class LlamaSwapModelEntity(CoordinatorEntity[LlamaSwapCoordinator]):
    """An entity belonging to a single model, on its own device."""

    _attr_has_entity_name = True

    def __init__(
        self, coordinator: LlamaSwapCoordinator, model_id: str, key: str
    ) -> None:
        """Initialise the entity."""
        super().__init__(coordinator)
        self.model_id = model_id
        entry_id = coordinator.config_entry.entry_id
        self._attr_unique_id = f"{entry_id}_model_{model_id}_{key}"
        self._attr_device_info = model_device_info(coordinator, model_id)

    @property
    def model(self) -> ModelInfo | None:
        """Return the current data for this model, if it still exists."""
        return self.coordinator.data.models.get(self.model_id)

    @property
    def available(self) -> bool:
        """Return True while the server responds and still lists the model."""
        return super().available and self.model is not None


@callback
def async_setup_dynamic_entities[KeyT: Hashable](
    coordinator: LlamaSwapCoordinator,
    async_add_entities: AddConfigEntryEntitiesCallback,
    keys_fn: Callable[[LlamaSwapData], Iterable[KeyT]],
    build: Callable[[KeyT], Iterable[Entity]],
) -> None:
    """Create entities for each key, including keys that show up later.

    What a llama-swap server exposes is not fixed at setup: reloading its
    config changes the model list, and turning on performance monitoring or
    adding a first profile makes whole groups of entities become possible.
    Watching the coordinator means none of that needs an integration reload.

    Entities are never removed here. A key that disappears leaves its entities
    unavailable rather than deleted, so a model that is temporarily missing
    during a llama-swap config reload does not lose its history or its
    customisations.
    """
    known: set[KeyT] = set()

    @callback
    def _async_add_new() -> None:
        entities: list[Entity] = []
        for key in keys_fn(coordinator.data):
            if key in known:
                continue
            known.add(key)
            entities.extend(build(key))
        if entities:
            async_add_entities(entities)

    _async_add_new()
    coordinator.config_entry.async_on_unload(
        coordinator.async_add_listener(_async_add_new)
    )


@callback
def async_setup_model_entities(
    coordinator: LlamaSwapCoordinator,
    async_add_entities: AddConfigEntryEntitiesCallback,
    build: Callable[[str], Iterable[Entity]],
) -> None:
    """Create per-model entities, including for models added later."""
    async_setup_dynamic_entities(
        coordinator, async_add_entities, lambda data: data.models, build
    )


def server_device_info(coordinator: LlamaSwapCoordinator) -> DeviceInfo:
    """Return the device entry representing the llama-swap server."""
    entry = coordinator.config_entry
    version = coordinator.data.version if coordinator.data else {}
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
        manufacturer=MANUFACTURER,
        model="llama-swap server",
        sw_version=version.get("version") or None,
        configuration_url=coordinator.client.base_url,
    )


def model_device_info(coordinator: LlamaSwapCoordinator, model_id: str) -> DeviceInfo:
    """Return the device entry representing one model, tied to the server."""
    entry = coordinator.config_entry
    info = coordinator.data.models.get(model_id) if coordinator.data else None
    return DeviceInfo(
        identifiers={(DOMAIN, f"{entry.entry_id}_model_{model_id}")},
        name=info.display_name if info else model_id,
        manufacturer=MANUFACTURER,
        model=(info.kind if info else "model").capitalize(),
        via_device=(DOMAIN, entry.entry_id),
        configuration_url=f"{coordinator.client.base_url}/ui/models",
    )


def current_device_identifiers(coordinator: LlamaSwapCoordinator) -> set[str]:
    """Return the identifiers of every device this entry should own now."""
    entry_id = coordinator.config_entry.entry_id
    data = coordinator.data
    identifiers = {entry_id}
    if data is None:
        return identifiers
    identifiers.update(f"{entry_id}_model_{model_id}" for model_id in data.models)
    identifiers.update(f"{entry_id}_gpu_{stat['id']}" for stat in data.gpu_stats)
    return identifiers
