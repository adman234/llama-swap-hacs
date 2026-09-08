"""Sensor platform for llama-swap."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
    UnitOfInformation,
    UnitOfPower,
    UnitOfTemperature,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN, MANUFACTURER, MODEL_STATES
from .coordinator import (
    LlamaSwapConfigEntry,
    LlamaSwapCoordinator,
    LlamaSwapData,
    ModelInfo,
)
from .entity import (
    LlamaSwapEntity,
    LlamaSwapModelEntity,
    async_setup_model_entities,
)

PARALLEL_UPDATES = 0


@dataclass(frozen=True, kw_only=True)
class LlamaSwapSensorDescription(SensorEntityDescription):
    """Describes a sensor derived from the whole-server payload."""

    value_fn: Callable[[LlamaSwapData], Any]
    attributes_fn: Callable[[LlamaSwapData], dict[str, Any]] | None = None
    exists_fn: Callable[[LlamaSwapData], bool] = lambda _: True


@dataclass(frozen=True, kw_only=True)
class LlamaSwapModelSensorDescription(SensorEntityDescription):
    """Describes a sensor derived from a single model."""

    value_fn: Callable[[ModelInfo], Any]
    attributes_fn: Callable[[ModelInfo], dict[str, Any]] | None = None


@dataclass(frozen=True, kw_only=True)
class LlamaSwapGpuSensorDescription(SensorEntityDescription):
    """Describes a sensor derived from one GPU's stats."""

    value_fn: Callable[[dict[str, Any]], Any]


def _model_attributes(model: ModelInfo) -> dict[str, Any]:
    """Return the full detail set for a model, for use in automations."""
    attrs: dict[str, Any] = {
        "model_id": model.id,
        "type": model.kind,
        "loaded": model.is_loaded,
        "unlisted": model.unlisted,
    }
    if model.name:
        attrs["model_name"] = model.name
    if model.description:
        attrs["description"] = model.description
    if model.aliases:
        attrs["aliases"] = model.aliases
    if model.capabilities:
        attrs["capabilities"] = model.capabilities
    if model.architecture:
        attrs["architecture"] = model.architecture
    if model.supported_parameters:
        attrs["supported_parameters"] = model.supported_parameters
    if model.context_length is not None:
        attrs["context_length"] = model.context_length
    if model.cmd:
        attrs["cmd"] = model.cmd
    if model.proxy:
        attrs["proxy"] = model.proxy
    if model.ttl is not None:
        attrs["ttl"] = model.ttl
    extra = {
        key: value
        for key, value in model.metadata.items()
        if key not in ("type", "aliases")
    }
    if extra:
        attrs["metadata"] = extra
    return attrs


def _active_model(data: LlamaSwapData) -> str | None:
    """Return the loaded model ID, or None when nothing is loaded.

    llama-swap can run several models at once when its groups allow it. The
    lowest-sorting ID is reported here; the full set is in the attributes.
    """
    loaded = data.loaded_models
    return loaded[0].id if loaded else None


def _active_model_attributes(data: LlamaSwapData) -> dict[str, Any]:
    """Return every loaded model and its details."""
    loaded = data.loaded_models
    return {
        "loaded_models": [model.id for model in loaded],
        "loaded_count": len(loaded),
        "models": [_model_attributes(model) for model in loaded],
    }


def _all_models_attributes(data: LlamaSwapData) -> dict[str, Any]:
    """Return details for every configured model."""
    models = sorted(data.models.values(), key=lambda model: model.id)
    return {
        "model_ids": [model.id for model in models],
        "models": [_model_attributes(model) for model in models],
    }


def _cpu_average(data: LlamaSwapData) -> float | None:
    """Return mean CPU utilisation across cores."""
    cores = data.sys_stats.get("cpu_util_per_core")
    if not isinstance(cores, list):
        return None
    values = [core for core in cores if isinstance(core, (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def _memory_percent(data: LlamaSwapData) -> float | None:
    """Return used system memory as a percentage of the total."""
    total = data.sys_stats.get("mem_total_mb")
    used = data.sys_stats.get("mem_used_mb")
    if not isinstance(total, (int, float)) or not total:
        return None
    if not isinstance(used, (int, float)):
        return None
    return round(used / total * 100, 1)


def _sys_value(key: str) -> Callable[[LlamaSwapData], Any]:
    """Return a getter for one system-stats field."""

    def _value(data: LlamaSwapData) -> Any:
        value = data.sys_stats.get(key)
        return value if isinstance(value, (int, float)) else None

    return _value


def _has_sys_stats(data: LlamaSwapData) -> bool:
    """Return True when the server reported system stats."""
    return bool(data.sys_stats)


SERVER_SENSORS: tuple[LlamaSwapSensorDescription, ...] = (
    LlamaSwapSensorDescription(
        key="active_model",
        translation_key="active_model",
        icon="mdi:brain",
        value_fn=_active_model,
        attributes_fn=_active_model_attributes,
    ),
    LlamaSwapSensorDescription(
        key="loaded_models",
        translation_key="loaded_models",
        icon="mdi:memory",
        native_unit_of_measurement="models",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: len(data.loaded_models),
        attributes_fn=_active_model_attributes,
    ),
    LlamaSwapSensorDescription(
        key="total_models",
        translation_key="total_models",
        icon="mdi:format-list-bulleted",
        native_unit_of_measurement="models",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda data: len(data.models),
        attributes_fn=_all_models_attributes,
    ),
    LlamaSwapSensorDescription(
        key="version",
        translation_key="version",
        icon="mdi:tag-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda data: data.version.get("version") or None,
        attributes_fn=lambda data: {
            "commit": data.version.get("commit"),
            "build_date": data.version.get("build_date"),
        },
        exists_fn=lambda data: bool(data.version),
    ),
    LlamaSwapSensorDescription(
        key="cpu_utilization",
        translation_key="cpu_utilization",
        icon="mdi:cpu-64-bit",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=_cpu_average,
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="memory_used",
        translation_key="memory_used",
        device_class=SensorDeviceClass.DATA_SIZE,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        suggested_unit_of_measurement=UnitOfInformation.GIGABYTES,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=_sys_value("mem_used_mb"),
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="memory_utilization",
        translation_key="memory_utilization",
        icon="mdi:memory",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=_memory_percent,
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="memory_total",
        translation_key="memory_total",
        device_class=SensorDeviceClass.DATA_SIZE,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        suggested_unit_of_measurement=UnitOfInformation.GIGABYTES,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        suggested_display_precision=1,
        value_fn=_sys_value("mem_total_mb"),
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="swap_used",
        translation_key="swap_used",
        device_class=SensorDeviceClass.DATA_SIZE,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        suggested_unit_of_measurement=UnitOfInformation.GIGABYTES,
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=False,
        suggested_display_precision=1,
        value_fn=_sys_value("swap_used_mb"),
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="load_1m",
        translation_key="load_1m",
        icon="mdi:gauge",
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=2,
        value_fn=_sys_value("load_avg_1"),
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="load_5m",
        translation_key="load_5m",
        icon="mdi:gauge",
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=False,
        suggested_display_precision=2,
        value_fn=_sys_value("load_avg_5"),
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="load_15m",
        translation_key="load_15m",
        icon="mdi:gauge",
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=False,
        suggested_display_precision=2,
        value_fn=_sys_value("load_avg_15"),
        exists_fn=_has_sys_stats,
    ),
    LlamaSwapSensorDescription(
        key="active_profile",
        translation_key="active_profile",
        icon="mdi:account-switch",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda data: data.active_profile,
        attributes_fn=lambda data: {
            "profiles": [profile.get("id") for profile in data.profiles],
        },
        exists_fn=lambda data: bool(data.profiles),
    ),
)


MODEL_SENSORS: tuple[LlamaSwapModelSensorDescription, ...] = (
    LlamaSwapModelSensorDescription(
        key="state",
        translation_key="model_state",
        icon="mdi:state-machine",
        device_class=SensorDeviceClass.ENUM,
        options=MODEL_STATES,
        value_fn=lambda model: model.state,
        attributes_fn=_model_attributes,
    ),
    LlamaSwapModelSensorDescription(
        key="context_length",
        translation_key="context_length",
        icon="mdi:text-long",
        native_unit_of_measurement="tokens",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda model: model.context_length,
    ),
    LlamaSwapModelSensorDescription(
        key="ttl",
        translation_key="ttl",
        icon="mdi:timer-sand",
        native_unit_of_measurement=UnitOfTime.SECONDS,
        device_class=SensorDeviceClass.DURATION,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda model: model.ttl,
    ),
)


def _gpu_value(key: str) -> Callable[[dict[str, Any]], Any]:
    """Return a getter for one GPU-stats field."""

    def _value(stat: dict[str, Any]) -> Any:
        value = stat.get(key)
        return value if isinstance(value, (int, float)) else None

    return _value


GPU_SENSORS: tuple[LlamaSwapGpuSensorDescription, ...] = (
    LlamaSwapGpuSensorDescription(
        key="gpu_utilization",
        translation_key="gpu_utilization",
        icon="mdi:expansion-card",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=0,
        value_fn=_gpu_value("gpu_util_pct"),
    ),
    LlamaSwapGpuSensorDescription(
        key="vram_used",
        translation_key="vram_used",
        device_class=SensorDeviceClass.DATA_SIZE,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        suggested_unit_of_measurement=UnitOfInformation.GIGABYTES,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=_gpu_value("mem_used_mb"),
    ),
    LlamaSwapGpuSensorDescription(
        key="vram_total",
        translation_key="vram_total",
        device_class=SensorDeviceClass.DATA_SIZE,
        native_unit_of_measurement=UnitOfInformation.MEGABYTES,
        suggested_unit_of_measurement=UnitOfInformation.GIGABYTES,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=1,
        value_fn=_gpu_value("mem_total_mb"),
    ),
    LlamaSwapGpuSensorDescription(
        key="vram_utilization",
        translation_key="vram_utilization",
        icon="mdi:memory",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=0,
        value_fn=_gpu_value("mem_util_pct"),
    ),
    LlamaSwapGpuSensorDescription(
        key="gpu_temperature",
        translation_key="gpu_temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=_gpu_value("temp_c"),
    ),
    LlamaSwapGpuSensorDescription(
        key="vram_temperature",
        translation_key="vram_temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=False,
        value_fn=_gpu_value("vram_temp_c"),
    ),
    LlamaSwapGpuSensorDescription(
        key="gpu_power",
        translation_key="gpu_power",
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=_gpu_value("power_draw_w"),
    ),
    LlamaSwapGpuSensorDescription(
        key="gpu_fan_speed",
        translation_key="gpu_fan_speed",
        icon="mdi:fan",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_registry_enabled_default=False,
        suggested_display_precision=0,
        value_fn=_gpu_value("fan_speed_pct"),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LlamaSwapConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the llama-swap sensors."""
    coordinator = entry.runtime_data

    entities: list[Entity] = [
        LlamaSwapServerSensor(coordinator, description)
        for description in SERVER_SENSORS
        if description.exists_fn(coordinator.data)
    ]
    entities.extend(
        LlamaSwapGpuSensor(coordinator, stat["id"], description)
        for stat in coordinator.data.gpu_stats
        for description in GPU_SENSORS
    )
    async_add_entities(entities)

    def _build(model_id: str) -> Iterable[Entity]:
        return (
            LlamaSwapModelSensor(coordinator, model_id, description)
            for description in MODEL_SENSORS
        )

    async_setup_model_entities(coordinator, async_add_entities, _build)


class LlamaSwapServerSensor(LlamaSwapEntity, SensorEntity):
    """A sensor reporting whole-server state."""

    entity_description: LlamaSwapSensorDescription

    def __init__(
        self,
        coordinator: LlamaSwapCoordinator,
        description: LlamaSwapSensorDescription,
    ) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> Any:
        """Return the sensor value."""
        return self.entity_description.value_fn(self.coordinator.data)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the sensor's extra detail."""
        if self.entity_description.attributes_fn is None:
            return None
        return self.entity_description.attributes_fn(self.coordinator.data)


class LlamaSwapModelSensor(LlamaSwapModelEntity, SensorEntity):
    """A sensor reporting the state of one model."""

    entity_description: LlamaSwapModelSensorDescription

    def __init__(
        self,
        coordinator: LlamaSwapCoordinator,
        model_id: str,
        description: LlamaSwapModelSensorDescription,
    ) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, model_id, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> Any:
        """Return the sensor value."""
        model = self.model
        return self.entity_description.value_fn(model) if model else None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the model's full detail set."""
        model = self.model
        if model is None or self.entity_description.attributes_fn is None:
            return None
        return self.entity_description.attributes_fn(model)


class LlamaSwapGpuSensor(LlamaSwapEntity, SensorEntity):
    """A sensor reporting one GPU's stats."""

    entity_description: LlamaSwapGpuSensorDescription

    def __init__(
        self,
        coordinator: LlamaSwapCoordinator,
        gpu_id: int,
        description: LlamaSwapGpuSensorDescription,
    ) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, f"gpu{gpu_id}_{description.key}")
        self.entity_description = description
        self._gpu_id = gpu_id
        entry = coordinator.config_entry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry.entry_id}_gpu_{gpu_id}")},
            name=self._stat.get("name") or f"GPU {gpu_id}",
            manufacturer=MANUFACTURER,
            model="GPU",
            via_device=(DOMAIN, entry.entry_id),
        )

    @property
    def _stat(self) -> dict[str, Any]:
        """Return the latest sample for this GPU."""
        for stat in self.coordinator.data.gpu_stats:
            if stat.get("id") == self._gpu_id:
                return stat
        return {}

    @property
    def available(self) -> bool:
        """Return True while the server still reports this GPU."""
        return super().available and bool(self._stat)

    @property
    def native_value(self) -> Any:
        """Return the sensor value."""
        return self.entity_description.value_fn(self._stat)
