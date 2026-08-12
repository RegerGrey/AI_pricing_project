"""Configuration objects and YAML loading for ConjointKit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

AttributeType = Literal["categorical", "numeric", "price"]
PreferenceDirection = Literal["higher", "lower", "neutral"]


@dataclass(frozen=True)
class AttributeConfig:
    """One product attribute and its permitted levels."""

    name: str
    type: AttributeType
    levels: tuple[Any, ...]
    preference_direction: PreferenceDirection = "neutral"


@dataclass(frozen=True)
class DesignConfig:
    """Settings used when creating a CBC design."""

    num_tasks: int = 10
    alternatives_per_task: int = 2
    random_seed: int | None = None


@dataclass(frozen=True)
class ConjointConfig:
    """Complete configuration for a ConjointKit experiment."""

    product_name: str
    attributes: Mapping[str, AttributeConfig]
    include_none: bool = True
    design: DesignConfig = field(default_factory=DesignConfig)

    @property
    def price_attribute(self) -> str:
        """Return the configured price attribute name."""
        return next(name for name, attribute in self.attributes.items() if attribute.type == "price")


def _as_attribute(name: str, value: Mapping[str, Any]) -> AttributeConfig:
    attribute_type = value.get("type", "categorical")
    direction = value.get("preference_direction")
    if direction is None:
        direction = "lower" if attribute_type == "price" else "neutral"
    return AttributeConfig(
        name=name,
        type=attribute_type,
        levels=tuple(value.get("levels", [])),
        preference_direction=direction,
    )


def _as_config(payload: Mapping[str, Any]) -> ConjointConfig:
    attributes_payload = payload.get("attributes", {})
    if not isinstance(attributes_payload, Mapping):
        raise ValueError("'attributes' must be a mapping of attribute names to definitions.")
    attributes = {
        name: _as_attribute(name, value)
        for name, value in attributes_payload.items()
        if isinstance(value, Mapping)
    }
    if len(attributes) != len(attributes_payload):
        raise ValueError("Each attribute definition must be a mapping.")
    options = payload.get("options", {})
    design_payload = payload.get("design", {})
    config = ConjointConfig(
        product_name=str(payload.get("product_name", "Unnamed product")),
        attributes=attributes,
        include_none=bool(options.get("include_none", True)),
        design=DesignConfig(
            num_tasks=int(design_payload.get("num_tasks", 10)),
            alternatives_per_task=int(design_payload.get("alternatives_per_task", 2)),
            random_seed=design_payload.get("random_seed"),
        ),
    )
    validate_config(config)
    return config


def load_config(source: str | Path | Mapping[str, Any]) -> ConjointConfig:
    """Load and validate a ConjointKit configuration from YAML or a mapping."""
    if isinstance(source, Mapping):
        return _as_config(source)
    path = Path(source)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Configuration YAML must contain a mapping at its top level.")
    return _as_config(payload)


def validate_config(config: ConjointConfig) -> None:
    """Raise ``ValueError`` when a configuration cannot support CBC and WTP analysis."""
    if len(config.attributes) < 2:
        raise ValueError("A CBC experiment requires at least two attributes.")
    if config.design.num_tasks <= 0:
        raise ValueError("design.num_tasks must be greater than zero.")
    if config.design.alternatives_per_task < 2:
        raise ValueError("design.alternatives_per_task must be at least two.")

    price_attributes: list[str] = []
    allowed_types = {"categorical", "numeric", "price"}
    allowed_directions = {"higher", "lower", "neutral"}
    for name, attribute in config.attributes.items():
        if attribute.type not in allowed_types:
            raise ValueError(f"Attribute '{name}' has unsupported type '{attribute.type}'.")
        if attribute.preference_direction not in allowed_directions:
            raise ValueError(
                f"Attribute '{name}' has unsupported preference_direction "
                f"'{attribute.preference_direction}'."
            )
        if len(attribute.levels) < 2:
            raise ValueError(f"Attribute '{name}' must define at least two levels.")
        if attribute.type in {"numeric", "price"} and not all(
            isinstance(level, (int, float)) and not isinstance(level, bool)
            for level in attribute.levels
        ):
            raise ValueError(f"Attribute '{name}' must use numeric levels.")
        if attribute.type == "price":
            price_attributes.append(name)

    if len(price_attributes) != 1:
        raise ValueError("Exactly one attribute with type 'price' is required for WTP analysis.")
