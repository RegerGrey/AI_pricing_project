"""Quality diagnostics for CBC designs and standardized response data."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .config import ConjointConfig


def _attribute_value(value: Any, levels: tuple[Any, ...]) -> float:
    """Encode a configured level in its specified order for heuristic comparisons."""
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(levels.index(value))
    except ValueError as error:
        raise ValueError(f"Value {value!r} is not a configured level.") from error


def check_dominance(tasks: pd.DataFrame, config: ConjointConfig) -> list[dict[str, Any]]:
    """Return pairwise dominance findings, ignoring attributes marked ``neutral``.

    A profile dominates another only if it is at least as preferred on every
    non-neutral attribute and strictly preferred on one of them.
    """
    findings: list[dict[str, Any]] = []
    active_attributes = [
        attribute
        for attribute in config.attributes.values()
        if attribute.preference_direction != "neutral"
    ]
    if not active_attributes:
        return findings

    for task_id, group in tasks.groupby("task_id", sort=False):
        for (_, first), (_, second) in combinations(group.iterrows(), 2):
            first_better = True
            second_better = True
            first_strict = False
            second_strict = False
            for attribute in active_attributes:
                first_value = _attribute_value(first[attribute.name], attribute.levels)
                second_value = _attribute_value(second[attribute.name], attribute.levels)
                if attribute.preference_direction == "lower":
                    first_value, second_value = -first_value, -second_value
                first_better &= first_value >= second_value
                second_better &= second_value >= first_value
                first_strict |= first_value > second_value
                second_strict |= second_value > first_value
            if first_better and first_strict:
                findings.append(
                    {
                        "task_id": task_id,
                        "dominant_alternative": first["alternative_id"],
                        "dominated_alternative": second["alternative_id"],
                    }
                )
            elif second_better and second_strict:
                findings.append(
                    {
                        "task_id": task_id,
                        "dominant_alternative": second["alternative_id"],
                        "dominated_alternative": first["alternative_id"],
                    }
                )
    return findings


def check_duplicates(tasks: pd.DataFrame, config: ConjointConfig) -> list[dict[str, Any]]:
    """Return duplicate profile pairs within the same task."""
    attribute_names = list(config.attributes)
    duplicates: list[dict[str, Any]] = []
    for task_id, group in tasks.groupby("task_id", sort=False):
        duplicate_rows = group[group.duplicated(attribute_names, keep=False)]
        if not duplicate_rows.empty:
            duplicates.append(
                {
                    "task_id": task_id,
                    "alternatives": duplicate_rows["alternative_id"].tolist(),
                }
            )
    return duplicates


def calculate_level_balance(tasks: pd.DataFrame, config: ConjointConfig) -> pd.DataFrame:
    """Return the count and target count for every configured attribute level."""
    rows: list[dict[str, Any]] = []
    for name, attribute in config.attributes.items():
        target = len(tasks) / len(attribute.levels)
        for level in attribute.levels:
            count = int((tasks[name] == level).sum())
            rows.append(
                {
                    "attribute": name,
                    "level": level,
                    "count": count,
                    "target_count": target,
                    "absolute_deviation": abs(count - target),
                }
            )
    return pd.DataFrame(rows)


def calculate_attribute_correlation(tasks: pd.DataFrame, config: ConjointConfig) -> pd.DataFrame:
    """Return an ordinal-encoded attribute correlation matrix for design diagnosis.

    Categorical level order is used only as a transparent balance heuristic; it
    is not a claim that nominal levels are numerically spaced.
    """
    encoded: dict[str, pd.Series] = {}
    for name, attribute in config.attributes.items():
        encoded[name] = tasks[name].map(
            {level: index for index, level in enumerate(attribute.levels)}
        )
    return pd.DataFrame(encoded).corr()


def validate_design(tasks: pd.DataFrame, config: ConjointConfig) -> dict[str, Any]:
    """Compute transparent CBC design diagnostics without inventing a quality score."""
    required = {"task_id", "alternative_id", *config.attributes.keys()}
    missing = required.difference(tasks.columns)
    if missing:
        raise ValueError(f"Design is missing required columns: {sorted(missing)}")
    task_sizes = tasks.groupby("task_id")["alternative_id"].nunique()
    if task_sizes.empty or task_sizes.nunique() != 1:
        raise ValueError("Every task must contain the same number of alternatives.")
    duplicates = check_duplicates(tasks, config)
    dominated = check_dominance(tasks, config)
    balance = calculate_level_balance(tasks, config)
    correlation = calculate_attribute_correlation(tasks, config)
    if len(correlation) > 1:
        correlation_values = correlation.abs().to_numpy(copy=True)
        np.fill_diagonal(correlation_values, 0.0)
        max_correlation = float(np.nanmax(correlation_values))
    else:
        max_correlation = 0.0
    max_possible_deviation = max(len(tasks) * (1 - 1 / len(attribute.levels)) for attribute in config.attributes.values())
    mean_deviation = float(balance["absolute_deviation"].mean()) if not balance.empty else 0.0
    level_balance_score = max(0.0, 1 - mean_deviation / max_possible_deviation)
    return {
        "num_tasks": int(tasks["task_id"].nunique()),
        "alternatives_per_task": int(task_sizes.iloc[0]),
        "duplicate_profiles": len(duplicates),
        "dominated_tasks": len({row["task_id"] for row in dominated}),
        "max_attribute_correlation": max_correlation,
        "level_balance_score": level_balance_score,
        "level_balance": balance,
        "attribute_correlation": correlation,
        "duplicate_details": duplicates,
        "dominance_details": dominated,
    }
