"""Balanced randomized CBC design generation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .config import ConjointConfig
from .validation import check_dominance, validate_design


@dataclass
class CBCDesign:
    """A generated CBC design and its transparent quality diagnostics."""

    tasks: pd.DataFrame
    profiles: pd.DataFrame
    quality_metrics: dict[str, object]


def generate_full_factorial(config: ConjointConfig) -> pd.DataFrame:
    """Return every possible product profile defined in ``config``."""
    names = list(config.attributes)
    level_lists = [config.attributes[name].levels for name in names]
    return pd.DataFrame(product(*level_lists), columns=names)


def _candidate_tasks(
    profiles: pd.DataFrame,
    config: ConjointConfig,
    num_tasks: int,
    alternatives_per_task: int,
    rng: np.random.Generator,
) -> pd.DataFrame | None:
    """Construct pairwise non-dominated tasks without reusing a profile."""
    available = list(rng.permutation(len(profiles)))
    rows: list[pd.DataFrame] = []
    alternative_labels = [chr(65 + index) if index < 26 else f"Alt{index + 1}" for index in range(alternatives_per_task)]
    for task_id in range(1, num_tasks + 1):
        selected: list[int] = []
        candidates = list(rng.permutation(available))
        for candidate_index in candidates:
            if not selected:
                selected.append(int(candidate_index))
                continue
            trial_indices = [*selected, int(candidate_index)]
            trial = profiles.iloc[trial_indices].reset_index(drop=True).copy()
            trial.insert(0, "alternative_id", alternative_labels[: len(trial)])
            trial.insert(0, "task_id", task_id)
            if not check_dominance(trial, config):
                selected.append(int(candidate_index))
            if len(selected) == alternatives_per_task:
                break
        if len(selected) != alternatives_per_task:
            return None
        task = profiles.iloc[selected].reset_index(drop=True).copy()
        task.insert(0, "alternative_id", alternative_labels)
        task.insert(0, "task_id", task_id)
        rows.append(task)
        selected_set = set(selected)
        available = [index for index in available if index not in selected_set]
    return pd.concat(rows, ignore_index=True)


def _objective(metrics: dict[str, object]) -> float:
    """A documented heuristic used only to rank randomized candidates."""
    return (
        1_000 * float(metrics["dominated_tasks"])
        + 100 * float(metrics["duplicate_profiles"])
        + 10 * float(metrics["max_attribute_correlation"])
        + (1 - float(metrics["level_balance_score"]))
    )


def generate_design(
    config: ConjointConfig,
    num_tasks: int | None = None,
    alternatives_per_task: int | None = None,
    random_seed: int | None = None,
    search_iterations: int = 100,
) -> CBCDesign:
    """Create a balanced randomized CBC design.

    This is a randomized search over unique full-factorial profiles.  It ranks
    valid candidates by level balance and ordinal-encoded correlation; it does
    not claim D-efficiency or another optimal-design criterion.
    """
    task_count = num_tasks if num_tasks is not None else config.design.num_tasks
    alternative_count = (
        alternatives_per_task
        if alternatives_per_task is not None
        else config.design.alternatives_per_task
    )
    if task_count <= 0 or alternative_count < 2:
        raise ValueError("num_tasks must be positive and alternatives_per_task must be at least two.")
    profiles = generate_full_factorial(config)
    required_profiles = task_count * alternative_count
    if required_profiles > len(profiles):
        raise ValueError(
            f"The requested design needs {required_profiles} unique profiles, but the "
            f"full factorial contains only {len(profiles)}. Reduce tasks or add levels."
        )
    if search_iterations <= 0:
        raise ValueError("search_iterations must be greater than zero.")

    seed = random_seed if random_seed is not None else config.design.random_seed
    rng = np.random.default_rng(seed)
    best_tasks: pd.DataFrame | None = None
    best_metrics: dict[str, object] | None = None
    best_score = float("inf")
    for _ in range(search_iterations):
        candidate = _candidate_tasks(profiles, config, task_count, alternative_count, rng)
        if candidate is None:
            continue
        metrics = validate_design(candidate, config)
        score = _objective(metrics)
        if score < best_score:
            best_tasks, best_metrics, best_score = candidate, metrics, score
        if metrics["dominated_tasks"] == 0 and metrics["max_attribute_correlation"] < 0.1:
            break
    if best_tasks is None or best_metrics is None:
        raise RuntimeError("Could not generate a CBC design.")
    if best_metrics["dominated_tasks"]:
        raise ValueError(
            "No non-dominated design was found. Increase search_iterations or revise "
            "preference directions and levels so each task can contain trade-offs."
        )
    return CBCDesign(tasks=best_tasks, profiles=profiles, quality_metrics=best_metrics)
