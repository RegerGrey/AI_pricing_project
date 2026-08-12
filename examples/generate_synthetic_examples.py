"""Regenerate the synthetic response files shipped with the examples folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from conjointkit import generate_design, load_config

ROOT = Path(__file__).parent


def _utilities(products: pd.DataFrame, weights: dict[str, dict[object, float] | float]) -> np.ndarray:
    values = np.zeros(len(products))
    for attribute, weight in weights.items():
        if isinstance(weight, dict):
            values += products[attribute].map(weight).fillna(0.0).to_numpy(dtype=float)
        else:
            values += products[attribute].to_numpy(dtype=float) * weight
    return values


def create_example(name: str, weights: dict[str, dict[object, float] | float], seed: int = 7) -> None:
    """Create 80 simulated standardized responses for one example configuration."""
    config = load_config(ROOT / name / "config.yaml")
    design = generate_design(config, search_iterations=100)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for respondent_id in range(1, 81):
        for task_id, task in design.tasks.groupby("task_id", sort=False):
            alternatives = task.copy()
            none = {name: None for name in config.attributes}
            none.update({"task_id": task_id, "alternative_id": "None"})
            alternatives = pd.concat([alternatives, pd.DataFrame([none])], ignore_index=True)
            alternatives["alternative_id"] = alternatives["alternative_id"].astype(object)
            alternatives.loc[alternatives["alternative_id"].isna(), "alternative_id"] = "None"
            utility = _utilities(alternatives.fillna({config.price_attribute: 0}), weights)
            utility[-1] = -0.2
            probabilities = np.exp(utility - utility.max())
            chosen_index = rng.choice(len(alternatives), p=probabilities / probabilities.sum())
            for index, alternative in alternatives.iterrows():
                record = {
                    "respondent_id": respondent_id,
                    "task_id": task_id,
                    "alternative_id": alternative["alternative_id"],
                    "choice": int(index == chosen_index),
                }
                record.update({attribute: alternative[attribute] for attribute in config.attributes})
                rows.append(record)
    pd.DataFrame(rows).to_csv(ROOT / name / "example_responses.csv", index=False)


if __name__ == "__main__":
    create_example(
        "coffee",
        {
            "size": {"Small": 0.0, "Medium": 0.3, "Large": 0.6},
            "milk": {"Regular": 0.0, "Oat": 0.35, "Almond": 0.2},
            "roast": {"Light": 0.0, "Medium": 0.15, "Dark": 0.1},
            "price": -0.06,
        },
    )
    create_example(
        "ai_subscription",
        {
            "intelligence": {"Basic": 0.0, "Advanced": 0.5, "Expert": 0.9},
            "context_window": {"Standard": 0.0, "Extended": 0.3, "Long": 0.6},
            "privacy": {"Default": 0.0, "Private": 0.5},
            "price": -0.025,
        },
        seed=11,
    )
