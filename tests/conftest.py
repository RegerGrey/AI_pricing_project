from __future__ import annotations

import pandas as pd
import pytest

from conjointkit import generate_design, load_config


@pytest.fixture
def config():
    return load_config(
        {
            "product_name": "Test product",
            "attributes": {
                "quality": {
                    "type": "categorical",
                    "levels": ["Basic", "Premium"],
                    "preference_direction": "higher",
                },
                "color": {
                    "type": "categorical",
                    "levels": ["Red", "Blue"],
                    "preference_direction": "neutral",
                },
                "price": {
                    "type": "price",
                    "levels": [10, 20, 30],
                    "preference_direction": "lower",
                },
            },
            "design": {"num_tasks": 3, "alternatives_per_task": 2, "random_seed": 4},
        }
    )


@pytest.fixture
def response_data(config):
    design = generate_design(config, search_iterations=300)
    rows = []
    for respondent_id in range(1, 31):
        for task_id, task in design.tasks.groupby("task_id", sort=False):
            alternatives = task.copy()
            none = {name: None for name in config.attributes}
            none.update({"task_id": task_id, "alternative_id": "None"})
            alternatives = pd.concat([alternatives, pd.DataFrame([none])], ignore_index=True)
            alternatives["alternative_id"] = alternatives["alternative_id"].astype(object)
            alternatives.loc[alternatives["alternative_id"].isna(), "alternative_id"] = "None"
            chosen = (respondent_id + int(task_id)) % len(alternatives)
            for index, alternative in alternatives.iterrows():
                rows.append(
                    {
                        "respondent_id": respondent_id,
                        "task_id": task_id,
                        "alternative_id": alternative["alternative_id"],
                        "choice": int(index == chosen),
                        "quality": alternative["quality"],
                        "color": alternative["color"],
                        "price": alternative["price"],
                    }
                )
    return pd.DataFrame(rows)
