import pandas as pd

from conjointkit import validate_design
from conjointkit.validation import check_dominance


def test_dominance_detection(config):
    tasks = pd.DataFrame(
        {
            "task_id": [1, 1],
            "alternative_id": ["A", "B"],
            "quality": ["Premium", "Basic"],
            "color": ["Red", "Blue"],
            "price": [10, 30],
        }
    )
    finding = check_dominance(tasks, config)
    assert finding[0]["dominant_alternative"] == "A"


def test_neutral_attribute_not_used_for_dominance(config):
    tasks = pd.DataFrame(
        {
            "task_id": [1, 1],
            "alternative_id": ["A", "B"],
            "quality": ["Basic", "Basic"],
            "color": ["Red", "Blue"],
            "price": [20, 20],
        }
    )
    assert check_dominance(tasks, config) == []


def test_design_metrics_are_transparent(config):
    tasks = pd.DataFrame(
        {
            "task_id": [1, 1, 2, 2],
            "alternative_id": ["A", "B", "A", "B"],
            "quality": ["Basic", "Premium", "Premium", "Basic"],
            "color": ["Red", "Blue", "Red", "Blue"],
            "price": [10, 30, 30, 10],
        }
    )
    metrics = validate_design(tasks, config)
    assert "level_balance_score" in metrics
    assert "overall_quality" not in metrics
