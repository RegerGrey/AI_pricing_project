import pandas as pd
import pytest

from conjointkit import validate_response_data
from conjointkit.models import _unique_choice_set_ids


def test_response_choice_set_validation(response_data, config):
    validate_response_data(response_data, config=config)


def test_exactly_one_choice_per_task(response_data, config):
    broken = response_data.copy()
    first_set = (broken["respondent_id"] == 1) & (broken["task_id"] == 1)
    broken.loc[first_set, "choice"] = 0
    with pytest.raises(ValueError, match="exactly one"):
        validate_response_data(broken, config=config)


def test_unique_choice_set_id_for_same_text_prefix_regression():
    data = pd.DataFrame(
        {
            "respondent_id": [1, 1, 1, 1, 1, 1],
            "task_id": ["Question ABC 1", "Question ABC 1", "Question ABC 1", "Question ABC 2", "Question ABC 2", "Question ABC 2"],
            "alternative_id": ["A", "B", "None", "A", "B", "None"],
            "choice": [1, 0, 0, 0, 1, 0],
        }
    )
    groups = _unique_choice_set_ids(data, "respondent_id", "task_id")
    assert groups.nunique() == 2
    assert groups.iloc[0] != groups.iloc[-1]
