from pathlib import Path

import pytest

from conjointkit import (
    fit_conditional_logit,
    load_config,
    load_responses,
    predict_choice_probabilities,
)


@pytest.mark.parametrize("example", ["coffee", "ai_subscription"])
def test_examples_run_end_to_end(example):
    root = Path("examples") / example
    config = load_config(root / "config.yaml")
    responses = load_responses(root / "example_responses.csv", config)
    result = fit_conditional_logit(responses, config=config)
    first_task = responses.query("respondent_id == 1 and task_id == 1")
    products = first_task.loc[
        ~first_task["alternative_id"].astype(str).str.lower().eq("none"),
        ["alternative_id", *config.attributes],
    ].rename(columns={"alternative_id": "product"})
    probabilities = predict_choice_probabilities(result, products)
    assert result.n_choice_sets > 0
    assert probabilities["choice_probability"].sum() == pytest.approx(1.0)
