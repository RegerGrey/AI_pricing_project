import pytest

from conjointkit import fit_conditional_logit, predict_choice_probabilities, simulate_revenue_curve


def test_probability_sum_equals_one(response_data, config):
    result = fit_conditional_logit(response_data, config=config)
    probabilities = predict_choice_probabilities(
        result,
        [
            {"product": "A", "quality": "Premium", "color": "Red", "price": 20},
            {"product": "B", "quality": "Basic", "color": "Blue", "price": 10},
        ],
    )
    assert probabilities["choice_probability"].sum() == pytest.approx(1.0)


def test_revenue_curve():
    revenue = simulate_revenue_curve([10, 20], [0.5, 0.25], market_size=100)
    assert revenue["revenue_index"].tolist() == [500.0, 500.0]
