import pytest

from conjointkit import fit_conditional_logit


def test_conditional_logit_runs_on_synthetic_data(response_data, config):
    result = fit_conditional_logit(response_data, config=config)
    assert result.n_choice_sets == 90
    assert result.n_observations == 270
    assert "price" in result.coefficients


def test_zero_variance_feature_has_clear_error(response_data, config):
    response_data["color"] = "Red"
    with pytest.raises(ValueError, match="zero variance"):
        fit_conditional_logit(response_data, config=config)
