import pandas as pd
import pytest

from conjointkit import calculate_wtp
from conjointkit.models import ConjointResult


def _result(price: float) -> ConjointResult:
    return ConjointResult(
        coefficients=pd.Series({"price": price, "quality[Premium]": 0.4}),
        standard_errors=pd.Series({"price": 0.1, "quality[Premium]": 0.1}),
        pvalues=pd.Series({"price": 0.01, "quality[Premium]": 0.01}),
        confidence_intervals=pd.DataFrame(
            {"lower": [-0.3, 0.2], "upper": [-0.1, 0.6]}, index=["price", "quality[Premium]"]
        ),
        log_likelihood=-1.0,
        n_observations=3,
        n_choice_sets=1,
        feature_metadata={
            "price": {"attribute": "price", "level": None, "reference_level": None},
            "quality[Premium]": {
                "attribute": "quality",
                "level": "Premium",
                "reference_level": "Basic",
            },
        },
        price_variable="price",
    )


def test_wtp_calculation():
    wtp = calculate_wtp(_result(-0.2))
    assert wtp.loc[0, "wtp"] == pytest.approx(2.0)
    assert wtp.loc[0, "reference_level"] == "Basic"


def test_price_coefficient_warning():
    with pytest.warns(UserWarning, match="non-negative"):
        calculate_wtp(_result(0.2))
