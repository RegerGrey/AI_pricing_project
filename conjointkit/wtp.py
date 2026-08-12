"""Willingness-to-pay calculations."""

from __future__ import annotations

import warnings

import pandas as pd

from .models import ConjointResult


def calculate_wtp(model_result: ConjointResult, price_variable: str | None = None) -> pd.DataFrame:
    """Calculate feature WTP as ``-beta_feature / beta_price``.

    A non-negative price coefficient is retained in the result but warned about,
    because its economic interpretation makes conventional WTP unreliable.
    """
    price_name = price_variable or model_result.price_variable
    if price_name not in model_result.coefficients.index:
        raise ValueError(f"Price variable '{price_name}' is absent from the fitted model.")
    price_coefficient = float(model_result.coefficients[price_name])
    if price_coefficient == 0:
        raise ValueError("The estimated price coefficient is zero; WTP is undefined.")
    if price_coefficient >= 0:
        warnings.warn(
            "The estimated price coefficient is non-negative. WTP interpretation may not be economically meaningful.",
            UserWarning,
            stacklevel=2,
        )
    rows: list[dict[str, object]] = []
    for feature, coefficient in model_result.coefficients.items():
        if feature == price_name:
            continue
        metadata = model_result.feature_metadata.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "attribute": metadata.get("attribute", feature),
                "level": metadata.get("level"),
                "coefficient": float(coefficient),
                "wtp": -float(coefficient) / price_coefficient,
                "reference_level": metadata.get("reference_level"),
            }
        )
    return pd.DataFrame(rows)
