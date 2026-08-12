"""Model-based choice-probability and pricing simulation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .models import ConjointResult


def _as_products(products: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(products, pd.DataFrame):
        frame = products.copy()
    else:
        frame = pd.DataFrame(list(products))
    if "product" not in frame:
        raise ValueError("Products must include a 'product' column.")
    if frame.empty:
        raise ValueError("At least one product is required for a simulation.")
    return frame


def _feature_matrix(model_result: ConjointResult, products: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=products.index, columns=model_result.coefficients.index)
    for feature, metadata in model_result.feature_metadata.items():
        attribute = metadata.get("attribute")
        if attribute is None:
            continue
        if attribute not in products:
            raise ValueError(f"Products are missing required attribute '{attribute}'.")
        level = metadata.get("level")
        if level is None:
            numeric = pd.to_numeric(products[attribute], errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"Product attribute '{attribute}' must be numeric.")
            matrix[feature] = numeric
        else:
            matrix[feature] = (products[attribute].astype(str) == level).astype(float)
    return matrix


def predict_choice_probabilities(
    model_result: ConjointResult,
    products: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Calculate multinomial choice probabilities for a supplied product set."""
    product_frame = _as_products(products)
    matrix = _feature_matrix(model_result, product_frame)
    utilities = matrix.to_numpy() @ model_result.coefficients.to_numpy(dtype=float)
    shifted = utilities - np.max(utilities)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return pd.DataFrame(
        {
            "product": product_frame["product"].to_numpy(),
            "utility": utilities,
            "choice_probability": probabilities,
        }
    )


def simulate_price_curve(
    model_result: ConjointResult,
    products: pd.DataFrame | Iterable[Mapping[str, Any]],
    product: str,
    prices: Iterable[float],
    price_column: str | None = None,
) -> pd.DataFrame:
    """Vary one product's price and return its predicted choice probability."""
    price_name = price_column or model_result.price_variable
    product_frame = _as_products(products)
    if price_name not in product_frame:
        raise ValueError(f"Products are missing price column '{price_name}'.")
    if product not in set(product_frame["product"]):
        raise ValueError(f"Product '{product}' is not in the supplied product set.")
    rows: list[dict[str, float | str]] = []
    for price in prices:
        scenario = product_frame.copy()
        scenario.loc[scenario["product"] == product, price_name] = price
        predicted = predict_choice_probabilities(model_result, scenario)
        probability = float(
            predicted.loc[predicted["product"] == product, "choice_probability"].iloc[0]
        )
        rows.append({"product": product, "price": float(price), "choice_probability": probability})
    return pd.DataFrame(rows)


def simulate_revenue_curve(
    price: float | pd.Series | np.ndarray,
    probability: float | pd.Series | np.ndarray,
    market_size: float = 1.0,
) -> pd.DataFrame:
    """Return model-based demand and revenue index for supplied price scenarios."""
    prices = np.atleast_1d(np.asarray(price, dtype=float))
    probabilities = np.atleast_1d(np.asarray(probability, dtype=float))
    if prices.shape != probabilities.shape:
        raise ValueError("price and probability must have the same shape.")
    if market_size <= 0:
        raise ValueError("market_size must be greater than zero.")
    demand = probabilities * market_size
    return pd.DataFrame({"price": prices, "predicted_demand": demand, "revenue_index": prices * demand})
