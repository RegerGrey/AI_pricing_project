"""Conditional-logit preference estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

from .config import ConjointConfig
from .data import validate_response_data


@dataclass
class ConjointResult:
    """Estimated conditional-logit parameters plus design metadata."""

    coefficients: pd.Series
    standard_errors: pd.Series
    pvalues: pd.Series
    confidence_intervals: pd.DataFrame
    log_likelihood: float
    n_observations: int
    n_choice_sets: int
    feature_metadata: dict[str, dict[str, str | None]]
    price_variable: str
    fitted_model: object | None = None

    def summary_frame(self) -> pd.DataFrame:
        """Return the standard coefficient summary as a DataFrame."""
        return pd.DataFrame(
            {
                "coefficient": self.coefficients,
                "standard_error": self.standard_errors,
                "pvalue": self.pvalues,
                "ci_lower": self.confidence_intervals["lower"],
                "ci_upper": self.confidence_intervals["upper"],
            }
        )


def _unique_choice_set_ids(data: pd.DataFrame, respondent_column: str, task_column: str) -> pd.Series:
    """Build collision-free choice-set identifiers from the two canonical IDs."""
    pairs = list(zip(data[respondent_column].astype(str), data[task_column].astype(str)))
    codes, _ = pd.factorize(pd.Series(pairs, index=data.index), sort=False)
    return pd.Series(codes, index=data.index, name="choice_set_id")


def _build_model_matrix(
    data: pd.DataFrame,
    attributes: list[str],
    price_column: str,
    config: ConjointConfig | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, str | None]]]:
    matrix = pd.DataFrame(index=data.index)
    metadata: dict[str, dict[str, str | None]] = {}
    for attribute in attributes:
        if attribute == price_column:
            numeric = pd.to_numeric(data[attribute], errors="coerce")
            none_rows = data["alternative_id"].astype(str).str.lower().eq("none")
            if numeric[~none_rows].isna().any():
                raise ValueError(f"Price attribute '{attribute}' must be numeric.")
            matrix[attribute] = numeric.fillna(0.0)
            metadata[attribute] = {"attribute": attribute, "level": None, "reference_level": None}
            continue
        none_rows = data["alternative_id"].astype(str).str.lower().eq("none")
        if config is not None and attribute in config.attributes:
            configured_levels = list(config.attributes[attribute].levels)
            observed_levels = set(data.loc[~none_rows, attribute].dropna().unique())
            unexpected = observed_levels.difference(configured_levels)
            if unexpected:
                raise ValueError(f"Attribute '{attribute}' contains unconfigured levels: {sorted(unexpected)}")
            categories = configured_levels
        else:
            categories = list(pd.unique(data[attribute].dropna()))
        if len(categories) < 2:
            raise ValueError(f"Attribute '{attribute}' has fewer than two levels and cannot be estimated.")
        reference = categories[0]
        for level in categories[1:]:
            name = f"{attribute}[{level}]"
            matrix[name] = (data[attribute] == level).astype(float)
            metadata[name] = {
                "attribute": attribute,
                "level": str(level),
                "reference_level": str(reference),
            }
    zero_variance = [name for name in matrix if matrix[name].nunique(dropna=False) <= 1]
    if zero_variance:
        raise ValueError(
            "The following variables have zero variance and cannot be estimated: "
            f"{', '.join(zero_variance)}."
        )
    return matrix, metadata


def fit_conditional_logit(
    data: pd.DataFrame,
    attributes: list[str] | None = None,
    price_column: str = "price",
    respondent_column: str = "respondent_id",
    task_column: str = "task_id",
    config: ConjointConfig | None = None,
) -> ConjointResult:
    """Fit a conditional logit using the explicit ``respondent_id × task_id`` key.

    Categorical features are dummy-coded with the first configured level as
    reference.  The function fails rather than silently dropping zero-variance
    features or otherwise changing the requested model.
    """
    if respondent_column != "respondent_id" or task_column != "task_id":
        renamed = data.rename(
            columns={respondent_column: "respondent_id", task_column: "task_id"}
        )
    else:
        renamed = data.copy()
    validate_response_data(renamed, config=config)
    if attributes is None:
        attributes = list(config.attributes) if config is not None else [price_column]
    if price_column not in attributes:
        attributes = [*attributes, price_column]
    missing = sorted(set(attributes).difference(data.columns))
    if missing:
        raise ValueError(f"Model attributes are missing from response data: {missing}")
    if price_column not in data:
        raise ValueError(f"Price column '{price_column}' is missing from response data.")

    model_data = data.copy()
    model_data["choice_set_id"] = _unique_choice_set_ids(data, respondent_column, task_column)
    matrix, metadata = _build_model_matrix(model_data, attributes, price_column, config)
    if matrix.shape[1] == 0:
        raise ValueError("No estimable model variables were created.")
    rank = int(np.linalg.matrix_rank(matrix.to_numpy(dtype=float)))
    if rank < matrix.shape[1]:
        raise ValueError(
            "Model matrix is singular or perfectly collinear. Revise the design or remove "
            "the collinear variable explicitly."
        )
    try:
        fitted = ConditionalLogit(
            model_data["choice"].astype(int),
            matrix.astype(float),
            groups=model_data["choice_set_id"],
        ).fit(disp=False)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ValueError(
            "Conditional Logit could not be estimated. Check choice-set validation, "
            "zero-variance variables, and collinearity."
        ) from error
    confidence = fitted.conf_int()
    confidence.columns = ["lower", "upper"]
    return ConjointResult(
        coefficients=fitted.params,
        standard_errors=fitted.bse,
        pvalues=fitted.pvalues,
        confidence_intervals=confidence,
        log_likelihood=float(fitted.llf),
        n_observations=int(len(model_data)),
        n_choice_sets=int(model_data["choice_set_id"].nunique()),
        feature_metadata=metadata,
        price_variable=price_column,
        fitted_model=fitted,
    )
