"""Standardized response-data loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConjointConfig

BASE_COLUMNS = ("respondent_id", "task_id", "alternative_id", "choice")


def load_responses(source: str | Path | pd.DataFrame, config: ConjointConfig | None = None) -> pd.DataFrame:
    """Load standardized long-format CSV data and validate it when configured."""
    if isinstance(source, pd.DataFrame):
        responses = source.copy()
    else:
        responses = pd.read_csv(source, keep_default_na=False)
    validate_response_data(responses, config=config)
    return responses


def _format_set_examples(choice_sets: pd.DataFrame) -> str:
    examples = [f"({row.respondent_id!r}, {row.task_id!r})" for row in choice_sets.itertuples(index=False)]
    return ", ".join(examples[:5])


def validate_response_data(
    data: pd.DataFrame,
    config: ConjointConfig | None = None,
    expected_alternatives: int | None = None,
) -> None:
    """Validate the canonical CBC long format.

    Choice-set identity is always the two-column key ``(respondent_id,
    task_id)``.  No question text is parsed or truncated.
    """
    required = set(BASE_COLUMNS)
    if config is not None:
        required.update(config.attributes)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Response data is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Response data contains no rows.")
    identifier_columns = ["respondent_id", "task_id", "alternative_id"]
    if data[["respondent_id", "task_id"]].isna().any().any():
        raise ValueError("respondent_id, task_id, and alternative_id cannot be missing.")
    if data["alternative_id"].isna().any() or (data["alternative_id"].astype(str).str.strip() == "").any():
        raise ValueError("respondent_id, task_id, and alternative_id cannot be missing.")
    if data.duplicated(identifier_columns).any():
        raise ValueError("Each respondent_id, task_id, alternative_id combination must be unique.")
    if not data["choice"].isin([0, 1, False, True]).all():
        raise ValueError("choice must contain only 0 or 1.")
    choice_sets = data.groupby(["respondent_id", "task_id"], sort=False)
    choice_counts = choice_sets["choice"].sum()
    invalid_choices = choice_counts[choice_counts != 1]
    if not invalid_choices.empty:
        examples = _format_set_examples(invalid_choices.reset_index())
        raise ValueError(
            "Each respondent_id × task_id choice set must contain exactly one choice == 1. "
            f"Invalid sets include: {examples}."
        )
    alternative_counts = choice_sets["alternative_id"].nunique()
    expected = expected_alternatives
    if expected is None and alternative_counts.nunique() == 1:
        expected = int(alternative_counts.iloc[0])
    if expected is not None:
        invalid_alternatives = alternative_counts[alternative_counts != expected]
        if not invalid_alternatives.empty:
            examples = _format_set_examples(invalid_alternatives.reset_index())
            raise ValueError(
                f"Each choice set must contain {expected} alternatives. Invalid sets include: {examples}."
            )
    else:
        raise ValueError("Each choice set must contain the same number of alternatives.")

    if config is None:
        return
    price_name = config.price_attribute
    numeric_price = pd.to_numeric(data[price_name], errors="coerce")
    none_rows = data["alternative_id"].astype(str).str.lower().eq("none")
    if numeric_price[~none_rows].isna().any():
        raise ValueError(f"Price column '{price_name}' must be numeric for non-None alternatives.")
    for name, attribute in config.attributes.items():
        values = data.loc[~none_rows, name]
        if attribute.type in {"numeric", "price"}:
            numeric_values = pd.to_numeric(values, errors="coerce")
            valid_levels = {float(level) for level in attribute.levels}
            invalid_levels = values[
                numeric_values.isna() | ~numeric_values.astype(float).isin(valid_levels)
            ]
        else:
            invalid_levels = values[~values.isin(attribute.levels)]
        if not invalid_levels.empty:
            found = sorted(map(str, invalid_levels.dropna().unique()))
            raise ValueError(
                f"Attribute '{name}' contains values outside its configured levels: {found}."
            )


def wide_to_long(
    data: pd.DataFrame,
    attributes: list[str],
    choice_columns: Mapping[Any, str],
    respondent_column: str = "respondent_id",
    task_column: str = "task_id",
) -> pd.DataFrame:
    """Convert an explicitly named wide layout to the canonical CBC long format.

    ``choice_columns`` maps an alternative label such as ``"A"`` to its wide
    choice indicator.  Attribute columns must be named ``{attribute}_{alt}``.
    Explicit mappings intentionally replace question-text parsing.
    """
    if respondent_column not in data or task_column not in data:
        raise ValueError("Wide data must include respondent and task identifier columns.")
    rows: list[dict[str, Any]] = []
    for row in data.to_dict(orient="records"):
        for alternative_id, choice_column in choice_columns.items():
            required = [choice_column, *(f"{attribute}_{alternative_id}" for attribute in attributes)]
            missing = [column for column in required if column not in data.columns]
            if missing:
                raise ValueError(f"Wide data is missing columns for alternative {alternative_id!r}: {missing}")
            rows.append(
                {
                    "respondent_id": row[respondent_column],
                    "task_id": row[task_column],
                    "alternative_id": alternative_id,
                    "choice": row[choice_column],
                    **{attribute: row[f"{attribute}_{alternative_id}"] for attribute in attributes},
                }
            )
    return pd.DataFrame(rows)
