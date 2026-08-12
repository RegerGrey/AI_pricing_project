"""ConjointKit: a compact toolkit for CBC design and pricing analysis."""

from .config import AttributeConfig, ConjointConfig, DesignConfig, load_config, validate_config
from .data import load_responses, validate_response_data, wide_to_long
from .design import CBCDesign, generate_design, generate_full_factorial
from .models import ConjointResult, fit_conditional_logit
from .simulation import predict_choice_probabilities, simulate_price_curve, simulate_revenue_curve
from .validation import validate_design
from .wtp import calculate_wtp

__all__ = [
    "AttributeConfig",
    "ConjointConfig",
    "DesignConfig",
    "CBCDesign",
    "ConjointResult",
    "load_config",
    "validate_config",
    "generate_full_factorial",
    "generate_design",
    "validate_design",
    "load_responses",
    "validate_response_data",
    "wide_to_long",
    "fit_conditional_logit",
    "calculate_wtp",
    "predict_choice_probabilities",
    "simulate_price_curve",
    "simulate_revenue_curve",
]
