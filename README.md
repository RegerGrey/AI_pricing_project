# ConjointKit

Design CBC experiments, estimate preferences, calculate willingness to pay, and simulate pricing — in Python.

ConjointKit is a small, open-source Python toolkit for Choice-Based Conjoint (CBC) work. It replaces product-specific survey parsing with a clear configuration file and a standard long response format. Version 0.1 focuses on a correct, inspectable workflow: a balanced randomized design, Conditional Logit estimation, WTP, and simple scenario simulation.

## Why ConjointKit

- General configuration: attributes, levels, price, task count, and preference direction are supplied by YAML or a Python dictionary.
- Explicit choice sets: every model group is defined by `(respondent_id, task_id)`, never by a truncated question label.
- Data-quality checks: each choice set must have a consistent number of alternatives and exactly one selected alternative.
- Transparent methods: the design is labelled *balanced randomized CBC design*, not an unimplemented optimal-design method.
- Runnable examples: AI subscription and coffee subscription examples use synthetic, non-respondent data.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/RegerGrey/AI_pricing_project.git
cd AI_pricing_project
pip install -e .[dev]
```

For the interface, the base installation already includes Streamlit and Matplotlib.

## Quick start

```python
from pathlib import Path

from conjointkit import (
    calculate_wtp,
    fit_conditional_logit,
    generate_design,
    load_config,
    load_responses,
    predict_choice_probabilities,
)

config = load_config(Path("examples/coffee/config.yaml"))
design = generate_design(config)
responses = load_responses("examples/coffee/example_responses.csv", config)
result = fit_conditional_logit(responses, config=config)

print(result.summary_frame())
print(calculate_wtp(result))
print(
    predict_choice_probabilities(
        result,
        [
            {"product": "House blend", "size": "Large", "milk": "Oat", "roast": "Medium", "price": 22},
            {"product": "Value blend", "size": "Small", "milk": "Regular", "roast": "Light", "price": 15},
        ],
    )
)
```

## Define attributes

Write a YAML configuration. There must be at least two attributes, every attribute needs at least two levels, and exactly one `price` attribute is required for WTP analysis.

```yaml
product_name: Coffee Subscription
attributes:
  size:
    type: categorical
    levels: [Small, Medium, Large]
    preference_direction: higher
  milk:
    type: categorical
    levels: [Regular, Oat, Almond]
    preference_direction: neutral
  price:
    type: price
    levels: [15, 22, 30]
    preference_direction: lower
options:
  include_none: true
design:
  num_tasks: 10
  alternatives_per_task: 2
  random_seed: 42
```

`higher` and `lower` are used only for dominance diagnostics. `neutral` attributes are deliberately excluded from that check. For categorical features, configured level order is used as a transparent ordinal heuristic in design diagnostics; it does not turn nominal levels into a cardinal scale.

## Generate a CBC design

```python
from conjointkit import generate_design, load_config

config = load_config("examples/coffee/config.yaml")
design = generate_design(config)
design.tasks.to_csv("cbc_design.csv", index=False)
print(design.quality_metrics["level_balance_score"])
```

`generate_design()` samples unique profiles from the full factorial and ranks candidates using dominance count, duplicate count, level balance, and ordinal-encoded attribute correlation. It is **not** D-efficient, A-efficient, or otherwise statistically optimal.

## Analyze responses

The core API requires canonical long data. It does not parse question text or infer task identifiers.

```csv
respondent_id,task_id,alternative_id,choice,size,milk,roast,price
1,1,A,1,Large,Oat,Medium,30
1,1,B,0,Small,Regular,Light,15
1,1,None,0,,,,
```

Use a unique `(respondent_id, task_id)` pair for every choice set. ConjointKit verifies that all sets contain the same number of alternatives and that exactly one row has `choice == 1`. This prevents accidental merging of separate questions whose labels share a text prefix.

```python
from conjointkit import fit_conditional_logit, load_responses

responses = load_responses("responses.csv", config)
result = fit_conditional_logit(responses, config=config)
print(result.summary_frame())
```

Categorical attributes are dummy-coded with the first configured level as the reference. The estimator does not silently remove zero-variance or collinear terms: it raises a clear error so the researcher can revise the design or specification intentionally.

`wide_to_long()` is available for explicitly mapped wide layouts. It expects columns such as `size_A`, `size_B`, and supplied identifiers; it is only a convenience helper, not a question-text parser.

## Calculate WTP

```python
from conjointkit import calculate_wtp

wtp = calculate_wtp(result)
print(wtp[["feature", "coefficient", "wtp", "reference_level"]])
```

For each non-price coefficient, ConjointKit calculates `WTP = -beta_feature / beta_price`. If the price coefficient is non-negative, it emits a warning because the usual economic WTP interpretation may not be meaningful.

## Simulate pricing

```python
from conjointkit import (
    predict_choice_probabilities,
    simulate_price_curve,
    simulate_revenue_curve,
)

products = [
    {"product": "Product A", "size": "Large", "milk": "Oat", "roast": "Medium", "price": 22},
    {"product": "Product B", "size": "Small", "milk": "Regular", "roast": "Light", "price": 15},
]
probabilities = predict_choice_probabilities(result, products)
price_curve = simulate_price_curve(result, products, "Product A", prices=range(15, 36, 5))
revenue = simulate_revenue_curve(
    price_curve["price"], price_curve["choice_probability"], market_size=1_000
)
```

Probabilities are model predictions conditional on the products supplied to the scenario. The revenue index is `price × predicted demand`; it is not a guarantee of real-world market share, sales, or revenue.

## Streamlit app

Run the minimal UI from the repository root:

```bash
streamlit run app/streamlit_app.py
```

The app provides four tabs: Design, Analyze, WTP, and Simulator. It is intentionally small in v0.1 and uses the same public APIs as the Python workflow.

## Examples

- `examples/coffee/`: a coffee subscription configuration and synthetic responses.
- `examples/ai_subscription/`: an AI subscription configuration retained as a generic example, without AI-specific assumptions in library code.

To regenerate both synthetic CSV files after installation:

```bash
python examples/generate_synthetic_examples.py
```

## Methodology

The estimation model is `statsmodels.discrete.conditional_models.ConditionalLogit`. A choice set is the explicit respondent-task pair, and the model compares alternatives only within that set. The current design generator uses a full-factorial candidate universe, then a reproducible randomized search. It avoids duplicate profiles within a task, rejects designs with detected dominance, and reports balance and correlation diagnostics.

## Limitations

- v0.1 has Conditional Logit only; it does not include mixed logit, latent class, hierarchical Bayes, segmentation, or Bayesian estimation.
- The design search is heuristic and does not claim D-, A-, or C-efficiency.
- WTP inherits the assumptions and uncertainty of the estimated model and requires an economically interpretable price coefficient.
- Simulations are model-based scenarios, not market forecasts.
- Researchers remain responsible for sampling, questionnaire wording, experimental validity, consent, privacy, and interpretation.

## Roadmap

- Design diagnostics that do not rely on ordinal encoding for nominal attributes.
- Optional resampling-based uncertainty intervals for WTP.
- Import adapters for commonly exported survey formats, while retaining explicit task identifiers.
- Additional discrete-choice estimators after their assumptions and tests are clearly specified.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `pytest` and `ruff check .` before opening a pull request.

## License

MIT License. See [LICENSE](LICENSE).

## Repository hygiene

The v0.1 working tree removes prior survey spreadsheets and related research or competition documents because their anonymization status or redistribution rights cannot be confirmed. Historical commits may still contain previously committed research files. Consider `git filter-repo` if permanent removal is required.
