"""Minimal Streamlit UI for the ConjointKit v0.1 workflow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
import yaml

from conjointkit import (
    calculate_wtp,
    fit_conditional_logit,
    generate_design,
    load_config,
    load_responses,
    predict_choice_probabilities,
)

st.set_page_config(page_title="ConjointKit", layout="wide")
st.title("ConjointKit")
st.caption("Design CBC experiments, estimate preferences, calculate WTP, and simulate pricing.")


def _config_from_editor() -> object:
    product_name = st.text_input("Product name", value="Coffee Subscription")
    attribute_text = st.text_area(
        "Attributes (one per line: name | type | preference direction | comma-separated levels)",
        value=(
            "size | categorical | higher | Small, Medium, Large\n"
            "milk | categorical | neutral | Regular, Oat, Almond\n"
            "price | price | lower | 15, 22, 30"
        ),
        height=130,
    )
    task_count = st.number_input("Number of tasks", min_value=1, value=9)
    alternative_count = st.number_input("Alternatives per task", min_value=2, value=2)
    attributes: dict[str, dict[str, object]] = {}
    for line in attribute_text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4:
            raise ValueError("Each attribute line must have four fields separated by |.")
        name, attribute_type, direction, raw_levels = parts
        levels: list[object] = [level.strip() for level in raw_levels.split(",")]
        if attribute_type in {"numeric", "price"}:
            levels = [float(level) for level in levels]
        attributes[name] = {
            "type": attribute_type,
            "preference_direction": direction,
            "levels": levels,
        }
    return load_config(
        {
            "product_name": product_name,
            "attributes": attributes,
            "options": {"include_none": True},
            "design": {
                "num_tasks": int(task_count),
                "alternatives_per_task": int(alternative_count),
                "random_seed": 42,
            },
        }
    )


design_tab, analyze_tab, wtp_tab, simulator_tab = st.tabs(["Design", "Analyze", "WTP", "Simulator"])

with design_tab:
    st.subheader("Balanced randomized CBC design")
    try:
        design_config = _config_from_editor()
        if st.button("Generate design", type="primary"):
            design = generate_design(design_config)
            st.session_state["config"] = design_config
            st.session_state["design"] = design
            st.success("Design generated.")
        if "design" in st.session_state:
            design = st.session_state["design"]
            st.dataframe(design.tasks, use_container_width=True)
            metrics = design.quality_metrics
            first, second, third = st.columns(3)
            first.metric("Level-balance heuristic", f"{metrics['level_balance_score']:.2f}")
            second.metric("Max attribute correlation", f"{metrics['max_attribute_correlation']:.2f}")
            third.metric("Dominated tasks", metrics["dominated_tasks"])
            st.download_button(
                "Download design CSV",
                data=design.tasks.to_csv(index=False),
                file_name="cbc_design.csv",
                mime="text/csv",
            )
    except ValueError as error:
        st.error(str(error))

with analyze_tab:
    st.subheader("Estimate a Conditional Logit model")
    st.write("Upload canonical long data: respondent_id, task_id, alternative_id, choice, and attributes.")
    uploaded_config = st.file_uploader("Optional YAML configuration", type=["yaml", "yml"], key="config_upload")
    if uploaded_config is not None:
        try:
            st.session_state["config"] = load_config(
                yaml.safe_load(uploaded_config.getvalue().decode("utf-8"))
            )
        except ValueError as error:
            st.error(str(error))
    uploaded_responses = st.file_uploader("Response CSV", type=["csv"], key="response_upload")
    if uploaded_responses is not None and st.button("Validate data and fit model", type="primary"):
        config = st.session_state.get("config")
        if config is None:
            st.error("Generate a design first or upload its YAML configuration.")
        else:
            try:
                responses = load_responses(uploaded_responses, config=config)
                result = fit_conditional_logit(responses, config=config)
                st.session_state["result"] = result
                st.success(f"Fitted {result.n_choice_sets} choice sets and {result.n_observations} rows.")
                st.dataframe(result.summary_frame(), use_container_width=True)
            except ValueError as error:
                st.error(str(error))

with wtp_tab:
    st.subheader("Willingness to pay")
    result = st.session_state.get("result")
    if result is None:
        st.info("Fit a model in Analyze first.")
    else:
        try:
            wtp = calculate_wtp(result)
            st.dataframe(wtp, use_container_width=True)
            figure, axis = plt.subplots()
            axis.bar(wtp["feature"], wtp["wtp"])
            axis.set_ylabel("WTP")
            axis.tick_params(axis="x", rotation=45)
            st.pyplot(figure)
        except ValueError as error:
            st.error(str(error))

with simulator_tab:
    st.subheader("Choice probability simulator")
    result = st.session_state.get("result")
    config = st.session_state.get("config")
    if result is None or config is None:
        st.info("Generate/upload a configuration and fit a model first.")
    else:
        products: list[dict[str, object]] = []
        columns = st.columns(2)
        for index, label in enumerate(["Product A", "Product B"]):
            with columns[index]:
                st.markdown(f"**{label}**")
                product: dict[str, object] = {"product": label}
                for name, attribute in config.attributes.items():
                    if attribute.type == "price":
                        product[name] = st.number_input(
                            f"{label} {name}", value=float(attribute.levels[0]), key=f"{label}_{name}"
                        )
                    else:
                        product[name] = st.selectbox(
                            f"{label} {name}", attribute.levels, key=f"{label}_{name}"
                        )
                products.append(product)
        if st.button("Simulate probabilities", type="primary"):
            try:
                st.dataframe(predict_choice_probabilities(result, products), use_container_width=True)
            except ValueError as error:
                st.error(str(error))
