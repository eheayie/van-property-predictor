import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from common import CAT_COLS, FEATURES, explain_rows, make_shap_explainer

st.set_page_config(page_title="Vancouver Property Value Predictor", page_icon="🏠")


@st.cache_resource
def load_artifacts():
    if not os.path.exists("model.joblib"):
        import train
        train.main()  # train on first launch (useful on Streamlit Cloud)
    bundle = joblib.load("model.joblib")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return bundle, metrics


@st.cache_resource
def get_explainer(_model, _background):
    return make_shap_explainer(_model, _background)


bundle, metrics = load_artifacts()
model, cats, defaults = bundle["model"], bundle["categories"], bundle["defaults"]
year = bundle["assessment_year"]

st.title("Vancouver Property Value Predictor")
st.write(
    f"Estimates the **BC Assessment value** of a Vancouver property ({year} assessments) "
    "from its type, zoning, location, and age, using a gradient boosting model trained on the "
    "City of Vancouver Open Data property tax report."
)


def pick(label, col):
    options = cats[col]
    default = options.index(defaults[col]) if defaults[col] in options else 0
    return st.sidebar.selectbox(label, options, index=default)


st.sidebar.header("Property details")
legal_type = pick("Property type (STRATA = condo/townhouse)", "legal_type")
zoning = pick("Zoning classification", "zoning_classification")
neighbourhood = pick("BC Assessment neighbourhood code", "neighbourhood_code")
fsa = pick("Postal code area (first 3 characters)", "postal_fsa")
year_built = st.sidebar.slider("Year built", 1890, year, int(defaults["year_built"]))
renovated = st.sidebar.checkbox("Had a major improvement/renovation")
improvement_year = (st.sidebar.slider("Year of major improvement", year_built, year, max(year_built, 2005))
                    if renovated else None)

row = pd.DataFrame([{
    "legal_type": legal_type,
    "zoning_classification": zoning,
    "neighbourhood_code": neighbourhood,
    "postal_fsa": fsa,
    "year_built": float(year_built),
    "property_age": float(year - year_built),
    "years_since_improvement": float(year - improvement_year) if improvement_year else np.nan,
}])[FEATURES]

log_pred = float(model.predict(row)[0])
prediction = float(np.exp(log_pred))
typical_err = metrics["results"][metrics["best"]]["median_abs_pct_error"]

st.metric("Predicted assessed value (CAD)", f"${prediction:,.0f}")
st.caption(f"Typical error on held-out properties: about {typical_err:.0f}% (median absolute % error).")

st.subheader("Why this prediction?")
explainer = get_explainer(model, bundle["background"])
explanation = explain_rows(explainer, model, row)
fig = plt.figure()
shap.plots.waterfall(explanation[0], show=False)
st.pyplot(fig, clear_figure=True)
st.caption("Bars show how each feature pushed the prediction up or down, on a log scale "
           "(a +0.10 push is roughly +10%) relative to the average Vancouver property.")

with st.expander("What drives the model overall? (SHAP)"):
    if os.path.exists("shap_importance.png"):
        st.image("shap_importance.png")

with st.expander("Model performance"):
    res = pd.DataFrame(metrics["results"]).T
    table = res[["mae", "median_abs_pct_error", "r2_log"]].rename(columns={
        "mae": "MAE ($)", "median_abs_pct_error": "Median % error", "r2_log": "R² (log value)"})
    st.dataframe(table.style.format({"MAE ($)": "{:,.0f}", "Median % error": "{:.1f}",
                                     "R² (log value)": "{:.3f}"}))
    st.write(f"Trained on **{metrics['n_properties']:,}** properties. Gradient boosting cuts MAE by "
             f"**{metrics['mae_improvement_pct_vs_baseline']:.1f}%** vs. a baseline that predicts the "
             "median value for each neighbourhood and property type.")

with st.expander("Limitations"):
    st.write(
        "- Covers the **City of Vancouver only**, not all of British Columbia.\n"
        "- Predicts BC Assessment *assessed* values, which approximate but do not equal sale prices.\n"
        "- No square footage, lot size, bedrooms, or condition, so very similar-looking "
        "properties can have very different values.\n"
        "- Neighbourhood codes are BC Assessment codes; names are not provided in the open data."
    )
