"""Shared code for training and the Streamlit app.

Data: City of Vancouver "Property Tax Report" (open data, Open Government Licence - Vancouver).
Each row is a property with its BC Assessment values, zoning, year built, etc.
Target: total assessed value = current_land_value + current_improvement_value.
"""
import io
import os
from datetime import date

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

API_BASE = "https://opendata.vancouver.ca/api/explore/v2.1"
DATASET = "property-tax-report"
LOCAL_CSV = os.path.join("data", "property-tax-report.csv")

CAT_COLS = ["legal_type", "zoning_classification", "neighbourhood_code", "postal_fsa"]
NUM_COLS = ["year_built", "property_age", "years_since_improvement"]
FEATURES = CAT_COLS + NUM_COLS
USE_COLS = [
    "pid", "legal_type", "zoning_classification", "neighbourhood_code",
    "property_postal_code", "year_built", "big_improvement_year",
    "current_land_value", "current_improvement_value",
    "tax_assessment_year", "report_year",
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _download_year(year: int) -> pd.DataFrame:
    url = f"{API_BASE}/catalog/datasets/{DATASET}/exports/csv"
    params = {
        "where": f'report_year="{year}"',
        "select": ",".join(c for c in USE_COLS),
        "delimiter": ",",
        "limit": -1,
    }
    resp = requests.get(url, params=params, timeout=300)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), dtype=str)


def load_raw() -> pd.DataFrame:
    """Use data/property-tax-report.csv if present, otherwise download the latest year."""
    if os.path.exists(LOCAL_CSV):
        with open(LOCAL_CSV, encoding="utf-8-sig") as f:
            first = f.readline()
        sep = ";" if first.count(";") > first.count(",") else ","
        df = pd.read_csv(LOCAL_CSV, sep=sep, dtype=str, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    for year in range(date.today().year, date.today().year - 4, -1):
        try:
            df = _download_year(year)
        except requests.RequestException as exc:
            print(f"Download for {year} failed: {exc}")
            continue
        if len(df) > 1000:
            print(f"Downloaded {len(df):,} rows for report year {year}.")
            os.makedirs("data", exist_ok=True)
            df.to_csv(LOCAL_CSV, index=False)
            return df
    raise RuntimeError(
        "Could not download the data. Download the CSV manually from "
        "https://opendata.vancouver.ca/explore/dataset/property-tax-report/ "
        "(Export tab) and save it as data/property-tax-report.csv"
    )


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw CSV rows -> model features (used for training AND app inference)."""
    out = pd.DataFrame(index=df.index)
    out["legal_type"] = df["legal_type"].fillna("Unknown").str.upper()
    out["zoning_classification"] = df["zoning_classification"].fillna("Unknown")
    out["neighbourhood_code"] = df["neighbourhood_code"].fillna("Unknown").astype(str)
    out["postal_fsa"] = (
        df["property_postal_code"].fillna("Unknown").astype(str).str.replace(" ", "").str[:3].str.upper()
    )

    year_built = pd.to_numeric(df["year_built"], errors="coerce")
    year_built = year_built.where((year_built >= 1850) & (year_built <= date.today().year))
    big_impr = pd.to_numeric(df["big_improvement_year"], errors="coerce")
    big_impr = big_impr.where((big_impr >= 1850) & (big_impr <= date.today().year))
    assess_year = pd.to_numeric(df["tax_assessment_year"], errors="coerce")

    out["year_built"] = year_built
    out["property_age"] = assess_year - year_built
    out["years_since_improvement"] = assess_year - big_impr
    return out[FEATURES]


def build_target(df: pd.DataFrame) -> pd.Series:
    land = pd.to_numeric(df["current_land_value"], errors="coerce")
    impr = pd.to_numeric(df["current_improvement_value"], errors="coerce")
    return land + impr


def prepare_dataset():
    raw = load_raw()
    raw.columns = [c.strip().lower() for c in raw.columns]
    # Keep only the most recent report year so values are comparable
    if "report_year" in raw.columns:
        latest = raw["report_year"].dropna().astype(str).max()
        raw = raw[raw["report_year"].astype(str) == latest]
    y = build_target(raw)
    mask = y.notna() & (y > 50_000)  # drop unassessed / placeholder rows
    raw, y = raw[mask], y[mask]
    X = build_features(raw)
    assess_year = pd.to_numeric(raw["tax_assessment_year"], errors="coerce").max()
    return X, y, int(assess_year) if pd.notna(assess_year) else date.today().year


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def make_model(random_state: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), CAT_COLS)],
        remainder="passthrough",
    )
    gbm = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CAT_COLS))),  # first columns after encoding
        max_iter=400, learning_rate=0.06, random_state=random_state,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def encode_for_shap(model, df: pd.DataFrame) -> np.ndarray:
    """Numeric version of the features (categories ordinal-encoded) so SHAP can mask them."""
    return model.named_steps["pre"].transform(df[FEATURES]).astype(float)


def make_shap_explainer(model, background_df: pd.DataFrame):
    """Explainer over the encoded feature space (columns are in FEATURES order)."""
    import shap

    gbm = model.named_steps["gbm"]
    return shap.Explainer(gbm.predict, encode_for_shap(model, background_df),
                          feature_names=FEATURES)


def explain_rows(explainer, model, df: pd.DataFrame):
    """SHAP values for rows of df. Shows the original (readable) values in plots."""
    import shap

    sv = explainer(encode_for_shap(model, df))
    return shap.Explanation(
        values=sv.values, base_values=sv.base_values,
        data=df[FEATURES].to_numpy(dtype=object), feature_names=FEATURES,
    )
