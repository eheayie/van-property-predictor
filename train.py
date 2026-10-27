"""Train and evaluate the Vancouver property value model.

Run:  python train.py
Outputs: model.joblib, metrics.json, shap_summary.png, shap_importance.png
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from common import CAT_COLS, FEATURES, NUM_COLS, make_model, predict_log_fn, prepare_dataset

RANDOM_STATE = 42


def money_metrics(y_true, pred):
    """Metrics in dollars, given true values and predicted values (both in $)."""
    err = pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "median_abs_pct_error": float(np.median(np.abs(err) / y_true) * 100),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2_log": float(1 - np.sum((np.log(pred) - np.log(y_true)) ** 2)
                        / np.sum((np.log(y_true) - np.log(y_true).mean()) ** 2)),
    }


def ridge_model():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_COLS),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_COLS),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def explain(model, X_train, X_test):
    import shap

    background = X_train.sample(100, random_state=RANDOM_STATE)
    sample = X_test.sample(300, random_state=RANDOM_STATE)
    # Mixed-type data -> hand SHAP plain arrays and rebuild the DataFrame inside predict
    explainer = shap.Explainer(predict_log_fn(model), background.to_numpy(dtype=object),
                               feature_names=FEATURES)
    sv = explainer(sample.to_numpy(dtype=object))

    plt.figure()
    shap.plots.bar(sv, show=False)
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    ranking = sorted(zip(FEATURES, np.abs(sv.values).mean(axis=0)), key=lambda p: -p[1])
    print("\nTop features by mean |SHAP| (log-value units):")
    for name, val in ranking:
        print(f"  {name:26s} {val:.3f}")
    return background


def main():
    X, y, assess_year = prepare_dataset()
    print(f"{len(X):,} properties, assessment year {assess_year}, "
          f"median assessed value ${y.median():,.0f}")
    y_log = np.log(y)

    X_train, X_test, ylog_train, ylog_test = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )
    y_test = np.exp(ylog_test)

    results = {}

    # Baseline 1: predict the overall median
    pred = np.full(len(X_test), np.exp(ylog_train.median()))
    results["Baseline: overall median"] = money_metrics(y_test.values, pred)

    # Baseline 2: median by neighbourhood + property type (a strong, simple baseline)
    keys = ["neighbourhood_code", "legal_type"]
    med = ylog_train.groupby([X_train[k] for k in keys]).median()
    idx = pd.MultiIndex.from_frame(X_test[keys])
    pred = np.exp(med.reindex(idx).fillna(ylog_train.median()).to_numpy())
    results["Baseline: neighbourhood + type median"] = money_metrics(y_test.values, pred)

    # Models (trained on log value, evaluated in dollars)
    models = {"Ridge regression": ridge_model(), "Gradient boosting": make_model(RANDOM_STATE)}
    for name, model in models.items():
        cv = -cross_val_score(model, X_train, ylog_train, cv=3,
                              scoring="neg_root_mean_squared_error").mean()
        model.fit(X_train, ylog_train)
        pred = np.exp(model.predict(X_test))
        results[name] = {**money_metrics(y_test.values, pred), "cv_rmse_log": float(cv)}

    print(f"\n{'Model':40s} {'MAE':>12s} {'MedAPE':>8s} {'R2(log)':>8s}")
    for name, r in results.items():
        print(f"{name:40s} ${r['mae']:>11,.0f} {r['median_abs_pct_error']:>7.1f}% {r['r2_log']:>8.3f}")

    best_name = "Gradient boosting"
    ref = results["Baseline: neighbourhood + type median"]["mae"]
    improvement = 100 * (1 - results[best_name]["mae"] / ref)
    print(f"\n{best_name} cuts MAE by {improvement:.1f}% vs. the neighbourhood+type baseline.")

    best = models[best_name]
    background = explain(best, X_train, X_test)

    # Error analysis by property type
    pred_best = np.exp(best.predict(X_test))
    by_type = (
        pd.DataFrame({"type": X_test["legal_type"].values,
                      "ape": np.abs(pred_best - y_test.values) / y_test.values * 100})
        .groupby("type")["ape"].agg(["median", "count"])
    )
    print("\nMedian % error by property type:\n", by_type.round(1))

    categories = {c: sorted(X_train[c].dropna().unique().tolist()) for c in CAT_COLS}
    defaults = {c: X_train[c].mode().iloc[0] for c in CAT_COLS}
    defaults.update({c: float(X_train[c].median()) for c in NUM_COLS})
    joblib.dump({"model": best, "background": background, "categories": categories,
                 "defaults": defaults, "assessment_year": assess_year}, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump({"results": results, "best": best_name, "n_properties": int(len(X)),
                   "mae_improvement_pct_vs_baseline": improvement,
                   "assessment_year": assess_year,
                   "median_pct_error_by_type": by_type["median"].round(2).to_dict()}, f, indent=2)


if __name__ == "__main__":
    main()
