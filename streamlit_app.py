from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer

MODEL_PATH = Path("model.pkl")

# The 10 inputs the UI will collect - matches the column names in the notebook.
FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]

FEATURE_LABELS = {
    "radius_mean": "Mean radius (mm)",
    "texture_mean": "Mean texture",
    "perimeter_mean": "Mean perimeter (mm)",
    "area_mean": "Mean area (mm^2)",
    "smoothness_mean": "Mean smoothness",
    "compactness_mean": "Mean compactness",
    "concavity_mean": "Mean concavity",
    "concave points_mean": "Mean concave points",
    "symmetry_mean": "Mean symmetry",
    "fractal_dimension_mean": "Mean fractal dimension",
}

# For the fallback explainer: whether larger values usually increase malignancy risk.
FEATURE_TRENDS = {
    "radius_mean": "higher",
    "texture_mean": "higher",
    "perimeter_mean": "higher",
    "area_mean": "higher",
    "smoothness_mean": "higher",
    "compactness_mean": "higher",
    "concavity_mean": "higher",
    "concave points_mean": "higher",
    "symmetry_mean": "higher",
    "fractal_dimension_mean": "higher",
}

SKLEARN_TO_DATASET_NAMES = {
    "mean radius": "radius_mean",
    "mean texture": "texture_mean",
    "mean perimeter": "perimeter_mean",
    "mean area": "area_mean",
    "mean smoothness": "smoothness_mean",
    "mean compactness": "compactness_mean",
    "mean concavity": "concavity_mean",
    "mean concave points": "concave points_mean",
    "mean symmetry": "symmetry_mean",
    "mean fractal dimension": "fractal_dimension_mean",
}

LABEL_MAP = {0: "Benign", 1: "Malignant"}


@st.cache_resource(show_spinner=False)
def load_reference_frame() -> pd.DataFrame:
    """Use sklearn's built-in dataset to infer sensible ranges/medians."""
    dataset = load_breast_cancer(as_frame=True)
    frame: pd.DataFrame = dataset.frame.rename(columns=SKLEARN_TO_DATASET_NAMES)
    return frame[FEATURES]


@st.cache_resource(show_spinner=False)
def get_reference_assets() -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    frame = load_reference_frame()
    stats: Dict[str, Dict[str, float]] = {}
    for feature in FEATURES:
        series = frame[feature]
        std = float(series.std(ddof=0))
        stats[feature] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "std": std if std > 0 else 1.0,
        }
    return frame, stats


@st.cache_resource(show_spinner=False)
def load_model_from_disk():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(str(MODEL_PATH))
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


def resolve_model():
    """Try loading model.pkl; fall back to sidebar upload."""
    if "uploaded_model" in st.session_state:
        return st.session_state["uploaded_model"]
    try:
        return load_model_from_disk()
    except FileNotFoundError:
        uploaded = st.sidebar.file_uploader(
            "Upload your trained model (.pkl/.joblib)", type=["pkl", "joblib"]
        )
        if uploaded is None:
            st.info(
                "Place `model.pkl` next to this script or upload it via the sidebar to get started."
            )
            st.stop()
        uploaded.seek(0)
        model = pickle.load(uploaded)
        st.session_state["uploaded_model"] = model
        return model


def build_explainer(model, reference_frame: pd.DataFrame):
    """Create a SHAP explainer; raise if it fails."""
    background = shap.sample(reference_frame, min(len(reference_frame), 200))
    background_values = background.values if hasattr(background, "values") else np.asarray(background)
    masker = shap.maskers.Independent(background_values)

    def predict_fn(data_as_numpy: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data_as_numpy, columns=FEATURES)
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
            except Exception:
                proba = model.predict_proba(df.values)
            return proba[:, 1]
        if hasattr(model, "decision_function"):
            try:
                decision = model.decision_function(df)
            except Exception:
                decision = model.decision_function(df.values)
            return 1.0 / (1.0 + np.exp(-decision))
        try:
            preds = model.predict(df)
        except Exception:
            preds = model.predict(df.values)
        preds = np.asarray(preds, dtype=float)
        if preds.ndim > 1:
            preds = preds[:, 0]
        return preds

    return shap.Explainer(predict_fn, masker, feature_names=FEATURES, algorithm="permutation")


def ensure_explainer(model, reference_frame: pd.DataFrame):
    """Cache the explainer in session_state so we don't rebuild on each rerun."""
    cache_key = st.session_state.get("explainer_model_id")
    model_id = id(model)
    if cache_key != model_id:
        try:
            st.session_state["explainer"] = build_explainer(model, reference_frame)
            st.session_state["explainer_error"] = ""
            st.session_state["explainer_model_id"] = model_id
        except Exception as exc:  # noqa: BLE001 - show the message to the user
            st.session_state["explainer"] = None
            st.session_state["explainer_error"] = str(exc)
            st.session_state["explainer_model_id"] = model_id
    return st.session_state.get("explainer"), st.session_state.get("explainer_error", "")


def model_call(model, method: str, data: pd.DataFrame):
    fn = getattr(model, method)
    try:
        return fn(data)
    except Exception:
        return fn(data.values)


def normalize_label(raw) -> int:
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, str):
        return 1 if raw.upper().startswith("M") else 0
    try:
        return 1 if int(raw) == 1 else 0
    except (ValueError, TypeError):
        return 1 if bool(raw) else 0


def run_inference(model, features_df: pd.DataFrame):
    raw_pred = model_call(model, "predict", features_df)
    label = normalize_label(np.asarray(raw_pred).ravel()[0])
    probability = None
    if hasattr(model, "predict_proba"):
        proba = model_call(model, "predict_proba", features_df)
        probability = float(proba[0, 1]) if proba.shape[-1] > 1 else float(proba[0, 0])
    elif hasattr(model, "decision_function"):
        decision = model_call(model, "decision_function", features_df)
        probability = float(1.0 / (1.0 + math.exp(-decision.ravel()[0])))
    return label, probability


def compute_shap_details(explainer, sample_df: pd.DataFrame):
    if explainer is None:
        return None
    explanation = explainer(sample_df)
    if isinstance(explanation, list):
        explanation = explanation[0]
    shap_values = np.atleast_2d(explanation.values)[0]
    data_row = np.atleast_2d(explanation.data)[0]
    base_value = float(np.atleast_1d(explanation.base_values)[0])
    contributions = (
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Value": data_row,
                "Contribution": shap_values,
            }
        )
        .assign(
            Impact=lambda df: np.where(df["Contribution"] >= 0, "Pushes malignant", "Pushes benign"),
            AbsContribution=lambda df: df["Contribution"].abs(),
        )
        .sort_values("AbsContribution", ascending=False)
    )
    return contributions, base_value, explanation


def fallback_reason(sample_df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    sample = sample_df.iloc[0]
    rows = []
    for feature in FEATURES:
        meta = stats[feature]
        value = float(sample[feature])
        median = meta["median"]
        std = meta["std"]
        z_score = (value - median) / std
        higher_is_risky = FEATURE_TRENDS[feature] == "higher"
        pushes_malignant = (z_score > 0 and higher_is_risky) or (z_score < 0 and not higher_is_risky)
        rows.append(
            {
                "Feature": feature,
                "Value": value,
                "Delta vs median (sigma)": z_score,
                "Direction": "towards malignant" if pushes_malignant else "towards benign",
                "AbsScore": abs(z_score),
            }
        )
    return pd.DataFrame(rows).sort_values("AbsScore", ascending=False)


def render_reasoning(section_title: str, df: pd.DataFrame, note: str):
    st.subheader(section_title)
    st.caption(note)
    st.dataframe(
        df[
            [
                "Feature",
                "Value",
                "Contribution" if "Contribution" in df else "Delta vs median (sigma)",
                "Impact" if "Impact" in df else "Direction",
            ]
        ]
    )


def feature_form(stats: Dict[str, Dict[str, float]]):
    with st.form("feature-inputs"):
        cols = st.columns(2)
        values = {}
        for idx, feature in enumerate(FEATURES):
            col = cols[idx % 2]
            meta = stats[feature]
            min_val = meta["min"]
            max_val = meta["max"]
            median = meta["median"]
            step = max((max_val - min_val) / 200, 0.001)
            label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
            help_text = f"Typical range: {min_val:.2f} - {max_val:.2f} (median {median:.2f})"
            values[feature] = col.number_input(
                label,
                value=float(median),
                min_value=float(min_val),
                max_value=float(max_val),
                step=float(step),
                help=help_text,
            )
        submitted = st.form_submit_button("Predict")
    return submitted, values


def main():
    st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
    st.title("Breast Cancer Prediction - Interactive Frontend")
    st.caption(
        "Enter the 10 diagnostic measurements for prediction and reasoning."
    )

    reference_frame, stats = get_reference_assets()
    model = resolve_model()
    explainer, explainer_error = ensure_explainer(model, reference_frame)

    submitted, inputs = feature_form(stats)
    if not submitted:
        st.info("Adjust the inputs and click **Predict** to run the model.")
        return

    sample_df = pd.DataFrame([inputs], columns=FEATURES)
    try:
        label, probability = run_inference(model, sample_df)
    except Exception as exc:  # noqa: BLE001 - show actionable feedback
        st.error(f"Model inference failed: {exc}")
        return

    outcome = LABEL_MAP.get(label, f"Class {label}")
    st.success(f"Prediction: **{outcome}**")
    if probability is not None:
        st.metric("Estimated probability of malignancy", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))
    else:
        st.caption("Probability unavailable because the model does not expose `predict_proba`.")

    shap_result = compute_shap_details(explainer, sample_df)
    if shap_result:
        contributions, base_value, explanation = shap_result
        render_reasoning(
            "Why the model chose this outcome",
            contributions.head(10),
            "Positive contributions push towards malignant; negative contributions push towards benign.",
        )
        with st.expander("Show SHAP waterfall plot"):
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
    else:
        if explainer_error:
            st.warning(f"SHAP explanation fallback in use: {explainer_error}")
        fallback_df = fallback_reason(sample_df, stats)
        render_reasoning(
            "Why (rule-based fallback)",
            fallback_df.head(10),
            "Top deviations from typical benign values (measured in standard deviations).",
        )

    st.caption(
        "Note: This interface is for research support only and is **not** a medical diagnostic device."
    )


if __name__ == "__main__":
    main()
