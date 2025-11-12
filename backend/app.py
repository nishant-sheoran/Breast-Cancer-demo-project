from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer

DEFAULT_MODEL_PATH = Path(__file__).with_name("breast_cancer_model.pkl")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]
LABELS = {0: "Benign", 1: "Malignant"}

sklearn_name_map = {
    "mean radius": "radius_mean",
    "mean texture": "texture_mean",
    "mean perimeter": "perimeter_mean",
    "mean area": "area_mean",
    "mean smoothness": "smoothness_mean",
    "mean compactness": "compactness_mean",
    "mean concavity": "concavity_mean",
    "mean concave points": "concave_points_mean",
    "mean symmetry": "symmetry_mean",
    "mean fractal dimension": "fractal_dimension_mean",
}


class FeaturePayload(BaseModel):
    radius_mean: float = Field(..., ge=0)
    texture_mean: float = Field(..., ge=0)
    perimeter_mean: float = Field(..., ge=0)
    area_mean: float = Field(..., ge=0)
    smoothness_mean: float = Field(..., ge=0)
    compactness_mean: float = Field(..., ge=0)
    concavity_mean: float = Field(..., ge=0)
    concave_points_mean: float = Field(..., ge=0)
    symmetry_mean: float = Field(..., ge=0)
    fractal_dimension_mean: float = Field(..., ge=0)

    def as_frame(self) -> pd.DataFrame:
        data = {feature: getattr(self, feature) for feature in FEATURES}
        return pd.DataFrame([[data[name] for name in FEATURES]], columns=FEATURES)


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float
    impact: str


class PredictionResponse(BaseModel):
    prediction: str
    label: int
    probability: Optional[float]
    reasons: List[Reason]
    base_value: Optional[float]


app = FastAPI(title="Breast Cancer Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
explainer = None
reference_frame, stats = None, None


def load_reference_frame() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.rename(columns=sklearn_name_map)
    return frame[FEATURES]


def compute_stats(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for feature in FEATURES:
        series = frame[feature]
        std = float(series.std(ddof=0))
        summary[feature] = {
            "median": float(series.median()),
            "std": std if std > 0 else 1.0,
        }
    return summary


def load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Drop your trained pipeline there or set MODEL_PATH."
        )
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


def build_explainer(loaded_model, frame: pd.DataFrame):
    background = shap.sample(frame, min(len(frame), 200))
    background_arr = background.values if hasattr(background, "values") else np.asarray(background)
    masker = shap.maskers.Independent(background_arr)

    def predict_fn(values: np.ndarray) -> np.ndarray:
        data = pd.DataFrame(values, columns=FEATURES)
        if hasattr(loaded_model, "predict_proba"):
            try:
                proba = loaded_model.predict_proba(data)
            except Exception:
                proba = loaded_model.predict_proba(data.values)
            return proba[:, 1]
        if hasattr(loaded_model, "decision_function"):
            decision = loaded_model.decision_function(data)
            decision = decision if isinstance(decision, np.ndarray) else np.asarray(decision)
            return 1.0 / (1.0 + np.exp(-decision))
        preds = loaded_model.predict(data)
        preds = np.asarray(preds, dtype=float).ravel()
        return preds

    return shap.Explainer(predict_fn, masker, feature_names=FEATURES, algorithm="permutation")


def ensure_loaded():
    global model, explainer, reference_frame, stats
    if reference_frame is None:
        reference_frame = load_reference_frame()
        stats = compute_stats(reference_frame)
    if model is None:
        model = load_model()
    if explainer is None and model is not None:
        explainer = build_explainer(model, reference_frame)


def normalize_label(raw) -> int:
    arr = np.asarray(raw).ravel()
    value = arr[0]
    if isinstance(value, str):
        return 1 if value.upper().startswith("M") else 0
    return 1 if int(value) == 1 else 0


def run_prediction(loaded_model, sample: pd.DataFrame) -> Tuple[int, Optional[float]]:
    prediction = loaded_model.predict(sample)
    label = normalize_label(prediction)
    probability = None
    if hasattr(loaded_model, "predict_proba"):
        proba = loaded_model.predict_proba(sample)
        probability = float(proba[0, 1] if proba.shape[-1] > 1 else proba[0, 0])
    elif hasattr(loaded_model, "decision_function"):
        decision = loaded_model.decision_function(sample)
        probability = float(1.0 / (1.0 + math.exp(-np.asarray(decision).ravel()[0])))
    return label, probability


def shap_reasons(shap_explainer, sample: pd.DataFrame):
    if shap_explainer is None:
        return None
    explanation = shap_explainer(sample)
    explanation = explanation[0] if isinstance(explanation, list) else explanation
    shap_values = np.atleast_2d(explanation.values)[0]
    data_row = np.atleast_2d(explanation.data)[0]
    base_value = float(np.atleast_1d(explanation.base_values)[0])
    rows = []
    for feature, value, contribution in zip(FEATURES, data_row, shap_values):
        rows.append(
            {
                "feature": feature,
                "value": float(value),
                "contribution": float(contribution),
                "impact": "pushes_malignant" if contribution >= 0 else "pushes_benign",
            }
        )
    rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return rows[:10], base_value


def fallback_reasons(sample: pd.DataFrame) -> List[Reason]:
    sample_row = sample.iloc[0]
    rows = []
    for feature in FEATURES:
        z_score = (float(sample_row[feature]) - stats[feature]["median"]) / stats[feature]["std"]
        rows.append(
            {
                "feature": feature,
                "value": float(sample_row[feature]),
                "contribution": float(z_score),
                "impact": "towards_malignant" if z_score >= 0 else "towards_benign",
            }
        )
    rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return rows[:10]


@app.on_event("startup")
def startup_event():
    try:
        ensure_loaded()
    except FileNotFoundError:
        # Model will be loaded lazily when available.
        pass


@app.get("/health")
def healthcheck():
    message = "ok" if model else "model not loaded"
    return {"status": message}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FeaturePayload):
    try:
        ensure_loaded()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sample = payload.as_frame()
    label, probability = run_prediction(model, sample)

    shap_output = shap_reasons(explainer, sample)
    if shap_output:
        reasons, base_value = shap_output
    else:
        reasons = fallback_reasons(sample)
        base_value = None

    return PredictionResponse(
        prediction=LABELS[label],
        label=label,
        probability=probability,
        reasons=reasons,
        base_value=base_value,
    )
