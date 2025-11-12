from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from backend.model_artifact import ScaledClassifier

RANDOM_STATE = 42
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

try:  # Optional dependency
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:  # pragma: no cover - runtime guard
    SMOTE = None
    RandomUnderSampler = None

try:  # Optional dependency
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - runtime guard
    XGBClassifier = None


@dataclass
class ModelResult:
    model_name: str
    sampling: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion: np.ndarray


def load_dataset(args) -> pd.DataFrame:
    """Load the Kaggle dataset; fall back to sklearn copy if needed."""
    if not args.use_sklearn_only:
        try:
            import kagglehub  # type: ignore

            dataset_path = Path(kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data"))
            csv_file = dataset_path / "data.csv"
            if not csv_file.exists():
                raise FileNotFoundError(csv_file)
            print(f"Dataset downloaded to: {dataset_path}")
            return pd.read_csv(csv_file)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Kaggle download failed ({exc!r}); falling back to sklearn dataset.")
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame
    frame["diagnosis"] = frame["target"]
    frame = frame.drop(columns=["target"])
    return frame


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df.columns = [column.replace(" ", "_") for column in df.columns]
    for column in ("Unnamed: 32", "id"):
        normalized = column.replace(" ", "_")
        if column in df.columns:
            df = df.drop(columns=column)
        elif normalized in df.columns:
            df = df.drop(columns=normalized)
    if "diagnosis" not in df.columns:
        raise ValueError("Expected 'diagnosis' column.")
    encoder = LabelEncoder()
    df["diagnosis"] = encoder.fit_transform(df["diagnosis"])
    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required features: {missing_features}")
    X = df[FEATURES].copy()
    y = df["diagnosis"].copy()
    return X, y


def available_models() -> Dict[str, ClassifierMixin]:
    models: Dict[str, ClassifierMixin] = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=200),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_estimators=400),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=1.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    return models


def sampling_strategies() -> Dict[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    strategies: Dict[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = {
        "Original": lambda X, y: (X, y),
    }
    if SMOTE is not None:
        strategies["SMOTE"] = lambda X, y: SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    if RandomUnderSampler is not None:
        strategies["Undersampling"] = lambda X, y: RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X, y)
    if SMOTE is not None and RandomUnderSampler is not None:
        def smote_then_under(X: np.ndarray, y: np.ndarray):
            X_smote, y_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
            return RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_smote, y_smote)

        strategies["SMOTE + Undersampling"] = smote_then_under
    return strategies


def evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> List[ModelResult]:
    models = available_models()
    samplers = sampling_strategies()
    print(f"Evaluating {len(models)} models across {len(samplers)} sampling strategies...")
    results: List[ModelResult] = []
    for sampling_name, sampler in samplers.items():
        X_resampled, y_resampled = sampler(X_train, y_train)
        for model_name, model in models.items():
            estimator = clone(model)
            estimator.fit(X_resampled, y_resampled)
            y_pred = estimator.predict(X_test)
            y_proba = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            cm = confusion_matrix(y_test, y_pred)
            print(
                f"[{sampling_name}] {model_name} -> Recall: {recall:.3f}, "
                f"F1: {f1:.3f}, Accuracy: {accuracy:.3f}"
            )
            results.append(
                ModelResult(
                    model_name=model_name,
                    sampling=sampling_name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    roc_auc=roc_auc,
                    confusion=cm,
                )
            )
    if not results:
        raise RuntimeError("No models were evaluated. Install optional dependencies?")
    return results


def select_best(results: List[ModelResult]) -> ModelResult:
    return max(results, key=lambda r: (r.recall, r.f1, r.accuracy))


def retrain_best(
    X: pd.DataFrame,
    y: pd.Series,
    best_result: ModelResult,
) -> ScaledClassifier:
    models = available_models()
    samplers = sampling_strategies()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    sampler_fn = samplers[best_result.sampling]
    X_balanced, y_balanced = sampler_fn(X_scaled, y.values)
    estimator = clone(models[best_result.model_name])
    estimator.fit(X_balanced, y_balanced)
    return ScaledClassifier(scaler=scaler, classifier=estimator)


def save_model(model: ScaledClassifier, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    print(f"Saved trained model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train and export the breast cancer classifier.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("breast_cancer_model.pkl"),
        help="Where to store the trained model.",
    )
    parser.add_argument(
        "--use-sklearn-only",
        action="store_true",
        help="Skip Kaggle download and rely on sklearn's built-in dataset.",
    )
    args = parser.parse_args()

    df = load_dataset(args)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = evaluate_models(X_train_scaled, X_test_scaled, y_train.values, y_test.values)
    best = select_best(results)
    print(
        f"\nBest configuration: {best.model_name} + {best.sampling} "
        f"(Recall={best.recall:.3f}, F1={best.f1:.3f}, Accuracy={best.accuracy:.3f})"
    )
    final_model = retrain_best(X, y, best)
    save_model(final_model, args.output)


if __name__ == "__main__":
    main()
