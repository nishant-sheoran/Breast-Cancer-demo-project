from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _to_numpy(X):
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)


class ScaledClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a fitted scaler + classifier so FastAPI can load a single object.

    The scaler is applied on every predict/predict_proba call, mirroring the
    preprocessing performed during training. The wrapped classifier must
    already be fitted.
    """

    def __init__(self, scaler: Any, classifier: Any):
        self.scaler = scaler
        self.classifier = classifier

    def _transform(self, X):
        return self.scaler.transform(_to_numpy(X))

    def predict(self, X):
        return self.classifier.predict(self._transform(X))

    def predict_proba(self, X):
        if not hasattr(self.classifier, "predict_proba"):
            raise AttributeError("Underlying classifier has no predict_proba.")
        return self.classifier.predict_proba(self._transform(X))

    def decision_function(self, X):
        if not hasattr(self.classifier, "decision_function"):
            raise AttributeError("Underlying classifier lacks decision_function.")
        return self.classifier.decision_function(self._transform(X))

    def __getattr__(self, item):
        # Delegate unfound attributes (e.g., classes_) to underlying estimator.
        try:
            return getattr(self.classifier, item)
        except AttributeError as exc:  # pragma: no cover - mirror AttributeError semantics
            raise exc
