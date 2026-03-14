"""Ensemble de modelos e calibração de probabilidades."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from config import SEED


class EnsemblePredictor:
    """Ensemble com média ponderada + calibração."""

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.calibrator = None

    def add_model(self, name: str, model, weight: float = 1.0):
        self.models[name] = model
        self.weights[name] = weight

    def predict_proba_raw(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Previsões individuais de cada modelo."""
        preds = {}
        for name, model in self.models.items():
            p = model.predict_proba(X)[:, 1]
            preds[name] = p
        return preds

    def predict_proba_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Média ponderada das previsões."""
        preds = self.predict_proba_raw(X)
        total_weight = sum(self.weights[n] for n in preds)
        ensemble = sum(preds[n] * self.weights[n] for n in preds) / total_weight
        return ensemble

    def fit_calibrator(self, X_val: np.ndarray, y_val: np.ndarray):
        """Ajustar Platt Scaling nos resíduos do ensemble."""
        raw_preds = self.predict_proba_ensemble(X_val).reshape(-1, 1)
        self.calibrator = LogisticRegression(C=1.0, random_state=SEED)
        self.calibrator.fit(raw_preds, y_val)

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Previsão calibrada final."""
        raw = self.predict_proba_ensemble(X).reshape(-1, 1)
        if self.calibrator is not None:
            calibrated = self.calibrator.predict_proba(raw)[:, 1]
        else:
            calibrated = raw.flatten()
        return np.clip(calibrated, 0.01, 0.99)


class StackingEnsemble:
    """Stacking: usa previsões dos modelos base como features para meta-learner."""

    def __init__(self):
        self.base_models = {}
        self.meta_model = LogisticRegression(C=1.0, random_state=SEED)
        self.calibrator = None

    def add_base_model(self, name: str, model):
        self.base_models[name] = model

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        meta = []
        for name, model in self.base_models.items():
            p = model.predict_proba(X)[:, 1]
            meta.append(p)
        return np.column_stack(meta)

    def fit_meta(self, X_val: np.ndarray, y_val: np.ndarray):
        meta_X = self._get_meta_features(X_val)
        self.meta_model.fit(meta_X, y_val)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._get_meta_features(X)
        preds = self.meta_model.predict_proba(meta_X)[:, 1]
        return np.clip(preds, 0.01, 0.99)
