"""Treino de múltiplos modelos."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from config import SEED


def get_lgb_model(n_estimators: int = 2000) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=SEED,
        verbose=-1,
    )


def get_xgb_model(n_estimators: int = 2000) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=SEED,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )


def get_catboost_model(n_estimators: int = 2000) -> cb.CatBoostClassifier:
    return cb.CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=10,
        random_seed=SEED,
        verbose=0,
        eval_metric="Logloss",
    )


def get_lr_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])


def train_model_with_early_stopping(
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
):
    """Treina modelo com early stopping (para GBMs)."""
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
    elif isinstance(model, xgb.XGBClassifier):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # XGBoost uses early_stopping_rounds in constructor or fit
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif isinstance(model, cb.CatBoostClassifier):
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
    else:
        # Logistic Regression - sem early stopping
        model.fit(X_train, y_train)

    return model
