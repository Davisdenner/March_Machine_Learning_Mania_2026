import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from config import SEED

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("   Optuna não instalado. Rode: pip install optuna")


#====== Modelos com hiperparâmetros padrão =======#

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


#====== Optuna - otimização de hiperparâmetros =======#

def optimize_lgb(X_train, y_train, X_val, y_val, n_trials: int = 30) -> dict:

    if not OPTUNA_AVAILABLE:
        print("   Optuna não disponível, usando params padrão para LGB")
        return {}

    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "num_leaves": trial.suggest_int("num_leaves", 8, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "random_state": SEED,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        preds = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"   LGB melhor Log Loss Optuna: {study.best_value:.5f} | params: {study.best_params}")
    return study.best_params

#otimizar hiperparâmetros do CatBoost com Optuna
def optimize_catboost(X_train, y_train, X_val, y_val, n_trials: int = 30) -> dict:

    if not OPTUNA_AVAILABLE:
        print("   Optuna não disponível, usando params padrão para CatBoost")
        return {}

    def objective(trial):
        params = {
            "iterations": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "random_seed": SEED,
            "verbose": 0,
            "eval_metric": "Logloss",
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
        preds = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"   CAT melhor Log Loss Optuna: {study.best_value:.5f} | params: {study.best_params}")
    return study.best_params

#retornar LGB com hiperparâmetros otimizados pelo Optuna
def get_lgb_model_tuned(best_params: dict, n_estimators: int = 2000) -> lgb.LGBMClassifier:

    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.02,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 40,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": SEED,
        "verbose": -1,
    }
    params.update(best_params)
    return lgb.LGBMClassifier(**params)

#retornar CatBoost com hiperparâmetros otimizados pelo Optuna
def get_catboost_model_tuned(best_params: dict, n_estimators: int = 2000) -> cb.CatBoostClassifier:

    params = {
        "iterations": n_estimators,
        "learning_rate": 0.02,
        "depth": 4,
        "l2_leaf_reg": 10,
        "random_seed": SEED,
        "verbose": 0,
        "eval_metric": "Logloss",
    }
    params.update(best_params)
    return cb.CatBoostClassifier(**params)


#====== Treino com early stopping =======#

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
        model.fit(X_train, y_train)

    return model