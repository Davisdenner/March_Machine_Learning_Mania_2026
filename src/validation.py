import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

#======== Validação temporal para evitar data leakage =======#
#gerar splits de validação temporal expandindo a janela de treino
def temporal_cv_splits(
        df: pd.DataFrame,
        first_train_end: int = 2014,
        last_val_season: int = 2024,
        min_train_seasons: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:

    seasons = sorted(df["Season"].unique())
    splits = []

    for val_season in range(first_train_end + 1, last_val_season + 1):
        if val_season not in seasons:
            continue
        train_mask = df["Season"] <= val_season - 1
        val_mask = df["Season"] == val_season

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        train_idx = df[train_mask].index.values
        val_idx = df[val_mask].index.values

        #checar mínimo
        n_train_seasons = df.loc[train_idx, "Season"].nunique()
        if n_train_seasons >= min_train_seasons:
            splits.append((train_idx, val_idx))

    return splits

#calcular Log Loss com clipping para segurança
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return log_loss(y_true, y_pred)
