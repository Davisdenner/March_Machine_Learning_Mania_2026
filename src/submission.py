"""Geração do arquivo de submissão."""
import pandas as pd
import numpy as np
from config import OUTPUT_DIR


def generate_submission(
    matchups: pd.DataFrame,
    predictions: np.ndarray,
    filename: str = "submission.csv",
) -> pd.DataFrame:
    """Gera e salva o arquivo de submissão."""
    sub = pd.DataFrame({
        "ID": matchups["ID"],
        "Pred": np.clip(predictions, 0.05, 0.95),
    })

    path = f"{OUTPUT_DIR}/{filename}"
    sub.to_csv(path, index=False)
    print(f"✅ Submissão salva em: {path}")
    print(f"   Shape: {sub.shape}")
    print(f"   Pred stats: mean={sub['Pred'].mean():.4f}, "
          f"std={sub['Pred'].std():.4f}, "
          f"min={sub['Pred'].min():.4f}, max={sub['Pred'].max():.4f}")
    return sub