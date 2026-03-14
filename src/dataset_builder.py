# src/dataset_builder.py
"""Construção dos datasets de confrontos para treino e submissão."""

import numpy as np
import pandas as pd


def _merge_features(df, team_features, seeds, elo_df, tourney_history, team_col, prefix,
                    massey_df=None):
    """Merge features de um time no DataFrame de confrontos."""
    # Team stats
    df = df.merge(
        team_features.add_prefix(f"{prefix}_").rename(
            columns={f"{prefix}_Season": "Season", f"{prefix}_TeamID": team_col}
        ),
        on=["Season", team_col],
        how="left",
    )
    # Seed
    seed_cols = seeds[["Season", "TeamID", "SeedNum"]].rename(
        columns={"TeamID": team_col, "SeedNum": f"{prefix}_Seed"}
    )
    df = df.merge(seed_cols, on=["Season", team_col], how="left")

    # ELO
    elo_cols = elo_df[["Season", "TeamID", "ELO"]].rename(
        columns={"TeamID": team_col, "ELO": f"{prefix}_ELO"}
    )
    df = df.merge(elo_cols, on=["Season", team_col], how="left")

    # Tourney history
    if tourney_history is not None and len(tourney_history) > 0:
        th_cols = tourney_history[["Season", "TeamID", "HistTourneyWins"]].rename(
            columns={"TeamID": team_col, "HistTourneyWins": f"{prefix}_HistTourneyWins"}
        )
        df = df.merge(th_cols, on=["Season", team_col], how="left")
        df[f"{prefix}_HistTourneyWins"] = df[f"{prefix}_HistTourneyWins"].fillna(0)

    # Massey Ordinals
    if massey_df is not None and len(massey_df) > 0:
        massey_cols = [c for c in massey_df.columns if c.startswith("Massey_")]
        m_rename = {"TeamID": team_col}
        m_rename.update({c: f"{prefix}_{c}" for c in massey_cols})
        m_df = massey_df[["Season", "TeamID"] + massey_cols].rename(columns=m_rename)
        df = df.merge(m_df, on=["Season", team_col], how="left")

    return df


def _compute_diffs(df):
    """Calcula features diferenciais (TeamA - TeamB)."""
    a_cols = [c for c in df.columns if c.startswith("A_")]
    for ac in a_cols:
        suffix = ac[2:]  # remove "A_"
        bc = f"B_{suffix}"
        if bc in df.columns:
            df[f"Diff_{suffix}"] = df[ac] - df[bc]
    return df

def build_training_matchups(
        tourney: pd.DataFrame,
        team_features: pd.DataFrame,
        seeds: pd.DataFrame,
        elo_df: pd.DataFrame,
        tourney_history: pd.DataFrame,
        massey_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Constrói dataset de treino a partir dos resultados do torneio.

    Cada jogo gera um registro com TeamA (menor ID) vs TeamB (maior ID).
    Result = 1 se TeamA venceu, 0 caso contrário.
    """
    rows = []
    for _, g in tourney.iterrows():
        season = g["Season"]
        w_id = g["WTeamID"]
        l_id = g["LTeamID"]

        # TeamA = menor ID (convenção Kaggle)
        if w_id < l_id:
            team_a, team_b, result = w_id, l_id, 1
        else:
            team_a, team_b, result = l_id, w_id, 0

        rows.append({
            "Season": season,
            "TeamA": team_a,
            "TeamB": team_b,
            "Result": result,
        })

    df = pd.DataFrame(rows)

    # Merge features para cada time
    df = _merge_features(df, team_features, seeds, elo_df, tourney_history, "TeamA", "A", massey_df)
    df = _merge_features(df, team_features, seeds, elo_df, tourney_history, "TeamB", "B", massey_df)

    # Features diferenciais
    df = _compute_diffs(df)

    # Preencher NaN
    df = df.fillna(0)

    return df

def build_submission_matchups(
        sample_sub: pd.DataFrame,
        team_features: pd.DataFrame,
        seeds: pd.DataFrame,
        elo_df: pd.DataFrame,
        tourney_history: pd.DataFrame,
        massey_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Constrói dataset de submissão a partir do sample submission."""
    # Parsear IDs: formato "2025_1101_1102"
    df = sample_sub.copy()
    parts = df["ID"].str.split("_", expand=True).astype(int)
    df["Season"] = parts[0]
    df["TeamA"] = parts[1]
    df["TeamB"] = parts[2]

    # Merge features
    df = _merge_features(df, team_features, seeds, elo_df, tourney_history, "TeamA", "A", massey_df)
    df = _merge_features(df, team_features, seeds, elo_df, tourney_history, "TeamB", "B", massey_df)

    # Features diferenciais
    df = _compute_diffs(df)

    # Preencher NaN
    df = df.fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna lista de colunas usáveis como features."""
    exclude = {"Season", "TeamA", "TeamB", "Result", "ID", "Pred"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]