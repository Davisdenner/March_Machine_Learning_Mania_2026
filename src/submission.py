import pandas as pd
import numpy as np
from config import OUTPUT_DIR, DATA_DIR

#====== geração do arquivo de submissão =======#
#carrega seeds diretamente do arquivo para garantir que 2026 está disponível
def _load_seeds_from_file() -> pd.DataFrame:

    try:
        seeds = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
        seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
        return seeds[["Season", "TeamID", "SeedNum"]]
    except Exception as e:
        print(f"   Override seed: erro ao carregar seeds — {e}")
        return pd.DataFrame()

# override conservador para confrontos extremos de seed.
#
# lógica:
# - Seed 1 vs Seed 16: clipar favorito entre 0.82 e 0.93
# - Seed 2 vs Seed 15: clipar favorito entre 0.78 e 0.93
# - Seed 1 vs Seed 15: clipar favorito entre 0.78 e 0.93
def _apply_seed_override(sub: pd.DataFrame, matchups: pd.DataFrame) -> pd.DataFrame:

    #carregar seeds diretamente do arquivo
    seeds_df = _load_seeds_from_file()
    if seeds_df.empty:
        print("   Override seed: seeds não disponíveis, pulando.")
        return sub

    #parsear ids da submissão para obter Season, TeamA, TeamB
    df = sub.reset_index(drop=True).copy()
    parts = df["ID"].str.split("_", expand=True).astype(int)
    df["Season"] = parts[0]
    df["TeamA"]  = parts[1]
    df["TeamB"]  = parts[2]

    #merge seeds para TeamA
    df = df.merge(
        seeds_df.rename(columns={"TeamID": "TeamA", "SeedNum": "A_Seed"}),
        on=["Season", "TeamA"], how="left"
    )
    #merge seeds para TeamB
    df = df.merge(
        seeds_df.rename(columns={"TeamID": "TeamB", "SeedNum": "B_Seed"}),
        on=["Season", "TeamB"], how="left"
    )

    overrides = 0

    for i in range(len(df)):
        sa = df.loc[i, "A_Seed"]
        sb = df.loc[i, "B_Seed"]

        if pd.isna(sa) or pd.isna(sb):
            continue

        sa, sb = int(sa), int(sb)
        pred = df.loc[i, "Pred"]

        #Seed 1 vs Seed 16
        if (sa == 1 and sb == 16) or (sa == 16 and sb == 1):
            if sa == 1:
                df.loc[i, "Pred"] = np.clip(pred, 0.82, 0.93)
            else:
                df.loc[i, "Pred"] = np.clip(pred, 0.07, 0.18)
            overrides += 1

        #Seed 2 vs Seed 15
        elif (sa == 2 and sb == 15) or (sa == 15 and sb == 2):
            if sa == 2:
                df.loc[i, "Pred"] = np.clip(pred, 0.78, 0.93)
            else:
                df.loc[i, "Pred"] = np.clip(pred, 0.07, 0.22)
            overrides += 1

        #Seed 1 vs Seed 15
        elif (sa == 1 and sb == 15) or (sa == 15 and sb == 1):
            if sa == 1:
                df.loc[i, "Pred"] = np.clip(pred, 0.78, 0.93)
            else:
                df.loc[i, "Pred"] = np.clip(pred, 0.07, 0.22)
            overrides += 1

    print(f"   Override seed aplicado em {overrides} confrontos")

    #retornar apenas ID e Pred
    sub["Pred"] = df["Pred"].values
    return sub

#gera e salva o arquivo de submissão
def generate_submission(matchups: pd.DataFrame, predictions: np.ndarray,filename: str = "submission.csv",apply_seed_override: bool = True,) -> pd.DataFrame:

    sub = pd.DataFrame({
        "ID": matchups["ID"].values,
        "Pred": np.clip(predictions, 0.05, 0.95),
    })

    if apply_seed_override:
        sub = _apply_seed_override(sub, matchups)

    path = f"{OUTPUT_DIR}/{filename}"
    sub.to_csv(path, index=False)
    print(f" Submissão salva em: {path}")
    print(f" Shape: {sub.shape}")
    print(f" Pred stats: mean={sub['Pred'].mean():.4f}, "
          f"std={sub['Pred'].std():.4f}, "
          f"min={sub['Pred'].min():.4f}, max={sub['Pred'].max():.4f}")
    return sub