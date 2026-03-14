#carregamento e integração de todos os datasets
import pandas as pd
from config import DATA_DIR, GENDER

def load_csv(name: str):
    path = f"{DATA_DIR}/{name}.csv"
    return pd.read_csv(path)

def load_regular_season_detailed():
    return load_csv(f"{GENDER}RegularSeasonDetailedResults")

def load_regular_season_compact():
    return load_csv(f"{GENDER}RegularSeasonCompactResults")


def load_tourney_compact():
    return load_csv(f"{GENDER}NCAATourneyCompactResults")


def load_tourney_detailed():
    return load_csv(f"{GENDER}NCAATourneyDetailedResults")

def load_seeds():
    df = load_csv(f"{GENDER}NCAATourneySeeds")
    #extrair seed numérica: 'W01' ->, 'X16a' -> 16
    df["SeedNum"] = df["Seed"].str[1:3].astype(int)
    return df

def load_teams():
    return load_csv(f"{GENDER}Teams")

def load_conferences():
    return load_csv(f"{GENDER}TeamConferences")

def load_massey_ordinals() -> pd.DataFrame:
    df = load_csv("MMasseyOrdinals")
    #manter apenas os sistemas mais preditivos e o último ranking da temporada
    top_systems = ["POM", "SAG", "MOR", "COL", "DOL"]
    df = df[df["SystemName"].isin(top_systems)]
    #pegar o ranking mais recente de cada temporada (maior RankingDayNum)
    df = df.sort_values("RankingDayNum").groupby(["Season", "TeamID", "SystemName"]).tail(1)
    #pivotar: uma coluna por sistema
    df = df.pivot_table(index=["Season", "TeamID"], columns="SystemName", values="OrdinalRank").reset_index()
    df.columns.name = None
    #renomear colunas
    df = df.rename(columns={s: f"Massey_{s}" for s in top_systems if s in df.columns})
    return df

def load_sample_submission(stage: int=1):
    return load_csv(f"SampleSubmissionStage{stage}")


#combinar temporada regular + torneio num formato unificado

#retornar um df com as colunas:
#Season, DayNum, WTeamID, WScore, LTeamID, LScore,
#WLoc, NumOT, e todas as detailed stats quando disponíveis.
#IsTourney (bool).
def build_all_games():
    reg = load_regular_season_detailed()
    reg["IsTourney"] = False

    try:
        trn = load_tourney_detailed()
    except FileNotFoundError:
        trn = load_tourney_compact()
    trn["IsTourney"] = True

    #garantir as colunas comuns
    common = list(set(reg.columns) & set(trn.columns))
    df = pd.concat([reg[common], trn[common]], ignore_index=True)
    df.sort_values(["Season", "DayNum"], inplace=True)
    return df.reset_index(drop=True)
