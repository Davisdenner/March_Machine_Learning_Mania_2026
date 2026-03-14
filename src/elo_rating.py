#sistema de ELO rating dinâmico com reset suave entre temporadas
import numpy as np
import pandas as pd
from config import ELO_INITIAL, ELO_K, ELO_HOME_ADVANTAGE

def expected_score(elo_a: float, elo_b:float):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def update_elo(winner_elo:float, loser_elo:float, k:float = ELO_K, mov: int= 0,):
    #atualizar o ELO com Margin-of-Victory multiplier
    #MOV multiplier (FiveThirtyEight style)
    mov_mult = np.log(max(abs(mov), 1) + 1) * (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))

    exp_w = expected_score(winner_elo, loser_elo)
    shift = k * mov_mult * (1 - exp_w)

    return winner_elo + shift, loser_elo - shift

def compute_elo_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Calcula ELO para todos os times em todas as temporadas.

    Retorna DataFrame: Season, TeamID, ELO (valor ao final da temporada regular).
    Também retorna um dict season→{teamid: elo} para uso downstream.
    """
    elos: dict[int, float] = {}  # teamid -> current elo
    season_elos: list[dict] = []

    prev_season = None

    for _, row in games.iterrows():
        season = row["Season"]

        # Reset suave entre temporadas: regride 1/3 ao mean
        if prev_season is not None and season != prev_season:
            # Salvar ELOs do final da temporada anterior
            for tid, elo in elos.items():
                season_elos.append({"Season": prev_season, "TeamID": tid, "ELO": elo})
            # Regressão ao mean
            for tid in elos:
                elos[tid] = elos[tid] * 0.75 + ELO_INITIAL * 0.25

        prev_season = season

        w_id = row["WTeamID"]
        l_id = row["LTeamID"]
        w_score = row.get("WScore", 0)
        l_score = row.get("LScore", 0)
        mov = w_score - l_score

        w_elo = elos.get(w_id, ELO_INITIAL)
        l_elo = elos.get(l_id, ELO_INITIAL)

        # Home-court adjustment para cálculo (não permanente)
        loc = row.get("WLoc", "N")
        if loc == "H":
            w_elo_adj = w_elo + ELO_HOME_ADVANTAGE
        elif loc == "A":
            w_elo_adj = w_elo - ELO_HOME_ADVANTAGE
        else:
            w_elo_adj = w_elo

        new_w, new_l = update_elo(w_elo_adj, l_elo, mov=mov)
        # Remap adjustment back
        elos[w_id] = new_w - (w_elo_adj - w_elo)
        elos[l_id] = new_l

    # Salvar última temporada
    for tid, elo in elos.items():
        season_elos.append({"Season": prev_season, "TeamID": tid, "ELO": elo})

    return pd.DataFrame(season_elos)


def compute_elo_per_game(games: pd.DataFrame) -> pd.DataFrame:
    """Retorna ELO de cada time ANTES de cada jogo (para rolling features)."""
    elos: dict[int, float] = {}
    records = []
    prev_season = None

    for idx, row in games.iterrows():
        season = row["Season"]

        if prev_season is not None and season != prev_season:
            for tid in elos:
                elos[tid] = elos[tid] * 0.75 + ELO_INITIAL * 0.25
        prev_season = season

        w_id, l_id = row["WTeamID"], row["LTeamID"]
        w_elo = elos.get(w_id, ELO_INITIAL)
        l_elo = elos.get(l_id, ELO_INITIAL)

        records.append({
            "GameIdx": idx,
            "Season": season,
            "DayNum": row["DayNum"],
            "WTeamID": w_id,
            "LTeamID": l_id,
            "W_ELO_pre": w_elo,
            "L_ELO_pre": l_elo,
        })

        mov = row.get("WScore", 0) - row.get("LScore", 0)
        loc = row.get("WLoc", "N")
        adj = ELO_HOME_ADVANTAGE if loc == "H" else (-ELO_HOME_ADVANTAGE if loc == "A" else 0)

        new_w, new_l = update_elo(w_elo + adj, l_elo, mov=mov)
        elos[w_id] = new_w - adj
        elos[l_id] = new_l

    return pd.DataFrame(records)