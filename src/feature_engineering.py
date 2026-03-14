"""Feature engineering avançado: estatísticas por time/temporada."""
import numpy as np
import pandas as pd
from config import ROLLING_WINDOW


# ── Helpers ────────────────────────────────────────────
def _possessions(row, prefix: str) -> float:
    """Estimativa de posses de bola (Dean Oliver)."""
    fga = row.get(f"{prefix}FGA", 0)
    fta = row.get(f"{prefix}FTA", 0)
    orb = row.get(f"{prefix}OR", 0)
    to = row.get(f"{prefix}TO", 0)
    return fga - orb + to + 0.475 * fta


# ── Estatísticas agregadas por time / temporada ──────
def compute_season_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas agregadas por time e temporada.

    Usa apenas jogos da temporada regular (IsTourney == False).
    """
    reg = games[~games["IsTourney"]].copy()

    has_detailed = "WFGM" in reg.columns

    # Construir registros por time (um row por time-jogo)
    rows = []
    for _, g in reg.iterrows():
        season = g["Season"]
        for role in ("W", "L"):
            opp_role = "L" if role == "W" else "W"
            tid = g[f"{role}TeamID"]
            opp_id = g[f"{opp_role}TeamID"]
            score = g[f"{role}Score"]
            opp_score = g[f"{opp_role}Score"]
            won = 1 if role == "W" else 0

            rec = {
                "Season": season,
                "TeamID": tid,
                "OppID": opp_id,
                "Score": score,
                "OppScore": opp_score,
                "Won": won,
                "DayNum": g["DayNum"],
                "ScoreDiff": score - opp_score,
            }

            if has_detailed:
                poss = _possessions(g, role)
                opp_poss = _possessions(g, opp_role)
                avg_poss = max((poss + opp_poss) / 2, 1)

                rec["Poss"] = avg_poss
                rec["OffEff"] = score / avg_poss * 100
                rec["DefEff"] = opp_score / avg_poss * 100
                rec["FGM"] = g.get(f"{role}FGM", 0)
                rec["FGA"] = g.get(f"{role}FGA", 0)
                rec["FGM3"] = g.get(f"{role}FGM3", 0)
                rec["FGA3"] = g.get(f"{role}FGA3", 0)
                rec["FTM"] = g.get(f"{role}FTM", 0)
                rec["FTA"] = g.get(f"{role}FTA", 0)
                rec["OR"] = g.get(f"{role}OR", 0)
                rec["DR"] = g.get(f"{role}DR", 0)
                rec["Ast"] = g.get(f"{role}Ast", 0)
                rec["TO"] = g.get(f"{role}TO", 0)
                rec["Stl"] = g.get(f"{role}Stl", 0)
                rec["Blk"] = g.get(f"{role}Blk", 0)
                rec["PF"] = g.get(f"{role}PF", 0)

            rows.append(rec)

    tg = pd.DataFrame(rows)

    # ── Agregações ──
    agg_dict = {
        "Won": ["sum", "count"],
        "Score": "mean",
        "OppScore": "mean",
        "ScoreDiff": "mean",
    }
    if has_detailed:
        agg_dict.update({
            "OffEff": "mean",
            "DefEff": "mean",
            "FGM": "mean",
            "FGA": "mean",
            "FGM3": "mean",
            "FGA3": "mean",
            "FTM": "mean",
            "FTA": "mean",
            "OR": "mean",
            "DR": "mean",
            "Ast": "mean",
            "TO": "mean",
            "Stl": "mean",
            "Blk": "mean",
            "PF": "mean",
            "Poss": "mean",
        })

    stats = tg.groupby(["Season", "TeamID"]).agg(agg_dict)
    stats.columns = ["_".join(c).strip("_") for c in stats.columns]
    stats = stats.rename(columns={"Won_sum": "Wins", "Won_count": "Games"})
    stats["WinPct"] = stats["Wins"] / stats["Games"]
    stats["AvgScoreDiff"] = stats["ScoreDiff_mean"]

    if has_detailed:
        stats["FGPct"] = stats["FGM_mean"] / stats["FGA_mean"].replace(0, 1)
        stats["FG3Pct"] = stats["FGM3_mean"] / stats["FGA3_mean"].replace(0, 1)
        stats["FTPct"] = stats["FTM_mean"] / stats["FTA_mean"].replace(0, 1)
        stats["AstTO"] = stats["Ast_mean"] / stats["TO_mean"].replace(0, 1)
        stats["NetEff"] = stats["OffEff_mean"] - stats["DefEff_mean"]

    stats = stats.reset_index()

    # ── Rolling (últimos N jogos da temporada) ────────
    tg_sorted = tg.sort_values(["Season", "TeamID", "DayNum"])
    rolling = (
        tg_sorted
        .groupby(["Season", "TeamID"])
        .tail(ROLLING_WINDOW)
        .groupby(["Season", "TeamID"])
        .agg({"Won": "mean", "ScoreDiff": "mean", "Score": "mean"})
        .rename(columns={
            "Won": "Rolling_WinPct",
            "ScoreDiff": "Rolling_ScoreDiff",
            "Score": "Rolling_Score",
        })
        .reset_index()
    )

    stats = stats.merge(rolling, on=["Season", "TeamID"], how="left")

    # ── Strength of Schedule ──────────────────────────
    # Primeiro: calcular WinPct de cada time
    winpct_map = stats.set_index(["Season", "TeamID"])["WinPct"].to_dict()

    def _opp_winpct(row):
        return winpct_map.get((row["Season"], row["OppID"]), 0.5)

    tg["OppWinPct"] = tg.apply(_opp_winpct, axis=1)
    sos = (
        tg.groupby(["Season", "TeamID"])["OppWinPct"]
        .mean()
        .rename("SOS")
        .reset_index()
    )
    stats = stats.merge(sos, on=["Season", "TeamID"], how="left")

    # Adjusted efficiency
    if has_detailed:
        off_eff_map = stats.set_index(["Season", "TeamID"]).get("OffEff_mean", pd.Series(dtype=float)).to_dict()

        def _opp_def_eff(row):
            return off_eff_map.get((row["Season"], row["OppID"]), 100)

        tg["OppOffEff"] = tg.apply(_opp_def_eff, axis=1)
        adj_off = (
            tg.groupby(["Season", "TeamID"])
            .apply(lambda x: (x["OffEff"] * x["OppOffEff"] / 100).mean() if "OffEff" in x else np.nan)
            .rename("AdjOffEff")
            .reset_index()
        )
        stats = stats.merge(adj_off, on=["Season", "TeamID"], how="left")

    return stats


def compute_tourney_history(tourney_results: pd.DataFrame) -> pd.DataFrame:
    """Calcular histórico de torneio: vitórias totais por time."""
    rows = []
    for _, g in tourney_results.iterrows():
        rows.append({"Season": g["Season"], "TeamID": g["WTeamID"], "TourneyWin": 1})
        rows.append({"Season": g["Season"], "TeamID": g["LTeamID"], "TourneyWin": 0})

    df = pd.DataFrame(rows)

    # Histórico acumulado até temporada anterior (evita leak)
    df_sorted = df.sort_values("Season")
    # Para cada (Season, TeamID), contar vitórias em torneios de temporadas ANTERIORES
    cumulative = []
    for (tid,), grp in df.groupby(["TeamID"]):
        yearly = grp.groupby("Season")["TourneyWin"].sum().sort_index()
        cum = yearly.cumsum().shift(1, fill_value=0)
        for s, v in cum.items():
            cumulative.append({"Season": s, "TeamID": tid, "HistTourneyWins": v})

    return pd.DataFrame(cumulative)
