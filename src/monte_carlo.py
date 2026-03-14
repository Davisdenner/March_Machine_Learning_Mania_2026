
import numpy as np
import pandas as pd
from config import SEED

#======== Simulação Monte Carlo do torneio NCAA ========#
def simulate_tournament(
        bracket: list[list[int]],
        prob_matrix: dict[tuple[int, int], float],
        n_simulations: int = 10000,
        rng_seed: int = SEED,
) -> pd.DataFrame:

    rng = np.random.RandomState(rng_seed)

    n_teams = len(bracket[0]) if bracket else 0
    advancement_counts = {tid: np.zeros(7) for tid in bracket[0]}  #até 6 rounds

    for _ in range(n_simulations):
        current_round = list(bracket[0])  #cópia

        round_num = 0
        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                if i + 1 >= len(current_round):
                    next_round.append(current_round[i])
                    continue

                t1 = min(current_round[i], current_round[i + 1])
                t2 = max(current_round[i], current_round[i + 1])

                p = prob_matrix.get((t1, t2), 0.5)

                if current_round[i] == t1:
                    winner = t1 if rng.random() < p else t2
                else:
                    winner = t2 if rng.random() < (1 - p) else t1

                next_round.append(winner)

            round_num += 1
            for tid in next_round:
                advancement_counts[tid][round_num] += 1

            current_round = next_round

    #resumo
    records = []
    for tid, counts in advancement_counts.items():
        rec = {"TeamID": tid}
        for r in range(1, 7):
            rec[f"Round{r}_Pct"] = counts[r] / n_simulations
        rec["Championship_Pct"] = counts[6] / n_simulations if len(counts) > 6 else 0
        records.append(rec)

    return pd.DataFrame(records).sort_values("Championship_Pct", ascending=False)

#======== construir um dicionário de probabilidades a partir do submission ========#
def build_prob_matrix(submission_df: pd.DataFrame) -> dict[tuple[int, int], float]:

    prob_matrix = {}
    for _, row in submission_df.iterrows():
        parts = row["ID"].split("_")
        t1, t2 = int(parts[1]), int(parts[2])
        prob_matrix[(t1, t2)] = row["Pred"]
    return prob_matrix
