# -*- coding: utf-8 -*-
import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

#garantir imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SEED, TRAIN_SEASONS_END, VAL_SEASON, OUTPUT_DIR
from data_loading import (
    build_all_games,
    load_seeds,
    load_tourney_compact,
    load_tourney_detailed,
    load_sample_submission,
    load_massey_ordinals,
)
from elo_rating import compute_elo_ratings
from feature_engineering import compute_season_stats, compute_tourney_history
from dataset_builder import (
    build_training_matchups,
    build_submission_matchups,
    get_feature_columns,
)
from validation import temporal_cv_splits, evaluate_predictions
from model_training import (
    get_lgb_model,
    get_xgb_model,
    get_catboost_model,
    get_lr_model,
    train_model_with_early_stopping,
)
from ensemble import EnsemblePredictor
from submission import generate_submission


#features mais importantes (seleção manual baseada em importância + estabilidade)
TOP_FEATURES = [
    "Diff_Seed", "Diff_ELO", "A_Seed", "B_Seed", "A_ELO", "B_ELO",
    "Diff_WinPct", "Diff_SOS", "Diff_NetEff", "Diff_OffEff_mean", "Diff_DefEff_mean",
    "Diff_Rolling_WinPct", "Diff_Rolling_ScoreDiff",
    "Diff_HistTourneyWins", "A_HistTourneyWins", "B_HistTourneyWins",
    "Diff_Massey_POM", "Diff_Massey_SAG", "Diff_Massey_MOR", "Diff_Massey_COL", "Diff_Massey_DOL",
    "A_Massey_POM", "B_Massey_POM",
    "Diff_FGPct", "Diff_FG3Pct", "Diff_AstTO",
    "Diff_Score_mean", "Diff_AvgScoreDiff",
    "Diff_OR_mean", "Diff_DR_mean", "Diff_Stl_mean",
    "Diff_Wins",
]


def main():
    print("=" * 60)
    print(" March Machine Learning Mania 2026")
    print("=" * 60)

    #1. CARREGAR DADOS
    print("\n Carregando dados...")
    games = build_all_games()
    seeds = load_seeds()
    tourney = load_tourney_compact()
    sample_sub = load_sample_submission(stage=2)

    print(f"   Jogos: {len(games):,}")
    print(f"   Temporadas: {games['Season'].nunique()}")
    print(f"   Torneio: {len(tourney):,} jogos")

    #2. ELO RATINGS
    print("\n Calculando ELO Ratings...")
    elo_df = compute_elo_ratings(games)
    print(f"   ELO ratings calculados para {elo_df['TeamID'].nunique()} times")

    #3. FEATURE ENGINEERING
    print("\n Engenharia de Features...")
    team_features = compute_season_stats(games)
    tourney_history = compute_tourney_history(tourney)
    massey_df = load_massey_ordinals()
    print(f"   Features por time: {team_features.shape[1]} colunas")
    print(f"   Massey Ordinals: {massey_df.shape[1] - 2} sistemas")
    print(f"   Temporadas com stats: {team_features['Season'].nunique()}")

    #4. DATASET DE CONFRONTOS
    print("\n Construindo dataset de confrontos...")
    train_matchups = build_training_matchups(
        tourney, team_features, seeds, elo_df, tourney_history, massey_df
    )
    all_feature_cols = get_feature_columns(train_matchups)

    #feature selection: usar apenas top features que existem no dataset
    feature_cols = [f for f in TOP_FEATURES if f in all_feature_cols]
    #adicionar qualquer Massey restante
    for c in all_feature_cols:
        if "Massey" in c and c not in feature_cols:
            feature_cols.append(c)

    print(f"   Matchups de treino: {len(train_matchups):,}")
    print(f"   Features totais: {len(all_feature_cols)}, selecionadas: {len(feature_cols)}")

    #5. VALIDAÇÃO TEMPORAL
    print("\n Validação Temporal...")
    splits = temporal_cv_splits(train_matchups, first_train_end=2014, last_val_season=2024)
    print(f"   Splits de validação: {len(splits)}")

    X = train_matchups[feature_cols].values
    y = train_matchups["Result"].values

    #6. TREINAR MODELOS COM CV
    print("\n Treinando modelos...")

    model_factories = {
        "lgb": get_lgb_model,
        "xgb": get_xgb_model,
        "cat": get_catboost_model,
        "lr": get_lr_model,
    }

    cv_scores = {name: [] for name in model_factories}
    cv_preds = {name: np.zeros(len(y)) for name in model_factories}

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        val_season = train_matchups.iloc[val_idx]["Season"].iloc[0]
        print(f"\n   Fold {fold_idx + 1}: Val Season = {val_season}")

        for name, factory in model_factories.items():
            model = factory()
            if name != "lr":
                model = train_model_with_early_stopping(model, X_tr, y_tr, X_val, y_val)
            else:
                model.fit(X_tr, y_tr)

            preds = model.predict_proba(X_val)[:, 1]
            score = evaluate_predictions(y_val, preds)
            cv_scores[name].append(score)
            cv_preds[name][val_idx] = preds
            print(f"      {name}: Log Loss = {score:.5f}")

    print("\n Resultados CV (média):")
    for name, scores in cv_scores.items():
        print(f"   {name}: {np.mean(scores):.5f} (±{np.std(scores):.5f})")

    #7. DETERMINAR PESOS DO ENSEMBLE
    print("\nCalculando pesos do ensemble...")
    mean_scores = {name: np.mean(scores) for name, scores in cv_scores.items()}

    #pesos inversamente proporcionais ao CV score, com LR floor
    inv = {n: 1.0 / s for n, s in mean_scores.items()}
    total_inv = sum(inv.values())
    weights = {n: v / total_inv for n, v in inv.items()}
    #garantir LR tenha pelo menos 0.35
    if weights["lr"] < 0.35:
        weights["lr"] = 0.40
        remaining = 0.60
        others = {n: v for n, v in weights.items() if n != "lr"}
        ot = sum(others.values())
        for n in others:
            weights[n] = others[n] / ot * remaining

    for name, w in weights.items():
        print(f"   {name}: {w:.3f} (CV: {mean_scores[name]:.5f})")

    #8. CALIBRAÇÃO OOF
    print("\n Calibrando com previsões Out-of-Fold...")
    oof_mask = np.zeros(len(y), dtype=bool)
    for _, val_idx in splits:
        oof_mask[val_idx] = True

    oof_ensemble = np.zeros(len(y))
    for name, w in weights.items():
        oof_ensemble += cv_preds[name] * w

    oof_ensemble_valid = oof_ensemble[oof_mask]
    y_oof_valid = y[oof_mask]

    oof_logloss_raw = evaluate_predictions(y_oof_valid, oof_ensemble_valid)
    print(f"   OOF Log Loss raw: {oof_logloss_raw:.5f}")

    #calibração: só usar se melhorar
    from sklearn.linear_model import LogisticRegression as LR_Cal
    calibrator = LR_Cal(C=1.0, random_state=SEED)
    calibrator.fit(oof_ensemble_valid.reshape(-1, 1), y_oof_valid)

    oof_calibrated = calibrator.predict_proba(oof_ensemble_valid.reshape(-1, 1))[:, 1]
    oof_logloss_cal = evaluate_predictions(y_oof_valid, oof_calibrated)
    print(f"   OOF Log Loss calibrado: {oof_logloss_cal:.5f}")

    use_calibrator = oof_logloss_cal < oof_logloss_raw
    print(f"   Usar calibração: {'SIM' if use_calibrator else 'NÃO  (raw é melhor)'}")

    #9. TREINAR MODELOS FINAIS
    print("\n️ Treinando modelos finais (full data)...")

    #usar TUDO para treino final (incluindo VAL_SEASON)
    train_mask = train_matchups["Season"] <= VAL_SEASON
    X_train_final = X[train_mask.values]
    y_train_final = y[train_mask.values]

    #Split interno para early stopping
    from sklearn.model_selection import train_test_split
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train_final, y_train_final, test_size=0.1, random_state=SEED, stratify=y_train_final
    )

    ensemble = EnsemblePredictor()

    for name, factory in model_factories.items():
        model = factory()
        if name != "lr":
            model = train_model_with_early_stopping(model, X_fit, y_fit, X_es, y_es)
        else:
            model.fit(X_train_final, y_train_final)

        ensemble.add_model(name, model, weight=weights[name])
        print(f"{name} treinado")

    #guardar calibrador só se melhorou
    if use_calibrator:
        ensemble.calibrator = calibrator
    else:
        ensemble.calibrator = None

    #10. GERAR SUBMISSÃO
    print("\n Gerando submissão...")
    sub_matchups = build_submission_matchups(
        sample_sub, team_features, seeds, elo_df, tourney_history, massey_df
    )

    sub_X = sub_matchups[feature_cols].values
    sub_preds = ensemble.predict_calibrated(sub_X)

    submission = generate_submission(sub_matchups, sub_preds, "submission.csv")

    #11. FEATURE IMPORTANCE
    print("\n Top 20 Features (LightGBM):")
    lgb_model = ensemble.models["lgb"]
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": lgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(20)

    for _, row in importance.iterrows():
        print(f"   {row['Feature']:40s} {row['Importance']:>6.0f}")

    print("\n" + "=" * 60)
    print(" Pipeline concluído com sucesso!")
    print("=" * 60)

    return submission, ensemble


if __name__ == "__main__":
    submission, ensemble = main()