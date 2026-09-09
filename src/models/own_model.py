"""
Fase 3 - Capa B: Modelo propio (XGBoost) con walk-forward validation.

Se entrenan y comparan DOS variantes:
  - Modelo A: solo Elo + EPA rolling + descanso (sin ver el mercado).
              Sirve para medir cuánta señal propia aportamos de verdad.
  - Modelo B: lo mismo + probabilidad del mercado como feature más.
              Esta es la versión que realmente se compite, porque no hay
              razón para ocultarle al modelo la mejor señal disponible.

Walk-forward: para cada temporada de test (2015-2024), se entrena con
TODAS las temporadas anteriores (ventana expansiva) y se predice esa
temporada completa. Nunca se usa información de una temporada futura
para entrenar. Los juegos de semana 1 de cada temporada (sin EPA rolling
previo) se excluyen del entrenamiento y de la evaluación.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "backtest_results"

FEATURES_NO_MARKET = [
    "elo_diff",
    "home_off_epa_play_roll4", "home_def_epa_play_roll4",
    "home_off_success_rate_roll4", "home_def_success_rate_roll4",
    "away_off_epa_play_roll4", "away_def_epa_play_roll4",
    "away_off_success_rate_roll4", "away_def_success_rate_roll4",
    "home_rest", "away_rest",
]
FEATURES_WITH_MARKET = FEATURES_NO_MARKET + ["p_home_market"]

TARGET = "home_win"
TEST_SEASONS = list(range(2015, 2025))  # 2015-2024, walk-forward expansivo


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "games_with_features.parquet")
    # Excluir semana 1 de cada temporada: sin EPA rolling previo (NaN por diseño)
    df = df.dropna(subset=FEATURES_WITH_MARKET).copy()
    return df


def walk_forward_backtest(df: pd.DataFrame, features: list, label: str) -> pd.DataFrame:
    results = []
    all_preds = []

    for test_season in TEST_SEASONS:
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]

        if len(train) < 200 or len(test) == 0:
            continue

        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(train[features], train[TARGET])

        preds_proba = model.predict_proba(test[features])[:, 1]
        preds = (preds_proba > 0.5).astype(int)
        acc = (preds == test[TARGET].values).mean()

        results.append({"season": test_season, "accuracy": acc, "n_games": len(test)})

        season_preds = test[["game_id", "season", "week", "home_team", "away_team", TARGET]].copy()
        season_preds["pred_proba"] = preds_proba
        season_preds["pred"] = preds
        all_preds.append(season_preds)

    results_df = pd.DataFrame(results)
    preds_df = pd.concat(all_preds, ignore_index=True)

    overall_acc = (preds_df["pred"] == preds_df[TARGET]).mean()
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(results_df.set_index("season").round(4))
    print(f"\nAccuracy global walk-forward ({label}): {overall_acc:.4f}  ({len(preds_df)} juegos)")

    return preds_df, overall_acc


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset para modelado (post-dropna semana 1): {len(df)} juegos")

    preds_no_market, acc_no_market = walk_forward_backtest(
        df, FEATURES_NO_MARKET, "MODELO A — Elo + EPA rolling (SIN mercado)"
    )
    preds_with_market, acc_with_market = walk_forward_backtest(
        df, FEATURES_WITH_MARKET, "MODELO B — Elo + EPA rolling + probabilidad de mercado"
    )

    # Baseline de mercado puro, sobre el MISMO subconjunto (post-dropna, mismas temporadas de test)
    test_subset = df[df["season"].isin(TEST_SEASONS)]
    market_pred = (test_subset["p_home_market"] > 0.5).astype(int)
    market_acc = (market_pred == test_subset[TARGET]).mean()

    print(f"\n{'=' * 60}\nRESUMEN COMPARATIVO (mismo subconjunto de test: 2015-2024, sin semana 1)\n{'=' * 60}")
    print(f"Mercado puro (Capa A):                          {market_acc:.4f}")
    print(f"Modelo A - Elo+EPA sin mercado:                 {acc_no_market:.4f}")
    print(f"Modelo B - Elo+EPA + mercado como feature:      {acc_with_market:.4f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_with_market.to_parquet(OUTPUTS_DIR / "walkforward_predictions_model_b.parquet", index=False)

    with open(OUTPUTS_DIR / "phase3_summary.txt", "w") as f:
        f.write("Fase 3 - Resumen walk-forward (test 2015-2024, excluye semana 1 de cada temporada)\n")
        f.write(f"Mercado puro (Capa A):                     {market_acc:.4f}\n")
        f.write(f"Modelo A - Elo+EPA sin mercado:             {acc_no_market:.4f}\n")
        f.write(f"Modelo B - Elo+EPA + mercado como feature:  {acc_with_market:.4f}\n")

    print(f"\nGuardado: {OUTPUTS_DIR / 'phase3_summary.txt'}")
    print(f"Guardado: {OUTPUTS_DIR / 'walkforward_predictions_model_b.parquet'}")
