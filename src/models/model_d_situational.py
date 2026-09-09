"""
Modelo D — Elo + EPA + QB rating + features situacionales (clima, juego
divisional, semana corta, ventaja de descanso extrema).

Investigación Fase 7 (2026-08-18): última pasada buscando edge real antes
de aceptar ~67% como techo. Ver PLAN_MAESTRO sección 14 para el contexto
completo de por qué se descartó movimiento de línea (opening vs closing)
como avenida — cobertura histórica insuficiente para validar sin
sobreajustar.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.own_model import TARGET, TEST_SEASONS
from models.model_c_qb import FEATURES_MODEL_C

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "backtest_results"

SITUATIONAL_FEATURES = [
    "div_game", "wind", "temp_extreme", "dome_or_closed",
    "home_short_week", "away_short_week", "rest_advantage_extreme",
]
FEATURES_MODEL_D = FEATURES_MODEL_C + SITUATIONAL_FEATURES


def build_situational_features() -> pd.DataFrame:
    sched = pd.read_csv(RAW_DIR / "games.csv")

    sched["dome_or_closed"] = sched["roof"].isin(["dome", "closed"]).astype(int)
    sched["wind"] = sched["wind"].fillna(0)
    sched["temp_extreme"] = (
        sched["temp"].notna() & ((sched["temp"] < 32) | (sched["temp"] > 90))
    ).astype(int)
    sched["home_short_week"] = (sched["home_rest"] <= 4).astype(int)
    sched["away_short_week"] = (sched["away_rest"] <= 4).astype(int)
    rest_diff = sched["home_rest"] - sched["away_rest"]
    sched["rest_advantage_extreme"] = (rest_diff.abs() >= 6).astype(int)

    return sched[["game_id", "div_game", "wind", "temp_extreme", "dome_or_closed",
                  "home_short_week", "away_short_week", "rest_advantage_extreme"]]


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "games_with_qb_features.parquet")
    situational = build_situational_features()
    df = df.merge(situational, on="game_id", how="left")
    df = df.dropna(subset=FEATURES_MODEL_D + ["p_home_market"]).copy()
    return df


def walk_forward(df: pd.DataFrame, features: list, label: str):
    all_preds = []
    for test_season in TEST_SEASONS:
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]
        if len(train) < 200 or len(test) == 0:
            continue
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=42,
        )
        model.fit(train[features], train[TARGET])
        proba = model.predict_proba(test[features])[:, 1]
        sp = test[["game_id", "season", "week", TARGET]].copy()
        sp["p_model"] = proba
        all_preds.append(sp)
    preds_df = pd.concat(all_preds, ignore_index=True)
    overall = ((preds_df["p_model"] > 0.5).astype(int) == preds_df[TARGET]).mean()
    print(f"{label}: accuracy standalone = {overall:.4f}")
    return preds_df


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset (con features situacionales): {len(df)} juegos\n")

    preds_c = walk_forward(df, FEATURES_MODEL_C, "Modelo C (Elo+EPA+QB, sin situacionales)")
    preds_d = walk_forward(df, FEATURES_MODEL_D, "Modelo D (+ clima, div_game, semana corta, descanso extremo)")

    test_subset = df[df["season"].isin(TEST_SEASONS)]
    market_acc = ((test_subset["p_home_market"] > 0.5).astype(int) == test_subset[TARGET]).mean()

    print(f"\nMercado puro: {market_acc:.4f}")

    for name, preds in [("Modelo C", preds_c), ("Modelo D", preds_d)]:
        merged = df.merge(preds[["game_id", "season", "week", "p_model"]], on=["game_id", "season", "week"])
        best_acc, best_w = 0, 0
        for w in np.arange(0.0, 1.01, 0.05):
            pf = w * merged["p_home_market"] + (1 - w) * merged["p_model"]
            acc = ((pf > 0.5).astype(int) == merged[TARGET]).mean()
            if acc > best_acc:
                best_acc, best_w = acc, w
        print(f"Ensemble con {name}: mejor accuracy={best_acc:.4f} (w_mercado={best_w:.2f}), "
              f"mejora sobre mercado={best_acc - market_acc:+.4f}")

    final_model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", random_state=42,
    )
    final_model.fit(df[FEATURES_MODEL_D], df[TARGET])
    importances = pd.Series(final_model.feature_importances_, index=FEATURES_MODEL_D).sort_values(ascending=False)
    print(f"\nImportancia de features (Modelo D, entrenado con todo el histórico):\n{importances}")
