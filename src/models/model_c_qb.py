"""
Modelo C — Elo + EPA rolling + QB rating (la mejora de mayor prioridad).

Compara, sobre el MISMO esquema walk-forward de Fase 3 (test 2015-2024,
entrenamiento expansivo con temporadas estrictamente anteriores):
  - Mercado puro (referencia)
  - Modelo A (Elo + EPA, sin QB)          — ya medido en Fase 3
  - Modelo C (Elo + EPA + QB rating)      — nuevo
  - Ensemble mercado + Modelo C, barrido de peso (igual que Fase 4)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.own_model import FEATURES_NO_MARKET, TARGET, TEST_SEASONS

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "backtest_results"

FEATURES_MODEL_C = FEATURES_NO_MARKET + [
    "home_qb_rating", "away_qb_rating", "qb_rating_diff",
    "home_qb_starts_prior", "away_qb_starts_prior",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "games_with_qb_features.parquet")
    df = df.dropna(subset=FEATURES_MODEL_C + ["p_home_market"]).copy()
    return df


def walk_forward(df: pd.DataFrame, features: list, label: str):
    results, all_preds = [], []
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
        pred = (proba > 0.5).astype(int)
        acc = (pred == test[TARGET].values).mean()
        results.append({"season": test_season, "accuracy": acc, "n_games": len(test)})

        sp = test[["game_id", "season", "week", TARGET]].copy()
        sp["pred_proba"] = proba
        all_preds.append(sp)

    results_df = pd.DataFrame(results)
    preds_df = pd.concat(all_preds, ignore_index=True)
    overall = (preds_df["pred_proba"] > 0.5).astype(int).eq(preds_df[TARGET]).mean()
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(results_df.set_index("season").round(4))
    print(f"Accuracy global: {overall:.4f}  ({len(preds_df)} juegos)")
    return preds_df.rename(columns={"pred_proba": "p_model"}), overall


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset (con QB rating, post-dropna): {len(df)} juegos")

    preds_a, acc_a = walk_forward(df, FEATURES_NO_MARKET, "MODELO A — Elo+EPA (sin QB) [recalculado sobre este subset]")
    preds_c, acc_c = walk_forward(df, FEATURES_MODEL_C, "MODELO C — Elo+EPA+QB rating")

    test_subset = df[df["season"].isin(TEST_SEASONS)]
    market_pred = (test_subset["p_home_market"] > 0.5).astype(int)
    market_acc = (market_pred == test_subset[TARGET]).mean()

    print(f"\n{'=' * 60}\nRESUMEN — mismo subconjunto de test (2015-2024)\n{'=' * 60}")
    print(f"Mercado puro:                    {market_acc:.4f}")
    print(f"Modelo A (Elo+EPA, sin QB):       {acc_a:.4f}")
    print(f"Modelo C (Elo+EPA+QB rating):     {acc_c:.4f}")

    # Ensemble mercado + Modelo C
    merged = df.merge(preds_c[["game_id", "season", "week", "p_model"]], on=["game_id", "season", "week"], how="inner")
    rows = []
    for w in np.arange(0.0, 1.01, 0.05):
        p_final = w * merged["p_home_market"] + (1 - w) * merged["p_model"]
        acc = ((p_final > 0.5).astype(int) == merged[TARGET]).mean()
        rows.append({"w_mercado": round(w, 2), "accuracy": acc})
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["accuracy"].idxmax()]

    print(f"\n{'=' * 60}\nENSEMBLE mercado + Modelo C — barrido de peso\n{'=' * 60}")
    print(sweep.to_string(index=False))
    print(f"\nMejor: w_mercado={best['w_mercado']}  accuracy={best['accuracy']:.4f}")
    print(f"Mejora sobre mercado puro: {best['accuracy'] - market_acc:+.4f}")
    print(f"Mejora sobre ensemble anterior (Fase 4, sin QB, 67.04%): {best['accuracy'] - 0.6704:+.4f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUTPUTS_DIR / "phase6_qb_ensemble_weight_sweep.csv", index=False)
    with open(OUTPUTS_DIR / "phase6_qb_summary.txt", "w") as f:
        f.write("Fase 6 - Modelo C (con QB rating) + ensemble\n")
        f.write(f"Mercado puro:                 {market_acc:.4f}\n")
        f.write(f"Modelo A (sin QB):            {acc_a:.4f}\n")
        f.write(f"Modelo C (con QB rating):     {acc_c:.4f}\n")
        f.write(f"Mejor ensemble (w={best['w_mercado']}):  {best['accuracy']:.4f}\n")
    print(f"\nGuardado: {OUTPUTS_DIR / 'phase6_qb_summary.txt'}")
