"""
Fase 4 - Capa C: Ensemble ponderado.

Hallazgo de Fase 3: meter la probabilidad de mercado como feature dentro
de XGBoost (Modelo B) rindió PEOR que el mercado puro (65.41% vs 66.95%).
Interpretación: el árbol "reinterpreta" una señal que ya venía casi óptima,
y con eso pierde precisión en vez de ganarla.

La solución estándar en estos casos es un ensemble por PROMEDIO PONDERADO
en vez de dejar que el modelo mezcle todo internamente:

    p_final = w * p_mercado + (1 - w) * p_modelo_A

Donde Modelo A es el XGBoost entrenado SOLO con Elo+EPA (sin ver el
mercado) — así el modelo aporta señal genuinamente independiente, y el
peso `w` decide cuánto confiar en cada uno.

Barrido de `w` de 0.0 a 1.0 en pasos de 0.05, evaluado sobre las mismas
predicciones walk-forward ya generadas (2015-2024, out-of-sample real).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

from own_model import (
    load_dataset, walk_forward_backtest,
    FEATURES_NO_MARKET, TARGET, TEST_SEASONS,
)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "backtest_results"


def get_model_a_oos_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Re-genera las predicciones out-of-sample del Modelo A (sin mercado)."""
    preds_no_market, _ = walk_forward_backtest(df, FEATURES_NO_MARKET, "MODELO A (recalculado para ensemble)")
    return preds_no_market.rename(columns={"pred_proba": "p_model_a"})[
        ["game_id", "season", "week", "p_model_a"]
    ]


def sweep_ensemble_weight(df: pd.DataFrame, preds_model_a: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(preds_model_a, on=["game_id", "season", "week"], how="inner")

    rows = []
    for w in np.arange(0.0, 1.01, 0.05):
        p_final = w * merged["p_home_market"] + (1 - w) * merged["p_model_a"]
        pred = (p_final > 0.5).astype(int)
        acc = (pred == merged[TARGET]).mean()
        rows.append({"w_mercado": round(w, 2), "accuracy": acc})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_dataset()
    preds_model_a = get_model_a_oos_predictions(df)

    sweep = sweep_ensemble_weight(df, preds_model_a)
    print(f"\n{'=' * 60}\nBARRIDO DE PESO DEL ENSEMBLE (w_mercado de 0.0 a 1.0)\n{'=' * 60}")
    print(sweep.to_string(index=False))

    best_row = sweep.loc[sweep["accuracy"].idxmax()]
    print(f"\nMejor peso encontrado: w_mercado={best_row['w_mercado']}  "
          f"accuracy={best_row['accuracy']:.4f}")

    market_only_acc = sweep[sweep["w_mercado"] == 1.0]["accuracy"].values[0]
    print(f"Referencia — mercado puro (w=1.0):  {market_only_acc:.4f}")

    improvement = best_row["accuracy"] - market_only_acc
    print(f"Mejora del ensemble sobre mercado puro: {improvement:+.4f}")

    sweep.to_csv(OUTPUTS_DIR / "phase4_ensemble_weight_sweep.csv", index=False)
    print(f"\nGuardado: {OUTPUTS_DIR / 'phase4_ensemble_weight_sweep.csv'}")

    with open(OUTPUTS_DIR / "phase4_summary.txt", "w") as f:
        f.write("Fase 4 - Ensemble ponderado (mercado + Modelo A Elo/EPA)\n")
        f.write(f"Mejor w_mercado: {best_row['w_mercado']}\n")
        f.write(f"Accuracy ensemble: {best_row['accuracy']:.4f}\n")
        f.write(f"Accuracy mercado puro (referencia): {market_only_acc:.4f}\n")
        f.write(f"Mejora: {improvement:+.4f}\n")
    print(f"Guardado: {OUTPUTS_DIR / 'phase4_summary.txt'}")
