"""
Fase 1 - Capa A: Baseline de mercado
Convierte el moneyline de cierre a probabilidad implícita, remueve el
vig (margen de la casa), y mide el accuracy histórico de "apostarle
siempre al favorito del mercado" por temporada y agregado.

Este número es la referencia que todo el resto del proyecto (Elo, EPA,
XGBoost, ensemble) tiene que superar para justificar su propia existencia.
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def moneyline_to_implied_prob(moneyline: pd.Series) -> pd.Series:
    """Convierte moneyline americano a probabilidad implícita CRUDA (con vig)."""
    ml = moneyline.astype(float)
    prob = pd.Series(index=ml.index, dtype=float)
    prob[ml < 0] = -ml[ml < 0] / (-ml[ml < 0] + 100)
    prob[ml > 0] = 100 / (ml[ml > 0] + 100)
    return prob


def devig(p_home_raw: pd.Series, p_away_raw: pd.Series) -> pd.Series:
    """Remueve el overround (vig) normalizando a que sumen 1."""
    overround = p_home_raw + p_away_raw
    return p_home_raw / overround


def add_market_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_home_raw"] = moneyline_to_implied_prob(df["home_moneyline"])
    df["p_away_raw"] = moneyline_to_implied_prob(df["away_moneyline"])
    df["p_home_market"] = devig(df["p_home_raw"], df["p_away_raw"])

    # Predicción binaria: le apostamos a quien el mercado favorece
    df["pred_market"] = (df["p_home_market"] > 0.5).astype(int)
    df["correct_market"] = (df["pred_market"] == df["home_win"]).astype(int)
    return df


def report_accuracy(df: pd.DataFrame):
    print("=" * 60)
    print("BASELINE DE MERCADO — accuracy por temporada")
    print("=" * 60)
    by_season = df.groupby("season")["correct_market"].agg(["mean", "count"])
    by_season.columns = ["accuracy", "n_juegos"]
    print(by_season.round(4))

    overall_acc = df["correct_market"].mean()
    overall_n = len(df)

    # Brier score como diagnóstico secundario (no es la métrica de la competencia,
    # pero indica qué tan bien calibrado está el mercado)
    brier = ((df["p_home_market"] - df["home_win"]) ** 2).mean()

    print("=" * 60)
    print(f"ACCURACY GLOBAL (2006-2024, {overall_n} juegos): {overall_acc:.4f}")
    print(f"Brier score (diagnóstico, no es la métrica de la competencia): {brier:.4f}")
    print("=" * 60)
    print("\n>>> Este es el número a vencer. Cualquier modelo propio que no")
    print(">>> supere este accuracy en el set de validación no debe usarse.")

    return overall_acc


if __name__ == "__main__":
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    games = add_market_baseline(games)
    games.to_parquet(PROCESSED_DIR / "games_with_market_baseline.parquet", index=False)
    report_accuracy(games)
