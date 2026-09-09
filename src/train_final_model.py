"""
Entrena el modelo final de producción — Modelo A (Elo + EPA rolling,
SIN ver el mercado), usando TODO el histórico disponible (2006-2024).

Este es el modelo que se carga en weekly_pipeline.py para generar
p_model_a de juegos nuevos. El walk-forward validation de Fase 3/4 ya
demostró su accuracy esperado (~61.8% standalone, aporta ~10% de peso
al ensemble final con el mercado).
"""

import joblib
from pathlib import Path
from xgboost import XGBClassifier

from models.own_model import load_dataset, FEATURES_NO_MARKET, TARGET

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_and_save():
    df = load_dataset()
    print(f"Entrenando Modelo A final sobre {len(df)} juegos (2006-2024, post-dropna semana 1)...")

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
    model.fit(df[FEATURES_NO_MARKET], df[TARGET])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "model_a_final.joblib"
    joblib.dump(model, out_path)
    print(f"Guardado: {out_path}")

    # Fix Bug 4 (auditoría 2026-08-18): antes weekly_pipeline.py rellenaba
    # features faltantes con 0.0 — para success_rate eso es un valor a
    # -8.7 desviaciones estándar (fuera de TODO el soporte de entrenamiento),
    # capaz de producir un pick catastrófico en un juego real. Se persisten
    # las medias reales de entrenamiento, feature por feature, para usarlas
    # como fallback correcto en producción.
    feature_means = df[FEATURES_NO_MARKET].mean().to_dict()
    means_path = MODELS_DIR / "feature_means.joblib"
    joblib.dump(feature_means, means_path)
    print(f"Guardado: {means_path} (fallback correcto para producción, reemplaza fillna(0.0))")

    return model


if __name__ == "__main__":
    train_and_save()
