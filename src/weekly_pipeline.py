"""
Fase 5 - Pipeline en vivo.

Corre esto tan cerca del kickoff de cada juego como sea posible (la app
de la competencia permite modificar predicciones justo antes de que
inicie cada partido — así que entre más tarde se corra, mejor, dentro
de lo permitido). Genera un CSV con la predicción final (ensemble
90% mercado / 10% modelo propio) para cada juego próximo.

Uso:
    export ODDS_API_KEY="tu-api-key"
    python3 src/weekly_pipeline.py

Requiere que ya se haya corrido una vez (y cada vez que se quiera
refrescar con resultados recientes):
    python3 src/data_ingestion.py       (si aún no existe games.parquet)
    python3 src/train_final_model.py    (si aún no existe el modelo)
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import compute_elo, ELO_START
from odds_client import fetch_current_odds
from models.own_model import FEATURES_NO_MARKET
from team_reloc import canonicalize_teams

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "weekly_predictions"

SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
PBP_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

ENSEMBLE_W_MARKET = 0.95  # decisión por convicción (auditoría 2026-08-18: w=0.90 y w=1.00
# son estadísticamente indistinguibles, p=0.83 — no hay base real para "encontrar" un peso
# óptimo vía backtest). w=0.95 deja al modelo como desempate en pick'em exactos y limita
# el daño potencial si el modelo propio se equivoca.


def get_current_season_schedule(current_year: int) -> pd.DataFrame:
    """
    Descarga el schedule más reciente (incluye juegos futuros sin score
    todavía) y lo filtra a temporada regular del año actual + histórico
    completo, para tener contexto de Elo actualizado.
    """
    sched = pd.read_csv(SCHEDULES_URL)
    sched = sched[sched["game_type"] == "REG"].copy()
    sched = sched[sched["season"] <= current_year]

    keep_cols = [
        "game_id", "season", "week", "gameday",
        "home_team", "away_team", "home_score", "away_score",
        "home_moneyline", "away_moneyline", "spread_line",
        "home_rest", "away_rest", "location",
    ]
    sched = sched[[c for c in keep_cols if c in sched.columns]]

    # Fix Bug 1/2 (auditoría): mismo fix de reubicación de franquicias que
    # en data_ingestion.py — sin esto el Elo/EPA de LA/LAC/LV se corrompe.
    sched = canonicalize_teams(sched)

    # home_win: NaN si el juego todavía no se juega (score faltante)
    played = sched["home_score"].notna() & sched["away_score"].notna()
    sched["home_win"] = np.nan
    sched.loc[played, "home_win"] = (
        sched.loc[played, "home_score"] > sched.loc[played, "away_score"]
    ).astype(float)

    return sched


def get_current_team_form(current_year: int, lookback: int = 4) -> pd.DataFrame:
    """
    Forma reciente de cada equipo (EPA/play y success rate promedio de
    sus últimos `lookback` juegos jugados, sin importar si cruzan la
    frontera de temporada). A diferencia del EPA rolling de entrenamiento
    (que resetea en cada temporada nueva), aquí SÍ se permite cruzar de
    temporada — es la forma más honesta de estimar "cómo viene jugando
    el equipo ahora mismo" al inicio de una temporada nueva, cuando el
    reset intra-temporada dejaría todo en NaN. Diferencia metodológica
    documentada — ver PLAN_MAESTRO sección 12/13.
    """
    frames = []
    for season in [current_year - 1, current_year]:
        try:
            url = PBP_URL_TEMPLATE.format(season=season)
            df = pd.read_parquet(url, columns=[
                "game_id", "season", "week", "posteam", "defteam",
                "play_type", "epa", "success",
            ])
            frames.append(df)
        except Exception as e:
            print(f"AVISO: no se pudo descargar pbp de {season} ({e}). Se omite.")

    if not frames:
        raise RuntimeError("No se pudo obtener play-by-play reciente para calcular forma de equipos.")

    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].dropna(subset=["epa", "posteam", "defteam"])

    off = (
        pbp.groupby(["season", "week", "posteam"])
        .agg(off_epa_play=("epa", "mean"), off_success_rate=("success", "mean"))
        .reset_index().rename(columns={"posteam": "team"})
    )
    deff = (
        pbp.groupby(["season", "week", "defteam"])
        .agg(def_epa_play=("epa", "mean"), def_success_rate=("success", "mean"))
        .reset_index().rename(columns={"defteam": "team"})
    )
    team_week = off.merge(deff, on=["season", "week", "team"], how="outer")
    team_week = team_week.sort_values(["team", "season", "week"])

    # Últimos `lookback` juegos jugados por equipo, sin importar la temporada
    latest_form = (
        team_week.groupby("team")
        .tail(lookback)
        .groupby("team")
        .agg(
            off_epa_play_roll4=("off_epa_play", "mean"),
            def_epa_play_roll4=("def_epa_play", "mean"),
            off_success_rate_roll4=("off_success_rate", "mean"),
            def_success_rate_roll4=("def_success_rate", "mean"),
        )
        .reset_index()
    )
    return latest_form


def build_weekly_predictions():
    now = datetime.now(timezone.utc)
    current_year = now.year if now.month >= 3 else now.year - 1  # temporada NFL cruza año calendario

    print(f"Obteniendo odds en vivo (temporada {current_year})...")
    odds = fetch_current_odds()
    if odds.empty:
        print("No hay juegos próximos disponibles en la API de odds en este momento.")
        return

    print("Actualizando Elo con resultados recientes...")
    sched = get_current_season_schedule(current_year)
    sched_with_elo, _ratings = compute_elo(sched)

    # Nos quedamos con el elo_pre calculado para cada juego futuro que
    # también viene en el feed de odds (join por equipos + temporada actual)
    upcoming_elo = sched_with_elo[sched_with_elo["home_win"].isna()][
        ["game_id", "home_team", "away_team", "home_elo_pre", "away_elo_pre", "elo_diff", "home_rest", "away_rest"]
    ]

    print("Calculando forma reciente de equipos (EPA rolling cross-temporada)...")
    form = get_current_team_form(current_year)

    df = odds.merge(upcoming_elo, on=["home_team", "away_team"], how="inner")
    df = df.merge(form.add_prefix("home_"), left_on="home_team", right_on="home_team", how="left")
    df = df.merge(form.add_prefix("away_"), left_on="away_team", right_on="away_team", how="left")

    missing_form = df[[c for c in df.columns if "roll4" in c]].isna().any(axis=1)
    if missing_form.any():
        # Fix Bug 4 (auditoría 2026-08-18): antes se rellenaba con 0.0, que
        # para success_rate es un valor a -8.7 desviaciones estándar del
        # promedio real (fuera de todo el soporte de entrenamiento) — capaz
        # de producir un pick catastrófico en un juego real si un equipo
        # nuevo o sin historial completo aparece en el feed de odds. Ahora
        # se usa la media real de entrenamiento por feature (persistida en
        # feature_means.joblib), un fallback neutral de verdad.
        feature_means = joblib.load(MODELS_DIR / "feature_means.joblib")
        print(f"AVISO: {missing_form.sum()} juego(s) sin forma reciente completa (equipo nuevo o sin historial "
              f"suficiente). Se rellenan con la media histórica de entrenamiento (no con 0.0) — revisar "
              f"manualmente antes de someter de todas formas.")
        for c in df.columns:
            if "roll4" in c:
                df[c] = df[c].fillna(feature_means.get(c, df[c].mean()))
    if df["home_rest"].isna().any() or df["away_rest"].isna().any():
        df["home_rest"] = df["home_rest"].fillna(7)
        df["away_rest"] = df["away_rest"].fillna(7)

    print("Cargando modelo propio (Modelo A) y generando p_model_a...")
    model = joblib.load(MODELS_DIR / "model_a_final.joblib")
    df["p_model_a"] = model.predict_proba(df[FEATURES_NO_MARKET])[:, 1]

    print(f"Combinando ensemble (w_mercado={ENSEMBLE_W_MARKET})...")
    df["p_home_final"] = ENSEMBLE_W_MARKET * df["p_home_market"] + (1 - ENSEMBLE_W_MARKET) * df["p_model_a"]

    # Desempate explícito para pick'em exactos (p_home_final == 0.5, antes
    # `>` favorecía silenciosamente al visitante por construcción — un
    # accidente de código, no una decisión). La auditoría mostró que estos
    # casos son estadísticamente una moneda al aire de todas formas, así
    # que el criterio de desempate no importa para el accuracy esperado,
    # pero que sea explícito y defendible (Elo) en vez de un accidente.
    is_tie = np.isclose(df["p_home_final"], 0.5)
    df["pick"] = np.where(df["p_home_final"] > 0.5, df["home_team"], df["away_team"])
    df.loc[is_tie, "pick"] = np.where(
        df.loc[is_tie, "elo_diff"] >= 0, df.loc[is_tie, "home_team"], df.loc[is_tie, "away_team"]
    )
    df["confidence"] = np.where(df["p_home_final"] > 0.5, df["p_home_final"], 1 - df["p_home_final"])

    # Bandas de confianza (2026-08-19) — calibradas sobre accuracy histórico
    # real 2006-2024, no sobre intuición. Ver PLAN_MAESTRO sección 16 para
    # la tabla completa y la guía de cuándo SÍ vale la pena que el criterio
    # manual de Edward intervenga sobre el pick del modelo.
    dist_from_toss = (df["confidence"] - 0.5).abs()
    conditions = [dist_from_toss > 0.20, dist_from_toss > 0.10, dist_from_toss > 0.06]
    choices = [
        "1_LOCK (hist. 79.0% acierto — confía en el modelo, no intervengas por narrativa/rivalidad)",
        "2_CONFIADO (hist. 63.4% acierto — confía en el modelo salvo info muy concreta y reciente)",
        "3_PAREJO (hist. 56.5% acierto — el modelo tampoco tiene edge probado aquí; tu criterio con info fresca puede ayudar)",
    ]
    df["tier"] = np.select(conditions, choices,
                            default="4_MONEDA_AL_AIRE (hist. 53.4% acierto — empate real; usa info de último minuto, no presentimiento)")

    out_cols = [
        "commence_time", "home_team", "away_team", "p_home_market", "p_model_a",
        "p_home_final", "pick", "confidence", "tier", "n_bookmakers",
    ]
    result = df[out_cols].sort_values("commence_time")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"predictions_{stamp}.csv"
    result.to_csv(out_path, index=False)

    print(f"\n{result.to_string(index=False)}")
    print(f"\nGuardado: {out_path}")
    return result


if __name__ == "__main__":
    build_weekly_predictions()
