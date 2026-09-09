"""
check_results.py — Compara el ÚLTIMO pick hecho ANTES del kickoff de
cada juego contra el resultado real, una vez que el juego ya se jugó.

Genera/actualiza outputs/track_record.csv, que alimenta la pestaña
"🎯 Track record" del dashboard.

Se puede correr solo:
    python3 check_results.py
o como parte de run_pipeline.py (que ya lo incluye como último paso).
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history_utils import load_all_snapshots, get_last_pre_kickoff_snapshot
from weekly_pipeline import get_current_season_schedule  # reutiliza la misma fuente de resultados reales

PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "weekly_predictions"
TRACK_RECORD_PATH = Path(__file__).resolve().parent.parent / "outputs" / "track_record.csv"

# El schedule de nflverse usa fecha de calendario US (gameday); las odds
# vienen en UTC. Un juego de domingo 8:20pm ET cae en 00:20 UTC del
# LUNES — sin este ajuste, el emparejamiento por fecha fallaría en todos
# los juegos nocturnos. -5h es una aproximación a hora del Este (no
# corrige DST con precisión, pero es más que suficiente para emparejar
# por día calendario).
US_OFFSET_HOURS = 5


def get_actual_results(current_year: int) -> pd.DataFrame:
    """Trae resultados reales ya jugados de la temporada actual."""
    sched = get_current_season_schedule(current_year)
    played = sched.dropna(subset=["home_score", "away_score"]).copy()
    played["gameday"] = pd.to_datetime(played["gameday"]).dt.date
    played["actual_winner"] = played.apply(
        lambda r: r["home_team"] if r["home_score"] > r["away_score"] else r["away_team"], axis=1
    )
    return played[["home_team", "away_team", "gameday", "home_score", "away_score", "actual_winner"]]


def build_track_record(current_year: int) -> pd.DataFrame:
    history = load_all_snapshots(PREDICTIONS_DIR)
    if history.empty:
        print("No hay snapshots de predicciones todavía — corre weekly_pipeline.py primero.")
        return pd.DataFrame()

    last_pre_kickoff = get_last_pre_kickoff_snapshot(history)
    if last_pre_kickoff.empty:
        print("Hay snapshots, pero ninguno se generó ANTES del kickoff de algún juego — "
              "nada que calificar todavía (¿corriste el pipeline después de que empezaran los juegos?).")
        return pd.DataFrame()

    last_pre_kickoff = last_pre_kickoff.copy()
    last_pre_kickoff["gameday"] = (
        last_pre_kickoff["commence_time"] - pd.Timedelta(hours=US_OFFSET_HOURS)
    ).dt.date

    results = get_actual_results(current_year)

    merged = last_pre_kickoff.merge(
        results, on=["home_team", "away_team", "gameday"], how="inner"
    )
    if merged.empty:
        print("Ningún juego con predicción pre-kickoff tiene resultado registrado todavía "
              "(¿ya se jugaron? puede tardar unas horas en publicarse el score final).")
        return pd.DataFrame()

    merged["correct"] = (merged["pick"] == merged["actual_winner"]).astype(int)

    out_cols = [
        "game_key", "home_team", "away_team", "commence_time", "generated_at",
        "pick", "confidence", "tier", "actual_winner", "home_score", "away_score", "correct",
    ]
    track = merged[out_cols].sort_values("commence_time").drop_duplicates(subset=["game_key"])

    TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(TRACK_RECORD_PATH, index=False)
    return track


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    current_year = now.year if now.month >= 3 else now.year - 1

    track = build_track_record(current_year)

    if not track.empty:
        acc = track["correct"].mean()
        print(f"\n{'=' * 70}\nTRACK RECORD — {len(track)} juegos calificados hasta ahora\n{'=' * 70}")
        display = track[["home_team", "away_team", "pick", "actual_winner", "tier", "correct"]].copy()
        display["correct"] = display["correct"].map({1: "✅", 0: "❌"})
        print(display.to_string(index=False))
        print(f"\nAccuracy real acumulado: {acc:.1%}  ({track['correct'].sum()}/{len(track)})")
        print(f"\nGuardado: {TRACK_RECORD_PATH}")
    else:
        print("\nTrack record vacío por ahora — normal si todavía no se ha jugado ningún "
              "partido con predicción pre-kickoff registrada.")
