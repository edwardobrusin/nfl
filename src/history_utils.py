"""
Utilidades compartidas para leer el historial de predicciones generado
por weekly_pipeline.py. Un solo lugar para esta lógica — la usan tanto
dashboard.py como check_results.py, para no mantenerla duplicada en dos
archivos (si mañana cambia el formato del nombre del CSV, se corrige
aquí una sola vez).
"""

import re
from pathlib import Path
from datetime import datetime

import pandas as pd

FILENAME_RE = re.compile(r"predictions_(\d{8})_(\d{4})\.csv$")


def load_all_snapshots(folder) -> pd.DataFrame:
    """Lee TODOS los predictions_*.csv de la carpeta y los concatena,
    agregando `generated_at` (parseado del nombre del archivo) y
    `game_key` (identificador único por partido)."""
    path = Path(folder)
    if not path.exists():
        return pd.DataFrame()

    frames = []
    for f in sorted(path.glob("predictions_*.csv")):
        m = FILENAME_RE.search(f.name)
        if not m:
            continue
        generated_at = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
        df = pd.read_csv(f)
        df["generated_at"] = generated_at
        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df["commence_time"] = pd.to_datetime(all_df["commence_time"], utc=True).dt.tz_localize(None)
    all_df["game_key"] = (
        all_df["home_team"] + " vs " + all_df["away_team"]
        + " (" + all_df["commence_time"].dt.strftime("%Y-%m-%d %H:%M") + ")"
    )
    return all_df.sort_values("generated_at")


def get_last_pre_kickoff_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada juego, la ÚLTIMA corrida que se hizo ANTES de su kickoff —
    es el pick que "cuenta" de verdad, porque es lo último que se pudo
    haber registrado en la app antes de que arrancara ese partido en
    particular. Corridas hechas DESPUÉS del kickoff (por error, o solo
    para revisar) se ignoran para efectos de calificar aciertos.
    """
    if history.empty:
        return history
    pre_kickoff = history[history["generated_at"] < history["commence_time"]]
    if pre_kickoff.empty:
        return pre_kickoff
    return pre_kickoff.sort_values("generated_at").groupby("game_key").tail(1).copy()
