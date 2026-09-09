"""
Fase 2 - Ingesta de play-by-play y agregación de EPA por equipo-semana.

Descarga cada temporada por separado desde el release "pbp" de nflverse-data
(mismo patrón de URL confirmado que funciona en este entorno) y colapsa
jugada-por-jugada a nivel equipo-semana: EPA ofensivo/defensivo por play,
success rate, etc. Esta tabla es el insumo para las rolling features de
features.py (Fase 2 - Elo y EPA rolling).

NOTA: esto es pesado (~20-25MB por temporada). Se cachea en
data/raw/pbp_{season}.parquet para no re-descargar en cada corrida.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

PBP_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

YEARS = list(range(2006, 2025))

# Columnas mínimas necesarias — no cargamos las 372 columnas completas
# para ahorrar memoria y tiempo de descarga.
PBP_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam",
    "play_type", "epa", "success", "pass", "rush",
    "down", "penalty",
]


def download_pbp_season(season: int) -> pd.DataFrame:
    cache_path = RAW_DIR / f"pbp_{season}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = PBP_URL_TEMPLATE.format(season=season)
    print(f"Descargando play-by-play {season}...")
    df = pd.read_parquet(url, columns=PBP_COLUMNS)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def download_all_pbp(years=YEARS) -> pd.DataFrame:
    frames = [download_pbp_season(y) for y in years]
    return pd.concat(frames, ignore_index=True)


def aggregate_team_week_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa play-by-play a nivel equipo-semana: EPA promedio por jugada
    (ofensivo, cuando el equipo tiene el balón) y EPA defensivo (promedio
    de EPA permitido cuando el equipo está defendiendo).

    Solo se consideran jugadas de scrimmage reales (pass/rush), excluyendo
    penalties sin jugada, kneels, spikes, etc., para no ensuciar el EPA/play.
    """
    df = pbp.copy()
    df = df[df["play_type"].isin(["pass", "run"])]
    df = df.dropna(subset=["epa", "posteam", "defteam"])

    # Ofensiva: EPA/play y success rate del equipo con el balón
    off = (
        df.groupby(["season", "week", "posteam"])
        .agg(off_epa_play=("epa", "mean"), off_success_rate=("success", "mean"), off_plays=("epa", "count"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Defensiva: EPA/play permitido (mismo epa, agrupado por defteam)
    deff = (
        df.groupby(["season", "week", "defteam"])
        .agg(def_epa_play=("epa", "mean"), def_success_rate=("success", "mean"), def_plays=("epa", "count"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    team_week = off.merge(deff, on=["season", "week", "team"], how="outer")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    team_week.to_parquet(PROCESSED_DIR / "team_week_epa.parquet", index=False)
    print(f"Guardado: {len(team_week)} filas equipo-semana en team_week_epa.parquet")
    return team_week


if __name__ == "__main__":
    pbp = download_all_pbp()
    print(f"Play-by-play total: {len(pbp)} jugadas, temporadas {pbp['season'].min()}-{pbp['season'].max()}")
    team_week = aggregate_team_week_epa(pbp)
    print(team_week.head())
