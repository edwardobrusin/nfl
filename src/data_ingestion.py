"""
Fase 1 - Data ingestion
Descarga schedules históricos directo del release de GitHub de nflverse-data
(mismo dataset que usan nfl_data_py / nflreadr por debajo) y los deja
listos en data/processed/games.parquet para el resto del pipeline.

NOTA: nfl_data_py.import_schedules() apunta a http://www.habitatring.com,
dominio no accesible en este entorno. Usamos en su lugar la URL del
release de GitHub, que es la fuente canónica de nflverse-data y sí
está permitida.

Solo temporada regular (sin playoffs), según decisión del plan maestro.
"""

import pandas as pd
from pathlib import Path
from team_reloc import canonicalize_teams

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

# Rango histórico: nflfastR tiene datos completos de spread/moneyline
# de forma confiable desde ~2006 en adelante (antes hay huecos).
YEARS = list(range(2006, 2025))


def download_schedules(years=YEARS) -> pd.DataFrame:
    """Descarga schedules crudos y filtra a temporada regular únicamente."""
    print(f"Descargando schedules desde {SCHEDULES_URL} ...")
    sched = pd.read_csv(SCHEDULES_URL)
    sched = sched[sched["season"].isin(years)]

    # Solo temporada regular (sin playoffs, según plan maestro)
    sched = sched[sched["game_type"] == "REG"].copy()

    keep_cols = [
        "game_id", "season", "week", "gameday",
        "home_team", "away_team",
        "home_score", "away_score",
        "home_moneyline", "away_moneyline",
        "spread_line",
        "home_rest", "away_rest",
        "location",  # fix auditoría: necesario para detectar sitios neutrales (Londres/CDMX/Múnich)
    ]
    sched = sched[[c for c in keep_cols if c in sched.columns]]

    # Fix Bug 1/2 (auditoría): canonicalizar abreviaciones de franquicias
    # reubicadas (STL->LA, SD->LAC, OAK->LV) ANTES de que nada más las use.
    sched = canonicalize_teams(sched)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sched.to_parquet(RAW_DIR / "schedules_raw.parquet", index=False)
    print(f"Guardado: {len(sched)} juegos en {RAW_DIR / 'schedules_raw.parquet'}")
    return sched


def build_games_table(sched: pd.DataFrame) -> pd.DataFrame:
    """Limpia y deja la tabla base de juegos con el target definido."""
    df = sched.copy()

    # Excluir juegos sin resultado (futuros) o sin moneyline (huecos históricos)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.dropna(subset=["home_moneyline", "away_moneyline"])

    # Target: 1 si gana el local, 0 si gana el visitante.
    # Empates (rarísimos) se excluyen del set de entrenamiento/backtest.
    df = df[df["home_score"] != df["away_score"]].copy()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "games.parquet", index=False)
    print(f"Guardado: {len(df)} juegos limpios en {PROCESSED_DIR / 'games.parquet'}")
    return df


if __name__ == "__main__":
    sched = download_schedules()
    games = build_games_table(sched)
    print(games.head())
    print(f"\nTemporadas cubiertas: {sorted(games['season'].unique())}")
