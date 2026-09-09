"""
QB Rating — la mejora de mayor prioridad (sección 13 del plan, ahora
promovida porque el tiempo lo permite).

Idea central (la misma que usa nfelo): el Elo de equipo mezcla la calidad
del QB con la del resto del roster, y reacciona LENTO a un cambio de QB
(titular lesionado → suplente) porque el rating de equipo se mueve
gradualmente juego a juego. Un rating que viaje CON el jugador, no con
el equipo, captura ese cambio de forma inmediata — que es precisamente
el escenario donde el mercado a veces tarda en reaccionar por completo
(sobre todo si la lesión se confirma a media semana).

Metodología:
1. Por jugada de pase, `qb_epa` (columna ya calculada por nflverse) mide
   la contribución del QB, incluyendo scrambles y sacks.
2. Se agrega a nivel QB-juego (promedio de qb_epa en sus dropbacks).
3. Rolling de sus últimos 8 juegos como QB, CRUZANDO equipo y temporada
   (si cambia de equipo, su rating viaja con él — otra diferencia clave
   frente al Elo de equipo).
4. Se une al home_qb_id / away_qb_id de cada partido (mismo esquema de
   ID gsis_id en schedule y pbp — verificado, se unen directo).
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

QB_ROLLING_WINDOW = 8
QB_RATING_PRIOR = 0.0  # EPA/dropback neutral para QBs sin historial (rookies, etc.)


def load_all_pbp(years) -> pd.DataFrame:
    """
    Carga passer_id/qb_epa por temporada. El cache de pbp_ingestion.py
    (Fase 2) no guardó estas columnas, así que se descarga un cache
    aparte específico para el rating de QB.
    """
    cols = ["game_id", "season", "week", "passer_id", "passer_player_name", "qb_epa"]
    frames = []
    for y in years:
        cache_path = RAW_DIR / f"pbp_qb_{y}.parquet"
        if cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
            continue
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
        print(f"Descargando columnas de QB para {y}...")
        df = pd.read_parquet(url, columns=cols)
        df.to_parquet(cache_path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_qb_game_ratings(pbp: pd.DataFrame) -> pd.DataFrame:
    """Agrega qb_epa a nivel QB-juego (solo dropbacks reales)."""
    df = pbp.dropna(subset=["passer_id", "qb_epa"])
    qb_game = (
        df.groupby(["season", "week", "game_id", "passer_id"])
        .agg(qb_epa_game=("qb_epa", "mean"), dropbacks=("qb_epa", "count"), name=("passer_player_name", "first"))
        .reset_index()
    )
    # Descarta apariciones muy cortas (mop-up duty, <5 dropbacks) que meten ruido al rating
    qb_game = qb_game[qb_game["dropbacks"] >= 5]
    return qb_game


def compute_qb_rolling_rating(qb_game: pd.DataFrame, window: int = QB_ROLLING_WINDOW) -> pd.DataFrame:
    """
    Rolling de los últimos `window` juegos de CADA QB, ordenado
    cronológicamente, cruzando equipo y temporada (sin reset). shift(1)
    para que el juego actual nunca se vea a sí mismo.
    """
    qb = qb_game.sort_values(["passer_id", "season", "week"]).copy()
    qb["qb_rating_pre"] = (
        qb.groupby("passer_id")["qb_epa_game"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    qb["qb_starts_prior"] = qb.groupby("passer_id").cumcount()  # juegos previos con >=5 dropbacks
    return qb[["season", "week", "game_id", "passer_id", "name", "qb_rating_pre", "qb_starts_prior"]]


def attach_qb_features_to_games(games: pd.DataFrame, qb_rolling: pd.DataFrame, schedule_qb_ids: pd.DataFrame) -> pd.DataFrame:
    """
    Une el rating pre-juego del QB titular de casa y visita a la tabla
    de juegos, usando home_qb_id / away_qb_id del schedule (el ID del
    QB que efectivamente arrancó ese partido).

    Fix Bug 3 (auditoría 2026-08-18): la versión anterior unía por
    (game_id, qb_id) exacto contra qb_rolling, que solo tiene una fila
    para ese (game_id, qb_id) si ESE QB acumuló >=5 dropbacks en ESE
    MISMO juego. Si salió lesionado temprano (o fue baja de última
    hora), el merge fallaba — y el modelo recibía esa falla como
    feature (`qb_unknown=1`), que es literalmente el resultado del
    juego filtrándose hacia atrás como input. Los equipos con ese flag
    perdían 7-8pp más de lo que el mercado ya implicaba: no era señal
    predictiva, era el modelo enterándose de la lesión.

    Fix: usar merge_asof por QB, tomando su rating de la aparición
    calificada (>=5 dropbacks) más reciente EN O ANTES de esta semana,
    sin exigir que el juego que se predice sea esa aparición. Así el
    rating nunca depende de lo que le pase al QB en el juego que se
    está evaluando.
    """
    df = games.merge(
        schedule_qb_ids[["game_id", "home_qb_id", "away_qb_id"]], on="game_id", how="left"
    )
    df["season_week_key"] = (df["season"] * 100 + df["week"]).astype("int64")

    qb_rolling = qb_rolling.copy()
    qb_rolling["season_week_key"] = (qb_rolling["season"] * 100 + qb_rolling["week"]).astype("int64")
    qb_rolling = qb_rolling.sort_values("season_week_key")

    def asof_join(df, qb_id_col, prefix):
        sub = df[["game_id", qb_id_col, "season_week_key"]].dropna(subset=[qb_id_col]).copy()
        sub = sub.rename(columns={qb_id_col: "passer_id"}).sort_values("season_week_key")
        joined = pd.merge_asof(
            sub, qb_rolling[["passer_id", "season_week_key", "qb_rating_pre", "qb_starts_prior"]],
            on="season_week_key", by="passer_id", direction="backward", allow_exact_matches=True,
        )
        joined = joined.rename(columns={
            "qb_rating_pre": f"{prefix}_qb_rating", "qb_starts_prior": f"{prefix}_qb_starts_prior",
        })
        return joined[["game_id", f"{prefix}_qb_rating", f"{prefix}_qb_starts_prior"]]

    df = df.merge(asof_join(df, "home_qb_id", "home"), on="game_id", how="left")
    df = df.merge(asof_join(df, "away_qb_id", "away"), on="game_id", how="left")
    df = df.drop(columns=["season_week_key"])

    # "unknown" ahora significa lo correcto: el QB no tiene NINGÚN start
    # calificado previo (rookie debutando, o el ID de QB simplemente no
    # viene en el schedule) — no "el merge falló porque salió lesionado
    # en este mismo juego".
    df["home_qb_unknown"] = df["home_qb_rating"].isna().astype(int)
    df["away_qb_unknown"] = df["away_qb_rating"].isna().astype(int)
    df["home_qb_rating"] = df["home_qb_rating"].fillna(QB_RATING_PRIOR)
    df["away_qb_rating"] = df["away_qb_rating"].fillna(QB_RATING_PRIOR)
    df["home_qb_starts_prior"] = df["home_qb_starts_prior"].fillna(0)
    df["away_qb_starts_prior"] = df["away_qb_starts_prior"].fillna(0)
    df["qb_rating_diff"] = df["home_qb_rating"] - df["away_qb_rating"]

    return df


if __name__ == "__main__":
    from data_ingestion import YEARS

    print("Cargando play-by-play cacheado...")
    pbp = load_all_pbp(YEARS)

    print("Agregando EPA por QB-juego...")
    qb_game = build_qb_game_ratings(pbp)
    print(f"{len(qb_game)} apariciones QB-juego con >=5 dropbacks")

    print("Calculando rolling de QB (rating pre-juego, cruza equipo/temporada)...")
    qb_rolling = compute_qb_rolling_rating(qb_game)

    print("Uniendo a la tabla de juegos...")
    schedule_qb_ids = pd.read_csv(RAW_DIR / "games.csv")[["game_id", "home_qb_id", "away_qb_id"]]
    games = pd.read_parquet(PROCESSED_DIR / "games_with_features.parquet")
    games = attach_qb_features_to_games(games, qb_rolling, schedule_qb_ids)

    out_path = PROCESSED_DIR / "games_with_qb_features.parquet"
    games.to_parquet(out_path, index=False)
    print(f"Guardado: {out_path}")
    print(f"Juegos con QB desconocido (home): {games['home_qb_unknown'].sum()}  "
          f"(away): {games['away_qb_unknown'].sum()}")
    print(games[["season", "week", "home_team", "home_qb_rating", "away_team", "away_qb_rating", "qb_rating_diff"]].tail(10))
