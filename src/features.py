"""
Fase 2 - Feature engineering: Elo propio + EPA rolling.

Construye dos familias de features, ambas SIN fuga de datos (usan solo
información disponible antes del kickoff de cada juego):

1. Elo propio: rating estilo 538/nfelo, actualizado juego a juego,
   con regresión a la media entre temporadas.
2. EPA rolling de 4 semanas: promedio de EPA/play ofensivo y defensivo
   de las últimas 4 semanas jugadas por cada equipo, shiftado para que
   la semana actual nunca se vea a sí misma.

Salida: data/processed/games_with_features.parquet, con una fila por
juego y las features de ambos equipos ya unidas.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------
# Elo propio
# ---------------------------------------------------------------------

ELO_START = 1500.0
K_FACTOR = 20.0
HOME_FIELD_ADV = 55.0  # puntos de Elo, valor de partida estándar en la literatura
REGRESSION_TO_MEAN = 1 / 3  # cuánto se regresa cada equipo hacia 1500 entre temporadas


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """
    Recorre los juegos en orden cronológico y calcula el Elo pre-juego
    de cada equipo (el rating ANTES de que se juegue ese partido, que es
    lo único válido como feature predictivo).

    Soporta filas sin resultado todavía (home_win = NaN, juegos futuros):
    se les calcula su elo_pre normalmente, pero no actualizan el rating
    de nadie (no hay nada que actualizar sin resultado real). Esto permite
    usar la misma función tanto para entrenar como para generar features
    de juegos que todavía no se juegan (pipeline en vivo, Fase 5).
    """
    games = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    ratings = {}
    current_season = None

    home_elo_pre = np.zeros(len(games))
    away_elo_pre = np.zeros(len(games))

    for i, row in games.iterrows():
        season = row["season"]
        home, away = row["home_team"], row["away_team"]

        # Regresión a la media al cambiar de temporada
        if season != current_season:
            for team in ratings:
                ratings[team] = ELO_START + (ratings[team] - ELO_START) * (1 - REGRESSION_TO_MEAN)
            current_season = season

        elo_home = ratings.get(home, ELO_START)
        elo_away = ratings.get(away, ELO_START)

        home_elo_pre[i] = elo_home
        away_elo_pre[i] = elo_away

        # Fix auditoría (hallazgo menor): sitios neutrales (Londres/CDMX/
        # Múnich) no deben recibir la ventaja de local en el cálculo de
        # probabilidad ni en la actualización del rating.
        is_neutral = str(row.get("location", "Home")).strip().lower() == "neutral"
        hfa = 0.0 if is_neutral else HOME_FIELD_ADV

        # Si no hay resultado todavía (juego futuro), no se actualiza nada
        if pd.isna(row.get("home_win")):
            continue

        # Actualización post-resultado
        exp_home = expected_score(elo_home + hfa, elo_away)
        actual_home = row["home_win"]

        change = K_FACTOR * (actual_home - exp_home)
        ratings[home] = elo_home + change
        ratings[away] = elo_away - change

    games["home_elo_pre"] = home_elo_pre
    games["away_elo_pre"] = away_elo_pre

    if "location" in games.columns:
        neutral_mask = games["location"].astype(str).str.strip().str.lower() == "neutral"
    else:
        neutral_mask = pd.Series(False, index=games.index)
    hfa_col = np.where(neutral_mask, 0.0, HOME_FIELD_ADV)

    games["elo_diff"] = games["home_elo_pre"] + hfa_col - games["away_elo_pre"]
    games["p_home_elo"] = 1 / (1 + 10 ** (-games["elo_diff"] / 400))

    return games, ratings


# ---------------------------------------------------------------------
# EPA rolling (4 semanas)
# ---------------------------------------------------------------------

def compute_epa_rolling(team_week: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Para cada equipo-semana, calcula el promedio de EPA/play (ofensivo y
    defensivo) de las `window` semanas PREVIAS. El shift(1) es lo que
    evita que la semana actual se vea a sí misma.

    Fix Bug 5 (auditoría 2026-08-18): el rolling CRUZA la frontera de
    temporada (no se reinicia en semana 1 de cada año), igual que hace
    get_current_team_form() en producción. Antes había una discrepancia:
    entrenamiento reseteaba por temporada (semana 2 veía en promedio 1
    juego), producción no reseteaba (vería hasta 4 juegos, la mayoría de
    la temporada anterior) — el modelo entrenaba con una distribución de
    features que nunca iba a ver en vivo. Este fix también recupera los
    juegos de semana 1 (excepto la semana 1 de la primera temporada de
    cada equipo en el dataset), que antes quedaban en NaN y se perdían
    del set de entrenamiento/evaluación.
    """
    tw = team_week.sort_values(["team", "season", "week"]).copy()

    for col in ["off_epa_play", "def_epa_play", "off_success_rate", "def_success_rate"]:
        tw[f"{col}_roll{window}"] = (
            tw.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    roll_cols = ["season", "week", "team"] + [
        f"{c}_roll{window}" for c in ["off_epa_play", "def_epa_play", "off_success_rate", "def_success_rate"]
    ]
    return tw[roll_cols]


def attach_epa_features(games: pd.DataFrame, epa_roll: pd.DataFrame) -> pd.DataFrame:
    games = games.merge(
        epa_roll.add_prefix("home_"), left_on=["season", "week", "home_team"],
        right_on=["home_season", "home_week", "home_team"], how="left",
    ).drop(columns=["home_season", "home_week"])

    games = games.merge(
        epa_roll.add_prefix("away_"), left_on=["season", "week", "away_team"],
        right_on=["away_season", "away_week", "away_team"], how="left",
    ).drop(columns=["away_season", "away_week"])

    return games


if __name__ == "__main__":
    games = pd.read_parquet(PROCESSED_DIR / "games_with_market_baseline.parquet")
    team_week = pd.read_parquet(PROCESSED_DIR / "team_week_epa.parquet")

    print("Calculando Elo propio...")
    games, _final_ratings = compute_elo(games)

    print("Calculando EPA rolling (4 semanas)...")
    epa_roll = compute_epa_rolling(team_week, window=4)
    games = attach_epa_features(games, epa_roll)

    out_path = PROCESSED_DIR / "games_with_features.parquet"
    games.to_parquet(out_path, index=False)
    print(f"Guardado: {len(games)} juegos con features en {out_path}")

    # Sanity check rápido: accuracy de Elo puro (sin ML todavía)
    elo_pred = (games["p_home_elo"] > 0.5).astype(int)
    elo_acc = (elo_pred == games["home_win"]).mean()
    print(f"\nSanity check — accuracy de Elo propio SOLO (sin EPA, sin ML): {elo_acc:.4f}")
    print("(referencia: baseline de mercado fue 0.6657)")
