"""
Fix Bug 1/2 (auditoría 2026-08-18): el schedule de nflverse usa
abreviaciones HISTÓRICAS (STL, SD, OAK) mientras que el play-by-play de
nflfastR usa las abreviaciones NORMALIZADAS MODERNAS (LA, LAC, LV) para
TODAS las temporadas, incluyendo las anteriores a la reubicación.

Esto hacía que el merge de EPA fallara en silencio para todos los juegos
de Rams pre-2016, Chargers pre-2017 y Raiders pre-2020, y que el Elo de
esas franquicias se reiniciara a 1500 en el año de la mudanza (el
diccionario de ratings las trataba como equipos nuevos).

Fix: canonicalizar las abreviaciones a su forma moderna INMEDIATAMENTE
al ingerir el schedule, antes de que toquen compute_elo() o
attach_epa_features(). Así todo el resto del pipeline ve un solo
identificador consistente por franquicia a través de toda su historia.
"""

RELOC = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def canonicalize_teams(df, columns=("home_team", "away_team")):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(RELOC)
    return df
