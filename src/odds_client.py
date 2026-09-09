"""
Cliente de The Odds API (the-odds-api.com) — obtiene moneylines actuales
de la NFL, tan cerca del kickoff como se corra este script (confirmado:
la app de la competencia permite modificar predicciones justo antes de
que inicie cada partido, así que este script se puede/debe correr lo más
tarde posible antes de cada juego individual, no una sola vez a inicio
de semana).

Requiere una API key gratuita de https://the-odds-api.com (free tier:
500 requests/mes — de sobra para uso semanal manual, hay que tener
cuidado si se automatiza con mucha frecuencia).

Uso:
    export ODDS_API_KEY="tu-api-key-aqui"
    python3 src/odds_client.py
"""

import os
import requests
import pandas as pd
from team_name_mapping import ODDS_API_TO_NFLVERSE

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


def fetch_current_odds(api_key: str = None) -> pd.DataFrame:
    """
    Devuelve un DataFrame con una fila por juego próximo, con la
    probabilidad implícita de-vigged promediada entre todas las casas
    disponibles (consenso de mercado — más robusto que una sola casa).
    """
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta ODDS_API_KEY. Consigue una gratis en https://the-odds-api.com "
            "y expórtala como variable de entorno: export ODDS_API_KEY='...'"
        )

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
    resp.raise_for_status()
    games = resp.json()

    # Aviso de cuota restante (viene en headers, útil para no quedarse sin requests)
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"[The Odds API] Requests restantes este mes: {remaining}")

    rows = []
    for game in games:
        home_full = game["home_team"]
        away_full = game["away_team"]
        home_abbr = ODDS_API_TO_NFLVERSE.get(home_full)
        away_abbr = ODDS_API_TO_NFLVERSE.get(away_full)

        if home_abbr is None or away_abbr is None:
            print(f"AVISO: equipo no reconocido en el mapeo — {home_full} / {away_full}. Se omite este juego.")
            continue

        home_probs, away_probs = [], []
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                if home_full not in outcomes or away_full not in outcomes:
                    continue

                ml_home, ml_away = outcomes[home_full], outcomes[away_full]
                p_home_raw = (-ml_home / (-ml_home + 100)) if ml_home < 0 else (100 / (ml_home + 100))
                p_away_raw = (-ml_away / (-ml_away + 100)) if ml_away < 0 else (100 / (ml_away + 100))
                overround = p_home_raw + p_away_raw
                home_probs.append(p_home_raw / overround)
                away_probs.append(p_away_raw / overround)

        if not home_probs:
            continue

        rows.append({
            "commence_time": game["commence_time"],
            "home_team": home_abbr,
            "away_team": away_abbr,
            "p_home_market": sum(home_probs) / len(home_probs),
            "n_bookmakers": len(home_probs),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    odds = fetch_current_odds()
    print(f"\n{len(odds)} juegos próximos encontrados:")
    print(odds.to_string(index=False))
