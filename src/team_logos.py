"""
Mapeo de logos y colores de equipo para el dashboard.

Nota de diseño: en vez de empaquetar los PNGs de los escudos (son marca
registrada de la NFL/cada franquicia — redistribuir 32 archivos de logo
cruza la línea de reproducir material con derechos de autor), este
módulo apunta a las URLs públicas que ESPN ya aloja en su propio CDN.
Es la misma técnica que usa el propio paquete `nflverse` (la tabla
`teams_colors_logos.csv` de nflverse-data trae exactamente estas mismas
URLs) — no es un truco, es el estándar de la industria de analítica NFL
open-source: apuntar al asset ya público, no redistribuirlo.

Streamlit puede recibir una URL directamente en st.image(), así que no
hace falta descargar nada — cada vez que corres el dashboard con
internet disponible, carga el logo actual directo de ESPN.
"""

# abbr -> (logo_url_espn, color_primario_hex, nombre_completo)
TEAM_INFO = {
    "ARI": ("https://a.espncdn.com/i/teamlogos/nfl/500/ari.png", "#97233F", "Arizona Cardinals"),
    "ATL": ("https://a.espncdn.com/i/teamlogos/nfl/500/atl.png", "#A71930", "Atlanta Falcons"),
    "BAL": ("https://a.espncdn.com/i/teamlogos/nfl/500/bal.png", "#241773", "Baltimore Ravens"),
    "BUF": ("https://a.espncdn.com/i/teamlogos/nfl/500/buf.png", "#00338D", "Buffalo Bills"),
    "CAR": ("https://a.espncdn.com/i/teamlogos/nfl/500-dark/car.png", "#0085CA", "Carolina Panthers"),
    "CHI": ("https://a.espncdn.com/i/teamlogos/nfl/500/chi.png", "#0B162A", "Chicago Bears"),
    "CIN": ("https://a.espncdn.com/i/teamlogos/nfl/500/cin.png", "#FB4F14", "Cincinnati Bengals"),
    "CLE": ("https://a.espncdn.com/i/teamlogos/nfl/500/cle.png", "#FF3C00", "Cleveland Browns"),
    "DAL": ("https://a.espncdn.com/i/teamlogos/nfl/500/dal.png", "#002244", "Dallas Cowboys"),
    "DEN": ("https://a.espncdn.com/i/teamlogos/nfl/500/den.png", "#002244", "Denver Broncos"),
    "DET": ("https://a.espncdn.com/i/teamlogos/nfl/500/det.png", "#0076B6", "Detroit Lions"),
    "GB":  ("https://a.espncdn.com/i/teamlogos/nfl/500/gb.png", "#203731", "Green Bay Packers"),
    "HOU": ("https://a.espncdn.com/i/teamlogos/nfl/500/hou.png", "#03202F", "Houston Texans"),
    "IND": ("https://a.espncdn.com/i/teamlogos/nfl/500/ind.png", "#002C5F", "Indianapolis Colts"),
    "JAX": ("https://a.espncdn.com/i/teamlogos/nfl/500/jax.png", "#006778", "Jacksonville Jaguars"),
    "KC":  ("https://a.espncdn.com/i/teamlogos/nfl/500/kc.png", "#E31837", "Kansas City Chiefs"),
    "LA":  ("https://a.espncdn.com/i/teamlogos/nfl/500/lar.png", "#003594", "Los Angeles Rams"),
    "LAC": ("https://a.espncdn.com/i/teamlogos/nfl/500/lac.png", "#007BC7", "Los Angeles Chargers"),
    "LV":  ("https://a.espncdn.com/i/teamlogos/nfl/500/lv.png", "#000000", "Las Vegas Raiders"),
    "MIA": ("https://a.espncdn.com/i/teamlogos/nfl/500/mia.png", "#008E97", "Miami Dolphins"),
    "MIN": ("https://a.espncdn.com/i/teamlogos/nfl/500/min.png", "#4F2683", "Minnesota Vikings"),
    "NE":  ("https://a.espncdn.com/i/teamlogos/nfl/500/ne.png", "#002244", "New England Patriots"),
    "NO":  ("https://a.espncdn.com/i/teamlogos/nfl/500/no.png", "#D3BC8D", "New Orleans Saints"),
    "NYG": ("https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png", "#0B2265", "New York Giants"),
    "NYJ": ("https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png", "#003F2D", "New York Jets"),
    "PHI": ("https://a.espncdn.com/i/teamlogos/nfl/500/phi.png", "#004C54", "Philadelphia Eagles"),
    "PIT": ("https://a.espncdn.com/i/teamlogos/nfl/500/pit.png", "#000000", "Pittsburgh Steelers"),
    "SEA": ("https://a.espncdn.com/i/teamlogos/nfl/500/sea.png", "#002244", "Seattle Seahawks"),
    "SF":  ("https://a.espncdn.com/i/teamlogos/nfl/500/sf.png", "#AA0000", "San Francisco 49ers"),
    "TB":  ("https://a.espncdn.com/i/teamlogos/nfl/500/tb.png", "#A71930", "Tampa Bay Buccaneers"),
    "TEN": ("https://a.espncdn.com/i/teamlogos/nfl/500/ten.png", "#4495D2", "Tennessee Titans"),
    "WAS": ("https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png", "#5A1414", "Washington Commanders"),
}

DEFAULT_LOGO = "https://a.espncdn.com/i/teamlogos/leagues/500/nfl.png"


def logo_url(abbr: str) -> str:
    return TEAM_INFO.get(abbr, (DEFAULT_LOGO, "#888888", abbr))[0]


def team_color(abbr: str) -> str:
    return TEAM_INFO.get(abbr, (DEFAULT_LOGO, "#888888", abbr))[1]


def team_name(abbr: str) -> str:
    return TEAM_INFO.get(abbr, (DEFAULT_LOGO, "#888888", abbr))[2]
