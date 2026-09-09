"""
Dashboard de predicciones NFL — Streamlit

Lee TODOS los CSV que ha generado weekly_pipeline.py en
outputs/weekly_predictions/ (uno por cada vez que lo corriste) y arma:

  1. Picks actuales: la corrida más reciente por juego, con logos,
     probabilidad, banda de confianza, y una alerta si el pick cambió
     desde la corrida anterior para ese mismo juego (el caso que
     preguntaste: hoy gana NE, mañana se lesiona el QB y cambia a SEA).
  2. Historial por juego: cómo evolucionó la probabilidad de un juego
     específico a lo largo de todas las veces que corriste el pipeline
     antes de su kickoff — para ver EXACTAMENTE en qué momento y cuánto
     se movió la línea.
  3. Tabla completa de todos los snapshots, por si quieres exportar o
     filtrar a mano.

CÓMO USAR:
  1. Coloca este archivo y team_logos.py en la carpeta `src/` de tu
     repo (junto a weekly_pipeline.py) — o en cualquier carpeta, solo
     ajusta la ruta en el sidebar la primera vez que lo corras.
  2. pip install streamlit plotly
  3. streamlit run dashboard.py

No modifica nada de tu pipeline existente — solo LEE los CSV que ya se
generan en outputs/weekly_predictions/.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from team_logos import logo_url, team_color, team_name
from history_utils import load_all_snapshots as _load_all_snapshots, get_last_pre_kickoff_snapshot
import re

st.set_page_config(page_title="NFL Predictor — Dashboard", layout="wide", page_icon="🏈")

FILENAME_RE = re.compile(r"predictions_(\d{8})_(\d{4})\.csv$")

TIER_STYLE = {
    "1": ("#1a7a3c", "🔒", "LOCK"),
    "2": ("#2f6fb0", "✅", "CONFIADO"),
    "3": ("#c98a1e", "⚖️", "PAREJO"),
    "4": ("#b0392f", "🪙", "MONEDA AL AIRE"),
}


def tier_style(tier_raw: str):
    if not isinstance(tier_raw, str):
        return "#888888", "❓", "DESCONOCIDO"
    key = tier_raw.split("_")[0].strip()
    return TIER_STYLE.get(key, ("#888888", "❓", tier_raw))


# ---------------------------------------------------------------------
# Carga de datos (la lógica de parseo vive en history_utils.py, compartida
# con check_results.py — aquí solo se envuelve con el cache de Streamlit)
# ---------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_all_snapshots(folder: str) -> pd.DataFrame:
    return _load_all_snapshots(folder)


@st.cache_data(ttl=60)
def load_track_record(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    return df


def make_demo_data() -> pd.DataFrame:
    """Datos de ejemplo para que el dashboard se vea funcionando aunque
    todavía no hayas corrido weekly_pipeline.py ninguna vez."""
    rows = [
        # (generated_at, home, away, commence, p_market, p_model, p_final, pick, conf, tier)
        ("2026-09-04 10:00", "NE", "SEA", "2026-09-07 17:00", 0.58, 0.55, 0.578, "NE", 0.578, "2_CONFIADO (demo)"),
        ("2026-09-06 09:00", "NE", "SEA", "2026-09-07 17:00", 0.60, 0.56, 0.598, "NE", 0.598, "2_CONFIADO (demo)"),
        ("2026-09-07 15:30", "NE", "SEA", "2026-09-07 17:00", 0.34, 0.50, 0.343, "SEA", 0.657, "2_CONFIADO (demo)"),
        ("2026-09-04 10:00", "KC", "LV", "2026-09-07 20:00", 0.82, 0.75, 0.817, "KC", 0.817, "1_LOCK (demo)"),
        ("2026-09-07 15:30", "KC", "LV", "2026-09-07 20:00", 0.83, 0.75, 0.827, "KC", 0.827, "1_LOCK (demo)"),
        ("2026-09-04 10:00", "DAL", "PHI", "2026-09-08 00:15", 0.51, 0.49, 0.509, "DAL", 0.509, "4_MONEDA_AL_AIRE (demo)"),
    ]
    df = pd.DataFrame(rows, columns=[
        "generated_at", "home_team", "away_team", "commence_time",
        "p_home_market", "p_model_a", "p_home_final", "pick", "confidence", "tier",
    ])
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["n_bookmakers"] = 6
    df["source_file"] = "demo"
    df["game_key"] = (
        df["home_team"] + " vs " + df["away_team"]
        + " (" + df["commence_time"].dt.strftime("%Y-%m-%d %H:%M") + ")"
    )
    return df


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("🏈 NFL Predictor")

# Fix: antes la ruta por defecto era relativa al directorio desde donde se
# ejecuta `streamlit run` (que puede no ser `src/`), lo que hacía que no
# encontrara los CSV si el comando se corría desde otra carpeta. Ahora se
# calcula relativa a la ubicación de ESTE archivo, igual que hace
# weekly_pipeline.py con OUT_DIR — así funciona sin importar desde dónde
# se invoque `streamlit run`.
_SCRIPT_DIR = Path(__file__).resolve().parent
folder = str(_SCRIPT_DIR.parent / "outputs" / "weekly_predictions")

if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()

data = load_all_snapshots(folder)

if data.empty:
    st.warning(
        f"No encontré archivos `predictions_*.csv` en `{folder}`. "
        "Asegúrate de correr `weekly_pipeline.py` para generar los datos."
    )
    st.stop()

n_snapshots = data["source_file"].nunique()
n_games = data["game_key"].nunique()
st.sidebar.markdown(f"**{n_snapshots}** corridas cargadas")
st.sidebar.markdown(f"**{n_games}** juegos distintos")
st.sidebar.caption(
    "Los logos se cargan en vivo desde el CDN público de ESPN — necesitas conexión "
    "a internet para verlos."
)

# ---------------------------------------------------------------------
# Preparar "latest" (última corrida por juego) + detección de cambios
# ---------------------------------------------------------------------

latest = data.sort_values("generated_at").groupby("game_key").tail(1).copy()
latest = latest.sort_values("commence_time")


def pick_changed_info(game_key: str):
    """Compara el pick más reciente contra el inmediatamente anterior
    para el mismo juego. Devuelve None si nunca cambió o solo hay 1 corrida."""
    hist = data[data["game_key"] == game_key].sort_values("generated_at")
    if len(hist) < 2:
        return None
    picks = hist["pick"].tolist()
    if len(set(picks)) == 1:
        return None
    prev, curr = hist.iloc[-2], hist.iloc[-1]
    if prev["pick"] != curr["pick"]:
        return {
            "prev_pick": prev["pick"], "prev_time": prev["generated_at"],
            "curr_pick": curr["pick"], "curr_time": curr["generated_at"],
            "prev_conf": prev["confidence"], "curr_conf": curr["confidence"],
        }
    return None


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Picks actuales", "📈 Historial por juego", "🗂️ Todos los snapshots", "🎯 Track record",
])

with tab1:
    st.subheader("Picks de la corrida más reciente, por juego")
    st.caption(
        "Ordenado por hora de kickoff. Si un pick cambió respecto a una corrida "
        "anterior de ESE juego, aparece resaltado arriba de la tarjeta."
    )

    for _, row in latest.iterrows():
        color, icon, label = tier_style(row["tier"])
        change = pick_changed_info(row["game_key"])

        if change is not None:
            st.markdown(
                f"<div style='background:#3a2a00;border-left:4px solid #d9a521;"
                f"padding:8px 12px;border-radius:4px;margin-bottom:4px;'>"
                f"⚠️ <b>El pick cambió</b> — antes <b>{change['prev_pick']}</b> "
                f"({change['prev_time'].strftime('%d/%m %H:%M')}, "
                f"{change['prev_conf']:.0%}) → ahora <b>{change['curr_pick']}</b> "
                f"({change['curr_time'].strftime('%d/%m %H:%M')}, {change['curr_conf']:.0%}). "
                f"Revisa si hubo una baja/lesión que lo explique."
                f"</div>",
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            c_home, c_vs, c_away, c_info = st.columns([2, 1, 2, 3])

            with c_home:
                st.image(logo_url(row["home_team"]), width=64)
                st.markdown(f"**{team_name(row['home_team'])}**")
                st.caption(f"Local · {row['p_home_market']:.0%} mercado")

            with c_vs:
                st.markdown("<div style='text-align:center;padding-top:20px;'>"
                             "<span style='font-size:20px;'>VS</span></div>", unsafe_allow_html=True)

            with c_away:
                st.image(logo_url(row["away_team"]), width=64)
                st.markdown(f"**{team_name(row['away_team'])}**")
                st.caption(f"Visitante · {1 - row['p_home_market']:.0%} mercado")

            with c_info:
                st.markdown(
                    f"<span style='background:{color};color:white;padding:3px 10px;"
                    f"border-radius:12px;font-size:13px;'>{icon} {label}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"### Pick: {team_name(row['pick'])}")
                st.progress(float(row["confidence"]), text=f"Confianza: {row['confidence']:.1%}")
                st.caption(
                    f"Kickoff: {row['commence_time'].strftime('%A %d/%m %H:%M')} · "
                    f"Última corrida: {row['generated_at'].strftime('%d/%m %H:%M')} · "
                    f"{int(row['n_bookmakers'])} casas de apuestas"
                )

with tab2:
    st.subheader("Cómo evolucionó la probabilidad de un juego a lo largo de tus corridas")

    game_options = latest.sort_values("commence_time")["game_key"].tolist()
    selected_game = st.selectbox("Elige un juego", game_options)

    hist = data[data["game_key"] == selected_game].sort_values("generated_at")

    if len(hist) < 2:
        st.info("Solo hay una corrida registrada para este juego todavía — corre el "
                "pipeline de nuevo más cerca del kickoff para empezar a ver el historial.")
    else:
        home_abbr = hist.iloc[0]["home_team"]
        away_abbr = hist.iloc[0]["away_team"]

        col_logos, col_chart = st.columns([1, 4])
        with col_logos:
            st.image(logo_url(home_abbr), width=50)
            st.caption(team_name(home_abbr))
            st.image(logo_url(away_abbr), width=50)
            st.caption(team_name(away_abbr))

        with col_chart:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["generated_at"], y=hist["p_home_final"],
                mode="lines+markers", name="Probabilidad final (ensemble)",
                line=dict(color=team_color(home_abbr), width=3),
                marker=dict(size=10),
            ))
            fig.add_trace(go.Scatter(
                x=hist["generated_at"], y=hist["p_home_market"],
                mode="lines+markers", name="Solo mercado",
                line=dict(color="gray", width=1, dash="dot"),
                marker=dict(size=6),
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4,
                          annotation_text="50/50")
            fig.update_layout(
                yaxis=dict(title=f"Prob. de que gane {home_abbr}", range=[0, 1], tickformat=".0%"),
                xaxis_title="Momento en que corriste el pipeline",
                height=420, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detectar el salto más grande entre corridas consecutivas — muy
        # probablemente el momento donde pasó algo real (lesión, clima, etc.)
        hist = hist.reset_index(drop=True)
        hist["delta"] = hist["p_home_final"].diff().abs()
        if hist["delta"].max(skipna=True) and hist["delta"].max() > 0.05:
            idx = hist["delta"].idxmax()
            st.warning(
                f"📍 El movimiento más grande fue entre "
                f"{hist.loc[idx-1, 'generated_at'].strftime('%d/%m %H:%M')} y "
                f"{hist.loc[idx, 'generated_at'].strftime('%d/%m %H:%M')}: "
                f"la probabilidad de {home_abbr} pasó de "
                f"{hist.loc[idx-1, 'p_home_final']:.0%} a {hist.loc[idx, 'p_home_final']:.0%} "
                f"({hist.loc[idx, 'delta']:+.0%} pp). Vale la pena revisar qué pasó en ese "
                f"lapso (inactive list, clima, movimiento de línea)."
            )

        st.markdown("##### Todas las corridas de este juego")
        display_cols = ["generated_at", "p_home_market", "p_model_a", "p_home_final", "pick", "confidence", "tier"]
        st.dataframe(hist[display_cols], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Todos los snapshots crudos")
    st.caption("Útil para exportar o filtrar a mano. Cada fila es una corrida de un juego.")
    st.dataframe(
        data.sort_values(["commence_time", "generated_at"]),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇️ Descargar todo como CSV",
        data.to_csv(index=False).encode("utf-8"),
        file_name="historial_completo_predicciones.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("¿El modelo está acertando de verdad?")
    st.caption(
        "Compara el ÚLTIMO pick hecho antes del kickoff de cada juego contra el resultado "
        "real. Se genera corriendo `check_results.py` (o `run_pipeline.py`, que ya lo "
        "incluye al final)."
    )

    track_path = str(_SCRIPT_DIR.parent / "outputs" / "track_record.csv")
    track = load_track_record(track_path)

    if track.empty:
        st.info(
            "Todavía no hay resultados calificados. Corre `python3 check_results.py` "
            "(o `python3 run_pipeline.py`) después de que se hayan jugado partidos con "
            "predicción registrada antes del kickoff."
        )
    else:
        acc = track["correct"].mean()
        n = len(track)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Accuracy real acumulado", f"{acc:.1%}", f"{track['correct'].sum()}/{n} juegos")
        col_b.metric("Juegos calificados", n)
        racha = track.sort_values("commence_time")["correct"].tail(5).tolist()
        col_c.metric("Últimos 5", " ".join("✅" if c else "❌" for c in racha))

        st.markdown("##### Accuracy real vs. lo esperado por banda de confianza")
        st.caption(
            "La línea punteada es el accuracy histórico 2006-2024 de cada banda "
            "(sección 16 del plan maestro). Con pocos juegos calificados, tu barra real "
            "puede estar lejos de la línea solo por tamaño de muestra chico — no saques "
            "conclusiones fuertes hasta tener varias semanas de datos."
        )
        expected = {"1": 0.790, "2": 0.634, "3": 0.565, "4": 0.534}
        track["tier_key"] = track["tier"].astype(str).str.split("_").str[0]
        by_tier = track.groupby("tier_key")["correct"].agg(["mean", "count"]).reindex(["1", "2", "3", "4"])

        fig_tier = go.Figure()
        fig_tier.add_trace(go.Bar(
            x=[TIER_STYLE[k][2] for k in by_tier.index], y=by_tier["mean"],
            name="Accuracy real", marker_color=[TIER_STYLE[k][0] for k in by_tier.index],
            text=[f"{v:.0%} (n={int(c)})" if not pd.isna(v) else "sin datos"
                  for v, c in zip(by_tier["mean"], by_tier["count"])],
            textposition="outside",
        ))
        fig_tier.add_trace(go.Scatter(
            x=[TIER_STYLE[k][2] for k in by_tier.index], y=[expected[k] for k in by_tier.index],
            mode="markers", name="Esperado (histórico 2006-2024)",
            marker=dict(symbol="line-ew", size=30, line=dict(width=3, color="white")),
        ))
        fig_tier.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]), height=380)
        st.plotly_chart(fig_tier, use_container_width=True)

        st.markdown("##### Detalle juego por juego")
        detail = track.sort_values("commence_time", ascending=False).copy()
        detail["Resultado"] = detail["correct"].map({1: "✅ Acertó", 0: "❌ Falló"})
        detail["Marcador"] = detail["home_team"] + " " + detail["home_score"].astype(int).astype(str) + " - " \
            + detail["away_score"].astype(int).astype(str) + " " + detail["away_team"]
        st.dataframe(
            detail[["commence_time", "home_team", "away_team", "pick", "actual_winner",
                    "Marcador", "tier", "Resultado"]],
            use_container_width=True, hide_index=True,
        )
