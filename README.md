# nfl-predictor

Predictor de ganador/perdedor (straight-up) para NFL. Ver `PLAN_MAESTRO_NFL_PREDICTOR.md`
en la raíz del proyecto para el plan completo, decisiones, y roadmap — ese documento
es la fuente de verdad, léelo antes de tocar código.

## Quickstart

```bash
pip install -r requirements.txt
python3 src/data_ingestion.py          # descarga + limpia schedules históricos 2006-2024
python3 src/models/market_baseline.py  # calcula baseline de mercado (Capa A)
```

## Resultado actual (Fase 1 completa)

Baseline de mercado (probabilidad implícita de-vigged del moneyline de cierre):
**66.57% accuracy** sobre 4,780 juegos de temporada regular, 2006-2024.
Este es el número de referencia — ver sección 2 y 11 del plan maestro.

## Nota importante sobre la fuente de datos

`nfl_data_py.import_schedules()` apunta por defecto a `habitatring.com`, que puede
no ser accesible dependiendo del entorno de red. `src/data_ingestion.py` usa en su
lugar la URL directa del release de GitHub de nflverse-data:

```
https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
```

Es la misma fuente canónica que usan nfl_data_py y nflreadr por debajo — mismos datos,
sin depender de un dominio intermedio.

## Cómo correr el pipeline en vivo (Fase 5)

1. Consigue una API key gratis en https://the-odds-api.com (free tier: 500 requests/mes).
2. `export ODDS_API_KEY="tu-api-key-aqui"`
3. Si es la primera vez, o quieres refrescar el modelo con resultados recientes:
   ```bash
   python3 src/data_ingestion.py
   python3 src/train_final_model.py
   ```
4. Justo antes de cada juego (la app de la competencia permite ajustar hasta ese momento —
   entre más tarde corras esto, mejor, porque la línea de mercado incorpora más información):
   ```bash
   python3 src/weekly_pipeline.py
   ```
5. El CSV con las picks queda en `outputs/weekly_predictions/predictions_{fecha}.csv`,
   con columnas `pick` (equipo favorecido) y `confidence` (probabilidad del ensemble).

**Nota:** `get_current_team_form()` cruza la frontera de temporada para calcular la forma
reciente de cada equipo (a diferencia del EPA rolling de entrenamiento, que resetea cada
temporada) — es la forma más honesta de tener una estimación no-NaN al inicio de una
temporada nueva, antes de que se acumulen 4 juegos. Ver `PLAN_MAESTRO` sección 12 para el
detalle completo de esta decisión de diseño.

## Estado del proyecto

Ver checklist en la sección 8/12 de `PLAN_MAESTRO_NFL_PREDICTOR.md`.
