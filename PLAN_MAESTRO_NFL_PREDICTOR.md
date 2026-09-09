# Plan Maestro — Modelo Predictivo NFL (Straight-Up Winner)

**Última actualización:** 2026-08-17
**Autor:** Edward + Claude (sesión inicial). Continuable con Gemini u otro LLM.
**Objetivo de la competencia:** Máximo % de aciertos en predicción de ganador/perdedor (straight-up), evaluado semana a semana. No hay jurado evaluando metodología — solo importa el número final. Sin restricciones de datos ni herramientas.

---

## 0. TL;DR — La tesis del proyecto

El mercado de apuestas (líneas de casas) es el predictor más fuerte que existe para NFL straight-up (~67-70% accuracy histórico). Nuestra estrategia NO es ignorarlo ni "competir" contra él con features caseras — es:

1. Usar la probabilidad implícita del mercado como **baseline y feature principal**.
2. Construir un modelo propio (Elo + EPA rolling + ML) que capture señal que el mercado no haya incorporado todavía (lesiones de último minuto, clima, momentum reciente).
3. Combinar ambos en un **ensemble ponderado**, con el peso óptimo determinado por backtest histórico, no por intuición.
4. Automatizar la ejecución semanal: jalar líneas actuales + injury report + features → generar predicción antes de cada juego.

Todo esto es reproducible, versionado en Git, y diseñado para que cualquier LLM (Claude o Gemini) pueda retomarlo leyendo este documento sin contexto previo.

---

## 1. Definición del problema

- **Target:** `home_win` (binario, 1 si local gana, 0 si visitante gana). Empates en temporada regular son rarísimos (post-2012 con OT de 10 min) — se excluyen del set de entrenamiento o se tratan como caso especial documentado aparte.
- **Granularidad de predicción:** por juego, generada antes del kickoff, usando solo información disponible en ese momento (sin fuga de datos del futuro).
- **Cadencia:** semanal durante la temporada regular (y playoffs si la competencia lo incluye — **pendiente confirmar con las reglas**).
- **Métrica de éxito primaria:** % de aciertos (accuracy) straight-up.
- **Métricas secundarias que monitoreamos igual (para diagnosticar, no para reportar):** Brier score, log-loss, calibración. Estas no importan para la competencia pero nos dicen si el modelo está bien construido o solo tuvo suerte.

---

## 2. Benchmarks de referencia (lo que hay que vencer o igualar)

| Estrategia | Accuracy histórica aprox. | Fuente |
|---|---|---|
| Favorito de Vegas siempre gana (naive) | ~65-67% | Consenso de la industria |
| Probabilidad implícita del moneyline de cierre | ~67-70% (referencia externa) | nflpredictr (regresión logística, 20 años de datos) |
| **Nuestro baseline real, calculado (Fase 1 completa)** | **66.57%** (2006-2024, 4,780 juegos, rango anual 58%-72%) | `src/models/market_baseline.py`, ver sección 12 |
| Elo simple (estilo 538/nfelo) sin ajustes | ~63-66% | nfelo, FiveThirtyEight NFL Elo |
| nfelo (Elo + ajustes de QB, clima, descanso, mercado) | Comparable o ligeramente superior al mercado en tramos específicos | nfeloapp.com |

**Regla de oro:** si nuestro modelo final no supera el baseline de "probabilidad implícita del moneyline de cierre" en el backtest, no tiene caso usarlo — mejor someter esa probabilidad directamente.

---

## 3. Fuentes de datos

### 3.1 Datos históricos (para entrenar y backtestear)

| Fuente | Qué provee | Cómo se accede |
|---|---|---|
| **nflreadpy** (o `nfl_data_py` como alternativa) | Play-by-play 1999-presente, schedules con `spread_line`, `home_moneyline`, `away_moneyline`, `home_rest`, `away_rest`, resultados reales | `pip install nflreadpy` → `nfl.load_schedules()`, `nfl.load_pbp()` |
| **nflverse team/player stats** | Stats agregadas por equipo/semana (EPA, éxito rate, etc.) ya calculadas | `nfl.load_team_stats()`, `nfl.load_player_stats()` |
| **FiveThirtyEight NFL Elo históricos** | Elo de referencia para validar nuestro propio cálculo de Elo | `github.com/fivethirtyeight/data/tree/master/nfl-elo` |
| **nfelo (greerreNFL)** | Predicciones semanales publicadas, código abierto, se puede usar directo como input de ensemble | `github.com/greerreNFL/nfelo`, salida visible en nfeloapp.com |
| **Kaggle — nfl-scores-and-betting-data** | Dataset histórico de resultados + líneas, útil como cruce de validación | `kaggle.com/tobycrabtree/nfl-scores-and-betting-data` |
| **Datos de clima históricos** | Viento, temperatura, precipitación por estadio/fecha | Visual Crossing API / NOAA (revisar límites de free tier) |

### 3.2 Datos en vivo (para predicciones semanales durante la temporada)

| Fuente | Qué provee | Notas |
|---|---|---|
| **The Odds API** (the-odds-api.com) | Moneylines/spreads actualizados cerca del kickoff, múltiples casas | Free tier limitado — revisar cuota semanal antes de automatizar |
| **ESPN API (no oficial)** | Odds, injury report, rosters actualizados | Sin API key, pero no documentada oficialmente — puede cambiar sin aviso |
| **nflreadpy en modo "current season"** | Schedule y rosters actualizados durante la temporada | Mismo paquete que el histórico |
| **nfelo weekly CSV** | Predicciones ya calculadas, actualizadas 2x por semana | Input directo al ensemble, cero esfuerzo |

**Pendiente de decidir contigo:** si la competencia da acceso a líneas "del momento antes de cada juego" — ¿ya viene ese dato en el formato de la competencia, o tenemos que jalarlo nosotros vía API externa? Esto define si necesitamos automatizar The Odds API o no.

---

## 4. Variables (features)

### 4.1 Nivel 0 — Mercado (el feature más fuerte, casi obligatorio)
- Probabilidad implícita del moneyline de cierre (de-vigged, ver fórmula abajo).
- Spread de cierre como feature numérico continuo (redundante con moneyline pero XGBoost puede encontrar no-linealidades).

**Fórmula de-vig (remover el margen de la casa):**
```
p_home_raw = |moneyline_home| / (|moneyline_home| + |moneyline_away|)  [ajustado según signo +/-]
overround = p_home_raw + p_away_raw
p_home_true = p_home_raw / overround
```

### 4.2 Nivel 1 — Fuerza de equipo
- Elo propio (recalculado desde cero, metodología tipo 538/nfelo, con K-factor ajustado y regresión a la media entre temporadas).
- EPA/play ofensivo y defensivo, rolling window de 4 y 8 semanas (no season-to-date completo — la NFL cambia rápido).
- Éxito rate (success rate) ofensivo/defensivo, mismo rolling.

### 4.3 Nivel 2 — Contexto del juego
- Local/visitante, descanso (`home_rest`, `away_rest` — ya vienen en el dataset).
- Distancia de viaje / cambio de zona horaria (derivarlo de coordenadas de estadios).
- Clima (viento, temperatura, si es indoor/outdoor — el viento es el que más afecta passing game).

### 4.4 Nivel 3 — Situacional / última hora
- Status del QB titular (activo/lesionado/dudoso) — el feature de mayor riesgo de fuga de información si no se maneja con el timestamp correcto.
- Cambios de línea entre apertura y el momento de la predicción (line movement) — puede señalar información privilegiada del mercado (lesiones, clima) que aún no está en nuestros otros features.

**Nota de disciplina:** cada feature debe tener timestamp de disponibilidad verificable. Si un dato "se supo" después del kickoff, no puede usarse — esto es la fuga de datos más común en proyectos de estudiantes y arruina el backtest.

---

## 5. Arquitectura del modelo — 3 capas

### Capa A — Baseline de mercado
Probabilidad implícita de-vigged. Cero modelo, es el piso.

### Capa B — Modelo propio
- **B1 (rápido de construir):** Regresión logística sobre Elo propio + EPA rolling. Sirve de sanity check y segunda opinión barata.
- **B2 (el que probablemente gana más):** Gradient Boosting (XGBoost o LightGBM) sobre el set completo de features de niveles 1-3. Tunear con `Optuna` o grid search simple sobre el backtest walk-forward (ver sección 6).

### Capa C — Ensemble final
Combinar Capa A + Capa B (mejor sub-modelo o promedio de B1/B2) con peso óptimo:
```
p_final = w * p_mercado + (1 - w) * p_modelo_propio
```
`w` se determina por backtest (probar de 0.0 a 1.0 en pasos de 0.05, quedarnos con el que maximice accuracy en el set de validación, NO en el de entrenamiento).

Opcional: meter la predicción de nfelo como tercer input y hacer stacking con una regresión logística simple en vez de un promedio fijo.

---

## 6. Metodología de validación (crítico — aquí es donde la mayoría de proyectos se auto-engañan)

- **Split por temporada, nunca aleatorio.** Ejemplo: entrenar 2010-2022, validar 2023, testear 2024-2025.
- **Walk-forward:** simular predicción semana a semana dentro de la temporada de test, sin que el modelo "vea" nada de semanas futuras de esa misma temporada al calcular rolling features.
- **Reportar accuracy por temporada de test**, no solo un número agregado — así se detecta si el modelo es inconsistente año a año.
- **Comparar siempre contra la Capa A sola** — si el ensemble no le gana consistentemente al mercado puro en el backtest, usar solo la Capa A en producción.

---

## 7. Estructura del repositorio

```
nfl-predictor/
├── PLAN_MAESTRO_NFL_PREDICTOR.md   ← este documento, fuente de verdad
├── README.md                        ← quickstart técnico
├── requirements.txt
├── data/
│   ├── raw/                         ← descargas crudas (gitignored si pesan mucho)
│   └── processed/                   ← features ya construidas, parquet
├── src/
│   ├── data_ingestion.py            ← wrappers de nflreadpy, odds API
│   ├── features.py                  ← cálculo de Elo, EPA rolling, de-vig
│   ├── models/
│   │   ├── market_baseline.py       ← Capa A
│   │   ├── own_model.py             ← Capa B (logística + XGBoost)
│   │   └── ensemble.py              ← Capa C
│   ├── backtest.py                  ← walk-forward validation
│   └── weekly_pipeline.py           ← script que corre cada semana en vivo
├── notebooks/
│   └── exploratory.ipynb            ← exploración libre, no producción
└── outputs/
    ├── backtest_results/
    └── weekly_predictions/
```

---

## 8. Roadmap / checklist de ejecución

- [ ] **Fase 1 — Baseline de mercado + backtest histórico**
  - [ ] Descargar schedules 2006-2024 con nflreadpy
  - [ ] Implementar de-vig del moneyline
  - [ ] Calcular accuracy histórico de la Capa A sola → **este número es la referencia de todo el proyecto**
- [ ] **Fase 2 — Feature engineering**
  - [ ] Elo propio (validar contra Elo de FiveThirtyEight como sanity check)
  - [ ] EPA rolling 4 y 8 semanas desde play-by-play
  - [ ] Clima, descanso, distancia de viaje
- [ ] **Fase 3 — Modelo propio (Capa B)**
  - [ ] B1: regresión logística
  - [ ] B2: XGBoost/LightGBM + tuning
  - [ ] Walk-forward validation de ambos
- [ ] **Fase 4 — Ensemble (Capa C)**
  - [ ] Optimizar peso `w` en set de validación
  - [ ] (Opcional) Meter nfelo como input adicional, stacking
  - [ ] Confirmar que le gana a la Capa A sola en el set de test
- [ ] **Fase 5 — Pipeline en vivo**
  - [ ] Conectar fuente de odds en tiempo real (decidir si es necesaria — depende del formato de la competencia)
  - [ ] Automatizar corrida semanal
  - [ ] Logging de predicciones vs resultados reales para trackear accuracy en vivo

---

## 9. Decisiones tomadas (2026-08-17)

1. **Acceso a líneas de apuestas:** no confirmado por la competencia. Se asume el escenario conservador — **nosotros las conseguimos vía API externa** (The Odds API, free tier). Si resulta que la competencia ya las provee, es una simplificación (menos trabajo), no un problema.
2. **Playoffs:** NO incluidos. Universo de datos = temporada regular únicamente, tanto para entrenar como para predecir.
3. **Formato de entrega:** sin restricción → se usa **CSV** por ser lo más simple y rápido de generar y depurar.
4. **Tiempo disponible: ~2 semanas.** Esto obliga a recortar alcance. Ver sección 12 (Roadmap ajustado a 2 semanas) — reemplaza el roadmap ambicioso de la sección 8 como plan operativo real.
5. **Timing de las predicciones — CONFIRMADO (2026-08-17):** los resultados se registran vía una app y **se pueden modificar juego por juego justo antes de que inicie cada partido.** Esto valida completamente el backtest hecho en Fases 1-4: usamos `spread_line`/`moneyline` de nflverse, que documentalmente son **líneas de cierre** ("closing line"), y ahora sabemos que en producción sí vamos a tener acceso a esa misma información al momento de predecir. El 67.04% del ensemble es una estimación realista, no optimista. (Nota histórica: hubo una confusión momentánea donde se pensó que las predicciones se congelaban semana completa por adelantado sin poder ajustar cerca del kickoff — quedó descartada por esta confirmación. Si esto cambia de nuevo, hay que revisar la sección 5 de este documento sobre disciplina de timestamps.)

---

## 12. Roadmap ajustado a 2 semanas (PLAN OPERATIVO REAL)

Con tiempo limitado, la prioridad es tener SIEMPRE algo funcional y sometible, mejorándolo incrementalmente — nunca quedarnos sin nada por perseguir la versión perfecta. Orden de prioridad si el tiempo se acaba antes: 1 → 2 → 3 → 4 → 5. Cada fase debe dejar un modelo *sometible* antes de pasar a la siguiente.

**Días 1-2 — Fase 1: Baseline de mercado + backtest — ✅ COMPLETA (2026-08-17)**
- Descargar schedules históricos (regular season, sin playoffs). Fuente real usada:
  `https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv`
  (no `nfl_data_py.import_schedules()` directo — apunta a habitatring.com, no accesible
  en el entorno de desarrollo; mismos datos, fuente distinta).
- Implementado de-vig y calculado accuracy histórico real de la Capa A.
- **Resultado: 66.57% accuracy global (2006-2024, 4,780 juegos).** Este es el número
  oficial a vencer por cualquier modelo propio.
- Repo funcional en `nfl-predictor/` con `src/data_ingestion.py` y
  `src/models/market_baseline.py`, ambos corren sin errores de punta a punta.
- **Esto ya es sometible si el tiempo se acaba aquí** — sigue siendo cierto.

**Días 3-5 — Fase 2: Feature engineering simplificado — ✅ COMPLETA (2026-08-17)**
- Elo propio implementado (`src/features.py`): K-factor=20, home field advantage=55 puntos
  de Elo, regresión a la media de 1/3 entre temporadas. Accuracy standalone: **61.95%**
  (por debajo del mercado, como se esperaba — su valor es como input del ensemble, no
  como reemplazo).
- EPA rolling de 4 semanas (ofensivo y defensivo, por equipo) desde play-by-play real
  (`src/pbp_ingestion.py`, 906,567 jugadas 2006-2024). Fuente: release `pbp` de
  nflverse-data (`.../releases/download/pbp/play_by_play_{season}.parquet`).
- Se omitió ventana de 8 semanas y clima/viaje por tiempo (quedan en sección 13).
- **Caveat documentado:** 748 de 4,780 juegos (semana 1 de cada temporada) quedan con
  NaN en las features de EPA rolling porque no hay historial previo esa temporada.
  Decisión para Fase 3: dropear esas filas del set de entrenamiento del modelo propio
  (no del baseline de mercado, que no las necesita).
- Salida: `data/processed/games_with_features.parquet` — un archivo con todo unido
  (mercado + Elo + EPA rolling) listo para el modelo de Fase 3.

**Días 6-8 — Fase 3: Modelo propio — ✅ COMPLETA (2026-08-17)**
- XGBoost entrenado con walk-forward validation real (ventana expansiva, test 2015-2024,
  entrenando solo con temporadas estrictamente anteriores). `src/models/own_model.py`.
- **Modelo A (Elo+EPA, sin ver el mercado): 61.82%** — confirma que sí hay señal real en
  las features propias (mejor que azar/50%), pero no le gana al mercado por sí solo.
- **Modelo B (mercado incluido como feature en XGBoost): 65.41%** — HALLAZGO IMPORTANTE:
  dejar que el árbol "reinterprete" la probabilidad de mercado la empeora respecto al
  mercado puro (66.95%). No se debe usar esta variante en producción.
- Se descartó B1 (regresión logística) como estaba planeado — se fue directo a XGBoost.

**Días 9-10 — Fase 4: Ensemble — ✅ COMPLETA (2026-08-17)**
- Ensemble por promedio ponderado (NO como feature dentro del árbol, por el hallazgo de
  Fase 3): `p_final = w * p_mercado + (1-w) * p_modelo_A`. `src/models/ensemble.py`.
- Barrido de `w` de 0.0 a 1.0 sobre las mismas 2,339 predicciones out-of-sample.
- **Mejor resultado: w=0.90 (90% mercado / 10% modelo propio) → 67.04% accuracy.**
- **Interpretación honesta: la mejora sobre el mercado puro (66.95%) es de solo +0.09pp,
  dentro del margen de ruido estadístico para este tamaño de muestra (~1pp de margen de
  error). No hay evidencia sólida de que nuestro modelo propio agregue edge real con el
  set de features actual (Elo simple + EPA rolling 4 semanas).**
- Decisión de producción: usar el ensemble w=0.90 de todas formas (no hace daño, y en el
  peor caso es equivalente al mercado puro) — pero no hay que invertir más tiempo
  esperando que este approach por sí solo genere una ventaja grande.
- **Redirección de esfuerzo recomendada:** las mejoras opcionales de la sección 13
  (status de QB, clima, nfelo como input adicional) son ahora la apuesta con más
  probabilidad de generar edge real, porque capturan información que el EPA rolling
  genérico no ve. Si el tiempo es escaso, priorizar **QB status** — es el que más peso
  tiene en la literatura para explicar sorpresas que el mercado tarda en incorporar.

**Días 11-13 — Fase 5: Pipeline en vivo — ✅ COMPLETA (2026-08-18)**
- `src/odds_client.py`: consume The Odds API (`americanfootball_nfl`, mercado `h2h`),
  promedia la probabilidad implícita de-vigged entre todas las casas disponibles
  (consenso de mercado, más robusto que una sola casa).
- `src/team_name_mapping.py`: mapeo de los 32 equipos entre nombres completos (Odds API)
  y abreviaciones (nflverse) — necesario para poder unir ambas fuentes.
- `src/train_final_model.py`: entrena y guarda (`models/model_a_final.joblib`) el Modelo A
  final sobre TODO el histórico 2006-2024, listo para predecir juegos nuevos.
- `compute_elo()` en `features.py` se extendió para soportar juegos futuros (sin
  resultado todavía) — calcula su elo_pre sin romper la actualización de ratings.
- `get_current_team_form()`: forma reciente de cada equipo cruzando la frontera de
  temporada (a diferencia del EPA rolling de entrenamiento, que resetea cada
  temporada) — evita NaN al inicio de temporada nueva. **Desviación metodológica
  documentada intencional:** en entrenamiento el rolling resetea por temporada: en
  producción no, porque en producción necesitamos SIEMPRE un número, no un NaN.
- `src/weekly_pipeline.py`: junta todo — odds en vivo + Elo actualizado + forma
  reciente + modelo propio → ensemble (w=0.90) → CSV en `outputs/weekly_predictions/`.
- **Probado en este entorno (sin API key real):** la extensión de Elo con la
  temporada 2026 (272 juegos futuros detectados correctamente) y el fallback de
  forma reciente (cayó automáticamente a datos de 2025 al no existir play-by-play
  de 2026 todavía — HTTP 404 manejado sin romper el pipeline).
- **Pendiente de probar por Edward, requiere su propia API key:** la conexión real
  con The Odds API (`fetch_current_odds()`). Instrucciones completas en `README.md`.

**Día 14 — Buffer**
- Colchón para lo que se haya atrasado. No se agenda nada nuevo aquí a propósito.

## 13. Mejoras opcionales (SOLO si sobra tiempo, en este orden de prioridad)

1. EPA rolling de 8 semanas además de 4.
2. Clima (viento) como feature.
3. Status de QB titular como feature de última hora.
4. Meter nfelo como tercer input del ensemble (stacking en vez de promedio ponderado).
5. Distancia de viaje / cambio de zona horaria.

## 14. Fase 6 — Investigación exhaustiva de edge adicional (2026-08-18, "exprimir el %")

Contexto: con 2 semanas de margen sobrante, se investigó a fondo si había edge real
adicional disponible más allá del ensemble de Fase 4, antes de dar por cerrado el
modelo. Se probaron 4 palancas de forma rigurosa (no especulativa):

### 14.1 QB Rating (cross-equipo, cross-temporada)
- **Metodología:** rating de EPA/dropback rolling de los últimos 8 starts de CADA QB,
  usando `qb_epa` y `passer_id` de play-by-play, viaja con el jugador aunque cambie de
  equipo (a diferencia del Elo de equipo). Es la misma idea central de nfelo.
  Implementado en `src/qb_ratings.py`.
- **Cobertura:** solo 84/78 juegos (home/away) con QB sin historial suficiente
  (rookies debutando) de 4,780 — cobertura muy alta, el rating es usable en casi
  todos los casos.
- **Resultado — Modelo standalone (sin ver mercado), mismo subconjunto de test:**
  - Modelo A (Elo+EPA, sin QB): 61.87%
  - Modelo C (Elo+EPA+QB rating): **62.74%** (+0.87pp — el QB SÍ aporta señal real
    a un modelo que no ve el mercado)
  - **PERO en el ensemble final con el mercado: 66.97% (w=0.75), prácticamente
    IDÉNTICO al ensemble sin QB (67.04%, Fase 4) — mejora neta ≈ 0.00pp.**
- **Interpretación:** el mercado YA incorpora casi toda la información de QB para
  cuando se cierra la línea (justo antes del kickoff, que es cuando nosotros también
  predecimos). La señal del QB es real pero redundante con lo que el mercado ya sabe
  en el momento en que nosotros predecimos.

### 14.2 Búsqueda de hiperparámetros (descartar mal ajuste como causa)
Se probaron 3 configuraciones de XGBoost (baseline, más profunda+regularizada, muy
simple+muy regularizada) sobre el Modelo C. Las tres convergen a resultados de
ensemble prácticamente idénticos (66.97%-67.02%). **Esto descarta que el techo se deba
a un modelo mal ajustado — es un techo real, no un problema de tuning.**

### 14.3 Calibración de mercado (búsqueda de sesgos explotables)
Se comparó, por bucket de probabilidad implícita del mercado (pasos de 5%), la
probabilidad implícita promedio vs. la tasa de victoria real observada (2006-2024).
**No se encontró ningún sesgo sistemático explotable** (ni siquiera el clásico "home
underdog bias" documentado en la literatura de apuestas) — las diferencias oscilan
alrededor de cero sin patrón consistente. El mercado está bien calibrado.

### 14.4 Conclusión honesta y razón estructural
**~67% parece ser el techo real alcanzable con información pública y modelado a nivel
equipo/QB, dado que la competencia exige predecir el 100% de los juegos.** Esto no es
falta de esfuerzo — es consistente con la literatura: incluso los grupos profesionales
de apuestas no le ganan al cierre por márgenes grandes, y ellos tienen la ventaja de
ser SELECTIVOS (solo apuestan donde ven valor claro). Nosotros no podemos elegir
nuestras batallas — tenemos que predecir cada juego, lo que nos fuerza a competir
directo contra la eficiencia del mercado en cada partido individual, sin poder
abstenernos en los casos difíciles.

**Recomendación de producción:** usar el ensemble con QB rating de todas formas
(w≈0.75-0.90 mercado), porque en el peor caso es equivalente al mercado puro y en
semanas con cambios de QB atípicos (lesiones, suplentes debutando) puede aportar en
casos puntuales aunque no se vea en el promedio agregado de 10 temporadas.

**Avenidas restantes con expectativa de retorno baja pero no nula (solo si sobra
tiempo, con disciplina anti-overfitting estricta — validar con holdout real, no
solo con el mismo backtest):**
- Sesgos situacionales específicos con respaldo en literatura (ej. equipos en semana
  corta tras partido internacional, "letdown games" tras victorias en primetime) —
  alto riesgo de overfitting a subgrupos pequeños, requiere mucho cuidado.
- Datos de movimiento de línea (opening vs closing) como señal de "steam moves" —
  requiere fuente de datos de line movement histórico, no solo el cierre.
- Clima en tiempo real al momento de la predicción (viento en estadios outdoor).

### 14.5 Fase 7 — Features situacionales (clima, divisional, descanso) + prueba de significancia estadística

Última pasada, con el objetivo explícito de Edward de superar 70% ("life changing
money" — se justifica la debida diligencia extra). Se agregaron 7 features más al
Modelo C: `div_game`, `wind`, `temp_extreme`, `dome_or_closed`, `home_short_week`,
`away_short_week`, `rest_advantage_extreme`. Datos ya disponibles en el schedule de
nflverse (roof/surface/temp/wind/div_game), sin necesidad de fuentes nuevas.
`src/models/model_d_situational.py`.

- **Modelo D standalone:** 63.35% (vs 62.74% del Modelo C — mejora real y esperable,
  el modelo sin ver el mercado sí aprovecha esta info).
- **Ensemble con Modelo D:** 67.19% (w=0.90) — **+0.22pp sobre el mercado puro**, la
  mejor cifra de todo el proyecto.
- **Prueba de significancia (McNemar exacto):** de 2,292 juegos de test, el ensemble
  solo difiere del mercado puro en 29 casos (1.3%). De esos 29, acierta 17 y falla 12.
  **p-value = 0.4583 — muy lejos de significancia estadística (se necesita p<0.05).**
  Conclusión: esta mejora de +0.22pp es indistinguible de suerte, no de señal real.

**Se investigó y descartó explícitamente movimiento de línea (opening vs closing)**
como avenida — The Odds API solo tiene histórico gratuito desde mediados de 2020
(menos de 5 temporadas completas), muestra insuficiente para validar sin sobreajustar,
y el costo en cuota de API es alto. No se pursuió más allá de la investigación inicial.

### 14.6 Veredicto final de la investigación de edge (Fases 6-7)

Se probaron, con rigor estadístico real (no solo backtest sin verificar), SEIS
palancas distintas: QB rating, 3 configuraciones de hiperparámetros, calibración de
mercado, y features situacionales con prueba de significancia. **Ninguna produjo una
mejora estadísticamente distinguible de ruido sobre el mercado puro (66.95-66.97%).**

Esto no es evidencia de que el modelo esté mal construido — es evidencia consistente
y convergente (desde ángulos completamente distintos) de que **~67% es el techo real
para este problema específico** (predicción straight-up del 100% de los juegos, con
datos públicos, prediciendo justo antes del kickoff — el mismo momento en que el
mercado ya incorporó toda la información disponible). Llegar establemente a 70%+
requeriría uno de estos ingredientes, ninguno realista en el marco de esta
competencia:
1. Información privilegiada que el mercado no tiene al momento del kickoff (no existe
   legalmente para un participante externo).
2. Datos de movimiento de línea de calidad institucional (de pago, fuera de alcance).
3. Que el mercado de NFL fuera menos eficiente de lo que es — no lo es; es uno de los
   mercados de apuestas más líquidos y estudiados que existen.

**Recomendación final: adoptar el ensemble del Modelo D (67.19%, w=0.90) como modelo
de producción** — no porque su mejora sobre el mercado esté probada, sino porque no
hay razón para NO incluir esas features (no hacen daño, son principiadas, y en el
peor caso el modelo converge a comportarse casi igual al mercado puro de todas
formas). Pero la expectativa honesta para comunicar es **~67%, no 70%.**

- Este documento es la **fuente única de verdad**. Cualquier LLM que retome el proyecto debe leer este archivo completo antes de tocar código.
- Al terminar cada sesión de trabajo, actualizar la sección 8 (checklist) marcando lo completado y anotar en la sección 9 cualquier decisión nueva que se haya tomado.
- El repo en `/mnt/user-data/outputs/nfl-predictor/` (o donde se aloje una vez subido a GitHub) debe reflejar siempre el estado real del checklist — si el checklist dice que Fase 1 está completa, el código de Fase 1 debe correr sin errores.
- Preferencias de trabajo de Edward (aplican para ambos LLMs): español como idioma principal, diffs quirúrgicos tipo "Reemplaza esto / Por esto" en vez de reescribir archivos completos, framing estratégico directo sin rodeos académicos.

---

## 15. AUDITORÍA EXTERNA (Claude Opus, 2026-08-18) — SECCIÓN DEFINITIVA, LEER PRIMERO

**Este documento completo vive también en `AUDITORIA_NFL_PREDICTOR.md` en la raíz del
repo — es lectura obligatoria antes de tocar el modelo. Lo que sigue es un resumen de
sus hallazgos y qué se corrigió en el código como resultado. Donde esta sección
contradice conclusiones de las secciones 6/13/14 (Fases 3-4-6-7, escritas por Claude
antes de la auditoría), **esta sección tiene autoridad** — la auditoría reprodujo el
walk-forward de forma independiente, con más rigor estadístico (McNemar, poder
estadístico, validación fuera de muestra del peso del ensemble) del que se aplicó
originalmente.

### 15.1 El hallazgo central: el ensemble NO le gana al mercado, con certeza estadística
- Sobre el universo completo 2015-2024 (2,613 juegos, sin excluir semana 1 ni años de
  reubicación — ver 15.2), el ensemble en su mejor peso vence al mercado en 12 juegos y
  pierde en 10. **McNemar p=0.83.** Esto no es una mejora real, es ruido.
- Se probó también un stacker logístico (meta-learner) en vez de promedio ponderado
  fijo: 67.65% vs 67.55%, gana 10 pierde 8, **p=0.81**. Tampoco es real.
- **Eligiendo el peso `w` de forma honesta (fuera de muestra: calibrar en 2015-19,
  evaluar en 2020-24) el procedimiento elige w=1.00 — es decir, mercado puro.** El
  w=0.90 que se reportó en Fase 4 se eligió mirando el mismo período que se usó para
  "validarlo" — es un error metodológico real de la Fase 4 original, aunque menor.

### 15.2 Bugs reales encontrados y CORREGIDOS en el código (2026-08-18)

| # | Bug | Efecto | Fix aplicado |
|---|---|---|---|
| 1-2 | `schedule` de nflverse usa abreviaciones históricas (STL/SD/OAK); `pbp` de nflfastR usa las modernas (LA/LAC/LV) para TODA su historia. El merge de EPA fallaba en silencio. | 483 juegos de entrenamiento y 118 de test desaparecían sin aviso (todos los Rams pre-2016, Chargers pre-2017, Raiders pre-2020). Elo de esas franquicias se reiniciaba a 1500 en el año de mudanza. | `src/team_reloc.py` (nuevo) — canonicaliza a abreviación moderna en el punto de ingesta, antes de que toque Elo o EPA. Aplicado en `data_ingestion.py` y `weekly_pipeline.py`. |
| 3 | El flag `qb_unknown` se activaba cuando el merge de QB fallaba — y fallaba exactamente cuando el QB titular no llegó a 5 dropbacks EN ESE MISMO JUEGO (ej. salió lesionado). El resultado del juego se filtraba hacia atrás como feature. | Equipos con el flag perdían 7-8pp más de lo que el mercado ya implicaba — no era señal predictiva, era fuga de datos. | `qb_ratings.py` — el join pasó de exacto por `(game_id, qb_id)` a `merge_asof` por QB (última aparición calificada ANTES o EN esta semana, sin exigir que el juego evaluado sea esa aparición). |
| 4 | `weekly_pipeline.py` rellenaba features de EPA faltantes con `fillna(0.0)`. | Para `success_rate`, 0.0 está a **-8.7 desviaciones estándar** del promedio real — fuera de todo el soporte de entrenamiento. Único bug con potencial de producir un pick catastrófico en un juego real. | `train_final_model.py` ahora persiste las medias reales de entrenamiento (`feature_means.joblib`); `weekly_pipeline.py` las usa como fallback en vez de 0.0. |
| 5 | El EPA rolling de entrenamiento reseteaba en cada temporada nueva (`groupby(team, season)`); el de producción (`get_current_team_form`) cruzaba temporadas. El modelo entrenaba con una distribución de features que nunca iba a ver en producción. | Discrepancia entrenamiento/producción; además perdía todos los juegos de semana 1 de cada temporada del set de entrenamiento/evaluación. | `features.py::compute_epa_rolling` — se quitó `season` del `groupby`, ahora cruza temporadas igual que producción. Recupera los juegos de semana 1. |
| 6 (menor) | Elo no distinguía sitios neutrales (Londres/CDMX/Múnich) — daba ventaja de local completa igual que en un estadio real. | Sesgo pequeño en Elo para ~1-2 juegos/temporada. | `features.py::compute_elo` — ventaja de local = 0 cuando `location == "Neutral"`. |
| 7 (menor) | Pick'em exacto (`p_home_final == 0.5`) se resolvía por accidente de código (`>` favorecía silenciosamente al visitante), no por decisión. | Ninguno en accuracy esperado (la auditoría confirma que estos casos son estadísticamente una moneda al aire de todas formas), pero no era defendible. | `weekly_pipeline.py` — desempate explícito usando el signo de `elo_diff`. |

**Después de corregir los bugs 1-2 y 5:** Modelo A standalone sube de 61.82% a
**63.34%** (recuperando los juegos y la señal que se perdían por el bug de
reubicación). **El ensemble final no se mueve de forma significativa** — los bugs
costaban DATOS y COBERTURA, no accuracy del ensemble. Esto es consistente: el modelo
propio mejora, pero como el ensemble ya pondera fuerte hacia el mercado, la mejora del
componente propio no se traduce en una mejora medible del ensemble.

### 15.3 El 67% original estaba medido sobre un subconjunto más fácil que la competencia real
- Universo completo 2015-2024 (con los fixes aplicados): mercado = **66.40%**.
- El subconjunto que se evaluó en Fases 3-4 (excluyendo semana 1 y años de
  reubicación, por los bugs 1-2-5 antes de corregirse): mercado = 66.95%.
- Los juegos EXCLUIDOS por esos bugs: mercado acertó solo **61.68%** en ellos.
- La semana 1 sola: mercado acertó **62.18%**.
- **Como la competencia exige predecir el 100% de los juegos (incluyendo semana 1),
  la expectativa honesta baja ~0.8pp respecto al 67% reportado originalmente — casi
  diez veces el tamaño de la "mejora" que el ensemble aparentaba tener sobre el
  mercado.**

### 15.4 Sobre llegar a 70%: veredicto cuantitativo, no solo cualitativo
- Con w=0.95, el modelo propio solo puede cambiar el pick del mercado en 415 de 2,613
  juegos (la banda 44%-56% de probabilidad implícita, donde el mercado está más
  indeciso). Dentro de esa banda se probó TODO lo disponible como criterio alternativo:
  quedarse con el mercado (52.5% de aciertos ahí), favorecer siempre al local (48.7%),
  favorecer mayor Elo (50.6%), mejor EPA (47.2%), más descanso (46.3%). **Ninguno con
  p<0.12.**
- Llegar a 70% global exigiría pasar de 52.5% a 76% de aciertos específicamente dentro
  de esa banda de 415 juegos — un salto que ninguna señal disponible sostiene.
- Se probaron 17 hipótesis de sesgos documentados en la literatura de apuestas
  (incluyendo altitud de Denver: residuo -1.96pp, p=0.61) — **ninguna sobrevive
  corrección de Bonferroni** para comparaciones múltiples.
- **El número más importante de toda la auditoría:** para validar con 80% de poder
  estadístico una mejora real de +1pp sobre el mercado, harían falta ~7,840 juegos ≈
  29 temporadas de datos. **Solo existen 19 temporadas con moneyline disponible en el
  mundo.** Una mejora de ese orden de magnitud es matemáticamente imposible de
  verificar con los datos que existen, sin importar cuánto tiempo o esfuerzo se le
  ponga.

### 15.5 70% SÍ es alcanzable — pero por varianza de temporada, no por modelado
La accuracy del favorito del mercado por temporada tiene media 66.6% y desviación
estándar 3.1pp. **P(una temporada dada supere 70%) ≈ 14%.** De hecho, 3 de las últimas
15 temporadas lo superaron. Esto es información valiosa para calibrar expectativas:
el resultado final de la competencia va a depender más de qué tan "predecible" resulte
ser la temporada 2026 en sí (varianza del mundo real) que de cualquier decisión de
modelado que tomemos nosotros.

### 15.6 Pendiente de decisión — RESUELTO (2026-08-18)

**Contexto confirmado:** 100+ participantes, premio único (winner-take-all).

**Análisis (teoría de torneos — Bronars 1986, Taylor 2003):** en contests de premio
único, quien va "adelante" en expectativa debe MINIMIZAR varianza (agregar riesgo solo
puede quitarle el primer lugar, no dárselo); quien va "atrás" debe MAXIMIZARLA (jugar
seguro garantiza perder).

**Conclusión:** Edward plausiblemente es el "líder" ex-ante, no el rezagado — la
premisa central de todo el proyecto (la mayoría de competidores estudiantiles no va a
usar el mercado como base, por desconocimiento o por evitarlo pensando que es "hacer
trampa") implica que su accuracy esperado ya está por encima de la mediana del campo
de 100+. Bajo esa premisa, la teoría indica lo contrario de la intuición: **NO inyectar
varianza artificial / picks de riesgo sin información real detrás.** Hacerlo solo
ayuda a quien va atrás; a quien va adelante, cualquier ruido no informativo solo
aumenta la probabilidad de que alguien con suerte lo alcance.

**Única excepción real (no verificable, no accionable):** si resulta que muchos de los
100+ competidores TAMBIÉN usan el mercado como base, Edward no es un líder distinto
sino parte de un grupo empatado — y ahí el ganador se decide por suerte pura, sin que
ninguna estrategia de modelado lo cambie. No hay forma de saber en qué escenario se
está, pero la estrategia recomendada (mantener el proceso de mayor valor esperado, sin
ruido artificial) es la dominante en AMBOS escenarios simultáneamente.

**Implicación operativa (esto sí es accionable):** con 100+ compitiendo por un premio
único, un solo error no forzado (pipeline caído un domingo, bug no detectado) tiene
costo desproporcionado — puede sacar a Edward de la contienda por completo, sin margen
para recuperarse. La prioridad para las semanas que quedan pasa de "seguir buscando
edge de modelado" a **confiabilidad operativa pura**: que `weekly_pipeline.py` no
falle en día de juego, tener un plan B manual si la API de odds falla.

- Este documento es la **fuente única de verdad**. Cualquier LLM que retome el proyecto debe leer este archivo completo antes de tocar código.
- Al terminar cada sesión de trabajo, actualizar la sección 8 (checklist) marcando lo completado y anotar en la sección 9 cualquier decisión nueva que se haya tomado.
- El repo en `/mnt/user-data/outputs/nfl-predictor/` (o donde se aloje una vez subido a GitHub) debe reflejar siempre el estado real del checklist — si el checklist dice que Fase 1 está completa, el código de Fase 1 debe correr sin errores.
- Preferencias de trabajo de Edward (aplican para ambos LLMs): español como idioma principal, diffs quirúrgicos tipo "Reemplaza esto / Por esto" en vez de reescribir archivos completos, framing estratégico directo sin rodeos académicos.

## 16. Bandas de confianza para intervención manual (2026-08-19)

Contexto: Edward registra los picks manualmente en la app de la competencia — tiene la
última palabra sobre cada uno, sin importar lo que diga el modelo. La pregunta que
resuelve esta sección: **¿en qué juegos vale la pena que su criterio (alineaciones,
lesiones de último minuto, rivalidad) anule al modelo, y en cuáles probablemente solo
va a restar valor?**

### 16.1 Calibración histórica real por banda (2006-2024, 4,780 juegos)

| Banda | Distancia de 50/50 | Accuracy histórico real | Prob. implícita del favorito | ~Juegos/temporada |
|---|---|---|---|---|
| 1. LOCK | >20pp (favorito ≥70%) | **79.0%** | 77.8% | ~92 |
| 2. CONFIADO | 10-20pp (favorito 60-70%) | **63.4%** | 64.8% | ~84 |
| 3. PAREJO | 6-10pp (favorito 56-60%) | **56.5%** | 58.0% | ~35 |
| 4. MONEDA AL AIRE | ≤6pp (favorito 50-56%) | **53.4%** | 53.2% | ~41 |

**El mercado está bien calibrado en las 4 bandas** (diferencia entre accuracy real y
probabilidad implícita ≤1.5pp en todas) — esto confirma que la banda "pareja" no es un
punto ciego del modelo, es que esos juegos genuinamente son casi 50/50 en la realidad.

### 16.2 Guía de intervención

- **Bandas 1-2 (Lock/Confiado, ~176 juegos/temporada, ~2/3 del calendario):** el
  criterio manual de Edward probablemente RESTA valor aquí, no suma. La sección 15.4
  ya probó formalmente que no hay sesgo sistemático explotable ni siquiera en cosas
  intuitivas (rivalidad divisional, home underdogs, altitud). Si la intuición dice
  "pero es juego de revancha" — eso es exactamente el tipo de narrativa que se probó
  y no sobrevivió ninguna prueba de significancia (17 hipótesis probadas, ninguna
  sobrevivió Bonferroni). **Regla: confiar en el modelo salvo evidencia concreta.**
- **Bandas 3-4 (Parejo/Moneda al aire, ~76 juegos/temporada):** aquí SÍ tiene sentido
  que el criterio manual intervenga, por una razón específica y no por "sensación": es
  la MISMA banda (44-56% de probabilidad implícita) donde la auditoría de la sección
  15.4 confirmó que ni el modelo propio tiene edge probado (p<0.12 en todo lo probado:
  Elo, EPA, descanso). El modelo no le está ganando a Edward ahí — están empatados.
  **Regla: si hay información genuinamente más reciente que la que tenía el pipeline
  al correr (baja de última hora anunciada después, inactive list visto en persona),
  vale la pena usarla. Si es solo presentimiento sin dato nuevo detrás, no.**

### 16.3 Implementación

`weekly_pipeline.py` ahora clasifica cada pick en una de las 4 bandas automáticamente
(columna `tier` en el CSV de salida), con el mensaje de guía correspondiente. La
categorización usa `confidence` (distancia de la probabilidad final del ensemble
respecto a 50/50), no solo la probabilidad de mercado — así que ya incorpora el ajuste
del modelo propio antes de clasificar.

- Este documento es la **fuente única de verdad**. Cualquier LLM que retome el proyecto debe leer este archivo completo antes de tocar código — y desde 2026-08-18, eso incluye obligatoriamente `AUDITORIA_NFL_PREDICTOR.md` y la sección 15.
- Al terminar cada sesión de trabajo, actualizar la sección 8 (checklist) marcando lo completado y anotar en la sección 9 cualquier decisión nueva que se haya tomado.
- El repo en `/mnt/user-data/outputs/nfl-predictor/` (o donde se aloje una vez subido a GitHub) debe reflejar siempre el estado real del checklist — si el checklist dice que Fase 1 está completa, el código de Fase 1 debe correr sin errores.
- Preferencias de trabajo de Edward (aplican para ambos LLMs): español como idioma principal, diffs quirúrgicos tipo "Reemplaza esto / Por esto" en vez de reescribir archivos completos, framing estratégico directo sin rodeos académicos.

## 11. Próximo paso inmediato

**Estado a 2026-08-18 (post-auditoría de Claude Opus): 7 bugs encontrados, 7
corregidos en el código real (`team_reloc.py` nuevo, fixes en `features.py`,
`qb_ratings.py`, `weekly_pipeline.py`, `train_final_model.py`). Pipeline completo
re-corrido de punta a punta con los fixes — resultados verificados, coinciden con
los que reportó la auditoría de forma independiente (66.40% de mercado puro en el
universo completo 2015-2024, mismo número en ambos análisis).**

**Expectativa honesta final: ~66.4%, no 67%, y definitivamente no 70% por modelado**
(sección 15.4 — matemáticamente no verificable con los datos que existen en el mundo,
independientemente del esfuerzo). 70% sí es alcanzable por varianza de temporada
(~14% de probabilidad, sección 15.5) — eso ya no depende de nosotros.

**Lo único pendiente de verdadero valor, en orden:**
1. **Responder la pregunta de la sección 15.6** (estrategia de torneo — ver
   pregunta pendiente de Edward sobre # de participantes y estructura del premio).
   La auditoría es explícita: esta decisión vale más que cualquier feature adicional.
2. Validar `weekly_pipeline.py` con datos reales y la API key de Edward en cuanto
   haya juegos próximos.
3. **Pendiente de implementar, no crítico:** `weekly_pipeline.py` sigue usando
   features del Modelo A (sin QB rating) — dado que el QB rating no mejora el
   ensemble de forma medible (sección 15.1-15.2), no es prioritario migrar
   producción a Modelo C/D. Dejar así es una decisión válida, no una tarea pendiente.

**Para continuidad entre sesiones:** todo el pipeline corre de punta a punta con los
fixes aplicados. `models/model_a_final.joblib` y `models/feature_means.joblib` están
regenerados con datos corregidos (4,780 juegos, ya no se pierden los de semana 1 ni
reubicación). Cualquier sesión nueva puede reproducir exactamente estos números
re-corriendo `data_ingestion.py` → `features.py` → `train_final_model.py`.
