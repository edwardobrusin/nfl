# Auditoría técnica — NFL Predictor
**Fecha:** 18 de agosto de 2026
**Alcance:** código fuente en `src/`, datos en `data/`, resultados en `outputs/`. Pipeline reproducido íntegro y re-ejecutado con modificaciones controladas.

---

## Resumen ejecutivo

Reproduje el walk-forward completo. El resultado reportado es correcto en el sentido de que el código hace lo que dice, pero **la mejora del ensemble sobre el mercado puro no es distinguible de ruido**:

| Comparación | Accuracy | McNemar vs mercado | p |
|---|---|---|---|
| Mercado puro (subset evaluado, n=2,339) | 66.95% | — | — |
| Ensemble w=0.90 (mejor del barrido) | 67.04% | gana 12, pierde 10 | **0.832** |
| Ensemble con bug de reubicación corregido (n=2,455) | 66.76% vs 66.68% | gana 14, pierde 12 | **0.845** |
| Stacker logístico walk-forward (meta-learner) | 67.65% vs 67.55% | gana 10, pierde 8 | **0.815** |

**+0.09 pp = 2 juegos netos en 10 temporadas.** No es una mejora; es la fluctuación esperada de tomar el máximo de 21 pesos candidatos evaluados sobre el mismo conjunto de prueba.

Además, el 67% está medido sobre un subconjunto **más fácil** que el universo real de la competencia (ver Supuesto A).

---

## (a) Bugs y supuestos cuestionables

### BUG 1 — CRÍTICO: abreviaciones de franquicias reubicadas no coinciden entre schedule y play-by-play

`data_ingestion.py` toma los equipos del schedule de nflverse, que usa las abreviaciones **históricas**: `STL`, `SD`, `OAK`. El play-by-play de nflfastR (`pbp_ingestion.py`) usa las abreviaciones **normalizadas modernas** para todas las temporadas: `LA`, `LAC`, `LV`.

Evidencia directa:

```
equipos presentes en team_week_epa (pbp): ... LA, LAC, LV ...  (no hay STL/SD/OAK)
equipos presentes en games:               ... OAK (2006-2019), SD (2006-2016), STL (2006-2015) ...
```

Consecuencia: el `merge` de `attach_epa_features()` falla en silencio para **todos** los juegos de Rams pre-2016, Chargers pre-2017 y Raiders pre-2020. Esas filas quedan con `roll4 = NaN` y luego el `dropna()` de `load_dataset()` las elimina.

**Impacto medido:**

- Set modelable: **4,032 juegos en vez de 4,515** (−10.7%, 483 juegos perdidos).
- Set de prueba 2015-2024: **2,339 en vez de 2,455** (118 juegos de test desaparecen sin aviso).
- Juegos afectados por temporada: 43 en 2015, 28 en 2016, 17 en 2017, 15 en 2018, 15 en 2019.

**Fix:**

```python
RELOC = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
games["home_team"] = games["home_team"].replace(RELOC)
games["away_team"] = games["away_team"].replace(RELOC)
# aplicar ANTES de compute_elo() y de attach_epa_features()
```

**Efecto sobre el resultado (medido, no estimado):** Modelo A standalone sube de 61.82% a 62.20%. El ensemble no cambia (66.76% vs mercado 66.68%, McNemar p=0.845). Es decir: el bug estaba costando datos, no accuracy.

---

### BUG 2 — Elo se reinicia a 1500 en cada reubicación

Origen idéntico al bug 1. En `compute_elo()`, el diccionario `ratings` se indexa por abreviación. Cuando `STL` se convierte en `LA` (2016), `SD` en `LAC` (2017) y `OAK` en `LV` (2020), la franquicia entra al `ratings.get(home, ELO_START)` como equipo nuevo y arranca en 1500, tirando a la basura su historial. Peor: el Elo es de suma cero, así que los rivales de esos equipos también quedan mal calibrados esa temporada.

Se corrige con el mismo mapeo del bug 1.

---

### BUG 3 — Fuga de datos (leakage) en `home_qb_unknown` / `away_qb_unknown`

En `qb_ratings.py`, `qb_rolling` solo contiene filas para apariciones con `dropbacks >= 5`. El merge se hace por `(game_id, qb_id)`, así que **el merge falla exactamente cuando el QB titular listado no acumuló 5 dropbacks en ese mismo partido** — o sea, cuando salió lesionado temprano, fue baja de último minuto, o lo sentaron. Eso es información posterior al kickoff.

El flag `qb_unknown` codifica ese evento y lo entrega al modelo como feature.

**Evidencia:**

| | n | win rate del equipo | p_mercado media | residuo |
|---|---|---|---|---|
| `home_qb_unknown = 1` | 84 | 36.90% | 44.71% | **−7.81 pp** |
| `away_qb_unknown = 1` | 78 | 21.79% | 28.89% | **−7.09 pp** |

El equipo cuyo QB queda "unknown" pierde ~7-8 pp más de lo que el mercado implicaba. Eso no es señal predictiva: es el modelo enterándose de que al QB lo sacaron del partido.

**Nota honesta:** verifiqué si esto explicaba la ganancia de +0.87 pp del Modelo C. **No la explica.** Al eliminar los juegos contaminados (n=3,914 de 3,971), la ganancia del QB rating crece de +1.18 pp a +1.95 pp standalone. La señal de QB es real; el flag es igualmente un bug y hay que quitarlo, sobre todo porque en producción se calcularía con una lógica distinta.

**Fix:** construir el rating de QB por `(qb_id, fecha)` sin exigir que el QB aparezca en el juego que se está prediciendo; usar el último rating disponible antes del kickoff, y derivar `unknown` de "este QB tiene menos de N starts previos", no de un merge fallido.

---

### BUG 4 — `fillna(0.0)` en el pipeline en vivo mete inputs imposibles

`weekly_pipeline.py`, líneas 160-162:

```python
for c in df.columns:
    if "roll4" in c:
        df[c] = df[c].fillna(0.0)
```

Distribución real de esas features en el set de entrenamiento:

| feature | media | desv. est. | mín |
|---|---|---|---|
| `home_off_success_rate_roll4` | 0.4285 | 0.0492 | 0.1639 |
| `home_def_success_rate_roll4` | 0.4275 | 0.0461 | 0.2373 |
| `away_off_success_rate_roll4` | 0.4283 | 0.0495 | 0.2373 |
| `away_def_success_rate_roll4` | 0.4270 | 0.0464 | 0.1639 |

Rellenar un success rate con `0.0` es un valor a **−8.7 desviaciones estándar**, fuera del soporte completo de los datos de entrenamiento. XGBoost extrapolará al nodo hoja más extremo. Para `off_epa_play` el 0.0 sí es aproximadamente neutral (media −0.009), pero para los success rates es un desastre.

**Fix:** rellenar con la media del set de entrenamiento, feature por feature, guardada junto al modelo:

```python
FILL = df_train[FEATURES_NO_MARKET].mean().to_dict()  # persistir con joblib
df[c] = df[c].fillna(FILL[c])
```

---

### BUG 5 — Skew entrenamiento/producción en el EPA rolling

- **Entrenamiento** (`features.py`): ventana de 4 semanas **dentro de la temporada**, con reset en cada temporada nueva y `min_periods=1`.
- **Producción** (`weekly_pipeline.py`): últimos 4 juegos **cruzando temporadas**.

Está documentado como decisión deliberada, y el razonamiento (evitar NaN al inicio de temporada) es sensato. Pero el efecto es que en las semanas 1-4 de cada temporada el modelo recibe una distribución de features que nunca vio durante el entrenamiento: en el entrenamiento, la semana 2 tenía un promedio de 1 juego de la temporada actual; en producción tendrá 4 juegos, 3 de ellos de la temporada anterior. Son variables con varianza y significado distintos.

**Fix limpio:** entrenar con la misma definición cross-temporada que se usa en producción. Es un cambio de una línea en `compute_epa_rolling` (quitar `season` del `groupby`) y elimina la discrepancia además de recuperar los juegos de semana 1.

---

### Supuesto A — El 67% está medido sobre un subconjunto más fácil que la competencia real

Esto es lo más importante de esta sección.

| Conjunto | n | Accuracy del mercado |
|---|---|---|
| **Universo real de la competencia** (todos los juegos REG 2015-2024) | 2,613 | **66.40%** |
| Subconjunto evaluado en el backtest | 2,339 | 66.95% |
| Juegos excluidos (semana 1 + reubicación) | 274 | **61.68%** |
| Solo semana 1 | 156 | **62.18%** |

La competencia obliga a predecir el 100% de los juegos. La semana 1 es sistemáticamente más difícil (62.2% vs 66.4%) y está completamente ausente de la validación. **La expectativa honesta para la competencia es ~66.4%, no 67.2%** — una diferencia de 0.8 pp, casi diez veces el tamaño de la "mejora" que aporta el ensemble.

---

### Supuesto B — El peso w=0.90 se eligió por argmax sobre el mismo conjunto de prueba

`ensemble.py` barre 21 valores de w sobre las predicciones de 2015-2024 y se queda con el máximo. El barrido tiene jitter no monótono de ±0.4 pp entre pesos adyacentes (w=0.85 → 66.70%, w=0.90 → 67.04%, w=0.95 → 66.99%). Tomar el máximo de 21 estadísticos correlacionados y reportarlo como resultado sobreestima por construcción.

**Prueba fuera de muestra** (elegir w con 2015-2019, evaluar en 2020-2024, n=1,260):

| Dataset | w óptimo en train | Accuracy 2020-24 | Mercado puro | Delta |
|---|---|---|---|---|
| Original | 1.00 | 67.54% | 67.54% | **+0.00 pp** |
| Con fix de reubicación | 0.90 | 67.62% | 67.54% | **+0.08 pp** |

En la mitad de los casos el procedimiento honesto elige **w=1.0, es decir, mercado puro**.

---

### Supuesto C — El método de de-vig no es el problema (y hay un argumento formal para ello)

Tu hipótesis era que el de-vig proporcional podría estar sesgando favoritos/underdogs extremos y que Shin lo arreglaría. **No puede, y esto es demostrable sin correr nada:** tanto la normalización proporcional como el método de Shin son transformaciones monótonas del par de probabilidades crudas. `p_home/(p_home+p_away) > 0.5` si y solo si `p_home_raw > p_away_raw`. Ningún método de de-vig puede cambiar **cuál lado es favorito**; solo cambia la magnitud.

Lo implementé de todas formas (Shin con solución numérica de z por Brent):

- Picks distintos entre proporcional y Shin: **15 de 2,613** — y son exactamente los 15 juegos con `p = 0.5000` donde el desempate es arbitrario.
- Accuracy: proporcional 66.40%, Shin 66.36%.
- Ensemble: **idéntico con ambos** (mejor w=0.90, 67.04%).
- Overround medio del dataset: 1.0297 (vig de ~2.97%, normal).

Shin es teóricamente superior para estimar probabilidades. Para una métrica de acierto binario es irrelevante.

---

### Hallazgos menores

- **Pick'em exactos:** 15 juegos con `p_home_market == 0.5000`. `(p > 0.5)` los asigna al visitante por accidente del operador. El local ganó 7 de 15. Es un empate estadístico, pero conviene desempatar explícitamente con el spread o el Elo en vez de dejarlo al azar del código.
- **Sitios neutrales:** 56 juegos (Londres, Ciudad de México, Múnich) reciben los +55 puntos de Elo de ventaja de local. La columna `location` del schedule ya distingue `Home`/`Neutral`. Efecto residual medido: −0.81 pp, n=56, p=0.89 — irrelevante, pero es gratis corregirlo.
- **Cobertura de moneyline:** 134 juegos sin moneyline en el histórico, concentrados en 2006 (47) y 2008 (72). Solo afecta a los primeros folds de entrenamiento.
- **Sin margen de victoria en el Elo:** el Elo actual actualiza solo con el resultado binario. El Elo de 538/nfelo usa un multiplicador de margen de victoria con corrección de autocorrelación. Es una mejora real del Elo — pero como el Elo entra con 10% de peso y solo puede actuar en la banda coin-flip, el efecto en accuracy final es prácticamente nulo.

### Lo que está bien hecho (verificado, sin fuga)

- El `shift(1)` antes del `rolling()` en `compute_epa_rolling` y en `compute_qb_rolling_rating` es correcto. Verificado por semana: 100% de NaN en semana 1, patrón consistente después.
- El Elo es estrictamente causal: recorre en orden cronológico y usa `elo_pre` antes de actualizar. El manejo de juegos futuros (`home_win` NaN → no actualiza) es correcto.
- El walk-forward es expansivo estricto: `df[df.season < test_season]`. No hay fuga temporal.
- El merge de EPA no genera duplicados (0 duplicados en `(season, week, team)`, 0 `game_id` duplicados en el resultado).
- La comparación contra el baseline de mercado se hace sobre el mismo subconjunto post-`dropna`, que es lo correcto.

---

## (b) Avenidas nuevas, priorizadas con estimación honesta

### Antes: por qué la mayoría de las avenidas están muertas de entrada

**Restricción estructural del ensemble.** Con `w = 0.90`, el pick final solo puede diferir del mercado cuando `0.9·p_mkt + 0.1·p_mod > 0.5` cambia de signo, lo que exige `p_mkt ∈ (0.4444, 0.5556)`:

| w | Banda donde el modelo puede cambiar el pick | Juegos (2015-24) | Accuracy del mercado ahí |
|---|---|---|---|
| 0.90 | (0.444, 0.556) | 415 (15.9%) | 52.53% |
| 0.80 | (0.375, 0.625) | 1,084 (41.5%) | 57.01% |
| 0.70 | (0.286, 0.714) | 1,828 (70.0%) | 60.39% |

Bajar w expande la zona de acción pero mete al modelo (61-62% standalone) a decidir juegos donde el mercado acierta 57-60%. Por eso el barrido converge a w alto: es la única configuración donde el modelo no destruye valor.

**Y en la banda donde sí puede actuar, no hay nada que predecir.** Probé todos los predictores disponibles sobre los 415 juegos coin-flip:

| Regla de decisión | Accuracy | z vs 50% | p |
|---|---|---|---|
| Mercado (favorito por ML) | 52.53% | +1.03 | 0.303 |
| Siempre el local | 48.67% | −0.54 | 0.589 |
| El de mayor Elo | 50.60% | +0.25 | 0.806 |
| El de mejor EPA neta | 47.23% | −1.13 | 0.259 |
| El de más descanso | 46.27% | −1.52 | 0.128 |

Ni el mercado mismo supera significativamente al 50% en esa banda. Son partidos genuinamente aleatorios.

**Batería de sesgos documentados en la literatura** (residuo = `home_win − p_mercado`, 2006-2024, 17 hipótesis con corrección de Bonferroni):

| Hipótesis | n | Residuo (pp) | p crudo | p Bonferroni |
|---|---|---|---|---|
| Visitante = equipo "público" (DAL/GB/PIT/NE/SF/KC/PHI/CHI) | 1,196 | −3.43 | 0.011 | 0.182 |
| Juego divisional | 1,744 | −2.13 | 0.051 | 0.864 |
| Local favorito fuerte (>75%) | 834 | +2.28 | 0.072 | 1.000 |
| Césped natural | 2,569 | −1.42 | 0.122 | 1.000 |
| Visitante saliendo de bye | 310 | −3.85 | 0.141 | 1.000 |
| Domo / techo cerrado | 1,258 | −1.58 | 0.222 | 1.000 |
| Lunes por la noche | 324 | −2.94 | 0.246 | 1.000 |
| **Local = Denver (altitud)** | 149 | −1.96 | 0.611 | 1.000 |
| Jueves (semana corta) | 279 | +1.36 | 0.606 | 1.000 |
| Local saliendo de bye | 292 | +0.78 | 0.765 | 1.000 |
| Local underdog claro (<40%) | 1,066 | −0.41 | 0.767 | 1.000 |
| Sitio neutral | 56 | −0.81 | 0.893 | 1.000 |
| Semanas 1-2 | 525 | −0.01 | 0.997 | 1.000 |

**Ninguna sobrevive la corrección por comparaciones múltiples.** La altitud de Denver, que era una de tus candidatas explícitas, da residuo negativo y p=0.61.

Y aunque alguna fuera real: un residuo de 3 pp solo puede cambiar un pick en juegos que estén a menos de 3 pp del 50/50, que son 194 de 2,613 (7.4%). El techo aritmético de explotar perfectamente un sesgo de 3 pp es **+3.7 pp**, y eso exigiría acertar el 100% de esos 194 juegos.

---

### Lista priorizada

**1. Consenso multi-casa lo más cerca posible del kickoff — probabilidad de edge real: ~40%, magnitud: +0.1 a +0.3 pp**

No es una fuente de señal nueva; es reducción de error de medición. El dataset histórico usa la línea de una fuente; `odds_client.py` ya promedia las probabilidades de-viggeadas de todas las casas disponibles. Esa es la mejor idea del proyecto y **ya está implementada**. Asegúrense de que corra lo más tarde posible antes de cada juego individual, y de que si una casa aislada da un valor absurdo se filtre por mediana en vez de media. Es la única "mejora" de esta lista que recomendaría implementar sin reservas, porque no depende de encontrar ineficiencia alguna.

**2. Lesiones de no-QB en tiempo real (línea ofensiva, CB1, edge rusher) — probabilidad de edge real: 10-15%**

Es tu mejor candidata conceptual y la única que justifica trabajo serio. Datos disponibles: `nflverse` publica `injuries` (2009+) y `snap_counts` (2012+); los inactivos oficiales salen 90 minutos antes del kickoff.

**Pero el formato de la competencia mata la mayor parte de la ventaja.** El argumento a favor de las lesiones es que la línea de apertura no las incorpora completamente. La línea de **cierre** sí: los inactivos se publican antes del kickoff y el mercado los mueve en minutos. Como ustedes pueden editar hasta el kickoff, están compitiendo contra la línea que ya vio la misma información. La ventaja que quedaría sería sobre lesiones parciales (jugador activo pero limitado), que es exactamente donde el ruido de medición es máximo.

**Validación exigida antes de adoptarla:** ver protocolo abajo. Mi predicción es que no alcanza significancia.

**3. Estrategia de torneo en vez de maximización de accuracy — probabilidad de valor real: alta, pero depende de información que no tengo**

Esta no es una mejora del modelo; es un cambio de función objetivo, y creo que es la avenida de mayor valor esperado de toda la lista.

Si la competencia es "gana quien tenga el mayor porcentaje" con premio único, **maximizar el accuracy esperado no es óptimo**. Si muchos participantes también le apuestan al favorito del mercado (que es la jugada obvia), todos convergen a ~66% y el ganador se decide por ruido. Desviarse deliberadamente del consenso en los juegos coin-flip baja tu media unas décimas pero **sube tu varianza**, y en un torneo winner-take-all la varianza tiene valor positivo. Es el mismo razonamiento que hace óptimas las picks contrarias en las quinielas de supervivencia.

Si el premio es proporcional, o se reparte entre los primeros lugares, el cálculo se invierte por completo: conviene minimizar varianza, es decir, mercado puro y nada más.

**Lo que necesitan averiguar antes de decidir:** cuántos participantes hay, si el premio es único o escalonado, y si se puede ver el marcador de los rivales durante la temporada. Esa respuesta vale más que cualquier feature que puedan construir.

**4. Ajuste dinámico de varianza a mitad de temporada — probabilidad de valor real: alta si (3) aplica, costo: cero**

Corolario de lo anterior. Si pueden ver el marcador de los rivales: yendo adelante en las últimas semanas, minimicen varianza (mercado puro, sin desviaciones). Yendo atrás, aumenten varianza (contrarian en los coin-flips). Es matemáticamente sólido, no requiere modelar nada, y es gratis.

**5. Stacking / meta-learner / bayesiano jerárquico / redes neuronales — probabilidad de edge real: <5%**

Ya lo probé por ustedes. Regresión logística walk-forward sobre `[logit(p_mercado), logit(p_modelo)]`, entrenada expansivamente 2016-2024: **67.65% vs 67.55% del mercado, gana 10 pierde 8, McNemar p=0.815.** Mezcla en espacio log-odds con peso fijo: igual o peor que el promedio lineal en todos los pesos probados.

El problema no es la forma funcional del blend. Es que no hay señal residual que mezclar. Ninguna arquitectura arregla eso, y modelos con más parámetros solo aumentan el riesgo de que el barrido encuentre ruido que parezca señal.

**6. Tendencias de árbitros — probabilidad de edge real: <5%**

La columna `referee` existe en el schedule. Los efectos de árbitro documentados en la literatura son sobre totales de puntos y tasas de penalización, no sobre quién gana. Con ~15-20 juegos por árbitro por temporada, el error estándar de cualquier efecto estimado supera cualquier efecto plausible. No lo recomiendo.

**7. Margen de victoria en el Elo — probabilidad de edge real: <5% sobre el resultado final**

Mejoraría el Elo como estimador (es la diferencia entre el Elo ingenuo y el de 538). Pero el Elo entra al ensemble con 10% de peso y solo puede actuar en la banda coin-flip, donde ya vimos que el Elo acierta 50.6%. Mejorar un componente que no mueve el resultado no mueve el resultado.

---

## Protocolo de validación (para cualquier candidata que decidan probar)

Esto responde a tu punto 4, y contiene el número más incómodo del informe.

**Split:** walk-forward expansivo idéntico al actual, pero con **holdout final intacto**. Elijan hiperparámetros y pesos usando solo 2015-2020. No toquen 2021-2024 hasta tener la configuración congelada. Una sola evaluación en el holdout, sin vuelta atrás.

**Prueba estadística:** McNemar sobre pares discordantes (prueba binomial exacta, dos colas) comparando el modelo nuevo contra el actual. **No comparen accuracies globales** — con predicciones tan correlacionadas, la diferencia de accuracies tiene un error estándar mucho menor que el de dos muestras independientes, y compararlas directamente da falsos positivos. La prueba tiene que ser sobre los juegos donde los picks difieren.

**Criterio de adopción:** p < 0.01, no p < 0.05. Están haciendo comparaciones múltiples implícitas a lo largo de todo el proyecto; ya llevan al menos 6 avenidas probadas.

**Tamaño de muestra mínimo — el número incómodo:**

Para detectar una mejora real de **+1 pp** con 80% de poder y α=0.05, cuando la modificación cambia el pick en el 10% de los juegos:

```
Se necesitan D ≥ 784 pares discordantes  →  n ≥ 7,840 juegos  ≈  29 temporadas de NFL
```

Solo existen **19 temporadas** con cobertura de moneyline (2006-2024), y de esas ~10 son utilizables como test.

**Implicación directa:** una mejora genuina de +1 pp es *matemáticamente imposible de verificar* con los datos que existen en el mundo. Cualquier "mejora" de ese orden que aparezca en el backtest es indistinguible de ruido, y siempre lo será. Esto no es pesimismo: es el poder estadístico del diseño experimental disponible.

Para que una mejora sea detectable con 10 temporadas necesitaría ser de **+2.5 pp o más**. Eso significaría vencer a la línea de cierre por un margen que nadie en la industria consigue.

---

## (c) Veredicto sobre el 70%

### Por modelado: no. Y no está cerca.

La aritmética:

- Base honesta sobre el 100% de los juegos: **66.4%**.
- 70% requiere **+3.6 pp = 10 juegos extra** en una temporada de 272.
- Los únicos juegos donde un modelo puede intervenir sin destruir valor son los ~42 por temporada dentro de la banda coin-flip.
- Sacar 10 juegos netos de ahí exige pasar de **52.5% a 76.1%** en partidos que son literalmente monedas al aire.

Eso implicaría que la línea de cierre está equivocada por ~24 puntos porcentuales precisamente en su región mejor calibrada. No hay nada en la literatura de mercados deportivos que sugiera algo remotamente parecido: los apostadores profesionales que ganan de forma sostenida lo hacen con 53-55% contra el spread, lo que traducido a accuracy straight-up es una fracción de punto porcentual.

Y hay una ironía en el formato: la posibilidad de editar hasta el kickoff parece una ventaja, pero significa que compiten contra la **línea de cierre**, que es el estimador más eficiente conocido para este problema. Renunciaron a la única ventaja estructural que tiene un apostador profesional, que es la selectividad — poder abstenerse en los juegos difíciles. Este formato los obliga a jugar precisamente donde no hay edge.

### Por suerte: sí, y con probabilidad nada despreciable.

Distribución histórica de la accuracy del favorito de mercado por temporada (2010-2024):

- Media: **66.63%**
- Desviación estándar entre temporadas: **3.14 pp** (mayor que la binomial pura de 2.86 pp — hay varianza real a nivel de temporada, no solo muestreo)
- Rango observado 2015-2024: **61.7% – 71.4%**

| Objetivo | Probabilidad de alcanzarlo en una temporada |
|---|---|
| ≥ 68% | ~33% |
| **≥ 70%** | **~14%** |
| ≥ 72% | ~4% |

Tres de las últimas quince temporadas superaron el 70%: 2013 (71.8%), 2017 (71.4%), 2024 (71.3%).

### Lo que esto significa en la práctica

**La diferencia entre el modelo actual y el mercado puro vale ~2 juegos por década. La diferencia entre una temporada afortunada y una desafortunada vale ~30 juegos.** El resultado de la competencia está determinado casi por completo por qué temporada les toque, no por la calidad del modelo. Un IC del 95% para una temporada de 272 juegos va aproximadamente de **60.8% a 72.0%**.

Dicho de la forma más útil posible: ya construyeron un sistema que captura esencialmente todo el accuracy que es capturable. El techo no es un problema de ingeniería que les falte resolver; es una propiedad del mercado contra el que están compitiendo.

---

## Recomendaciones operativas

1. **Arreglar los bugs 1-4.** Es una tarde de trabajo. No van a subir el accuracy, pero eliminan modos de falla en producción, que es el único lugar donde todavía se puede perder de verdad. El `fillna(0.0)` en particular puede producir un pick catastróficamente equivocado en un juego real.

2. **Decidir el peso `w` por convicción, no por backtest.** Estadísticamente, w=0.90 y w=1.00 son indistinguibles (p=0.83). Si dejan 0.90, que sea porque creen que el modelo aporta algo, no porque el barrido lo "encontró". Yo dejaría w=0.95: mantiene el modelo como desempate en los pick'em exactos y limita el daño potencial.

3. **Poner el esfuerzo restante en robustez operativa, no en features.** Verificación de que ningún juego se quede sin predicción, manejo explícito de pick'em, alertas si el feed de odds falla, corrida lo más tardía posible antes de cada kickoff, filtrado por mediana entre casas. Eso vale más que cualquier feature nueva.

4. **Averiguar la estructura de premios y el número de participantes antes de decidir nada más.** Si es winner-take-all contra muchos rivales que también usan el mercado, la estrategia óptima no es maximizar el accuracy esperado, y esa decisión vale más que todo lo demás en esta lista junta.

5. **Calibrar expectativas con quien haya que calibrarlas.** Si hay dinero de la familia de por medio, lo más valioso que puede entregar este informe no es una mejora: es que nadie apueste esperando 70% cuando la expectativa real es 66.4% ± 5.6 pp.
