# Guía paso a paso — NFL Predictor

Esta guía asume que ya tienes el archivo `nfl-predictor.zip` y `PLAN_MAESTRO_NFL_PREDICTOR.md`
descargados. Está pensada para seguirse una vez de principio a fin (setup) y después
volver solo a la sección 5 (uso semanal) cada semana de la temporada.

---

## 0. Requisitos previos

- **Python 3.10 o superior** instalado. Verifica con:
  ```bash
  python3 --version
  ```
  Si no lo tienes, instálalo desde [python.org](https://python.org) antes de continuar.
- **Una terminal** (Terminal en Mac/Linux, PowerShell o CMD en Windows).
- **Una cuenta gratuita en The Odds API** — la vas a crear en el paso 3.

---

## 1. Descomprimir e instalar

```bash
# 1a. Descomprime el zip donde quieras trabajar (ejemplo: tu carpeta de Documentos)
unzip nfl-predictor.zip
cd nfl-predictor

# 1b. Instala las dependencias de Python
pip install -r requirements.txt --break-system-packages
```

Si `pip install` te da un error de `pkg_resources` o `setuptools`, corre esto primero y
reintenta:
```bash
pip install "setuptools<81" --break-system-packages
pip install -r requirements.txt --break-system-packages
```

**Verificación:** corre esto y confirma que no hay errores:
```bash
python3 -c "import pandas, numpy, xgboost, sklearn, requests, joblib; print('Todo instalado correctamente')"
```

---

## 2. Lee el plan maestro y la auditoría (una sola vez, pero no te lo saltes)

Antes de tocar código, abre y lee:
1. `PLAN_MAESTRO_NFL_PREDICTOR.md` — completo, pero si tienes prisa, prioriza las
   secciones 15 (auditoría) y 16 (bandas de confianza) — son las que más cambian cómo
   usas el modelo en la práctica.
2. `AUDITORIA_NFL_PREDICTOR.md` — el reporte original de la auditoría externa.

Esto no es un trámite — el modelo tiene decisiones importantes (por qué w=0.95, por
qué ciertas bandas de confianza importan más que otras) que necesitas entender para
usarlo bien, no solo correrlo.

---

## 3. Consigue tu API key de odds (una sola vez)

1. Ve a [the-odds-api.com](https://the-odds-api.com) y regístrate (plan gratis: 500
   requests/mes — más que suficiente para uso semanal manual, pero ver nota de cuota
   en la sección 8).
2. Copia tu API key desde el dashboard.
3. Expórtala como variable de entorno en tu terminal:

   **Mac/Linux:**
   ```bash
   export ODDS_API_KEY="tu-api-key-aqui"
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:ODDS_API_KEY="tu-api-key-aqui"
   ```

   ⚠️ **Esto solo dura mientras esa ventana de terminal esté abierta.** Si la cierras,
   tienes que volver a correr ese comando la próxima vez. Si quieres que sea
   permanente, agrégalo a tu archivo `.bashrc`, `.zshrc`, o a las variables de entorno
   del sistema en Windows.

**Verificación:**
```bash
python3 -c "import os; print('OK' if os.environ.get('ODDS_API_KEY') else 'FALTA LA API KEY')"
```

---

## 4. Setup inicial de datos y modelo (una sola vez, tarda ~10-15 minutos)

Corre estos comandos EN ORDEN, desde la carpeta `nfl-predictor/src`:

```bash
cd src

# 4a. Descarga y limpia el histórico de partidos (2006-2024)
python3 data_ingestion.py

# 4b. Descarga play-by-play histórico (pesado, ~20MB x 19 temporadas — tarda varios minutos)
python3 pbp_ingestion.py

# 4c. Calcula Elo propio + EPA rolling
python3 features.py

# 4d. Calcula el rating de QB (necesita el pbp del paso 4b)
python3 qb_ratings.py

# 4e. Entrena y guarda el modelo final de producción
python3 train_final_model.py
```

**Qué esperar de cada uno:**

| Comando | Debe terminar con... |
|---|---|
| `data_ingestion.py` | "Guardado: 4780 juegos limpios..." |
| `pbp_ingestion.py` | "Play-by-play total: ~906,000 jugadas..." |
| `features.py` | "Sanity check — accuracy de Elo propio SOLO: 0.62xx" |
| `qb_ratings.py` | "Guardado: .../games_with_qb_features.parquet" |
| `train_final_model.py` | "Guardado: .../model_a_final.joblib" y "...feature_means.joblib" |

Si alguno falla, antes de seguir revisa el mensaje de error completo — casi siempre es
un problema de conexión a internet o de un paquete faltante, no del código.

**No necesitas repetir este paso 4 seguido.** Solo vuelve a correrlo si quieres que el
modelo entrenado se actualice con resultados más recientes ya jugados de la temporada
2026 (ver sección 7). El Elo y la forma de equipos "en vivo" ya se recalculan solos
cada vez que corres `weekly_pipeline.py` (sección 5), sin necesidad de repetir esto.

---

## 5. Uso semanal — esto SÍ lo repites cada vez que quieras generar picks

### 5.1 Cuándo correrlo

Como confirmaste que puedes modificar cada pick justo antes de que arranque ESE juego
en particular (no una sola vez por semana), la estrategia correcta es:

**Corre `weekly_pipeline.py` lo más tarde posible antes de cada partido individual —
no una sola vez el lunes para toda la semana.** Entre más tarde lo corras, la línea de
mercado que capturas tiene más información incorporada (lesiones confirmadas, clima
real, movimientos de dinero) — eso es directamente lo que te da el ~66-67% de accuracy
histórico, no una versión más vieja y menos informada de la misma línea.

En la práctica, esto significa correrlo varias veces por semana: por ejemplo, una vez
el jueves para el Thursday Night Football, otra vez el domingo temprano para los
juegos de la 1pm, otra antes de los de las 4pm, etc.

### 5.2 El comando

Desde `nfl-predictor/src` (con la `ODDS_API_KEY` ya exportada — ver sección 3):

```bash
python3 weekly_pipeline.py
```

### 5.3 Qué hace, en orden (por si algo falla y necesitas saber dónde)

1. Pide las odds actuales a The Odds API.
2. Descarga el schedule más reciente y recalcula el Elo con los resultados ya jugados
   de la temporada actual.
3. Calcula la forma reciente de cada equipo (EPA rolling, con fallback automático a la
   temporada anterior si aún no hay suficientes juegos jugados en la actual).
4. Carga el modelo entrenado y genera la probabilidad del modelo propio.
5. Combina mercado + modelo propio (ensemble, 95% mercado / 5% modelo).
6. Clasifica cada juego en una banda de confianza (ver sección 6).
7. Guarda un CSV en `outputs/weekly_predictions/predictions_{fecha}_{hora}.csv` y lo
   imprime en pantalla.

### 5.4 Cómo leer el CSV resultante

Columnas relevantes:

- **`pick`**: el equipo que el modelo favorece.
- **`confidence`**: qué tan seguro está el modelo (0.5 = totalmente parejo, 1.0 =
  certeza total).
- **`tier`**: la banda de confianza (ver sección 6 — esta es la columna que te dice
  cuándo confiar ciegamente y cuándo vale la pena que apliques tu criterio).
- **`p_home_market`** / **`p_model_a`**: las dos probabilidades por separado, por si
  quieres ver si el mercado y el modelo propio están de acuerdo o no.

---

## 6. Cómo registrar los picks en la app — la guía de intervención manual

Para cada juego del CSV, mira la columna `tier`:

| Tier | Qué significa | Qué hacer |
|---|---|---|
| **1_LOCK** | Favorito claro, ~79% de acierto histórico en esta banda | Regístralo tal cual. No lo cuestiones por "corazonada" — ya probamos que la intuición (rivalidad, revancha, etc.) no le gana al mercado aquí. |
| **2_CONFIADO** | Favorito sólido, ~63% de acierto histórico | Regístralo tal cual, salvo que tengas un dato MUY concreto y reciente (lesión confirmada después de correr el pipeline). |
| **3_PAREJO** | Juego parejo, ~56-58% de acierto histórico | Aquí SÍ vale la pena tu criterio si tienes información fresca (inactive list visto en persona, algo que pasó después de correr el pipeline). No por presentimiento sin dato nuevo. |
| **4_MONEDA_AL_AIRE** | Prácticamente 50/50, ~53% de acierto histórico | Mismo criterio que el tier 3 — es la banda donde ni el modelo tiene ventaja probada, así que tu información de último minuto no compite contra nada, compite contra un empate real. |

**Regla general: tu ventaja real sobre el modelo es tener información MÁS RECIENTE que
la que el pipeline capturó al correr — no tener mejor intuición sobre el juego.** Si
vas a intervenir, pregúntate: "¿sé algo que pasó DESPUÉS de que corrí el pipeline?" Si
la respuesta es sí, interviene. Si es solo una corazonada, no.

---

## 7. Mantenimiento durante la temporada (opcional pero recomendado)

Cada 3-4 semanas, o si notas que el modelo empieza a fallar en equipos que has visto
jugar mucho mejor/peor de lo que refleja, refresca los datos y reentrena:

```bash
cd src
python3 data_ingestion.py      # trae resultados nuevos ya jugados
python3 pbp_ingestion.py       # trae play-by-play nuevo (solo descarga temporadas
                                # que aún no tengas cacheadas)
python3 features.py
python3 qb_ratings.py
python3 train_final_model.py
```

No es obligatorio hacerlo cada semana — el Elo y la forma reciente ya se actualizan
automáticamente dentro de `weekly_pipeline.py` sin necesidad de reentrenar. Esto es
solo para que el modelo XGBoost en sí (no solo el Elo/EPA en vivo) vea ejemplos más
recientes al entrenar.

---

## 8. Problemas comunes y qué hacer

| Problema | Causa probable | Solución |
|---|---|---|
| `ValueError: Falta ODDS_API_KEY` | No exportaste la variable de entorno en esta sesión de terminal | Repite el paso 3.3 — se pierde cada vez que cierras la terminal |
| `AVISO: equipo no reconocido en el mapeo` | The Odds API cambió el nombre de un equipo, o hay un typo | Revisa `src/team_name_mapping.py` y agrega el nombre exacto que aparece en el aviso |
| Se te acaba la cuota de The Odds API (500/mes) | Corriste el pipeline demasiadas veces | El plan gratis alcanza para ~3-4 corridas por juego/semana con margen. Si se te acaba, espera al siguiente mes o considera el plan de pago más barato |
| `No hay juegos próximos disponibles` | Corriste el pipeline muy lejos de cualquier partido, o ya pasó el kickoff de todos los juegos de la semana | Normal, no es error — vuelve a correrlo más cerca del próximo partido |
| Error de conexión / dominio bloqueado | Tu red no tiene acceso a `github.com` o `api.the-odds-api.com` | Revisa tu configuración de red/firewall |
| El pipeline corre pero el CSV sale vacío | Las odds llegaron pero el merge con Elo/schedule no encontró coincidencias | Revisa que el schedule ya tenga cargados los juegos de esta semana (a veces nflverse publica el schedule completo de la temporada con unos días de anticipación, pero puede haber un retraso puntual) |

**Plan B si la API de odds falla justo el día del juego:** entra manualmente a
cualquier casa de apuestas pública (o a un agregador como oddsshark.com) para ver el
moneyline actual, y usa la fórmula de-vig de la sección 4.1 del `PLAN_MAESTRO` para
convertirlo a probabilidad a mano — mejor tener un pick basado en mercado calculado a
mano que perder el juego por completo por una falla técnica.

---

## 9. Checklist rápido para cada semana de la temporada

- [ ] `ODDS_API_KEY` exportada en la terminal que vas a usar
- [ ] Corriste `weekly_pipeline.py` lo más cerca posible del kickoff de cada juego (no
      todo de una vez el lunes)
- [ ] Revisaste la columna `tier` de cada pick antes de registrar en la app
- [ ] En tiers 1-2: registraste el pick del modelo sin cuestionarlo por intuición
- [ ] En tiers 3-4: aplicaste tu criterio SOLO si tenías información más reciente que
      la que el pipeline capturó
- [ ] Guardaste el CSV de esa semana (queda automáticamente en
      `outputs/weekly_predictions/`) por si quieres revisar después qué tan bien
      calibrado estuvo
