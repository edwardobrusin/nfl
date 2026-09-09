"""
run_pipeline.py — Corre todo el pipeline en el orden correcto, en un
solo comando.

Uso normal (todo, de principio a fin):
    python3 run_pipeline.py

Solo actualizar picks de la semana (sin reentrenar el modelo):
    python3 run_pipeline.py --skip-retrain

Solo reentrenar (sin pedir odds en vivo — útil si no tienes tu
ODDS_API_KEY a la mano en este momento):
    python3 run_pipeline.py --skip-weekly

Sin calificar resultados al final:
    python3 run_pipeline.py --skip-check

Se detiene en el primer paso que falle — no sigue corriendo pasos
posteriores con datos a medias.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

RETRAIN_STEPS = [
    "data_ingestion.py",
    "pbp_ingestion.py",
    "features.py",
    "qb_ratings.py",
    "train_final_model.py",
]
WEEKLY_STEP = "weekly_pipeline.py"
CHECK_STEP = "check_results.py"


def run_step(script_name: str):
    script_path = SCRIPT_DIR / script_name
    print(f"\n{'=' * 70}\n▶ Corriendo {script_name}\n{'=' * 70}")
    start = time.time()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(SCRIPT_DIR))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ {script_name} falló (código de salida {result.returncode}) tras "
              f"{elapsed:.1f}s. Deteniendo el pipeline aquí — no se corren los pasos "
              f"siguientes con datos a medias.")
        sys.exit(result.returncode)

    print(f"✅ {script_name} terminó OK en {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corre el pipeline completo de NFL Predictor.")
    parser.add_argument("--skip-retrain", action="store_true",
                         help="Salta data_ingestion/pbp_ingestion/features/qb_ratings/train_final_model")
    parser.add_argument("--skip-weekly", action="store_true",
                         help="Salta weekly_pipeline.py (no pide odds en vivo)")
    parser.add_argument("--skip-check", action="store_true",
                         help="Salta check_results.py (no actualiza el track record)")
    args = parser.parse_args()

    steps = []
    if not args.skip_retrain:
        steps += RETRAIN_STEPS
    if not args.skip_weekly:
        if not os.environ.get("ODDS_API_KEY"):
            print("⚠️  ADVERTENCIA: no encuentro la variable de entorno ODDS_API_KEY en esta "
                  "sesión de terminal. weekly_pipeline.py va a fallar al pedir odds en vivo.")
            print("   Revisa la sección 3 de GUIA_PASO_A_PASO.md, o corre con --skip-weekly "
                  "si por ahora solo quieres reentrenar/calificar resultados.\n")
        steps += [WEEKLY_STEP]
    if not args.skip_check:
        steps += [CHECK_STEP]

    if not steps:
        print("No hay nada que correr — revisa las banderas que pasaste (--skip-*).")
        sys.exit(0)

    print(f"Pipeline a correr, en orden: {' → '.join(steps)}")

    overall_start = time.time()
    for step in steps:
        run_step(step)

    total_min = (time.time() - overall_start) / 60
    print(f"\n🎉 Pipeline completo terminado en {total_min:.1f} minutos.")
