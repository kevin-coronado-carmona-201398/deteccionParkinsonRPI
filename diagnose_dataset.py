#!/usr/bin/env python3
"""
Diagnóstico del dataset de eye tracking.

Este script NO modifica ningún archivo.
Analiza:
- estructura de columnas
- tipos de datos
- valores faltantes
- rangos
- estadísticas básicas
- timestamps
- frecuencia aproximada de muestreo
- diferencias entre pacientes (P_) y controles (N_)
"""

from pathlib import Path
import pandas as pd
import numpy as np


TRAINING_DIR = Path("training_data")


def detect_column(df, candidates):
    """Busca una columna ignorando mayúsculas/minúsculas y espacios."""
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def analyze_file(path):
    print("\n" + "=" * 80)
    print(f"ARCHIVO: {path.name}")
    print("=" * 80)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] No se pudo leer: {e}")
        return None

    print(f"Filas:    {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    print("\nColumnas:")
    for col in df.columns:
        print(f"  - {col}")

    # ------------------------------------------------------------------
    # Columnas importantes
    # ------------------------------------------------------------------

    important = {
        "gaze_x": ["gaze_x"],
        "gaze_y": ["gaze_y"],
        "saccade_velocity": ["saccade_velocity"],
        "pupil_size": ["pupil_size"],
        "blink": ["blink"],
        "timestamp": ["timestamp"],
        "recording_time": ["RecordingTime [ms]", "recordingtime [ms]"],
        "category": ["category_binocular", "category binocular"],
    }

    print("\n" + "-" * 80)
    print("VARIABLES IMPORTANTES")
    print("-" * 80)

    results = {}

    for name, candidates in important.items():
        col = detect_column(df, candidates)

        if col is None:
            print(f"\n{name}: NO ENCONTRADA")
            results[name] = None
            continue

        series = df[col]

        print(f"\n{name}: '{col}'")
        print(f"  dtype: {series.dtype}")
        print(f"  valores faltantes: {series.isna().sum():,} "
              f"({series.isna().mean() * 100:.2f}%)")

        numeric = pd.to_numeric(series, errors="coerce")

        if numeric.notna().sum() > 0:
            valid = numeric.dropna()

            print(f"  numéricos válidos: {len(valid):,}")
            print(f"  mínimo:  {valid.min():.6g}")
            print(f"  máximo:  {valid.max():.6g}")
            print(f"  media:   {valid.mean():.6g}")
            print(f"  mediana: {valid.median():.6g}")
            print(f"  std:     {valid.std():.6g}")

            results[name] = {
                "column": col,
                "numeric": True,
                "min": valid.min(),
                "max": valid.max(),
                "mean": valid.mean(),
                "median": valid.median(),
                "std": valid.std(),
            }
        else:
            print("  No parece ser una columna numérica.")

            unique = series.dropna().astype(str).str.strip().str.lower().unique()

            if len(unique) <= 20:
                print(f"  valores únicos: {list(unique)}")

            results[name] = {
                "column": col,
                "numeric": False,
            }

    # ------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("ANÁLISIS TEMPORAL")
    print("-" * 80)

    timestamp_col = results.get("timestamp")
    recording_col = results.get("recording_time")

    time_values = None
    time_source = None

    if recording_col is not None:
        col = recording_col["column"]
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) > 1:
            time_values = values.to_numpy()
            time_source = f"{col} (ms)"

    elif timestamp_col is not None:
        col = timestamp_col["column"]
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) > 1:
            time_values = values.to_numpy()
            time_source = f"{col} (numérico)"

    if time_values is not None and len(time_values) > 1:

        diffs = np.diff(time_values)

        # Eliminar diferencias inválidas o negativas
        positive_diffs = diffs[diffs > 0]

        if len(positive_diffs) > 0:
            median_dt = np.median(positive_diffs)
            mean_dt = np.mean(positive_diffs)

            print(f"Fuente temporal: {time_source}")
            print(f"Primer timestamp: {time_values[0]:.6f}")
            print(f"Último timestamp: {time_values[-1]:.6f}")
            print(f"Duración aproximada: {time_values[-1] - time_values[0]:.6f}")

            print(f"\nDelta temporal mediano: {median_dt:.6f}")
            print(f"Delta temporal medio:   {mean_dt:.6f}")

            if "ms" in time_source.lower():
                fps = 1000.0 / median_dt
            else:
                fps = 1.0 / median_dt

            print(f"Frecuencia aproximada: {fps:.2f} Hz")

            print(f"Delta mínimo: {positive_diffs.min():.6f}")
            print(f"Delta máximo: {positive_diffs.max():.6f}")

    else:
        print("No se pudo determinar una frecuencia temporal confiable.")

    # ------------------------------------------------------------------
    # Categorías
    # ------------------------------------------------------------------

    category_col = results.get("category")

    if category_col is not None:
        col = category_col["column"]

        print("\n" + "-" * 80)
        print("CATEGORÍAS")
        print("-" * 80)

        categories = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
        )

        total = categories.sum()

        for category, count in categories.items():
            percentage = count / total * 100
            print(f"  {category:25s} {count:8,} ({percentage:6.2f}%)")

    # ------------------------------------------------------------------
    # Outliers
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("VALORES EXTREMOS")
    print("-" * 80)

    for name in ["gaze_x", "gaze_y", "saccade_velocity", "pupil_size"]:

        info = results.get(name)

        if info is None or not info.get("numeric"):
            continue

        col = info["column"]
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) == 0:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((values < lower) | (values > upper)).sum()

        print(
            f"{name:20s} "
            f"outliers IQR: {outliers:,} "
            f"({outliers / len(values) * 100:.2f}%)"
        )

    return df


def main():

    if not TRAINING_DIR.exists():
        print(f"[ERROR] No existe: {TRAINING_DIR}")
        return

    files = sorted(TRAINING_DIR.glob("*.csv"))

    if not files:
        print(f"[ERROR] No se encontraron CSV en {TRAINING_DIR}")
        return

    patient_files = [f for f in files if f.name.startswith("P_")]
    control_files = [f for f in files if f.name.startswith("N_")]
    other_files = [
        f for f in files
        if not f.name.startswith("P_")
        and not f.name.startswith("N_")
    ]

    print("=" * 80)
    print("DIAGNÓSTICO DEL DATASET")
    print("=" * 80)

    print(f"Total CSV:       {len(files)}")
    print(f"Pacientes P_:    {len(patient_files)}")
    print(f"Controles N_:    {len(control_files)}")
    print(f"Otros archivos:  {len(other_files)}")

    # Analizar uno de cada grupo
    if patient_files:
        analyze_file(patient_files[0])

    if control_files:
        analyze_file(control_files[0])

    # ------------------------------------------------------------------
    # Comparación resumida entre todos los archivos
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("COMPARACIÓN GLOBAL")
    print("=" * 80)

    summary = []

    for path in files:

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        row = {
            "archivo": path.name,
            "grupo": "P" if path.name.startswith("P_") else
                     "N" if path.name.startswith("N_") else "?",
            "filas": len(df),
        }

        for name, candidates in {
            "gaze_x": ["gaze_x"],
            "gaze_y": ["gaze_y"],
            "saccade_velocity": ["saccade_velocity"],
            "pupil_size": ["pupil_size"],
        }.items():

            col = detect_column(df, candidates)

            if col is not None:
                values = pd.to_numeric(df[col], errors="coerce").dropna()

                if len(values) > 0:
                    row[f"{name}_median"] = values.median()
                    row[f"{name}_mean"] = values.mean()
                else:
                    row[f"{name}_median"] = np.nan
                    row[f"{name}_mean"] = np.nan
            else:
                row[f"{name}_median"] = np.nan
                row[f"{name}_mean"] = np.nan

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("FIN DEL DIAGNÓSTICO")
    print("=" * 80)


if __name__ == "__main__":
    main()