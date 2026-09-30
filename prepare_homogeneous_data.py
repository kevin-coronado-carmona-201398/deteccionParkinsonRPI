#!/usr/bin/env python3
"""
Construcción de un dataset homogéneo para el prototipo.

NO modifica training_data/.

Características finales:

    1. gaze_x
    2. gaze_y
    3. gaze_velocity
    4. pupil_size
    5. blink

Proceso:

    CSV original
        ↓
    reconstrucción temporal robusta
        ↓
    limpieza de NaN
        ↓
    normalización por grabación
        ↓
    remuestreo a 30 Hz
        ↓
    ventanas de 10 segundos
        ↓
    archivos NPZ
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("training_data")
OUTPUT_DIR = Path("homogeneous_data")

TARGET_HZ = 30.0
WINDOW_SECONDS = 10.0
WINDOW_SIZE = int(TARGET_HZ * WINDOW_SECONDS)


# ================================================================
# UTILIDADES
# ================================================================

def find_column(df, *names):
    """
    Busca una columna ignorando mayúsculas/minúsculas.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in names:

        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def parse_timestamp(value):
    """
    Convierte:

        HH:MM:SS:ms

    a segundos.

    También acepta valores numéricos.
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    parts = text.split(":")

    if len(parts) == 4:

        try:

            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            milliseconds = int(parts[3])

            return (
                hours * 3600
                + minutes * 60
                + seconds
                + milliseconds / 1000.0
            )

        except (ValueError, TypeError):
            return np.nan

    try:
        return float(value)

    except (ValueError, TypeError):
        return np.nan


def reconstruct_time(df):
    """
    Reconstruye una línea temporal robusta.

    Los timestamps de algunos archivos contienen saltos anómalos.
    En lugar de confiar en la mediana de todos los deltas, buscamos
    la cadencia dominante entre los intervalos plausibles.

    Para los archivos P_*:
        timestamp = HH:MM:SS:ms

    Para los archivos N_*:
        timestamp = segundos

    La línea temporal final se reconstruye usando la cadencia
    dominante de las muestras.
    """

    timestamp_col = find_column(
        df,
        "timestamp"
    )

    if timestamp_col is None:
        raise ValueError(
            "No existe la columna timestamp."
        )

    # ------------------------------------------------------------
    # Convertir timestamp a segundos
    # ------------------------------------------------------------

    raw_time = df[timestamp_col].apply(
        parse_timestamp
    )

    values = raw_time.to_numpy(
        dtype=float
    )

    valid = np.isfinite(values)

    if valid.sum() < 2:
        raise ValueError(
            "No hay suficientes timestamps válidos."
        )

    # ------------------------------------------------------------
    # Calcular diferencias entre muestras
    # ------------------------------------------------------------

    deltas = np.diff(values)

    positive_deltas = deltas[
        np.isfinite(deltas)
        & (deltas > 0)
    ]

    if len(positive_deltas) == 0:
        raise ValueError(
            "No se pudo determinar la cadencia temporal."
        )

    # ------------------------------------------------------------
    # Buscar la cadencia dominante
    #
    # Intervalos plausibles:
    #
    #   0.5 ms  -> 2000 Hz
    #   4 ms    -> 250 Hz
    #   17 ms   -> ~59 Hz
    #   100 ms  -> 10 Hz
    #
    # Los saltos de segundos/minutos quedan fuera.
    # ------------------------------------------------------------

    plausible = positive_deltas[
        (positive_deltas >= 0.0005)
        & (positive_deltas <= 0.100)
    ]

    if len(plausible) < 10:
        raise ValueError(
            "No se encontraron suficientes "
            "intervalos temporales plausibles."
        )

    # Agrupar pequeñas variaciones.
    rounded = np.round(
        plausible,
        4
    )

    unique_values, counts = np.unique(
        rounded,
        return_counts=True
    )

    dominant_index = np.argmax(
        counts
    )

    dominant_dt = float(
        unique_values[dominant_index]
    )

    if dominant_dt <= 0:
        raise ValueError(
            "Cadencia temporal inválida."
        )

    # ------------------------------------------------------------
    # Detectar anomalías
    # ------------------------------------------------------------

    anomaly_threshold = max(
        dominant_dt * 5.0,
        0.020
    )

    anomalies = positive_deltas[
        np.abs(
            positive_deltas
            - dominant_dt
        ) > anomaly_threshold
    ]

    frequency = 1.0 / dominant_dt

    print(
        f"  Delta dominante: "
        f"{dominant_dt * 1000:.3f} ms"
    )

    print(
        f"  Frecuencia estimada: "
        f"{frequency:.2f} Hz"
    )

    print(
        f"  Intervalos plausibles: "
        f"{len(plausible):,}"
    )

    print(
        f"  Saltos/anomalías detectadas: "
        f"{len(anomalies):,}"
    )

    # ------------------------------------------------------------
    # Reconstrucción temporal
    # ------------------------------------------------------------

    n = len(values)

    reconstructed = (
        np.arange(
            n,
            dtype=float
        )
        * dominant_dt
    )

    print(
        f"  Duración reconstruida: "
        f"{reconstructed[-1]:.2f} s"
    )

    return reconstructed


def fill_nan(values):
    """
    Interpola valores faltantes.
    """

    series = pd.Series(
        values
    )

    return (
        series
        .interpolate(
            method="linear",
            limit_direction="both"
        )
        .fillna(
            series.median()
        )
        .fillna(0)
        .to_numpy()
    )


def robust_zscore(values):
    """
    Normalización robusta mediante mediana y MAD.

    Finalmente se limita a [-5, 5] para evitar que outliers
    extremos dominen el modelo.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    median = np.nanmedian(
        values
    )

    mad = np.nanmedian(
        np.abs(
            values - median
        )
    )

    scale = 1.4826 * mad

    if (
        not np.isfinite(scale)
        or scale < 1e-8
    ):

        scale = np.nanstd(
            values
        )

    if (
        not np.isfinite(scale)
        or scale < 1e-8
    ):

        scale = 1.0

    normalized = (
        values - median
    ) / scale

    return np.clip(
        normalized,
        -5.0,
        5.0
    )


def interpolate_signal(
    time,
    values,
    new_time
):
    """
    Interpolación lineal.
    """

    valid = (
        np.isfinite(time)
        & np.isfinite(values)
    )

    if valid.sum() < 2:
        return np.zeros_like(
            new_time
        )

    t = time[valid]
    v = values[valid]

    unique_t, indices = np.unique(
        t,
        return_index=True
    )

    unique_v = v[indices]

    if len(unique_t) < 2:

        return np.full_like(
            new_time,
            unique_v[0]
        )

    return np.interp(
        new_time,
        unique_t,
        unique_v
    )


# ================================================================
# PROCESAMIENTO
# ================================================================

def process_file(path):

    print("\n" + "-" * 70)
    print(f"Procesando: {path.name}")
    print("-" * 70)

    df = pd.read_csv(path)

    # ------------------------------------------------------------
    # Columnas comunes
    # ------------------------------------------------------------

    gaze_x_col = find_column(
        df,
        "gaze_x"
    )

    gaze_y_col = find_column(
        df,
        "gaze_y"
    )

    pupil_col = find_column(
        df,
        "pupil_size"
    )

    blink_col = find_column(
        df,
        "blink"
    )

    required = {
        "gaze_x": gaze_x_col,
        "gaze_y": gaze_y_col,
        "pupil_size": pupil_col,
        "blink": blink_col,
    }

    missing = [
        name
        for name, col in required.items()
        if col is None
    ]

    if missing:

        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    # ------------------------------------------------------------
    # Tiempo reconstruido
    # ------------------------------------------------------------

    time = reconstruct_time(
        df
    )

    # ------------------------------------------------------------
    # Señales
    # ------------------------------------------------------------

    gaze_x = pd.to_numeric(
        df[gaze_x_col],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    gaze_y = pd.to_numeric(
        df[gaze_y_col],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    pupil = pd.to_numeric(
        df[pupil_col],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    blink = pd.to_numeric(
        df[blink_col],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    # ------------------------------------------------------------
    # NaN
    # ------------------------------------------------------------

    gaze_x = fill_nan(
        gaze_x
    )

    gaze_y = fill_nan(
        gaze_y
    )

    pupil = fill_nan(
        pupil
    )

    blink = np.nan_to_num(
        blink,
        nan=0.0
    )

    blink = (
        blink >= 0.5
    ).astype(float)

    # ------------------------------------------------------------
    # Normalización
    # ------------------------------------------------------------

    gaze_x = robust_zscore(
        gaze_x
    )

    gaze_y = robust_zscore(
        gaze_y
    )

    pupil = robust_zscore(
        pupil
    )

    # ------------------------------------------------------------
    # Remuestreo a 30 Hz
    # ------------------------------------------------------------

    duration = time[-1]

    if duration < WINDOW_SECONDS:

        print(
            f"  [WARNING] Duración insuficiente: "
            f"{duration:.2f} s"
        )

        return []

    new_time = np.arange(
        0.0,
        duration,
        1.0 / TARGET_HZ
    )

    resampled_x = interpolate_signal(
        time,
        gaze_x,
        new_time
    )

    resampled_y = interpolate_signal(
        time,
        gaze_y,
        new_time
    )

    resampled_pupil = interpolate_signal(
        time,
        pupil,
        new_time
    )

    # Blink mediante vecino más cercano.

    indices = np.searchsorted(
        time,
        new_time
    )

    indices = np.clip(
        indices,
        0,
        len(blink) - 1
    )

    resampled_blink = blink[
        indices
    ]

    # ------------------------------------------------------------
    # Gaze velocity
    # ------------------------------------------------------------

    dt = 1.0 / TARGET_HZ

    dx = np.diff(
        resampled_x,
        prepend=resampled_x[0]
    )

    dy = np.diff(
        resampled_y,
        prepend=resampled_y[0]
    )

    gaze_velocity = (
        np.sqrt(
            dx ** 2
            + dy ** 2
        )
        / dt
    )

    gaze_velocity = robust_zscore(
        gaze_velocity
    )

    # ------------------------------------------------------------
    # Matriz final
    # ------------------------------------------------------------

    features = np.column_stack([
        resampled_x,
        resampled_y,
        gaze_velocity,
        resampled_pupil,
        resampled_blink,
    ]).astype(
        np.float32
    )

    # ------------------------------------------------------------
    # Ventanas de 10 segundos
    # ------------------------------------------------------------

    windows = []

    for start in range(
        0,
        len(features) - WINDOW_SIZE + 1,
        WINDOW_SIZE
    ):

        end = start + WINDOW_SIZE

        window = features[
            start:end
        ]

        if window.shape == (
            WINDOW_SIZE,
            5
        ):

            windows.append(
                window
            )

    print(
        f"  Ventanas generadas: "
        f"{len(windows)}"
    )

    return windows


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print("CONSTRUCCIÓN DE DATASET HOMOGÉNEO")
    print("=" * 80)

    if not INPUT_DIR.exists():

        print(
            f"[ERROR] No existe {INPUT_DIR}"
        )

        return

    # Crear carpeta de salida.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    if not files:

        print(
            "[ERROR] No hay CSV."
        )

        return

    total_windows = 0

    for path in files:

        if not (
            path.name.startswith("P_")
            or path.name.startswith("N_")
        ):
            continue

        label = (
            1
            if path.name.startswith("P_")
            else 0
        )

        try:

            windows = process_file(
                path
            )

            for index, window in enumerate(
                windows,
                start=1
            ):

                output_name = (
                    f"{path.stem}_"
                    f"W{index:03d}.npz"
                )

                output_path = (
                    OUTPUT_DIR
                    / output_name
                )

                np.savez_compressed(
                    output_path,
                    X=window,
                    y=np.array(
                        label,
                        dtype=np.int64
                    )
                )

                total_windows += 1

        except Exception as error:

            print(
                f"  [ERROR] {error}"
            )

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)

    print(
        f"Ventanas totales: "
        f"{total_windows}"
    )

    print(
        f"Salida: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()