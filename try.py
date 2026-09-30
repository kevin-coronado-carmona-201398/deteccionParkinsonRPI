import cv2
import numpy as np
import pandas as pd
import os
import sys

from eye_tracking import EyeTracker


def detect_fixations(
    gaze_history,
    time_window=0.2,
    dispersion_thresh=5
):
    """
    Detecta fijaciones mediante un algoritmo de dispersión.

    Corregido respecto al código original:
    timestamps ahora es un numpy array y permite
    comparaciones vectoriales.
    """

    if not gaze_history:
        return []

    timestamps = np.array(
        [gaze[2] for gaze in gaze_history],
        dtype=float
    )

    coords = np.array(
        [
            [gaze[0], gaze[1]]
            for gaze in gaze_history
        ],
        dtype=float
    )

    fixations = []

    for i in range(len(gaze_history)):

        end_time = (
            timestamps[i] +
            time_window
        )

        mask = (
            (timestamps >= timestamps[i]) &
            (timestamps <= end_time)
        )

        window_coords = coords[mask]

        if len(window_coords) > 1:

            dispersion = np.max(
                np.ptp(
                    window_coords,
                    axis=0
                )
            )

            if dispersion < dispersion_thresh:

                fixations.append(
                    (
                        timestamps[i],
                        window_coords.mean(axis=0)
                    )
                )

    return fixations


def process_video(
    video_path,
    output_path="eye_metrics_output.csv"
):
    """
    Procesa un vídeo utilizando exactamente el mismo
    EyeTracker que posteriormente se utilizará con la cámara
    en vivo.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            f"[ERROR] Could not open video: {video_path}"
        )

        return False

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    print(
        f"[INFO] Processing: {video_path}"
    )

    print(
        f"[INFO] Video FPS: {fps:.2f}"
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"[INFO] Total frames: {total_frames}"
    )

    tracker = EyeTracker()

    gaze_history = []
    rows = []

    frame_num = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            # ------------------------------------------------
            # Timestamp basado en FPS real del vídeo.
            # ------------------------------------------------

            timestamp = frame_num / fps

            metrics = tracker.process_frame(
                frame,
                timestamp
            )

            if metrics is not None:

                gaze_history.append(
                    (
                        metrics["gaze_x"],
                        metrics["gaze_y"],
                        metrics["timestamp"]
                    )
                )

                rows.append(metrics)

            frame_num += 1

            if frame_num % 100 == 0:

                if total_frames > 0:

                    progress = (
                        frame_num /
                        total_frames
                    ) * 100

                    print(
                        f"[INFO] Progress: "
                        f"{progress:.1f}% "
                        f"({frame_num}/{total_frames})"
                    )

    finally:

        cap.release()
        tracker.close()

    if not rows:

        print(
            "[ERROR] No face landmarks detected."
        )

        return False

    # ========================================================
    # FIJACIONES
    # ========================================================

    fixations = detect_fixations(
        gaze_history
    )

    for row in rows:
        row["fixation"] = int(
            any(
                abs(
                    row["timestamp"] -
                    fixation[0]
                ) < 0.1
                for fixation in fixations
            )
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(rows)

    # Mantener compatibilidad con el CSV original.
    desired_columns = [
        "timestamp",
        "gaze_x",
        "gaze_y",
        "Eye aspect Ratio",
        "blink",
        "saccade_velocity",
        "fixation",
        "pupil_size",

        "left_pupil_x",
        "left_pupil_y",
        "left_pupil_diameter",

        "right_pupil_x",
        "right_pupil_y",
        "right_pupil_diameter",

        "PoR_binocular_x",
        "PoR_binocular_y",

        "Point of Regard Right X",
        "Point of Regard Right Y",

        "Point of Regard Left X",
        "Point of Regard Left Y",

        "Category Binocular",
        "Index Binocular"
    ]

    # Solo seleccionar columnas que existan.
    columns = [
        column
        for column in desired_columns
        if column in df.columns
    ]

    df = df[columns]

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"[SUCCESS] CSV saved: {output_path}"
    )

    print(
        f"[INFO] Frames read: {frame_num}"
    )

    print(
        f"[INFO] Valid eye-tracking rows: {len(df)}"
    )

    print(
        f"[INFO] Blinks: {tracker.blink_count}"
    )

    print(
        f"[INFO] Fixations: {len(fixations)}"
    )

    return True


if __name__ == "__main__":

    input_video = os.getenv(
        "INPUT_VIDEO",
        "camera_record.mp4"
    )

    output_csv = os.getenv(
        "OUTPUT_CSV",
        "eye_metrics_output.csv"
    )

    success = process_video(
        input_video,
        output_csv
    )

    sys.exit(
        0 if success else 1
    )