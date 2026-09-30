import os
import time
from collections import deque

import cv2
import numpy as np

from recording import open_camera
from eye_tracking import EyeTracker


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Índice utilizado únicamente como fallback
# cuando se utiliza una cámara USB.
CAMERA_INDEX = int(
    os.getenv("CAMERA_INDEX", "0")
)

CAMERA_WIDTH = int(
    os.getenv("CAMERA_WIDTH", "640")
)

CAMERA_HEIGHT = int(
    os.getenv("CAMERA_HEIGHT", "480")
)

CAMERA_FPS = int(
    os.getenv("CAMERA_FPS", "30")
)

# El LSTM original fue entrenado con 300 muestras.
WINDOW_SIZE = 300

# Realizar una predicción cada cierto número de muestras.
# No es necesario ejecutar el modelo en cada frame.
PREDICT_EVERY = 15

# Número de predicciones recientes utilizadas para
# estabilizar visualmente el resultado.
PREDICTION_SMOOTHING = 5

MODELS_DIR = os.getenv(
    "MODELS_DIR",
    "saved_models"
)

MODEL_TIMESTAMP = os.getenv(
    "MODEL_TIMESTAMP",
    ""
)


# ============================================================
# CARGAR MODELO LSTM
# ============================================================

def load_lstm_model(
    models_dir=MODELS_DIR,
    timestamp=MODEL_TIMESTAMP
):
    """
    Carga el modelo LSTM más reciente disponible.

    El modelo original utiliza:

        gaze_x
        gaze_y
        saccade_velocity
        pupil_size
        blink_bin

    con una secuencia de 300 muestras.
    """

    try:

        import tensorflow as tf

    except ImportError:

        print(
            "[WARNING] TensorFlow is not installed."
        )

        return None

    if not os.path.isdir(models_dir):

        print(
            f"[WARNING] Models directory not found: "
            f"{models_dir}"
        )

        return None

    # --------------------------------------------------------
    # Buscar archivos LSTM
    # --------------------------------------------------------

    if timestamp:

        model_filename = (
            f"lstm_{timestamp}.keras"
        )

        model_path = os.path.join(
            models_dir,
            model_filename
        )

        if not os.path.exists(model_path):

            print(
                f"[WARNING] Requested model not found: "
                f"{model_path}"
            )

            return None

    else:

        model_files = [

            filename

            for filename in os.listdir(models_dir)

            if (
                filename.startswith("lstm_")
                and filename.endswith(".keras")
            )
        ]

        if not model_files:

            print(
                "[WARNING] No LSTM models found."
            )

            return None

        # Los nombres contienen timestamp,
        # por lo que el orden lexicográfico funciona.
        model_files.sort(reverse=True)

        model_path = os.path.join(
            models_dir,
            model_files[0]
        )

    # --------------------------------------------------------
    # Cargar modelo
    # --------------------------------------------------------

    print(
        f"[MODEL] Loading LSTM: {model_path}"
    )

    try:

        model = tf.keras.models.load_model(
            model_path
        )

    except Exception as error:

        print(
            f"[ERROR] Could not load LSTM model: "
            f"{error}"
        )

        return None

    # --------------------------------------------------------
    # Verificar forma esperada
    # --------------------------------------------------------

    print(
        f"[MODEL] Input shape: {model.input_shape}"
    )

    print(
        f"[MODEL] Output shape: {model.output_shape}"
    )

    expected_shape = (
        None,
        WINDOW_SIZE,
        5
    )

    if model.input_shape != expected_shape:

        print(
            "[WARNING] The loaded model does not have "
            f"the expected input shape {expected_shape}."
        )

        print(
            "[WARNING] Real-time inference will continue, "
            "but compatibility should be checked."
        )

    print("[MODEL] LSTM loaded successfully.")

    return model


# ============================================================
# PREPARAR SECUENCIA PARA LSTM
# ============================================================

def build_lstm_input(buffer):
    """
    Convierte las últimas 300 métricas en el formato
    esperado por el LSTM.

    Orden de features:

        1. gaze_x
        2. gaze_y
        3. saccade_velocity
        4. pupil_size
        5. blink_bin
    """

    sequence = []

    for metrics in buffer:

        blink_bin = (
            1
            if metrics["blink"] != 0
            else 0
        )

        sequence.append(
            [
                metrics["gaze_x"],
                metrics["gaze_y"],
                metrics["saccade_velocity"],
                metrics["pupil_size"],
                blink_bin
            ]
        )

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    # Esperamos exactamente:
    #
    # (300, 5)

    if sequence.shape != (
        WINDOW_SIZE,
        5
    ):

        raise ValueError(
            "Invalid LSTM sequence shape: "
            f"{sequence.shape}"
        )

    # Añadir dimensión del batch:
    #
    # (1, 300, 5)

    return sequence[np.newaxis, :, :]


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_parkinsons(
    model,
    buffer
):
    """
    Ejecuta una predicción sobre una ventana de 300 muestras.

    La salida sigmoid del modelo se interpreta como el
    score asociado a la clase Parkinson (1), de acuerdo con
    el entrenamiento original.
    """

    if model is None:
        return None

    try:

        X = build_lstm_input(
            buffer
        )

        prediction = model.predict(
            X,
            verbose=0
        )[0][0]

        prediction = float(
            np.clip(
                prediction,
                0.0,
                1.0
            )
        )

        return prediction

    except Exception as error:

        print(
            f"[ERROR] Prediction failed: {error}"
        )

        return None


# ============================================================
# INTERPRETACIÓN DEL SCORE
# ============================================================

def classify_score(score):
    """
    Utiliza los mismos umbrales del repositorio original:

        > 0.60 -> Parkinson
        < 0.40 -> No Parkinson
        0.40-0.60 -> Uncertain
    """

    if score is None:

        return "MODEL UNAVAILABLE"

    if score > 0.60:

        return "PARKINSON SCORE HIGH"

    if score < 0.40:

        return "CONTROL SCORE HIGH"

    return "UNCERTAIN"


# ============================================================
# OVERLAY
# ============================================================

def draw_overlay(
    frame,
    tracking_active,
    metrics,
    sample_count,
    prediction,
    fps
):
    """
    Dibuja información del sistema sobre el feed.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Panel de información
    # --------------------------------------------------------

    panel_height = 185

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, panel_height),
        (0, 0, 0),
        -1
    )

    # Transparencia.
    frame[:] = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    # --------------------------------------------------------
    # Estado de tracking
    # --------------------------------------------------------

    tracking_text = (
        "TRACKING: OK"
        if tracking_active
        else "TRACKING: NO FACE"
    )

    cv2.putText(
        frame,
        tracking_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0)
        if tracking_active
        else (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Muestras
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"SAMPLES: {sample_count}/{WINDOW_SIZE}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Métricas actuales
    # --------------------------------------------------------

    if metrics is not None:

        cv2.putText(
            frame,
            f"GAZE: "
            f"{metrics['gaze_x']:.1f}, "
            f"{metrics['gaze_y']:.1f}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"BLINK: {metrics['blink']}   "
            f"VELOCITY: "
            f"{metrics['saccade_velocity']:.1f}",
            (15, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    if prediction is None:

        model_text = (
            "MODEL: WAITING FOR DATA"
        )

    else:

        model_text = (
            f"MODEL SCORE: "
            f"{prediction * 100:.1f}%"
        )

    cv2.putText(
        frame,
        model_text,
        (15, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    return frame


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print(
        "PARKINSON EYE TRACKING - REAL TIME"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Cámara
    # --------------------------------------------------------

    camera = open_camera(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        camera_index=CAMERA_INDEX
    )

    # --------------------------------------------------------
    # Eye tracker
    # --------------------------------------------------------

    tracker = EyeTracker()

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = load_lstm_model()

    if model is None:

        print(
            "[WARNING] Real-time model inference "
            "is disabled."
        )

    else:

        print(
            "[MODEL] Real-time inference enabled."
        )

    # --------------------------------------------------------
    # Buffer
    # --------------------------------------------------------

    feature_buffer = deque(
        maxlen=WINDOW_SIZE
    )

    prediction_history = deque(
        maxlen=PREDICTION_SMOOTHING
    )

    current_prediction = None

    frame_count = 0

    last_prediction_sample = 0

    start_time = time.perf_counter()

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "[ERROR] Failed to read camera frame."
                )

                break

            frame_count += 1

            # ------------------------------------------------
            # Timestamp REAL de captura/procesamiento.
            # ------------------------------------------------

            timestamp = (
                time.perf_counter()
                - start_time
            )

            # ------------------------------------------------
            # Eye tracking
            # ------------------------------------------------

            metrics = tracker.process_frame(
                frame,
                timestamp
            )

            tracking_active = (
                metrics is not None
            )

            # ------------------------------------------------
            # Guardar métricas
            # ------------------------------------------------

            if metrics is not None:

                feature_buffer.append(
                    metrics
                )

            # ------------------------------------------------
            # Predicción
            # ------------------------------------------------

            if (
                model is not None
                and len(feature_buffer)
                == WINDOW_SIZE
                and (
                    len(feature_buffer)
                    - last_prediction_sample
                    >= PREDICT_EVERY
                )
            ):

                prediction = predict_parkinsons(
                    model,
                    feature_buffer
                )

                if prediction is not None:

                    prediction_history.append(
                        prediction
                    )

                    # Promedio de las últimas
                    # predicciones para reducir
                    # oscilaciones visuales.

                    current_prediction = float(
                        np.mean(
                            prediction_history
                        )
                    )

                last_prediction_sample = len(
                    feature_buffer
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            elapsed = (
                time.perf_counter()
                - start_time
            )

            current_fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            draw_overlay(
                frame,
                tracking_active,
                metrics,
                len(feature_buffer),
                current_prediction,
                current_fps
            )

            # ------------------------------------------------
            # Resultado de clasificación
            # ------------------------------------------------

            if current_prediction is not None:

                result = classify_score(
                    current_prediction
                )

                cv2.putText(
                    frame,
                    result,
                    (
                        15,
                        CAMERA_HEIGHT - 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )

            # ------------------------------------------------
            # Mostrar
            # ------------------------------------------------

            cv2.imshow(
                "Parkinson Eye Tracking - REAL TIME",
                frame
            )

            # ------------------------------------------------
            # Tecla Q
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print(
                    "[SYSTEM] Stopping..."
                )

                break

    finally:

        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(
        "[SYSTEM] Real-time detection stopped."
    )


if __name__ == "__main__":
    main()