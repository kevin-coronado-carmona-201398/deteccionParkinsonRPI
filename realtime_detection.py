import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from recording import open_camera
from eye_tracking import EyeTracker


# ============================================================
# CONFIGURACIÓN
# ============================================================

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

# El modelo fue entrenado con ventanas de 300 muestras.
WINDOW_SIZE = 300

# Ejecutar una nueva predicción cada N muestras.
PREDICT_EVERY = 15

# Número de predicciones consecutivas utilizadas para suavizar
# el resultado mostrado.
PREDICTION_SMOOTHING = 5

# Umbrales de clasificación.
# > 0.60  -> Parkinson
# < 0.40  -> Control
# entre ambos -> Incertidumbre
PARKINSON_THRESHOLD = 0.60
CONTROL_THRESHOLD = 0.40

MODELS_DIR = "saved_models"

# Si se deja vacío, se selecciona automáticamente el .keras
# más reciente.
MODEL_TIMESTAMP = os.getenv("MODEL_TIMESTAMP", "")


# ============================================================
# CARGA DEL MODELO
# ============================================================

def load_lstm_model():
    """
    Carga el modelo LSTM de Keras.

    Si MODEL_TIMESTAMP está definido, intenta cargar
    específicamente ese modelo.

    Ejemplo:

        MODEL_TIMESTAMP=20260930_104209

    Si está vacío, selecciona el archivo .keras más reciente
    dentro de saved_models/.
    """

    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(
            f"No existe el directorio de modelos: {MODELS_DIR}"
        )

    model_files = [
        os.path.join(MODELS_DIR, filename)
        for filename in os.listdir(MODELS_DIR)
        if filename.endswith(".keras")
    ]

    if not model_files:
        raise FileNotFoundError(
            f"No se encontraron modelos .keras en '{MODELS_DIR}'"
        )

    # --------------------------------------------------------
    # Modelo específico
    # --------------------------------------------------------

    if MODEL_TIMESTAMP:

        matching = [
            path
            for path in model_files
            if MODEL_TIMESTAMP in os.path.basename(path)
        ]

        if not matching:
            raise FileNotFoundError(
                f"No se encontró un modelo con timestamp "
                f"'{MODEL_TIMESTAMP}'"
            )

        model_path = sorted(matching)[-1]

    # --------------------------------------------------------
    # Modelo más reciente
    # --------------------------------------------------------

    else:

        model_path = max(
            model_files,
            key=os.path.getmtime
        )

    print(
        f"[MODEL] Loading LSTM: {model_path}"
    )

    try:

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not load model '{model_path}': {error}"
        ) from error

    # --------------------------------------------------------
    # Validación de arquitectura
    # --------------------------------------------------------

    print(
        f"[MODEL] Input shape: {model.input_shape}"
    )

    print(
        f"[MODEL] Output shape: {model.output_shape}"
    )

    expected_shape = (None, WINDOW_SIZE, 5)

    if tuple(model.input_shape) != expected_shape:

        raise ValueError(
            "Modelo incompatible.\n"
            f"Esperado: {expected_shape}\n"
            f"Encontrado: {model.input_shape}"
        )

    print(
        "[MODEL] LSTM loaded successfully."
    )

    return model


# ============================================================
# CONSTRUCCIÓN DE LA SECUENCIA PARA EL LSTM
# ============================================================

def build_lstm_input(feature_buffer):
    """
    Convierte las 300 métricas obtenidas por EyeTracker
    en el tensor esperado por el LSTM.

    Features:

        0 -> gaze_x
        1 -> gaze_y
        2 -> saccade_velocity
        3 -> pupil_size
        4 -> blink

    Resultado:

        (1, 300, 5)
    """

    if len(feature_buffer) != WINDOW_SIZE:
        raise ValueError(
            f"Se requieren exactamente {WINDOW_SIZE} muestras. "
            f"Hay {len(feature_buffer)}."
        )

    sequence = []

    for metrics in feature_buffer:

        blink_value = (
            1.0
            if metrics["blink"] != 0
            else 0.0
        )

        sequence.append([
            float(metrics["gaze_x"]),
            float(metrics["gaze_y"]),
            float(metrics["saccade_velocity"]),
            float(metrics["pupil_size"]),
            blink_value
        ])

    array = np.asarray(
        sequence,
        dtype=np.float32
    )

    expected_shape = (
        WINDOW_SIZE,
        5
    )

    if array.shape != expected_shape:

        raise ValueError(
            f"Forma incorrecta de secuencia: {array.shape}. "
            f"Esperada: {expected_shape}"
        )

    # Añadir dimensión batch.
    array = np.expand_dims(
        array,
        axis=0
    )

    # Resultado:
    # (1, 300, 5)

    return array


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_parkinsons(model, feature_buffer):
    """
    Realiza una predicción.

    Devuelve una probabilidad entre 0 y 1.

    0 -> Control
    1 -> Parkinson
    """

    model_input = build_lstm_input(
        feature_buffer
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )

    score = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    score = float(
        np.clip(score, 0.0, 1.0)
    )

    return score


# ============================================================
# CLASIFICACIÓN
# ============================================================

def classify_score(score):
    """
    Convierte la probabilidad en una categoría visual.
    """

    if score >= PARKINSON_THRESHOLD:

        return "PARKINSON SCORE HIGH"

    if score <= CONTROL_THRESHOLD:

        return "CONTROL SCORE HIGH"

    return "UNCERTAIN"


# ============================================================
# FPS
# ============================================================

class FPSCounter:

    def __init__(self, averaging_window=30):

        self.timestamps = deque(
            maxlen=averaging_window
        )

    def update(self):

        self.timestamps.append(
            time.perf_counter()
        )

    def get_fps(self):

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = (
            self.timestamps[-1]
            - self.timestamps[0]
        )

        if elapsed <= 0:
            return 0.0

        return (
            (len(self.timestamps) - 1)
            / elapsed
        )


# ============================================================
# OVERLAY
# ============================================================

def draw_overlay(
    frame,
    metrics,
    feature_buffer,
    current_score,
    classification,
    fps
):
    """
    Dibuja información de seguimiento e inferencia.
    """

    output = frame.copy()

    # --------------------------------------------------------
    # Información de tracking
    # --------------------------------------------------------

    if metrics is None:

        tracking_text = "EYE TRACKING: NO FACE"

    else:

        tracking_text = "EYE TRACKING: ACTIVE"

    cv2.putText(
        output,
        tracking_text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if metrics is not None else (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Muestras
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"SAMPLES: {len(feature_buffer)}/{WINDOW_SIZE}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Métricas actuales
    # --------------------------------------------------------

    if metrics is not None:

        gaze_x = metrics["gaze_x"]
        gaze_y = metrics["gaze_y"]
        velocity = metrics["saccade_velocity"]
        pupil = metrics["pupil_size"]
        blink = metrics["blink"]

        cv2.putText(
            output,
            f"Gaze: ({gaze_x:.1f}, {gaze_y:.1f})",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Velocity: {velocity:.1f}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Pupil proxy: {pupil:.1f}",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Blink: {blink}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    if current_score is None:

        score_text = "MODEL: WAITING FOR 300 SAMPLES"

    else:

        score_text = (
            f"PARKINSON SCORE: "
            f"{current_score:.3f}"
        )

    cv2.putText(
        output,
        score_text,
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Clasificación
    # --------------------------------------------------------

    if classification is None:

        classification_text = "CLASSIFICATION: WAITING"

    else:

        classification_text = (
            f"CLASSIFICATION: {classification}"
        )

    cv2.putText(
        output,
        classification_text,
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Instrucción
    # --------------------------------------------------------

    cv2.putText(
        output,
        "Press Q to exit",
        (20, output.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PARKINSON EYE TRACKING - REAL TIME")
    print("=" * 60)

    camera = None
    tracker = None

    # --------------------------------------------------------
    # Cargar modelo
    # --------------------------------------------------------

    try:

        model = load_lstm_model()

    except Exception as error:

        print(
            f"[MODEL ERROR] {error}"
        )

        return

    # --------------------------------------------------------
    # Abrir cámara
    # --------------------------------------------------------

    try:

        camera = open_camera(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            camera_index=CAMERA_INDEX
        )

    except Exception as error:

        print(
            f"[CAMERA ERROR] {error}"
        )

        return

    # --------------------------------------------------------
    # Inicializar EyeTracker
    # --------------------------------------------------------

    try:

        tracker = EyeTracker()

    except Exception as error:

        print(
            f"[EYE TRACKING ERROR] {error}"
        )

        camera.release()

        return

    # --------------------------------------------------------
    # Estado
    # --------------------------------------------------------

    feature_buffer = deque(
        maxlen=WINDOW_SIZE
    )

    prediction_history = deque(
        maxlen=PREDICTION_SMOOTHING
    )

    current_score = None
    classification = None

    sample_counter = 0

    fps_counter = FPSCounter()

    # Timestamp relativo al inicio de la ejecución.
    start_time = time.perf_counter()

    print(
        "[MODEL] Real-time inference enabled."
    )

    print(
        "[SYSTEM] Starting real-time processing..."
    )

    # --------------------------------------------------------
    # Bucle principal
    # --------------------------------------------------------

    try:

        while True:

            success, frame = camera.read()

            if not success or frame is None:

                print(
                    "[CAMERA] Frame capture failed."
                )

                break

            fps_counter.update()

            timestamp = (
                time.perf_counter()
                - start_time
            )

            # ------------------------------------------------
            # Eye Tracking
            # ------------------------------------------------

            metrics = tracker.process_frame(
                frame,
                timestamp
            )

            # ------------------------------------------------
            # Agregar solamente frames con rostro válido
            # ------------------------------------------------

            if metrics is not None:

                feature_buffer.append(
                    metrics
                )

                sample_counter += 1

                # ------------------------------------------------
                # Primera predicción:
                # cuando existen 300 muestras.
                #
                # Posteriormente cada 15 muestras.
                # ------------------------------------------------

                if (
                    len(feature_buffer) == WINDOW_SIZE
                    and (
                        sample_counter == WINDOW_SIZE
                        or
                        (
                            sample_counter > WINDOW_SIZE
                            and
                            (
                                sample_counter
                                - WINDOW_SIZE
                            ) % PREDICT_EVERY == 0
                        )
                    )
                ):

                    try:

                        score = predict_parkinsons(
                            model,
                            feature_buffer
                        )

                        prediction_history.append(
                            score
                        )

                        # ------------------------------------------------
                        # Suavizado temporal
                        # ------------------------------------------------

                        current_score = float(
                            np.mean(
                                prediction_history
                            )
                        )

                        classification = classify_score(
                            current_score
                        )

                        print(
                            f"[MODEL] Score: "
                            f"{current_score:.4f} | "
                            f"{classification}"
                        )

                    except Exception as error:

                        print(
                            f"[MODEL ERROR] Prediction failed: {error}"
                        )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            output = draw_overlay(
                frame=frame,
                metrics=metrics,
                feature_buffer=feature_buffer,
                current_score=current_score,
                classification=classification,
                fps=fps_counter.get_fps()
            )

            cv2.imshow(
                "Parkinson Eye Tracking - Real Time",
                output
            )

            # ------------------------------------------------
            # Salir con Q
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(
                    "[SYSTEM] Q pressed. Exiting..."
                )
                break

    except KeyboardInterrupt:

        print(
            "\n[SYSTEM] Interrupted by user."
        )

    finally:

        print(
            "[SYSTEM] Releasing resources..."
        )

        if camera is not None:

            try:
                camera.release()
            except Exception:
                pass

        if tracker is not None:

            try:
                tracker.close()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print(
            "[SYSTEM] Shutdown complete."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()