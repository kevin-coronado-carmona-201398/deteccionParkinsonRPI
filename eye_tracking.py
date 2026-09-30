import cv2
import mediapipe as mp
import numpy as np
import os


class EyeTracker:
    """
    Procesador de frames para extracción de métricas oculares.

    Compatible con:
        - Raspberry Pi Camera
        - cámara USB
        - archivos de video

    Utiliza MediaPipe Face Landmarker (API Tasks).
    """

    # --------------------------------------------------------
    # Landmarks del contorno ocular.
    #
    # Estos índices son compatibles con la malla facial
    # de MediaPipe utilizada por el algoritmo original.
    # --------------------------------------------------------

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def __init__(
        self,
        blink_threshold=0.2,
        blink_window_size=5,
        blink_cooldown=0.5,
        model_path=None
    ):

        self.blink_threshold = blink_threshold
        self.blink_window_size = blink_window_size
        self.blink_cooldown = blink_cooldown

        # ----------------------------------------------------
        # Localizar modelo Face Landmarker
        # ----------------------------------------------------

        if model_path is None:

            possible_paths = [
                os.path.join(
                    os.path.dirname(__file__),
                    "models",
                    "face_landmarker.task"
                ),
                os.path.join(
                    os.getcwd(),
                    "models",
                    "face_landmarker.task"
                ),
                "models/face_landmarker.task",
                "face_landmarker.task"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                "No se encontró 'face_landmarker.task'. "
                "Colócalo en la carpeta 'models/'."
            )

        print(
            f"[EYE] Loading MediaPipe Face Landmarker: "
            f"{os.path.abspath(model_path)}"
        )

        # ----------------------------------------------------
        # MediaPipe Tasks
        # ----------------------------------------------------

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.face_landmarker = FaceLandmarker.create_from_options(
            options
        )

        print("[EYE] MediaPipe Face Landmarker loaded successfully.")

        # ----------------------------------------------------
        # Estado temporal
        # ----------------------------------------------------

        self.ear_history = []

        self.last_blink_time = -self.blink_cooldown
        self.blink_count = 0

        self.previous_gaze = None
        self.previous_timestamp = None

        self.gaze_history = []

        # ----------------------------------------------------
        # Último resultado de MediaPipe
        #
        # Se conserva para draw_debug() y evita volver a
        # ejecutar MediaPipe sobre el mismo frame.
        # ----------------------------------------------------

        self.last_landmarks = None

        # Timestamp utilizado por MediaPipe VIDEO mode.
        self.last_timestamp_ms = -1

    # ========================================================
    # UTILIDADES
    # ========================================================

    @staticmethod
    def calculate_ear(eye_landmarks):
        """
        Calcula Eye Aspect Ratio (EAR).

        Se utilizan 6 landmarks:
            0 y 3 -> extremos horizontales
            1 y 5 -> distancia vertical
            2 y 4 -> distancia vertical
        """

        A = np.linalg.norm(
            np.array(eye_landmarks[1]) -
            np.array(eye_landmarks[5])
        )

        B = np.linalg.norm(
            np.array(eye_landmarks[2]) -
            np.array(eye_landmarks[4])
        )

        C = np.linalg.norm(
            np.array(eye_landmarks[0]) -
            np.array(eye_landmarks[3])
        )

        if C == 0:
            return 0.0

        return (A + B) / (2.0 * C)

    # ========================================================
    # PROCESAMIENTO DE UN FRAME
    # ========================================================

    def process_frame(self, frame, timestamp):
        """
        Procesa un único frame.

        Parámetros:
            frame:
                Imagen BGR proveniente de OpenCV.

            timestamp:
                Tiempo del frame en segundos.

        Retorna:
            Diccionario con métricas o None si no se detectó
            ningún rostro.
        """

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Convertir a MediaPipe Image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # MediaPipe VIDEO mode requiere timestamps
        # estrictamente crecientes en milisegundos.
        timestamp_ms = int(timestamp * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        # ----------------------------------------------------
        # Detectar landmarks
        # ----------------------------------------------------

        results = self.face_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not results.face_landmarks:
            self.last_landmarks = None
            return None

        landmarks = results.face_landmarks[0]

        self.last_landmarks = landmarks

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # Convertir landmarks normalizados a píxeles
        # ----------------------------------------------------

        left_eye = [
            (
                int(landmarks[i].x * width),
                int(landmarks[i].y * height)
            )
            for i in self.LEFT_EYE
        ]

        right_eye = [
            (
                int(landmarks[i].x * width),
                int(landmarks[i].y * height)
            )
            for i in self.RIGHT_EYE
        ]

        # ====================================================
        # EAR / BLINK
        # ====================================================

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        self.ear_history.append(avg_ear)

        if len(self.ear_history) > self.blink_window_size:
            self.ear_history.pop(0)

        is_blink = 0

        if len(self.ear_history) >= self.blink_window_size:

            low_ear_count = sum(
                ear < self.blink_threshold
                for ear in self.ear_history
            )

            if low_ear_count >= self.blink_window_size // 2:

                if (
                    timestamp - self.last_blink_time
                    > self.blink_cooldown
                ):
                    self.blink_count += 1
                    is_blink = 1
                    self.last_blink_time = timestamp

        # ====================================================
        # "PUPIL" / IRIS PROXY
        # ====================================================
        #
        # IMPORTANTE:
        #
        # Esta característica NO es una medición real de la
        # pupila.
        #
        # Se conserva aquí la lógica original para mantener
        # compatibilidad con el código existente.
        # ====================================================

        lx0, ly0 = left_eye[0]
        lx3, ly3 = left_eye[3]

        rx0, ry0 = right_eye[0]
        rx3, ry3 = right_eye[3]

        left_pupil_x = (lx0 + lx3) / 2.0
        left_pupil_y = (ly0 + ly3) / 2.0

        right_pupil_x = (rx0 + rx3) / 2.0
        right_pupil_y = (ry0 + ry3) / 2.0

        left_pupil_diameter = np.linalg.norm(
            np.array([lx0, ly0]) -
            np.array([lx3, ly3])
        )

        right_pupil_diameter = np.linalg.norm(
            np.array([rx0, ry0]) -
            np.array([rx3, ry3])
        )

        pupil_size = (
            left_pupil_diameter +
            right_pupil_diameter
        ) / 2.0

        # ====================================================
        # POINT OF REGARD
        # ====================================================

        por_x = (
            left_pupil_x +
            right_pupil_x
        ) / 2.0

        por_y = (
            left_pupil_y +
            right_pupil_y
        ) / 2.0

        # ====================================================
        # GAZE PROXY
        # ====================================================

        left_eye_mean = np.mean(
            left_eye,
            axis=0
        )

        right_eye_mean = np.mean(
            right_eye,
            axis=0
        )

        gaze_x = (
            left_eye_mean[0] +
            right_eye_mean[0]
        ) / 2.0

        gaze_y = (
            left_eye_mean[1] +
            right_eye_mean[1]
        ) / 2.0

        # ====================================================
        # VELOCIDAD
        # ====================================================

        saccade_velocity = 0.0

        if (
            self.previous_gaze is not None
            and self.previous_timestamp is not None
        ):

            dt = timestamp - self.previous_timestamp

            if dt > 0:

                dx = gaze_x - self.previous_gaze[0]
                dy = gaze_y - self.previous_gaze[1]

                distance = np.sqrt(
                    dx ** 2 +
                    dy ** 2
                )

                saccade_velocity = distance / dt

        self.previous_gaze = (
            gaze_x,
            gaze_y
        )

        self.previous_timestamp = timestamp

        # ----------------------------------------------------
        # Historial
        # ----------------------------------------------------

        self.gaze_history.append(
            (
                gaze_x,
                gaze_y,
                timestamp
            )
        )

        # ====================================================
        # RESULTADO
        # ====================================================

        return {
            "timestamp": timestamp,

            "gaze_x": gaze_x,
            "gaze_y": gaze_y,

            "Eye aspect Ratio": avg_ear,

            "blink": is_blink,

            "saccade_velocity": saccade_velocity,

            "pupil_size": pupil_size,

            "left_pupil_x": left_pupil_x,
            "left_pupil_y": left_pupil_y,
            "left_pupil_diameter": left_pupil_diameter,

            "right_pupil_x": right_pupil_x,
            "right_pupil_y": right_pupil_y,
            "right_pupil_diameter": right_pupil_diameter,

            "PoR_binocular_x": por_x,
            "PoR_binocular_y": por_y,

            "Point of Regard Right X": right_pupil_x,
            "Point of Regard Right Y": right_pupil_y,

            "Point of Regard Left X": left_pupil_x,
            "Point of Regard Left Y": left_pupil_y,

            "Category Binocular": "BINOCULAR",

            "Index Binocular":
                len(self.gaze_history) - 1,

            "left_ear": left_ear,
            "right_ear": right_ear
        }

    # ========================================================
    # DIBUJO DE LANDMARKS
    # ========================================================

    def draw_debug(self, frame):
        """
        Dibuja los landmarks oculares del último frame procesado.

        No vuelve a ejecutar MediaPipe.
        """

        if self.last_landmarks is None:
            return frame

        height, width, _ = frame.shape

        for index in self.LEFT_EYE + self.RIGHT_EYE:

            x = int(
                self.last_landmarks[index].x * width
            )

            y = int(
                self.last_landmarks[index].y * height
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        return frame

    # ========================================================
    # CERRAR MEDIAPIPE
    # ========================================================

    def close(self):
        """Libera los recursos de MediaPipe."""

        if self.face_landmarker is not None:
            self.face_landmarker.close()