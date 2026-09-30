import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    """
    Procesador de frames para extracción de métricas oculares.

    Compatible con:
        - Raspberry Pi Camera
        - cámara USB
        - archivos de video

    Este archivo utiliza MediaPipe 0.10.x mediante:
        mp.solutions.face_mesh.FaceMesh

    Características principales utilizadas por el LSTM:
        - gaze_x
        - gaze_y
        - saccade_velocity
        - pupil_size
        - blink
    """

    # ========================================================
    # LANDMARKS OCULARES
    # ========================================================

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def __init__(
        self,
        blink_threshold=0.2,
        blink_window_size=5,
        blink_cooldown=0.5
    ):
        """
        Inicializa el rastreador ocular.

        Parámetros:
            blink_threshold:
                Umbral EAR utilizado para detectar cierre ocular.

            blink_window_size:
                Número de muestras utilizadas para estabilizar
                la detección de parpadeos.

            blink_cooldown:
                Tiempo mínimo entre eventos de parpadeo.
        """

        self.blink_threshold = float(
            blink_threshold
        )

        self.blink_window_size = int(
            blink_window_size
        )

        self.blink_cooldown = float(
            blink_cooldown
        )

        # ====================================================
        # MEDIAPIPE FACE MESH
        # ====================================================

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ====================================================
        # ESTADO TEMPORAL
        # ====================================================

        self.ear_history = []

        self.last_blink_time = (
            -self.blink_cooldown
        )

        self.blink_count = 0

        self.previous_gaze = None
        self.previous_timestamp = None

        self.gaze_history = []

        # Últimos landmarks detectados.
        # Se utiliza para draw_debug() sin volver a ejecutar
        # MediaPipe sobre el mismo frame.
        self.last_landmarks = None

    # ========================================================
    # CALCULAR EAR
    # ========================================================

    @staticmethod
    def calculate_ear(eye_landmarks):
        """
        Calcula el Eye Aspect Ratio (EAR).

        Se utilizan 6 puntos:

            0 y 3 -> extremos horizontales
            1 y 5 -> primera distancia vertical
            2 y 4 -> segunda distancia vertical
        """

        if len(eye_landmarks) != 6:
            return 0.0

        A = np.linalg.norm(
            np.array(eye_landmarks[1], dtype=np.float32)
            -
            np.array(eye_landmarks[5], dtype=np.float32)
        )

        B = np.linalg.norm(
            np.array(eye_landmarks[2], dtype=np.float32)
            -
            np.array(eye_landmarks[4], dtype=np.float32)
        )

        C = np.linalg.norm(
            np.array(eye_landmarks[0], dtype=np.float32)
            -
            np.array(eye_landmarks[3], dtype=np.float32)
        )

        if C <= 0:
            return 0.0

        return float(
            (A + B) / (2.0 * C)
        )

    # ========================================================
    # PROCESAR FRAME
    # ========================================================

    def process_frame(self, frame, timestamp):
        """
        Procesa un frame.

        Parámetros:
            frame:
                Frame BGR de OpenCV.

            timestamp:
                Timestamp en segundos.

        Retorna:
            Diccionario de métricas o None si no se detecta
            ningún rostro.
        """

        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            return None

        if frame.ndim != 3:
            return None

        # ====================================================
        # BGR -> RGB
        # ====================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # MEDIA PIPE
        # ====================================================

        results = self.face_mesh.process(
            frame_rgb
        )

        if not results.multi_face_landmarks:
            self.last_landmarks = None
            return None

        landmarks = (
            results.multi_face_landmarks[0].landmark
        )

        self.last_landmarks = landmarks

        height, width, _ = frame.shape

        # ====================================================
        # LANDMARKS OCULARES EN PIXELES
        # ====================================================

        left_eye = [
            (
                int(landmarks[index].x * width),
                int(landmarks[index].y * height)
            )
            for index in self.LEFT_EYE
        ]

        right_eye = [
            (
                int(landmarks[index].x * width),
                int(landmarks[index].y * height)
            )
            for index in self.RIGHT_EYE
        ]

        # ====================================================
        # EAR / BLINK
        # ====================================================

        left_ear = self.calculate_ear(
            left_eye
        )

        right_ear = self.calculate_ear(
            right_eye
        )

        avg_ear = (
            left_ear + right_ear
        ) / 2.0

        self.ear_history.append(
            avg_ear
        )

        if len(self.ear_history) > self.blink_window_size:
            self.ear_history.pop(0)

        is_blink = 0

        if len(self.ear_history) >= self.blink_window_size:

            low_ear_count = sum(
                ear < self.blink_threshold
                for ear in self.ear_history
            )

            if low_ear_count >= (
                self.blink_window_size // 2
            ):

                if (
                    timestamp - self.last_blink_time
                    > self.blink_cooldown
                ):
                    self.blink_count += 1
                    is_blink = 1
                    self.last_blink_time = timestamp

        # ====================================================
        # PUPIL SIZE - PROXY
        # ====================================================
        #
        # IMPORTANTE:
        #
        # Esta NO es una medición física del diámetro de la
        # pupila.
        #
        # Para mantener compatibilidad con el procesamiento
        # anterior, se utiliza la distancia entre los extremos
        # horizontales de cada ojo.
        #
        # Por ahora se conserva exactamente esta semántica
        # porque el modelo fue entrenado utilizando esta
        # característica.
        # ====================================================

        lx0, ly0 = left_eye[0]
        lx3, ly3 = left_eye[3]

        rx0, ry0 = right_eye[0]
        rx3, ry3 = right_eye[3]

        left_pupil_x = (
            lx0 + lx3
        ) / 2.0

        left_pupil_y = (
            ly0 + ly3
        ) / 2.0

        right_pupil_x = (
            rx0 + rx3
        ) / 2.0

        right_pupil_y = (
            ry0 + ry3
        ) / 2.0

        left_pupil_diameter = np.linalg.norm(
            np.array([lx0, ly0], dtype=np.float32)
            -
            np.array([lx3, ly3], dtype=np.float32)
        )

        right_pupil_diameter = np.linalg.norm(
            np.array([rx0, ry0], dtype=np.float32)
            -
            np.array([rx3, ry3], dtype=np.float32)
        )

        pupil_size = (
            left_pupil_diameter
            +
            right_pupil_diameter
        ) / 2.0

        # ====================================================
        # POINT OF REGARD
        # ====================================================

        por_x = (
            left_pupil_x
            +
            right_pupil_x
        ) / 2.0

        por_y = (
            left_pupil_y
            +
            right_pupil_y
        ) / 2.0

        # ====================================================
        # GAZE PROXY
        # ====================================================

        left_eye_mean = np.mean(
            np.asarray(left_eye, dtype=np.float32),
            axis=0
        )

        right_eye_mean = np.mean(
            np.asarray(right_eye, dtype=np.float32),
            axis=0
        )

        gaze_x = (
            left_eye_mean[0]
            +
            right_eye_mean[0]
        ) / 2.0

        gaze_y = (
            left_eye_mean[1]
            +
            right_eye_mean[1]
        ) / 2.0

        gaze_x = float(gaze_x)
        gaze_y = float(gaze_y)

        # ====================================================
        # VELOCIDAD DE GAZE
        # ====================================================

        saccade_velocity = 0.0

        if (
            self.previous_gaze is not None
            and self.previous_timestamp is not None
        ):

            dt = (
                float(timestamp)
                -
                float(self.previous_timestamp)
            )

            if dt > 0:

                dx = (
                    gaze_x
                    -
                    self.previous_gaze[0]
                )

                dy = (
                    gaze_y
                    -
                    self.previous_gaze[1]
                )

                distance = np.sqrt(
                    dx ** 2
                    +
                    dy ** 2
                )

                saccade_velocity = (
                    distance / dt
                )

        saccade_velocity = float(
            saccade_velocity
        )

        self.previous_gaze = (
            gaze_x,
            gaze_y
        )

        self.previous_timestamp = (
            float(timestamp)
        )

        # ====================================================
        # HISTORIAL
        # ====================================================

        self.gaze_history.append(
            (
                gaze_x,
                gaze_y,
                float(timestamp)
            )
        )

        # ====================================================
        # RESULTADO
        # ====================================================

        return {
            "timestamp": float(timestamp),

            "gaze_x": gaze_x,
            "gaze_y": gaze_y,

            "Eye aspect Ratio": float(avg_ear),

            "blink": int(is_blink),

            "saccade_velocity": saccade_velocity,

            "pupil_size": float(pupil_size),

            "left_pupil_x": float(left_pupil_x),
            "left_pupil_y": float(left_pupil_y),
            "left_pupil_diameter": float(
                left_pupil_diameter
            ),

            "right_pupil_x": float(right_pupil_x),
            "right_pupil_y": float(right_pupil_y),
            "right_pupil_diameter": float(
                right_pupil_diameter
            ),

            "PoR_binocular_x": float(por_x),
            "PoR_binocular_y": float(por_y),

            "Point of Regard Right X": float(
                right_pupil_x
            ),
            "Point of Regard Right Y": float(
                right_pupil_y
            ),

            "Point of Regard Left X": float(
                left_pupil_x
            ),
            "Point of Regard Left Y": float(
                left_pupil_y
            ),

            "Category Binocular": "BINOCULAR",

            "Index Binocular":
                len(self.gaze_history) - 1,

            # Información de depuración.
            "left_ear": float(left_ear),
            "right_ear": float(right_ear)
        }

    # ========================================================
    # DEBUG DRAW
    # ========================================================

    def draw_debug(self, frame):
        """
        Dibuja los landmarks oculares del último frame
        procesado.

        No vuelve a ejecutar MediaPipe.
        """

        if (
            frame is None
            or self.last_landmarks is None
        ):
            return frame

        output = frame.copy()

        height, width, _ = output.shape

        for index in (
            self.LEFT_EYE
            +
            self.RIGHT_EYE
        ):

            x = int(
                self.last_landmarks[index].x
                *
                width
            )

            y = int(
                self.last_landmarks[index].y
                *
                height
            )

            cv2.circle(
                output,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        return output

    # ========================================================
    # CERRAR
    # ========================================================

    def close(self):
        """
        Libera los recursos de MediaPipe.
        """

        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
