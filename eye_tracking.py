import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    """
    Procesador de frames para extracción de métricas oculares.

    Esta clase reemplaza la lógica de procesamiento que
    originalmente estaba contenida dentro de try.py.

    Puede recibir frames provenientes de:
        - un archivo de video
        - una cámara USB
        - una futura Raspberry Pi Camera
    """

    # --------------------------------------------------------
    # Landmarks del contorno ocular utilizados por el
    # algoritmo original.
    # --------------------------------------------------------

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def __init__(
        self,
        blink_threshold=0.2,
        blink_window_size=5,
        blink_cooldown=0.5
    ):

        self.blink_threshold = blink_threshold
        self.blink_window_size = blink_window_size
        self.blink_cooldown = blink_cooldown

        # ----------------------------------------------------
        # MediaPipe Face Mesh
        # ----------------------------------------------------

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ----------------------------------------------------
        # Estado temporal
        # ----------------------------------------------------

        self.ear_history = []

        self.last_blink_time = -self.blink_cooldown
        self.blink_count = 0

        self.previous_gaze = None
        self.previous_timestamp = None

        # Historial utilizado posteriormente para
        # detección de fijaciones.
        self.gaze_history = []

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

        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

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

        # IMPORTANTE:
        #
        # El repositorio original llama a estas medidas
        # "pupil_size", pero en realidad los landmarks 0 y 3
        # representan los extremos del ojo.
        #
        # Por compatibilidad con el dataset conservamos
        # temporalmente esta característica con el mismo
        # nombre.
        #
        # Más adelante debemos decidir si sustituirla por
        # una medida basada realmente en los landmarks del
        # iris.
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

        # Guardamos para futuras fijaciones.
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

            # Información útil para depuración.
            "left_ear": left_ear,
            "right_ear": right_ear
        }

    # ========================================================
    # DIBUJO DE LANDMARKS
    # ========================================================

    def draw_debug(self, frame):
        """
        Dibuja landmarks oculares sobre un frame.

        Esta función se utilizará posteriormente en el
        feed de tiempo real.
        """

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return frame

        landmarks = results.multi_face_landmarks[0].landmark

        height, width, _ = frame.shape

        for index in self.LEFT_EYE + self.RIGHT_EYE:

            x = int(
                landmarks[index].x * width
            )

            y = int(
                landmarks[index].y * height
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

        self.face_mesh.close()