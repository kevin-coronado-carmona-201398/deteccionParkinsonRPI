import cv2
import os
import sys
import time


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_DURATION = 60


# ============================================================
# CÁMARA USB
# ============================================================

def open_usb_camera(
        camera_index=0,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS):
    """
    Abre una cámara USB mediante OpenCV.

    Se utiliza como cámara secundaria/fallback cuando
    la Camera Module de Raspberry Pi no puede abrirse.
    """

    print(
        f"[CAMERA] Trying USB camera with index "
        f"{camera_index}..."
    )

    if os.name == "nt":

        camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():

            print(
                "[WARNING] CAP_DSHOW failed. "
                "Trying default backend..."
            )

            camera.release()

            camera = cv2.VideoCapture(
                camera_index
            )

    else:

        camera = cv2.VideoCapture(
            camera_index
        )

    if not camera.isOpened():

        raise RuntimeError(
            f"Could not open USB camera "
            f"with index {camera_index}"
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        width
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        height
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        fps
    )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = camera.get(
        cv2.CAP_PROP_FPS
    )

    if actual_fps <= 0:

        actual_fps = fps

    print(
        f"[CAMERA] USB camera opened: "
        f"{actual_width}x{actual_height} "
        f"@ {actual_fps:.2f} FPS"
    )

    return camera


# ============================================================
# ADAPTADOR PARA CÁMARA RASPBERRY PI
# ============================================================

class RaspberryPiCamera:
    """
    Adaptador que proporciona una interfaz similar a
    cv2.VideoCapture.

    Esto permite utilizar Picamera2 sin tener que cambiar
    el resto del sistema.

    Métodos principales:

        read()
        release()
    """

    def __init__(
            self,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            fps=DEFAULT_FPS):

        from picamera2 import Picamera2

        print(
            "[CAMERA] Initializing Raspberry Pi Camera "
            "with Picamera2..."
        )

        self.picam2 = Picamera2()

        config = self.picam2.create_video_configuration(
            main={
                "size": (
                    width,
                    height
                ),
                "format": "RGB888"
            },
            controls={
                "FrameRate": fps
            }
        )

        self.picam2.configure(
            config
        )

        self.picam2.start()

        # Dar tiempo para que la cámara estabilice
        # exposición y balance de blancos.
        time.sleep(2)

        self.width = width
        self.height = height
        self.fps = fps

        print(
            f"[CAMERA] Raspberry Pi Camera opened: "
            f"{width}x{height} @ {fps} FPS"
        )

    def read(self):
        """
        Captura un frame.

        Devuelve:

            (True, frame)

        de manera compatible con cv2.VideoCapture.
        """

        try:

            frame = self.picam2.capture_array(
                "main"
            )

            if frame is None:

                return False, None

            # Picamera2 entrega RGB888.
            # OpenCV trabaja habitualmente en BGR.
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            return True, frame

        except Exception as error:

            print(
                f"[CAMERA] Raspberry frame error: "
                f"{error}"
            )

            return False, None

    def release(self):
        """
        Detiene y libera Picamera2.
        """

        try:

            self.picam2.stop()

        except Exception:

            pass

        try:

            self.picam2.close()

        except Exception:

            pass

        print(
            "[CAMERA] Raspberry Pi Camera released."
        )


# ============================================================
# APERTURA AUTOMÁTICA DE CÁMARA
# ============================================================

def open_camera(
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS,
        camera_index=0):
    """
    Abre automáticamente la mejor cámara disponible.

    Prioridad:

        1. Camera Module oficial de Raspberry Pi
           utilizando Picamera2.

        2. Cámara USB mediante OpenCV.

    En Windows, Picamera2 normalmente no está disponible,
    por lo que se utilizará directamente la cámara USB.

    En Raspberry Pi, la Camera Module será la primera opción.
    """

    # ========================================================
    # 1. INTENTAR RASPBERRY PI CAMERA
    # ========================================================

    print(
        "[CAMERA] Trying Raspberry Pi Camera first..."
    )

    try:

        camera = RaspberryPiCamera(
            width=width,
            height=height,
            fps=fps
        )

        print(
            "[CAMERA] Camera selected: "
            "Raspberry Pi Camera"
        )

        return camera

    except Exception as error:

        print(
            "[WARNING] Raspberry Pi Camera "
            f"unavailable: {error}"
        )

        print(
            "[CAMERA] Falling back to USB camera..."
        )


    # ========================================================
    # 2. FALLBACK A USB
    # ========================================================

    try:

        camera = open_usb_camera(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps
        )

        print(
            "[CAMERA] Camera selected: USB camera"
        )

        return camera

    except Exception as error:

        raise RuntimeError(
            "No camera could be opened. "
            "Tried Raspberry Pi Camera and USB camera. "
            f"USB error: {error}"
        )


# ============================================================
# GRABACIÓN
# ============================================================

def record_video(
        camera,
        output_path="camera_record.mp4",
        max_duration=DEFAULT_DURATION,
        fps=DEFAULT_FPS):
    """
    Captura frames de cualquiera de nuestras cámaras y
    los guarda como MP4.

    Actualmente soporta:

        - RaspberryPiCamera
        - cv2.VideoCapture
    """

    print(
        "[RECORDING] Starting camera capture..."
    )

    print(
        "[RECORDING] Press 'q' to stop."
    )

    # --------------------------------------------------------
    # Obtener dimensiones
    # --------------------------------------------------------

    if isinstance(
        camera,
        RaspberryPiCamera
    ):

        frame_width = camera.width
        frame_height = camera.height
        actual_fps = camera.fps

    else:

        frame_width = int(
            camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        frame_height = int(
            camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        actual_fps = camera.get(
            cv2.CAP_PROP_FPS
        )

        if actual_fps <= 0:

            actual_fps = fps

    print(
        f"[RECORDING] Resolution: "
        f"{frame_width}x{frame_height}"
    )

    print(
        f"[RECORDING] FPS used for recording: "
        f"{actual_fps:.2f}"
    )

    # --------------------------------------------------------
    # VideoWriter
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        actual_fps,
        (
            frame_width,
            frame_height
        )
    )

    if not writer.isOpened():

        camera.release()

        raise RuntimeError(
            f"Could not create video file: "
            f"{output_path}"
        )

    start_time = time.time()

    frame_count = 0

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "[ERROR] Failed to capture frame."
                )

                break

            writer.write(
                frame
            )

            frame_count += 1

            # ------------------------------------------------
            # PREVISUALIZACIÓN
            # ------------------------------------------------

            preview = frame.copy()

            cv2.putText(
                preview,
                "CAMERA",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            elapsed = (
                time.time()
                - start_time
            )

            cv2.putText(
                preview,
                f"Time: {elapsed:.1f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                preview,
                f"Frames: {frame_count}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Parkinson Eye Tracking - Camera",
                preview
            )

            # ------------------------------------------------
            # Q
            # ------------------------------------------------

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                print(
                    "[RECORDING] Stopped by user."
                )

                break

            # ------------------------------------------------
            # Duración máxima
            # ------------------------------------------------

            if elapsed >= max_duration:

                print(
                    f"[RECORDING] Maximum duration "
                    f"({max_duration}s) reached."
                )

                break

    finally:

        camera.release()

        writer.release()

        cv2.destroyAllWindows()

    print()
    print(
        "[RECORDING] Capture completed."
    )

    print(
        f"[RECORDING] Frames captured: "
        f"{frame_count}"
    )

    print(
        f"[RECORDING] Output: {output_path}"
    )

    return frame_count > 0


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    camera_index = int(
        os.getenv(
            "CAMERA_INDEX",
            "0"
        )
    )

    width = int(
        os.getenv(
            "CAMERA_WIDTH",
            str(DEFAULT_WIDTH)
        )
    )

    height = int(
        os.getenv(
            "CAMERA_HEIGHT",
            str(DEFAULT_HEIGHT)
        )
    )

    fps = int(
        os.getenv(
            "CAMERA_FPS",
            str(DEFAULT_FPS)
        )
    )

    max_duration = int(
        os.getenv(
            "MAX_DURATION",
            str(DEFAULT_DURATION)
        )
    )

    output_file = os.getenv(
        "OUTPUT_FILE",
        "camera_record.mp4"
    )

    # --------------------------------------------------------
    # Apertura automática
    # --------------------------------------------------------

    camera = open_camera(
        width=width,
        height=height,
        fps=fps,
        camera_index=camera_index
    )

    # --------------------------------------------------------
    # Grabación
    # --------------------------------------------------------

    success = record_video(
        camera=camera,
        output_path=output_file,
        max_duration=max_duration,
        fps=fps
    )

    sys.exit(
        0 if success else 1
    )


if __name__ == "__main__":

    main()