import cv2
import os
import shutil
import subprocess
import time


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30


# ============================================================
# USB CAMERA
# ============================================================

def open_usb_camera(
    camera_index=0,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    fps=DEFAULT_FPS
):
    print(f"[CAMERA] Trying USB camera index {camera_index}...")

    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open USB camera index {camera_index}"
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)

    # Confirmar que realmente podemos obtener un frame.
    success, frame = camera.read()

    if not success or frame is None:
        camera.release()
        raise RuntimeError(
            f"USB camera index {camera_index} opened but returned no frame"
        )

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(
        f"[CAMERA] USB Camera opened: "
        f"{actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
    )

    return camera


# ============================================================
# RASPBERRY PI CAMERA THROUGH RPICAM-VID
# ============================================================

class RpiCamVideo:
    """
    Captura la cámara oficial de Raspberry Pi usando rpicam-vid.

    rpicam-vid genera un stream MJPEG y lo escribe directamente
    en stdout. Python recibe los JPEG, los decodifica con OpenCV
    y expone la interfaz:

        success, frame = camera.read()
    """

    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS
    ):
        self.width = width
        self.height = height
        self.fps = fps

        self.process = None
        self.buffer = bytearray()

        print("[CAMERA] Initializing Raspberry Pi Camera with rpicam-vid...")

        rpicam_path = shutil.which("rpicam-vid")

        if rpicam_path is None:
            raise RuntimeError(
                "rpicam-vid was not found in PATH"
            )

        command = [
            rpicam_path,
            "-n",
            "-t", "0",
            "--codec", "mjpeg",
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(fps),
            "--quality", "80",
            "--flush",
            "-o", "-"
        ]

        print("[CAMERA] Command:", " ".join(command))

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
        except Exception as error:
            self.process = None
            raise RuntimeError(
                f"Could not start rpicam-vid: {error}"
            ) from error

        time.sleep(1.0)

        if self.process.poll() is not None:
            return_code = self.process.returncode
            self.release()

            raise RuntimeError(
                f"rpicam-vid exited immediately "
                f"with return code {return_code}"
            )

        print(
            f"[CAMERA] Raspberry Pi Camera selected: "
            f"{width}x{height} @ {fps} FPS"
        )

    def read(self):
        """
        Lee el siguiente JPEG del stream MJPEG.

        Devuelve:
            (True, frame) si hay un frame válido.
            (False, None) si la cámara dejó de producir datos.
        """

        if self.process is None or self.process.stdout is None:
            return False, None

        while True:

            # ------------------------------------------------
            # Buscar inicio de JPEG.
            # ------------------------------------------------
            start = self.buffer.find(b"\xff\xd8")

            if start >= 0:

                # Descartar cualquier basura anterior al JPEG.
                if start > 0:
                    del self.buffer[:start]

                # Buscar final del JPEG.
                end = self.buffer.find(b"\xff\xd9", 2)

                if end >= 0:

                    jpeg_data = bytes(
                        self.buffer[:end + 2]
                    )

                    del self.buffer[:end + 2]

                    frame = cv2.imdecode(
                        __import__("numpy").frombuffer(
                            jpeg_data,
                            dtype=__import__("numpy").uint8
                        ),
                        cv2.IMREAD_COLOR
                    )

                    if frame is not None:
                        return True, frame

                    # JPEG inválido; buscar siguiente.
                    continue

            # ------------------------------------------------
            # Leer más datos desde rpicam-vid.
            # ------------------------------------------------
            chunk = self.process.stdout.read(65536)

            if not chunk:
                return False, None

            self.buffer.extend(chunk)

            # ------------------------------------------------
            # Protección contra crecimiento anormal.
            # ------------------------------------------------
            if len(self.buffer) > 10 * 1024 * 1024:

                start = self.buffer.find(b"\xff\xd8")

                if start >= 0:
                    del self.buffer[:start]
                else:
                    self.buffer.clear()

                if self.process.poll() is not None:
                    return False, None

    def release(self):
        """
        Detiene rpicam-vid y libera los recursos.
        """

        if self.process is None:
            return

        try:

            if self.process.poll() is None:
                self.process.terminate()

                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

        except Exception as error:
            print(
                f"[WARNING] Error stopping rpicam-vid: {error}"
            )

        finally:

            if self.process.stdout is not None:
                try:
                    self.process.stdout.close()
                except Exception:
                    pass

            self.process = None


# ============================================================
# AUTOMATIC CAMERA SELECTION
# ============================================================

def open_raspberry_camera(
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    fps=DEFAULT_FPS
):
    """
    Intenta abrir la cámara oficial utilizando rpicam-vid.
    """

    camera = RpiCamVideo(
        width=width,
        height=height,
        fps=fps
    )

    return camera


def open_camera(
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    fps=DEFAULT_FPS,
    camera_index=0
):
    """
    Selección automática:

        1. Raspberry Pi Camera mediante rpicam-vid.
        2. USB Camera mediante OpenCV.

    Esto permite utilizar el mismo código en la Raspberry
    y en Windows.
    """

    # --------------------------------------------------------
    # Intentar cámara oficial
    # --------------------------------------------------------

    print("[CAMERA] Trying Raspberry Pi Camera first...")

    try:

        camera = open_raspberry_camera(
            width=width,
            height=height,
            fps=fps
        )

        print("[CAMERA] Camera selected: Raspberry Pi Camera")

        return camera

    except Exception as error:

        print(
            "[WARNING] Raspberry Pi Camera unavailable:"
        )
        print(
            f"[WARNING] {error}"
        )

        print(
            "[CAMERA] Falling back to USB camera..."
        )

    # --------------------------------------------------------
    # Fallback USB
    # --------------------------------------------------------

    camera = open_usb_camera(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps
    )

    print("[CAMERA] Camera selected: USB camera")

    return camera


# ============================================================
# OPTIONAL VIDEO RECORDING
# ============================================================

def record_video(
    camera,
    output_file="camera_record.mp4",
    duration=30
):
    """
    Función auxiliar para grabar frames a un archivo.

    No forma parte del pipeline de detección en tiempo real.
    """

    writer = None

    start_time = time.time()

    frames = 0

    try:

        while time.time() - start_time < duration:

            success, frame = camera.read()

            if not success or frame is None:
                print("[WARNING] Failed to read camera frame")
                break

            height, width = frame.shape[:2]

            if writer is None:

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    DEFAULT_FPS,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open VideoWriter: {output_file}"
                    )

            writer.write(frame)

            frames += 1

            cv2.imshow(
                "Recording",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        print(
            f"[CAMERA] Recording finished: "
            f"{output_file}"
        )

        print(
            f"[CAMERA] Frames recorded: {frames}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    camera_index = int(
        os.getenv("CAMERA_INDEX", "0")
    )

    width = int(
        os.getenv("CAMERA_WIDTH", str(DEFAULT_WIDTH))
    )

    height = int(
        os.getenv("CAMERA_HEIGHT", str(DEFAULT_HEIGHT))
    )

    fps = int(
        os.getenv("CAMERA_FPS", str(DEFAULT_FPS))
    )

    duration = int(
        os.getenv("MAX_DURATION", "30")
    )

    camera = open_camera(
        width=width,
        height=height,
        fps=fps,
        camera_index=camera_index
    )

    try:

        record_video(
            camera,
            output_file="camera_record.mp4",
            duration=duration
        )

    finally:

        camera.release()


if __name__ == "__main__":
    main()