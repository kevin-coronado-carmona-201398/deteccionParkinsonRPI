import cv2
import requests
import numpy as np
import time
import signal
import sys
import os

def record_video(url, output_path="flask_record.mp4", max_duration=3000):
    """
    Record video from a Flask MJPEG stream
    
    Args:
        url: URL of the Flask video stream
        output_path: Path to save the recorded video
        max_duration: Maximum recording duration in seconds
    """
    print(f"Starting recording from {url}...")
    print("Press 'q' to stop recording window, or Ctrl+C to stop in terminal")
    # Open MJPEG stream
    try:
        stream = requests.get(url, stream=True, timeout=10)
        stream.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to video stream: {e}")
        return False

    bytes_data = b""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    start_time = time.time()
    frame_count = 0

    try:
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                if out is None:  # Initialize writer once we know frame size
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, 5.0, (w, h))  # ~5 FPS
                    print(f"Video writer initialized: {w}x{h} @ 5 FPS")

                out.write(frame)
                frame_count += 1

                # Show preview window
                try:
                    cv2.imshow("Recording", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Stop requested via 'q'")
                        break
                except Exception:
                    # Headless environments may not support imshow; ignore
                    pass

                # Stop after max_duration seconds
                if (time.time() - start_time) > max_duration:
                    print(f"Maximum duration ({max_duration}s) reached")
                    break

                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Recorded {frame_count} frames in {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        if out:
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    if frame_count > 0:
        print(f"Recording completed. Saved {frame_count} frames to {output_path}")
        return True
    else:
        print("No frames recorded")
        return False

if __name__ == "__main__":
    # Get URL from environment variable or use default
    video_url = os.getenv('VIDEO_URL', "http://172.20.10.2:5000/video_feed")
    output_file = os.getenv('OUTPUT_FILE', "flask_record.mp4")
    max_duration = int(os.getenv('MAX_DURATION', "3000"))
    
    success = record_video(video_url, output_file, max_duration)
    sys.exit(0 if success else 1)
