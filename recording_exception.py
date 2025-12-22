import cv2
import requests
import numpy as np
import time

url = "http://172.17.13.83:5000/video_feed"  # Flask video stream

# Open MJPEG stream
stream = requests.get(url, stream=True)
bytes_data = b""

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
start_time = time.time()

print("Recording started...")

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')  # JPEG start
    b = bytes_data.find(b'\xff\xd9')  # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if out is None:  # Initialize writer once we know frame size
            h, w = frame.shape[:2]
            # use ~5 FPS since Flask MJPEG stream is usually slow
            out = cv2.VideoWriter("flask_record.mp4", fourcc, 5.0, (w, h))

        out.write(frame)
        cv2.imshow("Recording", frame)

        # Stop after 30 seconds
        if (time.time() - start_time) > 3000:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if out:
    out.release()
cv2.destroyAllWindows()
print("Recording stopped. Saved as flask_record.mp4")
