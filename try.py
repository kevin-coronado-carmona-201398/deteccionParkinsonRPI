import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd
import os
import sys

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_fixations(gaze_history, time_window=0.2, dispersion_thresh=5):
    """Detect fixations in gaze history"""
    fixations = []
    timestamps = [g[2] for g in gaze_history]
    coords = np.array([[g[0], g[1]] for g in gaze_history])

    for i in range(len(gaze_history)):
        end_time = timestamps[i] + time_window
        window_coords = coords[(timestamps >= timestamps[i]) & (timestamps <= end_time)]
        if len(window_coords) > 1:
            disp = np.max(np.ptp(window_coords, axis=0))
            if disp < dispersion_thresh:
                fixations.append((timestamps[i], window_coords.mean(axis=0)))
    return fixations

def process_video(video_path, output_path="eye_metrics_output.csv"):
    """
    Process video file and extract eye tracking metrics
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output CSV
    """
    # MediaPipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Eye landmark indices
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Load video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    frame_num = 0
    blink_threshold = 0.2
    window_size = 5
    cooldown_time = 0.5
    last_blink_time = -cooldown_time

    blink_cnt = 0
    blink_flags = []
    ears = []
    pupil_sizes = []
    gaze_history = []
    left_pupil_data = []
    right_pupil_data = []
    por_data = []

    print(f"Processing video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Fixed timestamp increment of 0.004s (4ms)
        timestamp = frame_num * 0.004

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            left_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0
            ears.append(avg_ear)

            is_blink = 0
            if len(ears) >= window_size:
                recent_ears = ears[-window_size:]
                if sum(e < blink_threshold for e in recent_ears) >= int(window_size / 2):
                    if (timestamp - last_blink_time) > cooldown_time:
                        blink_cnt += 1
                        is_blink = 1
                        last_blink_time = timestamp
            blink_flags.append(is_blink)

            # Pupil centers and diameters
            lx0, ly0 = left_eye_pts[0]
            lx3, ly3 = left_eye_pts[3]
            rx0, ry0 = right_eye_pts[0]
            rx3, ry3 = right_eye_pts[3]

            left_pupil_x = (lx0 + lx3) / 2
            left_pupil_y = (ly0 + ly3) / 2
            right_pupil_x = (rx0 + rx3) / 2
            right_pupil_y = (ry0 + ry3) / 2

            left_pupil_diameter = dist.euclidean((lx0, ly0), (lx3, ly3))
            right_pupil_diameter = dist.euclidean((rx0, ry0), (rx3, ry3))
            avg_pupil_size = (left_pupil_diameter + right_pupil_diameter) / 2

            pupil_sizes.append(avg_pupil_size)
            left_pupil_data.append((left_pupil_x, left_pupil_y, left_pupil_diameter))
            right_pupil_data.append((right_pupil_x, right_pupil_y, right_pupil_diameter))

            por_x = (left_pupil_x + right_pupil_x) / 2
            por_y = (left_pupil_y + right_pupil_y) / 2
            por_data.append((por_x, por_y))

            gaze_x = (np.mean(left_eye_pts, axis=0)[0] + np.mean(right_eye_pts, axis=0)[0]) / 2
            gaze_y = (np.mean(left_eye_pts, axis=0)[1] + np.mean(right_eye_pts, axis=0)[1]) / 2
            gaze_history.append((gaze_x, gaze_y, timestamp))

        # Print progress every 100 frames
        if frame_num % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")

        frame_num += 1

    cap.release()
    face_mesh.close()

    if not gaze_history:
        print("No face landmarks detected in video")
        return False

    print("Processing gaze data...")

    # Detect fixations
    fixations = detect_fixations(np.array(gaze_history))

    # Saccade velocity computation
    sac_vel = []
    for i in range(1, len(gaze_history)):
        x1, y1, t1 = gaze_history[i - 1]
        x2, y2, t2 = gaze_history[i]
        dt = t2 - t1
        if dt > 0:
            vel = np.linalg.norm([x2 - x1, y2 - y1]) / dt
            sac_vel.append(vel)
        else:
            sac_vel.append(0)

    # Fixation flags
    fix_flags = []
    for i, (_, _, t) in enumerate(gaze_history):
        fix_flags.append(any(abs(t - f[0]) < 0.1 for f in fixations))

    # Write to CSV
    rows = []
    for i, (gx, gy, ts) in enumerate(gaze_history):
        if i < len(left_pupil_data) and i < len(right_pupil_data) and i < len(por_data):
            lp_x, lp_y, lp_d = left_pupil_data[i]
            rp_x, rp_y, rp_d = right_pupil_data[i]
            por_x, por_y = por_data[i]

            row = {
                "timestamp": ts,
                "gaze_x": gx,
                "gaze_y": gy,
                "Eye aspect Ratio": ears[i] if i < len(ears) else 0,
                "blink": blink_flags[i] if i < len(blink_flags) else 0,
                "saccade_velocity": sac_vel[i - 1] if i > 0 and i-1 < len(sac_vel) else 0,
                "fixation": int(fix_flags[i]) if i < len(fix_flags) else 0,
                "pupil_size": pupil_sizes[i] if i < len(pupil_sizes) else 0,
                "left_pupil_x": lp_x,
                "left_pupil_y": lp_y,
                "left_pupil_diameter": lp_d,
                "right_pupil_x": rp_x,
                "right_pupil_y": rp_y,
                "right_pupil_diameter": rp_d,
                "PoR_binocular_x": por_x,
                "PoR_binocular_y": por_y,
                "Point of Regard Right X": rp_x,
                "Point of Regard Right Y": rp_y,
                "Point of Regard Left X": lp_x,
                "Point of Regard Left Y": lp_y,
                "Category Binocular": "BINOCULAR",
                "Index Binocular": i
            }
            rows.append(row)

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] CSV saved as '{output_path}'")
    print(f"Total Blinks: {blink_cnt}")
    print(f"Total Fixations Detected: {len(fixations)}")
    print(f"Total frames processed: {len(rows)}")
    
    return True

if __name__ == "__main__":
    # Get input and output paths from environment variables or use defaults
    input_video = os.getenv('INPUT_VIDEO', 'flask_record.mp4')
    output_csv = os.getenv('OUTPUT_CSV', 'eye_metrics_output.csv')
    
    success = process_video(input_video, output_csv)
    sys.exit(0 if success else 1)