#!/usr/bin/env python3


import os
import sys
import time
import subprocess
import signal
from pathlib import Path

# Global variable to track if user wants to skip current step
skip_current_step = False
restart_current_step = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global skip_current_step, restart_current_step
    print("\n[INTERRUPTED] Interrupted by user (Ctrl+C)")
    print("Options:")
    print("1. Press Ctrl+C again to skip current step and continue")
    print("2. Press Ctrl+C three times to restart current step")
    print("3. Wait 5 seconds to continue with current step")
    
    # Set a flag to indicate interruption
    skip_current_step = True

def run_command(command, description, timeout=None, allow_skip=True):
    """Run a command and handle errors with graceful interruption"""
    global skip_current_step, restart_current_step
    
    print(f"\n{'='*60}")
    print(f"[RUNNING] {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    # Reset flags
    skip_current_step = False
    restart_current_step = False
    
    while True:
        try:
            # Set up signal handler for this command
            original_handler = signal.signal(signal.SIGINT, signal_handler)
            
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,  
                timeout=timeout
            )
            
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            
            if result.returncode == 0:
                print(f"[SUCCESS] {description} completed successfully")
                if result.stdout:
                    print("Output:", result.stdout)
                return True
            else:
                print(f"[ERROR] {description} failed")
                print("Error:", result.stderr)
                
                if allow_skip:
                    print("\nOptions:")
                    print("1. Press Enter to retry")
                    print("2. Type 'skip' to skip this step")
                    print("3. Type 'exit' to quit pipeline")
                    
                    choice = input("Your choice: ").strip().lower()
                    if choice == 'skip':
                        print(f"[SKIP] Skipping {description}")
                        return False
                    elif choice == 'exit':
                        print("[EXIT] Exiting pipeline")
                        sys.exit(0)
                    else:
                        print("[RETRY] Retrying...")
                        continue
                else:
                    return False
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {description} timed out")
            
            if allow_skip:
                print("\nOptions:")
                print("1. Press Enter to retry")
                print("2. Type 'skip' to skip this step")
                print("3. Type 'exit' to quit pipeline")
                
                choice = input("Your choice: ").strip().lower()
                if choice == 'skip':
                    print(f"[SKIP] Skipping {description}")
                    return False
                elif choice == 'exit':
                    print("[EXIT] Exiting pipeline")
                    sys.exit(0)
                else:
                    print("[RETRY] Retrying...")
                    continue
            else:
                return False
                
        except KeyboardInterrupt:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            
            if skip_current_step:
                print(f"[SKIP] Skipping {description}")
                return False
            elif restart_current_step:
                print("[RESTART] Restarting current step...")
                continue
            else:
                print("[CONTINUE] Continuing with current step...")
                time.sleep(1)
                continue
                
        except Exception as e:
            print(f"[ERROR] Error running {description}: {e}")
            
            if allow_skip:
                print("\nOptions:")
                print("1. Press Enter to retry")
                print("2. Type 'skip' to skip this step")
                print("3. Type 'exit' to quit pipeline")
                
                choice = input("Your choice: ").strip().lower()
                if choice == 'skip':
                    print(f"[SKIP] Skipping {description}")
                    return False
                elif choice == 'exit':
                    print("[EXIT] Exiting pipeline")
                    sys.exit(0)
                else:
                    print("[RETRY] Retrying...")
                    continue
            else:
                return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"[FOUND] {description} found: {file_path}")
        return True
    else:
        print(f"[MISSING] {description} not found: {file_path}")
        return False

def main():
    """Main pipeline execution"""
    print("[START] Starting Parkinson's Eye Tracking Detection Pipeline")
    print("="*60)
    print("[TIP] Tip: Press Ctrl+C during any step to skip it and continue")
    print("="*60)
    
    # Configuration
    video_url = os.getenv('VIDEO_URL', 'http://172.17.13.231:5000/video_feed')
    max_duration = int(os.getenv('MAX_DURATION', '3000'))
    output_dir = os.getenv('OUTPUT_DIR', '/app/output')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Record video from Raspberry Pi
    print("\n[STEP 1] Recording video from Raspberry Pi")
    recording_success = run_command(
        f"python recording.py",
        "Video Recording",
        timeout=max_duration + 60,  # Add 60 seconds buffer
        allow_skip=True
    )
    
    if not recording_success:
        print("[WARNING] Video recording was skipped or failed")
        # Check if video file exists anyway
        video_file = "flask_record.mp4"
        if not check_file_exists(video_file, "Recorded video"):
            print("[ERROR] No video file available. Cannot continue.")
            sys.exit(1)
    
    # Check if video file was created
    video_file = "flask_record.mp4"
    if not check_file_exists(video_file, "Recorded video"):
        print("[ERROR] Video file not found. Exiting pipeline.")
        sys.exit(1)
    
    # Step 2: Extract eye tracking features
    print("\n[STEP 2] Extracting eye tracking features")
    feature_extraction_success = run_command(
        f"python try.py",
        "Eye Tracking Feature Extraction",
        allow_skip=True
    )
    
    if not feature_extraction_success:
        print("[WARNING] Feature extraction was skipped or failed")
        # Check if CSV file exists anyway
        csv_file = "eye_metrics_output.csv"
        if not check_file_exists(csv_file, "Eye metrics CSV"):
            print("[ERROR] No eye metrics CSV available. Cannot continue.")
            sys.exit(1)
    
    # Check if CSV file was created
    csv_file = "eye_metrics_output.csv"
    if not check_file_exists(csv_file, "Eye metrics CSV"):
        print("[ERROR] Eye metrics CSV not found. Exiting pipeline.")
        sys.exit(1)
    
    # Step 3: Preprocess eye tracking data
    print("\n[STEP 3] Preprocessing eye tracking data")
    preprocessing_success = run_command(
        f"python \"{os.path.join('collab files', 'eye_tracking_data_preprocessor.py')}\"",
        "Data Preprocessing",
        allow_skip=True
    )
    
    if not preprocessing_success:
        print("[WARNING] Data preprocessing was skipped or failed")
        # Check if cleaned file exists anyway
        cleaned_file = "cleaned_file.csv"
        if not check_file_exists(cleaned_file, "Cleaned data CSV"):
            print("[ERROR] No cleaned data CSV available. Cannot continue.")
            sys.exit(1)
    
    # Check if cleaned file was created
    cleaned_file = "cleaned_file.csv"
    if not check_file_exists(cleaned_file, "Cleaned data CSV"):
        print("[ERROR] Cleaned data CSV not found. Exiting pipeline.")
        sys.exit(1)
    
    # Step 4: Run Parkinson's classification
    print("\n[STEP 4] Running Parkinson's classification")
    classification_success = run_command(
        f"python \"{os.path.join('collab files', 'parkinsons_eye_tracking_classifier.py')}\"",
        "Parkinson's Classification",
        allow_skip=True
    )
    
    if not classification_success:
        print("[WARNING] Classification was skipped or failed")
        # Check if prediction result exists anyway
        prediction_file = "prediction_result.txt"
        if not check_file_exists(prediction_file, "Prediction result"):
            print("[ERROR] No prediction result available.")
    
    # Check if prediction result was created
    prediction_file = "prediction_result.txt"
    if check_file_exists(prediction_file, "Prediction result"):
        print("\n[RESULTS] Final Results:")
        try:
            with open(prediction_file, 'r') as f:
                result_content = f.read()
                print(result_content)
        except Exception as e:
            print(f"[ERROR] Error reading prediction result: {e}")
    else:
        print("[WARNING] No prediction result file found")
    
    # Copy results to output directory
    print(f"\n[COPY] Copying results to {output_dir}")
    try:
        import shutil
        for file in [video_file, csv_file, cleaned_file, prediction_file]:
            if os.path.exists(file):
                shutil.copy2(file, output_dir)
                print(f"[COPIED] Copied {file} to {output_dir}")
    except Exception as e:
        print(f"[WARNING] Could not copy files to output directory: {e}")
    
    print("\n[SUCCESS] Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main() 