#!/usr/bin/env python3
"""
Utility script to run individual steps of the Parkinson's Eye Tracking Pipeline
"""

import os
import sys
import subprocess
import argparse
import signal
import time

# Global variable to track if user wants to skip current step
skip_current_step = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global skip_current_step
    print("\n[INTERRUPTED] Interrupted by user (Ctrl+C)")
    print("Press Ctrl+C again to skip this step, or wait to continue...")
    skip_current_step = True

def run_step(step_name, command, description, allow_skip=True):
    """Run a specific step of the pipeline with graceful interruption"""
    global skip_current_step
    
    print(f"\n{'='*60}")
    print(f"[RUNNING] Step: {step_name}")
    print(f"Description: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    # Reset flag
    skip_current_step = False
    
    while True:
        try:
            # Set up signal handler for this command
            original_handler = signal.signal(signal.SIGINT, signal_handler)
            
            result = subprocess.run(command, shell=True, check=True)
            
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            
            print(f"[SUCCESS] Step '{step_name}' completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            
            print(f"[ERROR] Step '{step_name}' failed with error code {e.returncode}")
            
            if allow_skip:
                print("\nOptions:")
                print("1. Press Enter to retry")
                print("2. Type 'skip' to skip this step")
                print("3. Type 'exit' to quit")
                
                choice = input("Your choice: ").strip().lower()
                if choice == 'skip':
                    print(f"[SKIP] Skipping step '{step_name}'")
                    return False
                elif choice == 'exit':
                    print("[EXIT] Exiting")
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
                print(f"[SKIP] Skipping step '{step_name}'")
                return False
            else:
                print("[CONTINUE] Continuing with current step...")
                time.sleep(1)
                continue
                
        except Exception as e:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            
            print(f"[ERROR] Error running step '{step_name}': {e}")
            
            if allow_skip:
                print("\nOptions:")
                print("1. Press Enter to retry")
                print("2. Type 'skip' to skip this step")
                print("3. Type 'exit' to quit")
                
                choice = input("Your choice: ").strip().lower()
                if choice == 'skip':
                    print(f"[SKIP] Skipping step '{step_name}'")
                    return False
                elif choice == 'exit':
                    print("[EXIT] Exiting")
                    sys.exit(0)
                else:
                    print("[RETRY] Retrying...")
                    continue
            else:
                return False

def main():
    parser = argparse.ArgumentParser(description='Run individual steps of the Parkinson\'s Eye Tracking Pipeline')
    parser.add_argument('step', choices=['record', 'extract', 'preprocess', 'classify', 'all'], 
                       help='Step to run')
    parser.add_argument('--video-url', default=os.getenv('VIDEO_URL', 'http://172.17.13.231:5000/video_feed'),
                       help='URL of the video stream')
    parser.add_argument('--max-duration', type=int, default=int(os.getenv('MAX_DURATION', '3000')),
                       help='Maximum recording duration in seconds')
    parser.add_argument('--input-video', default=os.getenv('INPUT_VIDEO', 'flask_record.mp4'),
                       help='Input video filename')
    parser.add_argument('--output-csv', default=os.getenv('OUTPUT_CSV', 'eye_metrics_output.csv'),
                       help='Output CSV filename')
    parser.add_argument('--input-csv', default=os.getenv('INPUT_CSV', 'eye_metrics_output.csv'),
                       help='Input CSV for preprocessing')
    parser.add_argument('--input-file', default=os.getenv('INPUT_FILE', 'cleaned_file.csv'),
                       help='Input file for classification')
    parser.add_argument('--output-file', default=os.getenv('OUTPUT_FILE', 'prediction_result.txt'),
                       help='Output prediction file')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['VIDEO_URL'] = args.video_url
    os.environ['MAX_DURATION'] = str(args.max_duration)
    os.environ['INPUT_VIDEO'] = args.input_video
    os.environ['OUTPUT_CSV'] = args.output_csv
    os.environ['INPUT_CSV'] = args.input_csv
    os.environ['INPUT_FILE'] = args.input_file
    os.environ['OUTPUT_FILE'] = args.output_file
    
    steps = {
        'record': {
            'command': 'python recording.py',
            'description': 'Record video from Raspberry Pi'
        },
        'extract': {
            'command': 'python try.py',
            'description': 'Extract eye tracking features from video'
        },
        'preprocess': {
            'command': f'python "{os.path.join("collab files", "eye_tracking_data_preprocessor.py")}"',
            'description': 'Preprocess eye tracking data'
        },
        'classify': {
            'command': f'python "{os.path.join("collab files", "parkinsons_eye_tracking_classifier.py")}"',
            'description': 'Run Parkinson\'s classification'
        }
    }
    
    if args.step == 'all':
        print("[START] Running complete pipeline...")
        print("[TIP] Tip: Press Ctrl+C during any step to skip it and continue")
        print("="*60)
        
        success = True
        for step_name, step_info in steps.items():
            if not run_step(step_name, step_info['command'], step_info['description']):
                success = False
                print(f"[WARNING] Step '{step_name}' was skipped or failed")
                
                # Check if we can continue with next steps
                if step_name == 'record':
                    if not os.path.exists('flask_record.mp4'):
                        print("[ERROR] No video file available. Cannot continue.")
                        break
                elif step_name == 'extract':
                    if not os.path.exists('eye_metrics_output.csv'):
                        print("[ERROR] No eye metrics CSV available. Cannot continue.")
                        break
                elif step_name == 'preprocess':
                    if not os.path.exists('cleaned_file.csv'):
                        print("[ERROR] No cleaned data CSV available. Cannot continue.")
                        break
                elif step_name == 'classify':
                    print("[WARNING] Classification step failed, but pipeline can continue")
        
        if success:
            print("\n[SUCCESS] Complete pipeline finished successfully!")
        else:
            print("\n[WARNING] Pipeline completed with some steps skipped or failed")
            sys.exit(1)
    else:
        step_info = steps[args.step]
        success = run_step(args.step, step_info['command'], step_info['description'])
        if not success:
            print(f"[WARNING] Step '{args.step}' was skipped or failed")
            sys.exit(1)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main() 