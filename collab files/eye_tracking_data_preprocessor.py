# -*- coding: utf-8 -*-
"""
Eye Tracking Data Preprocessor
Processes eye tracking data from CSV files
"""

import numpy as np
import pandas as pd
import os
import sys

def preprocess_eye_tracking_data(input_file, output_file="cleaned_file.csv"):
    """
    Preprocess eye tracking data from CSV file
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save cleaned CSV file
    """
    print(f"[INFO] Loading data from {input_file}")
    
    # Load the data
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    print(f"[INFO] Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"[INFO] Original columns: {df.columns.tolist()}")

    # Normalize column names: lowercase, strip, replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"[INFO] Normalized columns: {df.columns.tolist()}")

    # Check if required columns exist
    required_columns = ['por_binocular_x', 'por_binocular_y', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"[ERROR] Missing required columns: {missing_columns}")
        print(f"[INFO] Available columns: {df.columns.tolist()}")
        return False

    # Apply a 5‑sample median filter (~20 ms) to reduce jitter
    df['gx_med'] = df['por_binocular_x'].rolling(window=5, center=True, min_periods=1).median()
    df['gy_med'] = df['por_binocular_y'].rolling(window=5, center=True, min_periods=1).median()

    print(f"[INFO] Applied median filter to gaze coordinates")

    # Compute delta positions & times
    dx = df['gx_med'].diff()
    dy = df['gy_med'].diff()
    dt = df['timestamp'].diff()         # timestamp already in seconds

    # Avoid divide‑by‑zero
    dt = dt.replace(0, np.nan)

    # Velocity (px/sec)
    df['saccade_velocity'] = np.sqrt(dx**2 + dy**2) / dt
    df['saccade_velocity'].fillna(0, inplace=True)

    print(f"[INFO] Calculated saccade velocity")

    # Parameters: you can tune these
    VEL_THRESH = 500    # px/sec
    DUR_THRESH = 0.02   # seconds

    # Initial saccade mask
    sv_mask = df['saccade_velocity'] > VEL_THRESH

    # Label contiguous runs
    df['sv_grp'] = (sv_mask != sv_mask.shift(1)).cumsum()

    # Compute run durations
    grp_stats = df.groupby('sv_grp').agg(
        is_saccade=('sv_grp','size'),
        start_time=('timestamp','first'),
        end_time=('timestamp','last')
    )
    grp_stats['duration'] = grp_stats['end_time'] - grp_stats['start_time']

    # Identify short saccade groups
    short_grps = grp_stats[(grp_stats['is_saccade']>0) &
                           (grp_stats['duration'] < DUR_THRESH)].index

    # Zero‑out micro‑saccades
    df.loc[df['sv_grp'].isin(short_grps), 'saccade_velocity'] = 0

    print(f"[INFO] Filtered out micro-saccades")

    # Initialize
    df['category_binocular'] = 'uncategorized'

    # Check if blink column exists, if not create it
    if 'blink' not in df.columns:
        print(f"[WARNING] Blink column not found, creating default blink column")
        df['blink'] = 0

    # Blink: low velocity OR explicit blink flag
    blink_mask = (df['saccade_velocity'] < 10) | (df['blink'] != 0)
    df.loc[blink_mask, 'category_binocular'] = 'blink'

    # Saccade: high velocity & not blink
    saccade_mask = (df['saccade_velocity'] > VEL_THRESH) & (~blink_mask)
    df.loc[saccade_mask, 'category_binocular'] = 'saccade'

    # Visual intake: everything else
    visual_intake_mask = ~(blink_mask | saccade_mask)
    df.loc[visual_intake_mask, 'category_binocular'] = 'visual intake'

    print(f"[INFO] Categorized eye movements")

    # Create 'RecordingTime [ms]' by subtracting the first timestamp
    df['RecordingTime [ms]'] = (df['timestamp'] - df['timestamp'].iloc[0]) * 1000

    # Print statistics
    print("\n[STATS] Eye Movement Counts:")
    print(f"  Blinks:       {(df['category_binocular']=='blink').sum()}")
    print(f"  Saccades:     {(df['category_binocular']=='saccade').sum()}")
    print(f"  Visual intake:{(df['category_binocular']=='visual intake').sum()}")

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Saved processed data to {output_file}")
    
    return True

if __name__ == "__main__":
    # Get input and output paths from environment variables or use defaults
    input_file = os.getenv('INPUT_CSV', 'eye_metrics_output.csv')
    output_file = os.getenv('OUTPUT_CSV', 'cleaned_file.csv')
    
    success = preprocess_eye_tracking_data(input_file, output_file)
    sys.exit(0 if success else 1)

