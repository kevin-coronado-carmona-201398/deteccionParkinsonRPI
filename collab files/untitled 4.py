# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

def impute(series, short=4, med=20):
    """Impute missing values using linear interpolation and cubic spline"""
    s = series.copy()
    mask = s.isna()
    groups = (mask != mask.shift()).cumsum()
    for _, block in s.groupby(groups):
        if not block.isna().all(): continue
        start = block.index[0] - 1
        end = block.index[-1] + 1
        length = end - start - 1
        if start < 0 or end >= len(s): continue
        b, a = s.iat[start], s.iat[end]
        # Short gap: linear
        if length <= short:
            for i in range(1, length+1):
                ratio = i/(length+1)
                s.iat[start+i] = b + ratio*(a-b)
        # Medium gap: spline if enough neighbors
        elif length <= med:
            idxs = [i for i in range(start-3, end+4) if 0<=i<len(s) and not pd.isna(s.iat[i])]
            if len(idxs) >= 4:
                xs, ys = np.array(idxs), s.iloc[idxs].values
                cs = CubicSpline(xs, ys)
                for i in range(1, length+1): s.iat[start+i] = cs(start+i)
            else:
                for i in range(1, length+1):
                    ratio = i/(length+1)
                    s.iat[start+i] = b + ratio*(a-b)
        # Long gaps: leave NaN
    return s

def preprocess_eye_tracking_data(input_file, output_file="cleaned_file.csv"):
    """
    Preprocess eye tracking data from CSV file
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save cleaned CSV file
    """
    print(f"Loading data from {input_file}...")
    
    # Load data
    try:
        if input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return False
    
    print("Column names:")
    print(df.columns.tolist())

    # Drop irrelevant columns
    columns_to_drop = [
        "Video Time [h:m:s:ms]", "Annotation Name", "Annotation Description", "Annotation Tags",
        "Mouse Position X [px]", "Mouse Position Y [px]",
        "Scroll Direction zX", "Scroll Direction Y", "Content","Stimulus","Export Start Trial Time [ms]","AOI Name Binocular",
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Convert relevant columns to numeric data types
    columns_to_convert = [
        "Pupil Size Right X [px]", "Pupil Size Right Y [px]", "Pupil Diameter Right [mm]",
        "Pupil Size Left X [px]", "Pupil Size Left Y [px]", "Pupil Diameter Left [mm]",
        "Point of Regard Binocular X [px]", "Point of Regard Binocular Y [px]",
        "Point of Regard Right X [px]", "Point of Regard Right Y [px]",
        "Point of Regard Left X [px]", "Point of Regard Left Y [px]",
        "Gaze Vector Right X", "Gaze Vector Right Y", "Gaze Vector Right Z",
        "Gaze Vector Left X", "Gaze Vector Left Y", "Gaze Vector Left Z"
    ]
    
    # Only convert columns that exist
    existing_columns = [col for col in columns_to_convert if col in df.columns]
    df[existing_columns] = df[existing_columns].apply(pd.to_numeric, errors="coerce")

    # Impute missing values
    print("Imputing missing values...")
    for col in ['Pupil Diameter Left [mm]', 'Pupil Diameter Right [mm]',
                'Point of Regard Binocular X [px]', 'Point of Regard Binocular Y [px]',
                'Tracking Ratio [%]']:
        if col in df.columns:
            df[col] = impute(df[col])

    # Calculate derived features
    print("Calculating derived features...")
    
    # Calculate pupil diameter avg - use the actual column names from the CSV
    if 'left_pupil_diameter' in df.columns and 'right_pupil_diameter' in df.columns:
        pupil_diameter_avg = df[['left_pupil_diameter', 'right_pupil_diameter']].mean(axis=1)
    elif 'Pupil Diameter Left [mm]' in df.columns and 'Pupil Diameter Right [mm]' in df.columns:
        pupil_diameter_avg = df[['Pupil Diameter Left [mm]', 'Pupil Diameter Right [mm]']].mean(axis=1)
    else:
        print("[WARNING] Pupil diameter columns not found, using pupil_size column")
        pupil_diameter_avg = df.get('pupil_size', pd.Series([0] * len(df)))

    # Calculate time in seconds - handle different timestamp formats
    if 'timestamp' in df.columns:
        # If timestamp is already in seconds
        recording_time_s = df['timestamp']
    elif 'RecordingTime [ms]' in df.columns:
        recording_time_s = df['RecordingTime [ms]'] / 1000.0
    else:
        print("[WARNING] Time column not found, using index as time")
        recording_time_s = pd.Series(range(len(df))) * 0.004  # Assume 4ms intervals

    # Calculate Point of Regard velocity - handle different column names
    por_x_col = None
    por_y_col = None
    
    if 'PoR_binocular_x' in df.columns and 'PoR_binocular_y' in df.columns:
        por_x_col = 'PoR_binocular_x'
        por_y_col = 'PoR_binocular_y'
    elif 'Point of Regard Binocular X [px]' in df.columns and 'Point of Regard Binocular Y [px]' in df.columns:
        por_x_col = 'Point of Regard Binocular X [px]'
        por_y_col = 'Point of Regard Binocular Y [px]'
    elif 'gaze_x' in df.columns and 'gaze_y' in df.columns:
        por_x_col = 'gaze_x'
        por_y_col = 'gaze_y'
    
    if por_x_col and por_y_col:
        POR_X_diff = df[por_x_col].diff()
        POR_Y_diff = df[por_y_col].diff()
        time_diff = recording_time_s.diff()
        POR_velocity = np.sqrt(POR_X_diff**2 + POR_Y_diff**2) / time_diff
        POR_velocity = POR_velocity.fillna(0)
    else:
        print("[WARNING] Point of Regard columns not found, using saccade_velocity column")
        POR_velocity = df.get('saccade_velocity', pd.Series([0] * len(df)))

    # Initialize the category
    df['Category Binocular'] = 'uncategorized'

    # Blink detection: Pupil size drops or tracking ratio = 0
    # Check if we have tracking ratio column
    tracking_ratio_col = None
    if 'Tracking Ratio [%]' in df.columns:
        tracking_ratio_col = 'Tracking Ratio [%]'
    
    blink_mask = (pupil_diameter_avg < 1.5)
    if tracking_ratio_col:
        blink_mask = blink_mask | (df[tracking_ratio_col] < 10)
    
    df.loc[blink_mask, 'Category Binocular'] = 'blink'

    # Saccade detection: Higher velocity threshold, short duration
    saccade_mask = (
        (POR_velocity > 1200) &  # Increased from 100
        (time_diff < 0.15) &
        (~blink_mask)
    )
    df.loc[saccade_mask, 'Category Binocular'] = 'saccade'

    # Visual Intake: More tolerant velocity and tracking ratio
    visual_intake_mask = (
        (POR_velocity < 1200) &  # Increased from 100
        (~blink_mask) &
        (~saccade_mask)
    )
    if tracking_ratio_col:
        visual_intake_mask = visual_intake_mask & (df[tracking_ratio_col] > 5)
    
    df.loc[visual_intake_mask, 'Category Binocular'] = 'visual intake'

    # Count events
    visual_intake_count = (df['Category Binocular'] == 'visual intake').sum()
    print(f"Number of Visual Intake events: {visual_intake_count}")

    # Column mapping for standardization - updated to match actual CSV columns
    column_mapping = {
        'timestamp': 'timestamp',
        'RecordingTime [ms]': 'RecordingTime [ms]',
        'gaze_x': 'gaze_x',
        'gaze_y': 'gaze_y',
        'left_pupil_x': 'left_pupil_x',
        'left_pupil_y': 'left_pupil_y',
        'left_pupil_diameter': 'left_pupil_diameter',
        'right_pupil_x': 'right_pupil_x',
        'right_pupil_y': 'right_pupil_y',
        'right_pupil_diameter': 'right_pupil_diameter',
        'PoR_binocular_x': 'PoR_binocular_x',
        'PoR_binocular_y': 'PoR_binocular_y',
        'Point of Regard Right X': 'Point of Regard Right X',
        'Point of Regard Right Y': 'Point of Regard Right Y',
        'Point of Regard Left X': 'Point of Regard Left X',
        'Point of Regard Left Y': 'Point of Regard Left Y',
        'Category Binocular': 'Category Binocular',
        'Index Binocular': 'Index Binocular',
    }

    # Final column order - keep original column names
    final_columns = ['timestamp', 'gaze_x', 'gaze_y', 'blink',
                     'saccade_velocity', 'fixation', 'pupil_size',
                     'left_pupil_x', 'left_pupil_y', 'left_pupil_diameter',
                     'right_pupil_x', 'right_pupil_y', 'right_pupil_diameter',
                     'PoR_binocular_x', 'PoR_binocular_y',
                     'Point of Regard Right X', 'Point of Regard Right Y',
                     'Point of Regard Left X', 'Point of Regard Left Y',
                     'Category Binocular', 'Index Binocular']

    # Select and rename required columns - keep original names
    existing_columns = [col for col in final_columns if col in df.columns]
    df_cleaned = df[existing_columns].copy()
    
    # Add missing columns
    missing_columns = ['blink', 'saccade_velocity', 'fixation', 'pupil_size']
    for col in missing_columns:
        if col not in df_cleaned.columns:
            if col == 'blink':
                df_cleaned[col] = df_cleaned['Category Binocular'].apply(lambda x: 1 if str(x).lower() == 'blink' else 0)
            elif col == 'saccade_velocity':
                df_cleaned[col] = POR_velocity
            elif col == 'fixation':
                df_cleaned[col] = df_cleaned['Category Binocular'].apply(lambda x: 1 if str(x).lower() == 'visual intake' else 0)
            elif col == 'pupil_size':
                df_cleaned[col] = pupil_diameter_avg
            else:
                df_cleaned[col] = None

    # Mark blink rows
    df_cleaned['blink'] = df_cleaned['Category Binocular'].apply(lambda x: 1 if str(x).lower() == 'blink' else 0)

    # Calculate saccade velocity
    time_diff_s = df_cleaned['RecordingTime [ms]'].diff() / 1000.0
    gaze_x_diff = df_cleaned['gaze_x'].diff()
    gaze_y_diff = df_cleaned['gaze_y'].diff()

    saccade_velocity = np.sqrt(gaze_x_diff**2 + gaze_y_diff**2) / time_diff_s
    saccade_velocity = saccade_velocity.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_cleaned['saccade_velocity'] = saccade_velocity

    # Identify fixations: velocity < 50 pixels/second
    df_cleaned['fixation'] = df_cleaned.apply(
        lambda row: 1 if (row['saccade_velocity'] < 50 and row['saccade_velocity'] > 0 and row['blink'] == 0) else 0,
        axis=1
    )

    # Calculate pupil size as average of left and right diameters
    if 'left_pupil_diameter' in df_cleaned.columns and 'right_pupil_diameter' in df_cleaned.columns:
        df_cleaned['pupil_size'] = df_cleaned[['left_pupil_diameter', 'right_pupil_diameter']].mean(axis=1)
    else:
        df_cleaned['pupil_size'] = 0

    # Reorder columns to match final_columns
    existing_final_columns = [col for col in final_columns if col in df_cleaned.columns]
    df_cleaned = df_cleaned[existing_final_columns]

    # Sort by time
    df_cleaned = df_cleaned.sort_values('RecordingTime [ms]').reset_index(drop=True)

    # Clean possible spaces in Category Binocular
    df_cleaned['Category Binocular'] = df_cleaned['Category Binocular'].str.strip()

    # Mark groups of continuous blinks
    df_cleaned['group_change'] = (df_cleaned['blink'] != df_cleaned['blink'].shift()).cumsum()

    # Filter only Blink = 1 groups
    blink_groups = df_cleaned[df_cleaned['blink'] == 1].groupby('group_change').agg({
        'RecordingTime [ms]': ['first', 'last'],
        'blink': 'count'
    }).reset_index()

    blink_groups.columns = ['blink_group', 'start_time', 'end_time', 'blink_length']

    # Calculate blink duration in ms
    blink_groups['blink_duration'] = blink_groups['end_time'] - blink_groups['start_time']

    # Classify short and long blinks (threshold 250 ms)
    short_blinks = blink_groups[blink_groups['blink_duration'] < 250]
    long_blinks = blink_groups[blink_groups['blink_duration'] >= 250]

    print(f"Total short blinks: {len(short_blinks)}")
    print(f"Total long blinks: {len(long_blinks)}")

    # Mark short and long blink rows in the main DataFrame
    short_blink_groups = short_blinks['blink_group'].values
    long_blink_groups = long_blinks['blink_group'].values

    df_cleaned['is_short_blink'] = df_cleaned['group_change'].isin(short_blink_groups).astype(int)
    df_cleaned['is_long_blink'] = df_cleaned['group_change'].isin(long_blink_groups).astype(int)

    # Find best window (if data is large enough)
    window_size = 10000
    if len(df_cleaned) >= window_size:
        # Create binary indicators
        df_cleaned['is_uncategorized'] = (df_cleaned['Category Binocular'].str.lower() == 'uncategorized').astype(int)
        df_cleaned['is_long_blink'] = df_cleaned['is_long_blink'].fillna(0).astype(int)

        # Compute rolling sums
        long_blinks_rolling = df_cleaned['is_long_blink'].rolling(window=window_size).sum()
        uncategorized_rolling = df_cleaned['is_uncategorized'].rolling(window=window_size).sum()

        # Build a combined score
        penalty_weight = 5
        final_score = (long_blinks_rolling * penalty_weight) + uncategorized_rolling

        # Find the window with the minimum score
        best_window_end = final_score.dropna().idxmin()
        best_window_start = best_window_end - window_size + 1

        best_segment = df_cleaned.iloc[best_window_start:best_window_end + 1]

        print(f"✅ Best 10,000-row window found!")
        print(f"🔹 Start index: {best_window_start}")
        print(f"🔹 End index: {best_window_end}")
        print(f"🔹 Total long blinks: {best_segment['is_long_blink'].sum()}")
        print(f"🔹 Total uncategorized: {best_segment['is_uncategorized'].sum()}")

        # Save best segment
        best_segment.to_csv("best_10000_segment.csv", index=False)
        print("✅ Saved as 'best_10000_segment.csv'")
        
        # Use best segment for final output
        df_final = best_segment
    else:
        print(f"❌ Data too small for window size {window_size}, using all data")
        df_final = df_cleaned

    # Save final cleaned file
    df_final.to_csv(output_file, index=False)
    print(f"✅ Final cleaned file saved as '{output_file}'")
    
    return True

if __name__ == "__main__":
    # Get input and output paths from environment variables or use defaults
    input_file = os.getenv('INPUT_CSV', 'eye_metrics_output.csv')
    output_file = os.getenv('OUTPUT_CSV', 'cleaned_file.csv')
    
    success = preprocess_eye_tracking_data(input_file, output_file)
    sys.exit(0 if success else 1)