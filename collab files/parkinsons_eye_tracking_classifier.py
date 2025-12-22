# -*- coding: utf-8 -*-
"""
Parkinson's Eye Tracking Classifier
"""

import pandas as pd
import numpy as np
import os
import sys
import zipfile
import shutil
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def extract_features(file_path):
    """Extract features from eye tracking CSV file"""
    df = pd.read_csv(file_path)

    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Fix naming differences
    cat_col = 'category_binocular' if 'category_binocular' in df.columns else 'category binocular'

    # Basic derived features
    total_rows = len(df)
    blink_ratio = (df[cat_col] == 'Blink').sum() / total_rows
    saccade_ratio = (df[cat_col] == 'Saccade').sum() / total_rows
    visual_ratio = ((df[cat_col] == 'Visual Intake') | (df[cat_col] == 'Fixation')).sum() / total_rows

    mean_saccade_velocity = df['saccade_velocity'].mean()
    mean_pupil_size = df['pupil_size'].mean()
    std_pupil_size = df['pupil_size'].std()

    # Optional blink features
    short_blink_count = df['is_short_blink'].sum() if 'is_short_blink' in df.columns else 0
    long_blink_count = df['is_long_blink'].sum() if 'is_long_blink' in df.columns else 0

    return {
        'blink_ratio': blink_ratio,
        'saccade_ratio': saccade_ratio,
        'visual_ratio': visual_ratio,
        'mean_saccade_velocity': mean_saccade_velocity,
        'mean_pupil_size': mean_pupil_size,
        'std_pupil_size': std_pupil_size,
        'short_blink_count': short_blink_count,
        'long_blink_count': long_blink_count
    }

def load_sequence(file_path, max_len=300):
    """Load sequence data from CSV file"""
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Use category_binocular to define blink
    cat_col = 'category_binocular' if 'category_binocular' in df.columns else 'category binocular'
    df['blink_bin'] = df[cat_col].apply(lambda x: 1 if str(x).strip().lower() == 'blink' else 0)

    # Select features
    features = ['gaze_x', 'gaze_y', 'saccade_velocity', 'pupil_size', 'blink_bin']
    df = df[[f for f in features if f in df.columns]].copy()

    # Fill missing
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Pad or truncate
    if len(df) > max_len:
        df = df.iloc[:max_len]
    else:
        pad_len = max_len - len(df)
        df = pd.concat([df, pd.DataFrame(np.zeros((pad_len, df.shape[1])), columns=df.columns)], axis=0)

    return df.values

def load_windowed_sequences(file_path, label, window_size=500, step=500):
    """Load windowed sequences from CSV file"""
    df = pd.read_csv(file_path)
    df.columns = [c.lower().strip() for c in df.columns]

    cat_col = 'category_binocular' if 'category_binocular' in df.columns else 'category binocular'
    df['blink_bin'] = df[cat_col].apply(lambda x: 1 if str(x).strip().lower() == 'blink' else 0)

    features = ['gaze_x', 'gaze_y', 'saccade_velocity', 'pupil_size', 'blink_bin']
    df = df[[f for f in features if f in df.columns]].copy()
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    sequences = []
    labels = []

    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start + window_size].values
        sequences.append(window)
        labels.append(label)

    return sequences, labels

def train_models(data_folder):
    """Train multiple models on the dataset"""
    print("Training models...")
    
    # Extract features for traditional ML
    data = []
    labels = []
    
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(data_folder, file)
            features = extract_features(file_path)
            label = 1 if file.startswith('P_') else 0
            data.append(features)
            labels.append(label)

    X = pd.DataFrame(data)
    y = pd.Series(labels)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels: {y.value_counts().to_dict()}")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        y_true.append(y_test.values[0])
        y_pred.append(pred[0])

    print("Random Forest LOOCV Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # Train KNN
    print("\nTraining KNN...")
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        y_true.append(y_test.values[0])
        y_pred.append(pred[0])

    print("KNN LOOCV Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # Train LSTM
    print("\nTraining LSTM...")
    X_lstm = []
    y_lstm = []

    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(data_folder, file)
            label = 1 if file.startswith('P_') else 0
            X_lstm.append(load_sequence(file_path))
            y_lstm.append(label)

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    # LSTM model
    lstm_model = Sequential([
        LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_lstm, y_lstm, epochs=30, batch_size=1, verbose=1)

    # LSTM evaluation
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X_lstm):
        X_train, X_test = X_lstm[train_index], X_lstm[test_index]
        y_train, y_test = y_lstm[train_index], y_lstm[test_index]

        model = tf.keras.models.clone_model(lstm_model)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=62, batch_size=1, verbose=0)

        pred = (model.predict(X_test)[0][0] > 0.5).astype(int)

        y_true.append(y_test[0])
        y_pred.append(pred)

    print("LSTM Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    return {
        'random_forest': rf_model,
        'knn': KNeighborsClassifier(n_neighbors=1),
        'lstm': lstm_model,
        'feature_data': (X, y),
        'sequence_data': (X_lstm, y_lstm)
    }

def load_latest_saved_models(models_dir='saved_models', timestamp=None):
    """Load latest saved models (if available) from saved_models directory"""
    try:
        if not os.path.exists(models_dir):
            print(f"No saved models directory found at {models_dir}")
            return None
        
        metadata_files = [f for f in os.listdir(models_dir) if f.startswith('metadata_') and f.endswith('.pkl')]
        if not metadata_files:
            print("No saved model metadata files found")
            return None
        
        if timestamp:
            metadata_name = f"metadata_{timestamp}.pkl"
            if metadata_name not in metadata_files:
                print(f"Requested timestamp {timestamp} not found")
                return None
        else:
            metadata_files.sort(reverse=True)
            metadata_name = metadata_files[0]
        
        metadata_path = os.path.join(models_dir, metadata_name)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Loading saved models from timestamp: {metadata.get('timestamp', 'unknown')}")
        
        models = {}
        paths = metadata.get('model_paths', {})
        
        # Load Random Forest
        rf_path = paths.get('random_forest')
        if rf_path and os.path.exists(rf_path):
            try:
                models['random_forest'] = joblib.load(rf_path)
                print(f"Loaded Random Forest model: {rf_path}")
            except Exception as e:
                print(f"Warning: Failed to load Random Forest model: {e}")
        
        # Load KNN
        knn_path = paths.get('knn')
        if knn_path and os.path.exists(knn_path):
            try:
                models['knn'] = joblib.load(knn_path)
                print(f"Loaded KNN model: {knn_path}")
            except Exception as e:
                print(f"Warning: Failed to load KNN model: {e}")
        
        # Load LSTM
        lstm_path = paths.get('lstm')
        if lstm_path and os.path.exists(lstm_path):
            try:
                models['lstm'] = tf.keras.models.load_model(lstm_path)
                print(f"Loaded LSTM model: {lstm_path}")
            except Exception as e:
                print(f"Warning: Failed to load LSTM model: {e}")
        
        if not models:
            print("No models could be loaded from saved artifacts")
            return None
        
        return models
    except Exception as e:
        print(f"Error loading saved models: {e}")
        return None

def predict_parkinsons(input_file, models, output_file="prediction_result.txt"):
    """Predict Parkinson's disease from a single CSV file"""
    print(f"Making prediction for {input_file}...")
    
    try:
        # Load and preprocess the input file
        X_new = load_sequence(input_file)
        X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
        
        # Make prediction with LSTM model
        prediction = models['lstm'].predict(X_new)[0][0]
        
        # Determine result
        if prediction > 0.6:
            result = "Parkinson's detected"
            confidence = "High"
        elif prediction < 0.4:
            result = "Non-Parkinson's"
            confidence = "High"
        else:
            result = "Uncertain"
            confidence = "Low"
        
        # Save prediction result
        with open(output_file, 'w') as f:
            f.write(f"Input File: {input_file}\n")
            f.write(f"Prediction: {result}\n")
            f.write(f"Probability: {prediction:.4f}\n")
            f.write(f"Confidence: {confidence}\n")
        
        print(f"Prediction: {result} (Prob = {prediction:.4f})")
        print(f"Result saved to {output_file}")
        
        return prediction, result
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def main():
    """Main function to run the complete pipeline"""
    # Get paths from environment variables
    data_folder = os.getenv('DATA_FOLDER', 'new_folder')
    input_file = os.getenv('INPUT_FILE', 'cleaned_file.csv')
    output_file = os.getenv('OUTPUT_FILE', 'prediction_result.txt')
    models_dir = os.getenv('MODELS_DIR', 'saved_models')
    prefer_saved_models = os.getenv('USE_SAVED_MODELS', '1').lower() in ('1', 'true', 'yes', 'y')
    
    models = None
    
    if prefer_saved_models:
        print("Attempting to load saved models first...")
        models = load_latest_saved_models(models_dir=models_dir)
        if models is None:
            print("Saved models unavailable. Falling back to training if data is provided.")
    
    # If saved models were not loaded (or not requested), fall back to training
    if models is None:
        has_training_csv = os.path.exists(data_folder) and any(
            f.endswith('.csv') for f in os.listdir(data_folder)
        )
        if has_training_csv:
            print("Training models on available dataset...")
            models = train_models(data_folder)
        else:
            print("No training data found and no saved models available.")
            print("Please provide P_/N_ prefixed CSV files or run train_model.py first.")
            return
    
    # Make prediction on input file
    if os.path.exists(input_file):
        predict_parkinsons(input_file, models, output_file)
    else:
        print(f"Input file {input_file} not found. Set INPUT_FILE to a valid CSV.")

if __name__ == "__main__":
    main()