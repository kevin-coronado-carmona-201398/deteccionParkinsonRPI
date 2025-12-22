#!/usr/bin/env python3
"""
Helper script to load saved models and make predictions
"""

import os
import sys
import pickle
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

def load_models(models_dir="saved_models", timestamp=None):
    """
    Load saved models from the models directory
    
    Args:
        models_dir: Directory containing saved models
        timestamp: Specific timestamp to load (if None, loads most recent)
    
    Returns:
        Dictionary containing loaded models and metadata
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find metadata files
    metadata_files = [f for f in os.listdir(models_dir) if f.startswith('metadata_')]
    if len(metadata_files) == 0:
        raise FileNotFoundError(f"No metadata files found in {models_dir}")
    
    if timestamp:
        metadata_file = f"metadata_{timestamp}.pkl"
        if metadata_file not in metadata_files:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    else:
        # Load most recent
        metadata_files.sort(reverse=True)
        metadata_file = metadata_files[0]
    
    metadata_path = os.path.join(models_dir, metadata_file)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loading models from timestamp: {metadata['timestamp']}")
    print(f"Training data: {metadata['data_folder']}")
    print(f"Samples: {metadata['num_samples']} ({metadata['num_patients']} patients, {metadata['num_controls']} controls)")
    
    models = {}
    
    # Load Random Forest
    if metadata['model_paths']['random_forest'] and os.path.exists(metadata['model_paths']['random_forest']):
        models['random_forest'] = joblib.load(metadata['model_paths']['random_forest'])
        print("  ✓ Loaded Random Forest")
    
    # Load KNN
    if metadata['model_paths']['knn'] and os.path.exists(metadata['model_paths']['knn']):
        models['knn'] = joblib.load(metadata['model_paths']['knn'])
        print("  ✓ Loaded KNN")
    
    # Load LSTM
    if metadata['model_paths']['lstm'] and os.path.exists(metadata['model_paths']['lstm']):
        models['lstm'] = tf.keras.models.load_model(metadata['model_paths']['lstm'])
        print("  ✓ Loaded LSTM")
    
    return {
        'models': models,
        'metadata': metadata
    }

def predict_with_models(input_file, models_dict, model_type='lstm'):
    """
    Make prediction using loaded models
    
    Args:
        input_file: Path to CSV file for prediction
        models_dict: Dictionary returned from load_models()
        model_type: Which model to use ('random_forest', 'knn', or 'lstm')
    
    Returns:
        Prediction result
    """
    if model_type not in models_dict['models']:
        raise ValueError(f"Model type '{model_type}' not found in loaded models")
    
    model = models_dict['models'][model_type]
    
    # Load and preprocess input file
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'collab files'))
    from parkinsons_eye_tracking_classifier import load_sequence, extract_features
    
    if model_type == 'lstm':
        # For LSTM, use sequence data
        X_new = load_sequence(input_file)
        X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
        prediction = model.predict(X_new)[0][0]
    else:
        # For Random Forest and KNN, use extracted features
        features = extract_features(input_file)
        X_new = pd.DataFrame([features])
        prediction = model.predict_proba(X_new)[0][1]  # Probability of class 1 (Parkinson's)
    
    return prediction

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_model.py <input_csv_file> [model_type] [timestamp]")
        print("  model_type: 'random_forest', 'knn', or 'lstm' (default: 'lstm')")
        print("  timestamp: Specific model timestamp to load (default: most recent)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'lstm'
    timestamp = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        # Load models
        models_dict = load_models(timestamp=timestamp)
        
        # Make prediction
        prediction = predict_with_models(input_file, models_dict, model_type)
        
        # Interpret result
        if prediction > 0.6:
            result = "🧠 Parkinson's"
            confidence = "High"
        elif prediction < 0.4:
            result = "✅ Non-Parkinson's"
            confidence = "High"
        else:
            result = "🤔 Uncertain"
            confidence = "Low"
        
        print(f"\n{'='*60}")
        print("Prediction Result")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"Model used: {model_type}")
        print(f"Prediction: {result}")
        print(f"Probability: {prediction:.4f}")
        print(f"Confidence: {confidence}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

