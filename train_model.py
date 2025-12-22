#!/usr/bin/env python3
"""
Script to train the Parkinson's eye tracking classifier models
"""

import os
import sys
import pickle
import joblib
from datetime import datetime

# Add the collab files directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'collab files'))

from parkinsons_eye_tracking_classifier import train_models

def main():
    """Main training function"""
    # Get data folder from environment variable or use default
    data_folder = os.getenv('DATA_FOLDER', 'training_data')
    
    print("="*60)
    print("Parkinson's Eye Tracking Model Training")
    print("="*60)
    print(f"Data folder: {data_folder}")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"\n[ERROR] Data folder not found: {data_folder}")
        print("\nTo prepare training data, run:")
        print("  python prepare_training_data.py")
        print("\nOr set DATA_FOLDER environment variable:")
        print("  set DATA_FOLDER=path/to/your/data")
        sys.exit(1)
    
    # Check if folder has CSV files
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    if len(csv_files) == 0:
        print(f"\n[ERROR] No CSV files found in {data_folder}")
        print("\nTo prepare training data, run:")
        print("  python prepare_training_data.py")
        sys.exit(1)
    
    p_files = [f for f in csv_files if f.startswith('P_')]
    n_files = [f for f in csv_files if f.startswith('N_')]
    
    print(f"\nFound {len(csv_files)} CSV files:")
    print(f"  Patient files (P_): {len(p_files)}")
    print(f"  Non-patient files (N_): {len(n_files)}")
    
    if len(p_files) == 0 or len(n_files) == 0:
        print("\n[WARNING] Need both P_ and N_ prefixed files for training!")
        print("Files should be named:")
        print("  P_*.csv for Parkinson's patients")
        print("  N_*.csv for non-Parkinson's controls")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Starting model training...")
    print("="*60)
    
    # Train models
    try:
        models = train_models(data_folder)
        print("\n" + "="*60)
        print("[SUCCESS] Model training completed!")
        print("="*60)
        
        # Save models
        models_dir = "saved_models"
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\nSaving models...")
        
        # Save Random Forest model
        rf_path = os.path.join(models_dir, f"random_forest_{timestamp}.pkl")
        # Note: The returned rf_model is not the trained one from LOOCV, 
        # but we can save a final trained model
        if 'random_forest' in models:
            # Train a final model on all data for saving
            from sklearn.ensemble import RandomForestClassifier
            X, y = models['feature_data']
            final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            final_rf.fit(X, y)
            joblib.dump(final_rf, rf_path)
            print(f"  ✓ Saved Random Forest: {rf_path}")
        
        # Save KNN model
        knn_path = os.path.join(models_dir, f"knn_{timestamp}.pkl")
        if 'knn' in models:
            # Train a final model on all data for saving
            from sklearn.neighbors import KNeighborsClassifier
            X, y = models['feature_data']
            final_knn = KNeighborsClassifier(n_neighbors=1)
            final_knn.fit(X, y)
            joblib.dump(final_knn, knn_path)
            print(f"  ✓ Saved KNN: {knn_path}")
        
        # Save LSTM model (Keras now requires explicit extension)
        lstm_path = os.path.join(models_dir, f"lstm_{timestamp}.keras")
        if 'lstm' in models:
            models['lstm'].save(lstm_path)
            print(f"  ✓ Saved LSTM: {lstm_path}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'data_folder': data_folder,
            'num_samples': len(models['feature_data'][0]),
            'num_patients': int(models['feature_data'][1].sum()),
            'num_controls': int((models['feature_data'][1] == 0).sum()),
            'model_paths': {
                'random_forest': rf_path if 'random_forest' in models else None,
                'knn': knn_path if 'knn' in models else None,
                'lstm': lstm_path if 'lstm' in models else None
            }
        }
        metadata_path = os.path.join(models_dir, f"metadata_{timestamp}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  ✓ Saved metadata: {metadata_path}")
        
        print("\n" + "="*60)
        print("All models saved successfully!")
        print("="*60)
        print(f"\nModels saved in: {os.path.abspath(models_dir)}")
        print(f"Timestamp: {timestamp}")
        print("\nTo load models later:")
        print(f"  Random Forest: joblib.load('{rf_path}')")
        print(f"  KNN: joblib.load('{knn_path}')")
        print(f"  LSTM: tf.keras.models.load_model('{lstm_path}')")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

