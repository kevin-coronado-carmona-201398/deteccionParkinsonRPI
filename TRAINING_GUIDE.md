# Complete Training Guide - Step by Step

This guide will walk you through the entire process of training the Parkinson's eye tracking classifier models from start to finish.

## Prerequisites

Make sure you have:
- Python 3.7+ installed
- All required packages installed (run `pip install -r requirements.txt`)
- Your dataset in `dataset (3)/dataset/new folder/` with `patients/` and `non_patients/` subfolders

---

## Step 1: Prepare the Training Data

The training script expects all CSV files in one folder with specific naming:
- Files starting with `P_` = Parkinson's patients (label = 1)
- Files starting with `N_` = Non-Parkinson's controls (label = 0)

**Run this command:**

```bash
python prepare_training_data.py
```

**What this does:**
- Copies all CSV files from `dataset (3)/dataset/new folder/patients/` 
- Renames them with `P_` prefix (e.g., `P_001_best_segment_cleaned (16).csv`)
- Copies all CSV files from `dataset (3)/dataset/new folder/non_patients/`
- Renames them with `N_` prefix (e.g., `N_001_processed_eye_data (14).csv`)
- Saves everything to `training_data/` folder

**Expected output:**
```
[INFO] Preparing training data from dataset (3)/dataset/new folder
[INFO] Created output folder: training_data
[INFO] Processing patient files from dataset (3)/dataset/new folder/patients
[INFO] Found 11 patient files
  Copied: best_segment_cleaned (16).csv -> P_001_best_segment_cleaned (16).csv
  ...
[INFO] Processing non-patient files from dataset (3)/dataset/new folder/non_patients
[INFO] Found 11 non-patient files
  Copied: processed_eye_data (14).csv -> N_001_processed_eye_data (14).csv
  ...
[SUCCESS] Training data preparation complete!
  Total files: 22
  Patient files (P_): 11
  Non-patient files (N_): 11
```

---

## Step 2: Train the Models

Now train all three models (Random Forest, KNN, and LSTM):

**Run this command:**

```bash
python train_model.py
```

**What this does:**
1. Loads all CSV files from `training_data/` folder
2. Extracts features from each file
3. Trains Random Forest classifier
4. Trains K-Nearest Neighbors (KNN) classifier
5. Trains LSTM neural network
6. Evaluates each model using Leave-One-Out Cross-Validation
7. **Saves all models to `saved_models/` folder**

**Expected output:**
```
============================================================
Parkinson's Eye Tracking Model Training
============================================================
Data folder: training_data

Found 22 CSV files:
  Patient files (P_): 11
  Non-patient files (N_): 11

============================================================
Starting model training...
============================================================
Training models...
Dataset shape: (22, 8)
Labels: {0: 11, 1: 11}

Training Random Forest...
Random Forest LOOCV Accuracy: 0.XX
[Classification report]

Training KNN...
KNN LOOCV Accuracy: 0.XX
[Classification report]

Training LSTM...
Epoch 1/30
...
LSTM Accuracy: 0.XX
[Classification report]

============================================================
[SUCCESS] Model training completed!
============================================================

Saving models...
  ✓ Saved Random Forest: saved_models/random_forest_20241215_143022.pkl
  ✓ Saved KNN: saved_models/knn_20241215_143022.pkl
  ✓ Saved LSTM: saved_models/lstm_20241215_143022
  ✓ Saved metadata: saved_models/metadata_20241215_143022.pkl

============================================================
All models saved successfully!
============================================================

Models saved in: C:\Users\91981\PycharmProjects\prototype_1\saved_models
Timestamp: 20241215_143022
```

**Training time:** This may take several minutes depending on your dataset size and computer speed.

---

## Step 3: Verify Models Were Saved

Check that the models were saved:

**On Windows:**
```bash
dir saved_models
```

**On Linux/Mac:**
```bash
ls saved_models
```

You should see files like:
- `random_forest_YYYYMMDD_HHMMSS.pkl`
- `knn_YYYYMMDD_HHMMSS.pkl`
- `lstm_YYYYMMDD_HHMMSS/` (folder)
- `metadata_YYYYMMDD_HHMMSS.pkl`

---

## Step 4: Use the Trained Models for Prediction

Now you can use the saved models to make predictions on new data:

**Option A: Using the load_model.py script**

```bash
python load_model.py cleaned_file.csv lstm
```

This will:
- Load the most recent trained models
- Make a prediction on `cleaned_file.csv`
- Display the result

**Option B: Using the main classifier script**

The main classifier script can also use trained models. You'll need to modify it to load saved models, or it will train new ones if no saved models are found.

---

## Quick Reference: All Commands

```bash
# Step 1: Prepare data
python prepare_training_data.py

# Step 2: Train models
python train_model.py

# Step 3: Use models for prediction
python load_model.py cleaned_file.csv lstm
```

---

## Troubleshooting

### Error: "Data folder not found"
- Make sure you ran `prepare_training_data.py` first
- Or set the `DATA_FOLDER` environment variable:
  ```bash
  set DATA_FOLDER=path/to/your/data
  python train_model.py
  ```

### Error: "No CSV files found"
- Check that your CSV files are in the correct location
- Verify files have `.csv` extension
- Make sure files are named with `P_` or `N_` prefix

### Error: "Need both P_ and N_ prefixed files"
- You need both patient and non-patient data for training
- Check that you have files starting with both `P_` and `N_`

### Models not saving
- Check that you have write permissions in the project directory
- Make sure there's enough disk space
- Check the error messages for specific issues

---

## What Each Model Does

1. **Random Forest**: Traditional machine learning using engineered features (blink ratio, saccade velocity, etc.)
2. **KNN**: Simple distance-based classification
3. **LSTM**: Deep learning neural network that processes sequences of eye tracking data

All three models are trained and saved, but the LSTM is typically used for predictions as it's designed for sequence data.

---

## Next Steps

After training:
- Models are saved in `saved_models/` folder
- You can use them for predictions on new eye tracking data
- You can retrain anytime with new data
- Each training session creates a new timestamped set of models

