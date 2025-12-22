## Parkinson's Eye Tracking Detection Pipeline

This project implements a complete pipeline for detecting Parkinson's disease using eye‑tracking data.  
It records video from a Raspberry Pi camera, extracts eye‑movement metrics, preprocesses them, and then
uses traditional ML and deep learning models to classify whether a subject is likely to have Parkinson's.

The codebase is organized to support:
- **End‑to‑end prediction from camera to diagnosis**
- **Offline training on curated datasets**
- **Model loading and standalone prediction**
- **Model explainability with SHAP (`shap.py`)**

---

### Project Structure

- **`main_pipeline.py`**: Orchestrates the full recording → feature extraction → preprocessing → classification pipeline.
- **`recording.py`**: Records video from a Flask MJPEG stream (Raspberry Pi camera) to `flask_record.mp4`.
- **`try.py`**: Extracts eye‑tracking metrics from the recorded video and writes `eye_metrics_output.csv`.
- **`collab files/eye_tracking_data_preprocessor.py`**: Cleans and enriches eye‑tracking CSVs into `cleaned_file.csv`.
- **`collab files/parkinsons_eye_tracking_classifier.py`**: Trains models and/or runs classification on a cleaned CSV.
- **`train_model.py`**: Full training script for Random Forest, KNN, and LSTM on dataset in `training_data/`.
- **`prepare_training_data.py`**: Prepares training CSVs from `dataset (3)/dataset/new folder/` into `training_data/`.
- **`load_model.py`**: Utility to load previously saved models from `saved_models/` and run prediction on a CSV.
- **`shap.py`**: Optional SHAP analysis for explaining model predictions (feature importance and dependence plots).
- **`saved_models/`**: Stores trained models and metadata (`random_forest_*.pkl`, `knn_*.pkl`, `lstm_*.keras`, `metadata_*.pkl`).
- **`training_data/`**: Training CSVs (created by `prepare_training_data.py`).
- **`dataset (3)/dataset/new folder/`**: Original raw CSV datasets for patients and non‑patients.

---

### Requirements

- **OS**: Windows, Linux, or macOS
- **Python**: 3.7+
- **Hardware**:
  - For recording: Raspberry Pi (or any device) running a Flask MJPEG video server
  - For training/inference: CPU (GPU optional but recommended for TensorFlow)

#### Python Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

Core libraries (from `requirements.txt`):
- **Data / numerics**: `pandas`, `numpy`, `scipy`
- **Computer vision**: `opencv-python`, `mediapipe`
- **ML / DL**: `scikit-learn`, `tensorflow`
- **Plotting / utilities**: `matplotlib`, `seaborn`, `openpyxl`, `numba`, `requests`

Optional for explainability (`shap.py`):
- `shap`
- `torch`

Install optional extras if you want SHAP explanations:

```bash
pip install shap torch
```

---

### End‑to‑End Pipeline (Recommended)

The easiest way to run the full pipeline is via `main_pipeline.py`. It will:
1. Record a video from the Raspberry Pi stream (`recording.py`)
2. Extract eye‑tracking metrics into `eye_metrics_output.csv` (`try.py`)
3. Preprocess and categorize movements into `cleaned_file.csv` (`eye_tracking_data_preprocessor.py`)
4. Load or train models, then create `prediction_result.txt` (`parkinsons_eye_tracking_classifier.py`)

#### Run the pipeline directly (no Docker)

From the project root:

```bash
python main_pipeline.py
```

Key environment variables (all optional):

| Variable         | Default                                | Used by                         | Description                                      |
|------------------|----------------------------------------|---------------------------------|--------------------------------------------------|
| `VIDEO_URL`      | `http://172.17.13.231:5000/video_feed` | `recording.py` / `main_pipeline.py` | MJPEG stream URL from Raspberry Pi              |
| `MAX_DURATION`   | `3000`                                 | `recording.py` / `main_pipeline.py` | Max recording time in seconds                   |
| `OUTPUT_DIR`     | `/app/output` (when in container)      | `main_pipeline.py`              | Final output directory for copied artifacts      |
| `INPUT_VIDEO`    | `flask_record.mp4`                     | `try.py`                        | Input video path for feature extraction          |
| `OUTPUT_CSV`     | `eye_metrics_output.csv`               | `try.py` / preprocessor         | Output CSV from extraction / preprocessor        |
| `INPUT_CSV`      | `eye_metrics_output.csv`               | `eye_tracking_data_preprocessor.py` | Raw CSV for preprocessing                   |
| `INPUT_FILE`     | `cleaned_file.csv`                     | `parkinsons_eye_tracking_classifier.py` | Cleaned CSV for classification           |
| `OUTPUT_FILE`    | `prediction_result.txt`                | classifier / `recording.py`     | Prediction result text file                      |
| `MODELS_DIR`     | `saved_models`                         | classifier                      | Directory where models are stored                |
| `USE_SAVED_MODELS` | `1`                                  | classifier                      | Use existing models if available (`1/true/yes`)  |

Minimal example with a custom camera URL:

```bash
set VIDEO_URL=http://192.168.1.100:5000/video_feed  # Windows (cmd)
python main_pipeline.py
```

On Linux/macOS:

```bash
VIDEO_URL=http://192.168.1.100:5000/video_feed python main_pipeline.py
```

All intermediate and final files are copied to `OUTPUT_DIR` at the end of the pipeline.

---

### Running Steps Manually (Without `main_pipeline.py`)

You can run each stage by hand for debugging or experimentation.

#### 1. Record Video

```bash
python recording.py
```

- Reads `VIDEO_URL`, `OUTPUT_FILE` (defaults to `flask_record.mp4`), and `MAX_DURATION` from the environment.
- Produces: `flask_record.mp4`

#### 2. Extract Eye‑Tracking Metrics

```bash
python try.py
```

- Reads `INPUT_VIDEO` (default `flask_record.mp4`) and `OUTPUT_CSV` (default `eye_metrics_output.csv`).
- Produces: `eye_metrics_output.csv` containing per‑frame gaze, blinks, saccade velocity, etc.

#### 3. Preprocess & Categorize Eye Movements

```bash
python "collab files/eye_tracking_data_preprocessor.py"
```

- Reads `INPUT_CSV` (default `eye_metrics_output.csv`).
- Produces: `cleaned_file.csv` with:
  - Median‑filtered gaze (`gx_med`, `gy_med`)
  - `saccade_velocity` with micro‑saccades removed
  - `category_binocular` (`blink`, `saccade`, `visual intake`)
  - `RecordingTime [ms]`

#### 4. Classification (Using Combined Script)

```bash
python "collab files/parkinsons_eye_tracking_classifier.py"
```

Environment variables:

| Variable        | Default         | Description                                             |
|-----------------|-----------------|---------------------------------------------------------|
| `DATA_FOLDER`   | `new_folder`    | Folder with training CSVs (P_/N_ prefixed)             |
| `INPUT_FILE`    | `cleaned_file.csv` | Cleaned CSV for prediction                        |
| `OUTPUT_FILE`   | `prediction_result.txt` | Prediction output file                         |
| `MODELS_DIR`    | `saved_models`  | Directory of saved models                               |
| `USE_SAVED_MODELS` | `1`          | Try loading saved models before retraining              |

Behavior:
- If `USE_SAVED_MODELS` is true and `saved_models/metadata_*.pkl` exists → load models.
- Otherwise, if `DATA_FOLDER` has CSVs → train models on the fly.
- Uses the LSTM model to predict on `INPUT_FILE` and writes a human‑readable report to `OUTPUT_FILE`.

---

### Training the Models from Scratch

For offline training on your dataset, follow the steps from `TRAINING_GUIDE.md`.

#### 1. Prepare Training Data

Your original CSVs should be under:

- `dataset (3)/dataset/new folder/patients/`
- `dataset (3)/dataset/new folder/non_patients/`

Run:

```bash
python prepare_training_data.py
```

This will:
- Copy patient CSVs to `training_data/` with `P_*.csv` names.
- Copy non‑patient CSVs to `training_data/` with `N_*.csv` names.

The resulting structure:

```text
training_data/
  P_001_*.csv
  ...
  N_001_*.csv
  ...
```

#### 2. Train Models

```bash
python train_model.py
```

This script will:
- Load all CSVs from `training_data/`.
- Extract features using `extract_features()` in `parkinsons_eye_tracking_classifier.py`.
- Train and evaluate:
  - Random Forest (LOOCV)
  - KNN (LOOCV)
  - LSTM (sequence‑based, LOOCV)
- Save final models and metadata to `saved_models/`:
  - `random_forest_<timestamp>.pkl`
  - `knn_<timestamp>.pkl`
  - `lstm_<timestamp>.keras`
  - `metadata_<timestamp>.pkl`

You can confirm they exist with:

```bash
dir saved_models  # Windows
ls saved_models   # Linux/macOS
```

---

### Using Saved Models for Standalone Prediction

You can use `load_model.py` to load saved models and run a prediction on a cleaned CSV:

```bash
python load_model.py cleaned_file.csv lstm
```

Arguments:
- `input_csv_file`: path to the cleaned eye‑tracking CSV (e.g., `cleaned_file.csv`).
- `model_type`: one of `random_forest`, `knn`, or `lstm` (default: `lstm`).
- `timestamp` (optional): specific model timestamp to load; if omitted, the most recent models are used.

The script prints:
- Predicted class (Parkinson's, Non‑Parkinson's, or Uncertain)
- Probability and confidence level

---

### SHAP Explainability (`shap.py`)

The file `shap.py` provides an example of how to generate SHAP explanations for your model.
It assumes you have:
- A trained model (`model`) already loaded and in eval mode
- A validation dataframe `val_df` with feature columns `feature_cols`

The script:
- Builds a `KernelExplainer` around `model_forward`
- Computes SHAP values for a sample of validation points
- Produces:
  - Bar summary plot of feature importance
  - Beeswarm plot of feature impact
  - Dependence plots for the top 3 features

To run it, adapt it to your model objects and then execute:

```bash
python shap.py
```

(Make sure `shap`, `torch`, and `matplotlib` are installed and your model/data variables are defined correctly.)

---

### Output Files Summary

Key outputs produced by the pipeline:
- `flask_record.mp4` – Raw recorded video.
- `eye_metrics_output.csv` – Per‑frame eye‑tracking metrics from `try.py`.
- `cleaned_file.csv` – Preprocessed and categorized gaze data from the preprocessor.
- `prediction_result.txt` – Final classification and probability from the classifier.
- `saved_models/` – Persisted ML models and metadata for reuse.

---

### Understanding the Results

- **Eye‑tracking metrics** include blink rate, saccade velocity, fixation/visual‑intake distribution,
  pupil size statistics, and gaze positions.
- **Prediction report** (`prediction_result.txt` or console output from `load_model.py`) includes:
  - Classification label: Parkinson's / Non‑Parkinson's / Uncertain
  - Probability score (0–1)
  - Confidence level (High / Low)

---

### Troubleshooting

- **Video stream not accessible**:
  - Verify the Raspberry Pi Flask server is running.
  - Check the `VIDEO_URL` and network connectivity.
- **No face detected / empty CSV**:
  - Improve lighting and camera placement.
  - Ensure the subject is within frame and facing the camera.
- **Training errors**:
  - Confirm `training_data/` contains both `P_*.csv` and `N_*.csv`.
  - Check CSV headers and formats (see training scripts for expected columns).
- **No saved models found**:
  - Ensure you ran `train_model.py` successfully.
  - Check `saved_models/` for timestamped files and metadata.

For detailed training instructions, see `TRAINING_GUIDE.md`.

---

### License and Contribution

This project is currently intended for research/prototyping purposes only and is **not** a medical device.

- **License**: See `LICENSE` (if provided) or consult the project owner.
- **Contributions**: Please open an issue or pull request with proposed changes or enhancements.