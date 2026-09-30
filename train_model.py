#!/usr/bin/env python3

"""
Entrenamiento del LSTM utilizando homogeneous_data/*.npz.

Cada archivo NPZ contiene:

    X -> (300, 5)
    y -> 0 / 1

Features:

    1. gaze_x
    2. gaze_y
    3. gaze_velocity
    4. pupil_size
    5. blink

La división TRAIN/VALIDATION/TEST se realiza por sujeto.

IMPORTANTE:
- Ningún sujeto aparece en más de un conjunto.
- La estratificación se realiza a nivel de sujeto.
- La evaluación se muestra tanto por ventana como por sujeto.
"""

from pathlib import Path
from datetime import datetime
import json
import re

import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_DIR = Path("homogeneous_data")
MODELS_DIR = Path("saved_models")

WINDOW_SIZE = 300
N_FEATURES = 5

EPOCHS = 30
BATCH_SIZE = 8

TEST_SIZE = 0.25
VALIDATION_SIZE = 0.20

RANDOM_STATE_TEST = 42
RANDOM_STATE_VALIDATION = 43

EARLY_STOPPING_PATIENCE = 5

FEATURE_NAMES = [
    "gaze_x",
    "gaze_y",
    "gaze_velocity",
    "pupil_size",
    "blink"
]


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

np.random.seed(RANDOM_STATE_TEST)
tf.random.set_seed(RANDOM_STATE_TEST)


# ============================================================
# IDENTIFICAR SUJETO
# ============================================================

def get_subject_id(filename):
    """
    Ejemplo:

        P_001_best_segment_cleaned_W001.npz

    devuelve:

        P_001
    """

    match = re.match(
        r"^(P|N)_\d{3}",
        filename
    )

    if not match:
        raise ValueError(
            f"No se pudo identificar el sujeto: {filename}"
        )

    return match.group(0)


# ============================================================
# CARGAR DATASET
# ============================================================

def load_dataset():

    files = sorted(
        DATA_DIR.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No existen archivos NPZ en {DATA_DIR}"
        )

    X = []
    y = []
    groups = []

    for path in files:

        data = np.load(path)

        if "X" not in data:
            raise ValueError(
                f"{path.name} no contiene X"
            )

        if "y" not in data:
            raise ValueError(
                f"{path.name} no contiene y"
            )

        sequence = np.asarray(
            data["X"],
            dtype=np.float32
        )

        label = int(
            np.asarray(
                data["y"]
            ).item()
        )

        if label not in (0, 1):
            raise ValueError(
                f"{path.name}: etiqueta inválida: {label}. "
                f"Se esperaba 0 o 1."
            )

        if sequence.shape != (
            WINDOW_SIZE,
            N_FEATURES
        ):

            raise ValueError(
                f"{path.name}: "
                f"shape={sequence.shape}, "
                f"esperado="
                f"({WINDOW_SIZE}, {N_FEATURES})"
            )

        X.append(sequence)
        y.append(label)

        groups.append(
            get_subject_id(
                path.name
            )
        )

    return (
        np.stack(X),
        np.asarray(
            y,
            dtype=np.int64
        ),
        np.asarray(
            groups
        )
    )


# ============================================================
# INFORMACIÓN DE SUJETOS
# ============================================================

def build_subject_table(groups, y):

    subjects = np.unique(
        groups
    )

    subject_labels = []

    for subject in subjects:

        subject_mask = (
            groups == subject
        )

        labels = np.unique(
            y[subject_mask]
        )

        if len(labels) != 1:
            raise ValueError(
                f"El sujeto {subject} tiene etiquetas "
                f"inconsistentes: {labels.tolist()}"
            )

        subject_labels.append(
            int(labels[0])
        )

    return (
        subjects,
        np.asarray(
            subject_labels,
            dtype=np.int64
        )
    )


def print_subject_distribution(
    subjects,
    subject_labels,
    title
):

    print()
    print(title)

    control_subjects = subjects[
        subject_labels == 0
    ]

    patient_subjects = subjects[
        subject_labels == 1
    ]

    print(
        f"  Controles:  {len(control_subjects)}"
    )

    print(
        f"  Parkinson:  {len(patient_subjects)}"
    )

    print(
        f"  Total:      {len(subjects)}"
    )


# ============================================================
# CONVERTIR SUJETOS → ÍNDICES DE VENTANAS
# ============================================================

def indices_for_subjects(
    groups,
    subjects
):

    mask = np.isin(
        groups,
        subjects
    )

    return np.flatnonzero(
        mask
    )


# ============================================================
# SPLIT TRAIN / VALIDATION / TEST
# ============================================================

def create_subject_split(
    groups,
    y
):

    subjects, subject_labels = (
        build_subject_table(
            groups,
            y
        )
    )

    # --------------------------------------------------------
    # TEST
    #
    # 25% de los sujetos se reservan completamente para test.
    # --------------------------------------------------------

    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE_TEST
    )

    train_val_subject_idx, test_subject_idx = next(
        splitter_test.split(
            subjects,
            subject_labels
        )
    )

    train_val_subjects = subjects[
        train_val_subject_idx
    ]

    train_val_labels = subject_labels[
        train_val_subject_idx
    ]

    test_subjects = subjects[
        test_subject_idx
    ]

    # --------------------------------------------------------
    # VALIDATION
    #
    # 20% de los sujetos restantes.
    # --------------------------------------------------------

    splitter_validation = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE_VALIDATION
    )

    train_subject_idx, validation_subject_idx = next(
        splitter_validation.split(
            train_val_subjects,
            train_val_labels
        )
    )

    train_subjects = train_val_subjects[
        train_subject_idx
    ]

    validation_subjects = train_val_subjects[
        validation_subject_idx
    ]

    # --------------------------------------------------------
    # Convertir sujetos en índices de ventanas
    # --------------------------------------------------------

    train_idx = indices_for_subjects(
        groups,
        train_subjects
    )

    validation_idx = indices_for_subjects(
        groups,
        validation_subjects
    )

    test_idx = indices_for_subjects(
        groups,
        test_subjects
    )

    return (
        train_idx,
        validation_idx,
        test_idx,
        train_subjects,
        validation_subjects,
        test_subjects,
        subjects,
        subject_labels
    )


# ============================================================
# MODELO
# ============================================================

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                WINDOW_SIZE,
                N_FEATURES
            )
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=False
        ),

        tf.keras.layers.Dropout(
            0.3
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y_train):

    classes = np.unique(
        y_train
    )

    if set(classes) != {0, 1}:
        raise ValueError(
            "El conjunto de entrenamiento debe contener "
            "ambas clases: 0 y 1."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    return {
        int(cls): float(weight)
        for cls, weight in zip(
            classes,
            weights
        )
    }


# ============================================================
# EVALUACIÓN POR SUJETO
# ============================================================

def evaluate_subject_level(
    probabilities,
    y_true,
    groups
):

    subjects = np.unique(
        groups
    )

    subject_true = []
    subject_pred = []

    print()
    print(
        "Resultados por sujeto:"
    )

    print(
        f"{'Sujeto':<10}"
        f"{'Real':<12}"
        f"{'Prob. Parkinson':<20}"
        f"{'Predicción':<12}"
    )

    print("-" * 54)

    for subject in sorted(subjects):

        mask = (
            groups == subject
        )

        true_labels = np.unique(
            y_true[mask]
        )

        if len(true_labels) != 1:
            raise ValueError(
                f"Etiqueta inconsistente para {subject}"
            )

        true_label = int(
            true_labels[0]
        )

        mean_probability = float(
            np.mean(
                probabilities[mask]
            )
        )

        prediction = int(
            mean_probability >= 0.5
        )

        subject_true.append(
            true_label
        )

        subject_pred.append(
            prediction
        )

        real_text = (
            "Control"
            if true_label == 0
            else "Parkinson"
        )

        prediction_text = (
            "Control"
            if prediction == 0
            else "Parkinson"
        )

        print(
            f"{subject:<10}"
            f"{real_text:<12}"
            f"{mean_probability:<20.4f}"
            f"{prediction_text:<12}"
        )

    subject_true = np.asarray(
        subject_true,
        dtype=np.int64
    )

    subject_pred = np.asarray(
        subject_pred,
        dtype=np.int64
    )

    accuracy = accuracy_score(
        subject_true,
        subject_pred
    )

    print()
    print(
        f"Accuracy por sujeto: "
        f"{accuracy:.4f}"
    )

    print()
    print(
        "Classification report por sujeto:"
    )

    print(
        classification_report(
            subject_true,
            subject_pred,
            target_names=[
                "Control",
                "Parkinson"
            ],
            zero_division=0
        )
    )

    print(
        "Confusion matrix por sujeto:"
    )

    print(
        confusion_matrix(
            subject_true,
            subject_pred
        )
    )

    return (
        accuracy,
        subject_true,
        subject_pred
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PARKINSON EYE TRACKING - LSTM TRAINING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Cargar dataset
    # --------------------------------------------------------

    X, y, groups = load_dataset()

    subjects, subject_labels = (
        build_subject_table(
            groups,
            y
        )
    )

    patient_subjects = subjects[
        subject_labels == 1
    ]

    control_subjects = subjects[
        subject_labels == 0
    ]

    print()
    print(
        f"Ventanas totales: {len(X)}"
    )

    print(
        f"Shape X: {X.shape}"
    )

    print(
        f"Ventanas Parkinson: {(y == 1).sum()}"
    )

    print(
        f"Ventanas Control: {(y == 0).sum()}"
    )

    print(
        f"Sujetos totales: {len(subjects)}"
    )

    print(
        f"Sujetos Parkinson: {len(patient_subjects)}"
    )

    print(
        f"Sujetos Control: {len(control_subjects)}"
    )

    # --------------------------------------------------------
    # Split por sujeto
    # --------------------------------------------------------

    (
        train_idx,
        validation_idx,
        test_idx,
        train_subjects,
        validation_subjects,
        test_subjects,
        all_subjects,
        all_subject_labels
    ) = create_subject_split(
        groups,
        y
    )

    X_train = X[
        train_idx
    ]

    y_train = y[
        train_idx
    ]

    X_validation = X[
        validation_idx
    ]

    y_validation = y[
        validation_idx
    ]

    X_test = X[
        test_idx
    ]

    y_test = y[
        test_idx
    ]

    groups_test = groups[
        test_idx
    ]

    print()
    print("=" * 70)
    print("SPLIT")
    print("=" * 70)

    print(
        f"Train:      {len(X_train)} ventanas"
    )

    print(
        f"Validation: {len(X_validation)} ventanas"
    )

    print(
        f"Test:       {len(X_test)} ventanas"
    )

    print(
        f"\nSujetos train:      {len(train_subjects)}"
    )

    print(
        f"Sujetos validation: {len(validation_subjects)}"
    )

    print(
        f"Sujetos test:       {len(test_subjects)}"
    )

    print_subject_distribution(
        train_subjects,
        all_subject_labels[
            np.isin(
                all_subjects,
                train_subjects
            )
        ],
        "Distribución de sujetos - TRAIN"
    )

    print_subject_distribution(
        validation_subjects,
        all_subject_labels[
            np.isin(
                all_subjects,
                validation_subjects
            )
        ],
        "Distribución de sujetos - VALIDATION"
    )

    print_subject_distribution(
        test_subjects,
        all_subject_labels[
            np.isin(
                all_subjects,
                test_subjects
            )
        ],
        "Distribución de sujetos - TEST"
    )

    print()
    print(
        "Sujetos test:"
    )

    print(
        ", ".join(
            sorted(
                test_subjects
            )
        )
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weight = calculate_class_weights(
        y_train
    )

    print()
    print(
        f"Class weights: {class_weight}"
    )

    # --------------------------------------------------------
    # Modelo de evaluación
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ENTRENAMIENTO PARA EVALUACIÓN"
    )
    print("=" * 70)

    model_eval = create_model()

    model_eval.summary()

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
    )

    history = model_eval.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            early_stopping
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Época con mejor validation loss
    # --------------------------------------------------------

    best_epoch = int(
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )

    best_val_loss = float(
        np.min(
            history.history["val_loss"]
        )
    )

    print()
    print(
        f"Mejor época según validation loss: "
        f"{best_epoch}"
    )

    print(
        f"Mejor validation loss: "
        f"{best_val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Evaluación por ventana
    # --------------------------------------------------------

    probabilities = (
        model_eval.predict(
            X_test,
            verbose=0
        ).ravel()
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    window_accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 70)
    print("RESULTADOS POR VENTANA")
    print("=" * 70)

    print(
        f"Accuracy: {window_accuracy:.4f}"
    )

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Control",
                "Parkinson"
            ],
            zero_division=0
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # Evaluación por sujeto
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTADOS POR SUJETO")
    print("=" * 70)

    (
        subject_accuracy,
        subject_true,
        subject_pred
    ) = evaluate_subject_level(
        probabilities,
        y_test,
        groups_test
    )

    # --------------------------------------------------------
    # Modelo final
    #
    # Se usa el número de épocas seleccionado con validation.
    # NO se utiliza el conjunto test para decidir este número.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ENTRENAMIENTO DEL MODELO FINAL"
    )
    print("=" * 70)

    print(
        f"Épocas utilizadas: {best_epoch}"
    )

    final_model = create_model()

    classes_all = np.unique(
        y
    )

    weights_all = compute_class_weight(
        class_weight="balanced",
        classes=classes_all,
        y=y
    )

    class_weight_all = {
        int(cls): float(weight)
        for cls, weight in zip(
            classes_all,
            weights_all
        )
    }

    final_model.fit(
        X,
        y,
        epochs=best_epoch,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_all,
        verbose=1
    )

    # --------------------------------------------------------
    # Guardar modelo
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_path = (
        MODELS_DIR
        / f"lstm_{timestamp}.keras"
    )

    metadata_path = (
        MODELS_DIR
        / f"metadata_lstm_{timestamp}.json"
    )

    final_model.save(
        model_path
    )

    metadata = {
        "timestamp": timestamp,

        "data_dir": str(
            DATA_DIR
        ),

        "num_windows": int(
            len(X)
        ),

        "num_subjects": int(
            len(subjects)
        ),

        "num_patients": int(
            len(patient_subjects)
        ),

        "num_controls": int(
            len(control_subjects)
        ),

        "num_patient_windows": int(
            (y == 1).sum()
        ),

        "num_control_windows": int(
            (y == 0).sum()
        ),

        "window_size": WINDOW_SIZE,

        "num_features": N_FEATURES,

        "feature_order": FEATURE_NAMES,

        "train_subjects": sorted(
            train_subjects.tolist()
        ),

        "validation_subjects": sorted(
            validation_subjects.tolist()
        ),

        "test_subjects": sorted(
            test_subjects.tolist()
        ),

        "epochs_max": EPOCHS,

        "best_epoch": best_epoch,

        "best_validation_loss": best_val_loss,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "class_weight_train":
            class_weight,

        "class_weight_all":
            class_weight_all,

        "evaluation_window_accuracy":
            float(window_accuracy),

        "evaluation_subject_accuracy":
            float(subject_accuracy),

        "window_confusion_matrix":
            confusion_matrix(
                y_test,
                predictions
            ).tolist(),

        "subject_confusion_matrix":
            confusion_matrix(
                subject_true,
                subject_pred
            ).tolist()
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print(
        "ENTRENAMIENTO COMPLETADO"
    )
    print("=" * 70)

    print(
        f"Modelo: {model_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()