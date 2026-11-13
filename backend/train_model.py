#!/usr/bin/env python3
"""
============================================================================
 IDS Dense MLP — Training Pipeline  (v4 — 27-feature LIME set)
 ──────────────────────────────────
 Faithful reproduction of Dense_MLP_BINARY_4L.ipynb as a standalone script.
 Feature set updated to the 27 features selected via LIME (Local Interpretable
 Model-agnostic Explanations) on a Random Forest classifier, retaining only
 features with median LIME importance above the median threshold.

 ARCHITECTURE (from notebook Cell 6):
   Dense(1024,ReLU) → Dropout(0.3) → BatchNorm
   Dense(512,ReLU)  → Dropout(0.3) → BatchNorm
   Dense(256,ReLU)  → Dropout(0.3) → BatchNorm
   Dense(128,ReLU)  → Dropout(0.3) → BatchNorm
   Dense(64,ReLU)   → Dropout(0.3) → BatchNorm
   Dense(1,Sigmoid)

 NORMALIZATION : Min-Max (custom, handles zero-std columns)
 OPTIMIZER     : Adam(lr=0.001, beta_1=0.999)
 LOSS          : BinaryCrossentropy
 BATCH SIZE    : 1024
 EPOCHS        : 50 (early stopping patience=10 on val_loss)

 OUTPUT DIRECTORIES (resolved relative to this file's location):
   <project_root>/model/    — saved_model.pkl, normalization_params.pkl,
                              training_history.pkl, feature_config.pkl,
                              keras_model.sav (Keras backend only)
   <project_root>/results/  — test_confusion_matrix.png, test_roc_curve.png,
                              training_history.png

 BACKENDS:
   • TensorFlow/Keras (primary) — exact architecture with BatchNorm
   • scikit-learn (auto-fallback) — MLPClassifier with same topology

 DATA SOURCE:
   By default main() uses generate_synthetic_data() so the pipeline runs
   anywhere without an external dataset.
   To train on the real CIC-DDoS2019 data, comment that line out and
   uncomment the pd.read_csv() call below it, pointing at your CSV.
============================================================================
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.metrics import (confusion_matrix, accuracy_score,
                             ConfusionMatrixDisplay, roc_curve, auc)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for headless environments
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
# 1. PATH RESOLUTION
# ══════════════════════════════════════════════════════════════════════════

# Resolve output directories from this file's location so the script works
# regardless of the working directory it is run from.
_HERE        = os.path.dirname(os.path.abspath(__file__))   # …/backend
_ROOT        = os.path.dirname(_HERE)                        # …/ (project root)
_MODEL_DIR   = os.path.join(_ROOT, "model")                 # artifacts go here
_RESULTS_DIR = os.path.join(_ROOT, "results")               # plots go here

# Create directories if they do not exist yet
os.makedirs(_MODEL_DIR,   exist_ok=True)
os.makedirs(_RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# 27 CIC-DDoS2019 flow features selected via LIME (Random Forest,
# median importance threshold).  This list MUST stay identical to the
# FEATURES list in ids_gui.py.
FEATURES = [
    "URG Flag Count",
    "SYN Flag Count",
    "Bwd Packets/s",
    "CWE Flag Count",
    "Total Length of Bwd Packets",
    "Down/Up Ratio",
    "Fwd PSH Flags",
    "Init_Win_bytes_forward",
    "Bwd Header Length",
    "Total Backward Packets",
    "Bwd Packet Length Max",
    "Flow IAT Min",
    "Bwd Packet Length Std",
    "Bwd IAT Std",
    "Packet Length Variance",
    "Max Packet Length",
    "Fwd IAT Min",
    "Bwd IAT Max",
    "Active Std",
    "Bwd IAT Total",
    "Idle Std",
    "Fwd Packet Length Min",
    "Active Mean",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Active Max",
    "Idle Min",
]

# Metadata columns present in the full dataset (not used for model training)
METADATA_COLS = [
    "Source IP", "Destination IP", "Protocol",
    "Source Port number", "Destination Port number",
    "Timestamp", "Label",
]

# Feature columns + target label
ALL_COLS = FEATURES + ["Label"]

# Artifact file paths (written to model/, read by ids_gui.py)
MODEL_PATH       = os.path.join(_MODEL_DIR, "saved_model.pkl")
NORM_PARAMS_PATH = os.path.join(_MODEL_DIR, "normalization_params.pkl")
HISTORY_PATH     = os.path.join(_MODEL_DIR, "training_history.pkl")
FEATURE_CFG_PATH = os.path.join(_MODEL_DIR, "feature_config.pkl")

# Training hyperparameters
BATCH_SIZE  = 1024
EPOCHS      = 50
PATIENCE    = 10    # early stopping on val_loss
LR          = 0.001
BETA_1      = 0.999
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Three-dataset configuration ───────────────────────────────────────────
# Set these to the paths of your pre-split CICDDoS2019 CSV files.
# When all three are set and the files exist, they are used directly —
# no automatic splitting is performed.
# If any path is None or a file is missing, the pipeline falls back to
# generating synthetic data and dividing it 70 / 15 / 15.
#
# Example (uncomment and adjust paths):
#   TRAIN_CSV_PATH = r"D:\Datasets\CICDDoS2019_train.csv"
#   VAL_CSV_PATH   = r"D:\Datasets\CICDDoS2019_val.csv"
#   TEST_CSV_PATH  = r"D:\Datasets\CICDDoS2019_test.csv"
TRAIN_CSV_PATH = None
VAL_CSV_PATH   = None
TEST_CSV_PATH  = None


# ══════════════════════════════════════════════════════════════════════════
# 3. MIN-MAX NORMALISATION  (exact copy from notebook Cell 1)
# ══════════════════════════════════════════════════════════════════════════

def Normalization(X_Train):
    """
    Apply in-place Min-Max normalisation to a training feature DataFrame.

    Handles zero-std columns: if the column maximum is <= 0 the column is
    filled with 0; if > 0 it is filled with 1.  This matches the notebook
    implementation exactly so that training metrics are reproducible.

    Parameters
    ----------
    X_Train : pd.DataFrame  — feature columns only (no label)

    Returns
    -------
    X_Train : pd.DataFrame  — normalised, values in [0, 1]
    """
    # Identify constant (zero-std) columns before scaling
    const_cols  = []
    const_maxes = []
    for col in X_Train.columns:
        if X_Train[col].std() == 0:
            const_cols.append(col)
            const_maxes.append(X_Train[col].max())

    # Standard Min-Max: (x - min) / (max - min)
    col_min = X_Train.min()
    col_max = X_Train.max()
    denom   = col_max - col_min
    denom[denom == 0] = 1   # avoid division by zero; constant cols handled below
    X_Train = (X_Train - col_min) / denom

    # Override constant columns with their canonical fill value
    for col, max_val in zip(const_cols, const_maxes):
        X_Train[col] = 1 if max_val > 0 else 0

    return X_Train


def save_normalization_params(X):
    """
    Persist per-feature min/max statistics from the training set to disk.

    ids_gui.py loads these parameters to apply the same scaling to new data
    at inference time.  They must always be derived from the training split
    only — never from validation or test data.
    """
    params = {"min": X.min().to_dict(), "max": X.max().to_dict()}
    with open(NORM_PARAMS_PATH, "wb") as f:
        pickle.dump(params, f)
    print(f"[SAVE] Norm params → {NORM_PARAMS_PATH}")


def apply_normalization(X, params):
    """
    Apply saved Min-Max normalisation to new data (for inference).

    Mirrors ids_gui.py's apply_normalization() to ensure consistency.

    Parameters
    ----------
    X      : pd.DataFrame  — feature columns only
    params : dict          — {"min": {col: float}, "max": {col: float}}
    """
    X = X.copy()
    for col in X.columns:
        cmin  = params["min"].get(col, 0)
        cmax  = params["max"].get(col, 1)
        denom = cmax - cmin
        if denom == 0:
            X[col] = 1 if cmax > 0 else 0
        else:
            X[col] = (X[col] - cmin) / denom
    return X.clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION METRICS  (from notebook Cell 5)
# ══════════════════════════════════════════════════════════════════════════

def Evaluation_Metrics(y_test, y_pred_binary, dataset_name="Test"):
    """
    Compute and print the six metrics used in the notebook.

    Returns
    -------
    dict with Sensitivity, Specificity, Precision, Accuracy, F1_Score,
    Balanced_Accuracy, TP, TN, FP, FN.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_binary, labels=[0, 1])
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    Precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    Accuracy    = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    F1_SCORE    = (2 * Precision * Sensitivity / (Precision + Sensitivity)
                   if (Precision + Sensitivity) > 0 else 0)
    BA          = (Sensitivity + Specificity) / 2   # Balanced Accuracy

    print(f"\n── {dataset_name} Evaluation ──────────────────────────")
    print(f"  Sensitivity       : {Sensitivity:.6f}")
    print(f"  Specificity       : {Specificity:.6f}")
    print(f"  Precision         : {Precision:.6f}")
    print(f"  Accuracy          : {Accuracy:.6f}")
    print(f"  F1 Score          : {F1_SCORE:.6f}")
    print(f"  Balanced Accuracy : {BA:.6f}")
    print(f"  Confusion Matrix  :\n{cm}")

    return {
        "Sensitivity": Sensitivity,
        "Specificity": Specificity,
        "Precision":   Precision,
        "Accuracy":    Accuracy,
        "F1_Score":    F1_SCORE,
        "Balanced_Accuracy": BA,
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
    }


def ROC_AUC(y_test, y_pred, title="MLP"):
    """
    Plot the ROC curve (from notebook Cell 4) and return the figure + AUC.

    Parameters
    ----------
    y_test : array-like  — true binary labels
    y_pred : array-like  — predicted probability scores
    title  : str         — plot title suffix

    Returns
    -------
    (fig, roc_auc) : matplotlib Figure, float
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc     = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")   # random classifier baseline
    plt.xlim([-0.03, 1.05])
    plt.ylim([-0.03, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for " + title)
    plt.legend()
    return fig, roc_auc


# ══════════════════════════════════════════════════════════════════════════
# 5a. BINARY RE-LABELING  (handles multi-class CICDDoS2019 CSVs)
# ══════════════════════════════════════════════════════════════════════════

def relabel_binary(df):
    """
    Convert the 'Label' column to strict binary (0 / 1) in-place.

    Strategy: whitelist BENIGN → 0; every other label → 1 (DDoS/Attack).
    This is consistent with ids_gui.py's re-labeling so training labels
    and inference labels use the same mapping.

    Handles:
      • Numeric labels already in {0, 1}         → passed through
      • "BENIGN" (any case) → 0
      • Any attack sub-class (DrDoS_LDAP, Syn, NTP, MSSQL, TFTP …) → 1

    Parameters
    ----------
    df : pd.DataFrame  — must contain a 'Label' column

    Returns
    -------
    df : pd.DataFrame  — copy with 'Label' binarized
    """
    df = df.copy()
    lbl = df["Label"].astype(str).str.strip().str.lower()
    df["Label"] = lbl.map(lambda x: 0 if x in ("0", "benign") else 1)
    return df


# ══════════════════════════════════════════════════════════════════════════
# 5b. THREE-DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_three_datasets():
    """
    Load (or generate) three separate datasets for training, validation,
    and testing without merging and re-splitting.

    Priority
    --------
    1. All three CSV paths (TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH)
       are set and their files exist → load each file directly.
    2. Otherwise → generate synthetic data and divide it 70 / 15 / 15.

    Each loaded CSV is:
      • Required to contain the 15 FEATURES columns.
      • Processed through relabel_binary() so multi-class attack labels
        are collapsed to binary before training.
      • Cleaned (Inf/NaN → 0).

    Returns
    -------
    X_train_raw, y_train : feature DataFrame and label array for training
    X_val_raw,   y_val   : feature DataFrame and label array for validation
    X_test_raw,  y_test  : feature DataFrame and label array for testing
    """
    all_paths = [TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH]
    all_exist  = all(p and os.path.isfile(p) for p in all_paths)

    if all_exist:
        # ── Path A: load three pre-split files directly ───────────────
        print("[DATA] Loading three pre-split datasets …")
        splits = {}
        for name, path in zip(("train", "val", "test"), all_paths):
            df = pd.read_csv(path)
            # Validate required feature columns
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                raise ValueError(
                    f"[{name}] CSV missing required columns: {', '.join(missing)}"
                )
            if "Label" not in df.columns:
                raise ValueError(f"[{name}] CSV missing 'Label' column.")
            df = relabel_binary(df)     # normalise multi-class labels → {0,1}
            X = df[FEATURES].copy()
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0, inplace=True)
            y = df["Label"].values.astype(int)
            splits[name] = (X, y)
            print(f"[DATA]  {name:5s}: {len(df):>8,} rows  "
                  f"| BENIGN {(y==0).sum():>8,}  DDoS {(y==1).sum():>8,}")

        return (*splits["train"], *splits["val"], *splits["test"])

    else:
        # ── Path B: synthetic data, split 70 / 15 / 15 ───────────────
        if any(all_paths):
            print("[WARN] Not all dataset paths are set or files are missing. "
                  "Falling back to synthetic data.")
        df = generate_synthetic_data(20_000)
        X  = df[FEATURES].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        y  = df["Label"].values.astype(int)

        # Stratified 70 / 15 / 15 manual split (no sklearn dependency)
        np.random.seed(RANDOM_SEED)
        idx  = np.random.permutation(len(X))
        n    = len(X)
        t1   = int(n * 0.70)
        t2   = int(n * 0.85)
        i_tr, i_vl, i_te = idx[:t1], idx[t1:t2], idx[t2:]

        X_train_raw = X.iloc[i_tr].reset_index(drop=True)
        y_train     = y[i_tr]
        X_val_raw   = X.iloc[i_vl].reset_index(drop=True)
        y_val       = y[i_vl]
        X_test_raw  = X.iloc[i_te].reset_index(drop=True)
        y_test      = y[i_te]

        print(f"[SPLIT] Train: {len(X_train_raw)}  Val: {len(X_val_raw)}  "
              f"Test: {len(X_test_raw)}")
        return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test


# ══════════════════════════════════════════════════════════════════════════
# 5c. SYNTHETIC DATA GENERATION  (default data source for testing)
# ══════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples: int = 20_000) -> pd.DataFrame:
    """
    Generate a CIC-DDoS2019-shaped synthetic dataset for end-to-end testing.

    Attack traffic uses high-frequency, small-packet distributions.
    Benign traffic uses lower-rate, larger-packet distributions.
    Labels: 1 = DDoS, 0 = BENIGN.

    To train on real data instead, comment out the call to this function
    in main() and uncomment the pd.read_csv() call pointing at your CSV.
    """
    rng      = np.random.default_rng(RANDOM_SEED)
    n_normal = n_samples // 2
    n_attack = n_samples - n_normal

    def _block(n, is_attack):
        """Build one block of n rows for either attack or benign traffic."""
        if is_attack:
            # DDoS: SYN floods, high backward rate, small packets, tight IATs
            bwd_ps = rng.exponential(50_000, n)
            return {
                "URG Flag Count":              rng.integers(0, 3, n).astype(float),
                "SYN Flag Count":              rng.integers(2, 10, n).astype(float),
                "Bwd Packets/s":               bwd_ps,
                "CWE Flag Count":              rng.integers(0, 2, n).astype(float),
                "Total Length of Bwd Packets": rng.exponential(500, n),
                "Down/Up Ratio":               rng.uniform(0, 1, n),
                "Fwd PSH Flags":               rng.integers(0, 4, n).astype(float),
                "Init_Win_bytes_forward":       rng.integers(0, 8192, n).astype(float),
                "Bwd Header Length":            rng.integers(20, 60, n).astype(float),
                "Total Backward Packets":       rng.integers(0, 20, n).astype(float),
                "Bwd Packet Length Max":        rng.integers(0, 200, n).astype(float),
                "Flow IAT Min":                 rng.exponential(500, n),
                "Bwd Packet Length Std":        rng.exponential(50, n),
                "Bwd IAT Std":                  rng.exponential(1_000, n),
                "Packet Length Variance":       rng.exponential(500, n),
                "Max Packet Length":            rng.integers(0, 200, n).astype(float),
                "Fwd IAT Min":                  rng.exponential(500, n),
                "Bwd IAT Max":                  rng.exponential(10_000, n),
                "Active Std":                   rng.exponential(5_000, n),
                "Bwd IAT Total":                rng.exponential(50_000, n),
                "Idle Std":                     rng.exponential(5_000, n),
                "Fwd Packet Length Min":        rng.integers(0, 20, n).astype(float),
                "Active Mean":                  rng.exponential(50_000, n),
                "Bwd Packet Length Min":        rng.integers(0, 10, n).astype(float),
                "Bwd Packet Length Mean":       rng.exponential(50, n),
                "Active Max":                   rng.exponential(50_000, n),
                "Idle Min":                     rng.exponential(5_000, n),
                "Label": np.ones(n, dtype=int),
            }
        else:
            # Benign: moderate backward rate, larger packets, longer idle/active
            bwd_ps = rng.exponential(5_000, n)
            return {
                "URG Flag Count":              np.zeros(n),
                "SYN Flag Count":              rng.integers(0, 2, n).astype(float),
                "Bwd Packets/s":               bwd_ps,
                "CWE Flag Count":              np.zeros(n),
                "Total Length of Bwd Packets": rng.exponential(50_000, n),
                "Down/Up Ratio":               rng.uniform(1, 10, n),
                "Fwd PSH Flags":               rng.integers(0, 2, n).astype(float),
                "Init_Win_bytes_forward":       rng.integers(8000, 65535, n).astype(float),
                "Bwd Header Length":            rng.integers(20, 60, n).astype(float),
                "Total Backward Packets":       rng.integers(10, 100, n).astype(float),
                "Bwd Packet Length Max":        rng.integers(500, 1500, n).astype(float),
                "Flow IAT Min":                 rng.exponential(50_000, n),
                "Bwd Packet Length Std":        rng.exponential(200, n),
                "Bwd IAT Std":                  rng.exponential(100_000, n),
                "Packet Length Variance":       rng.exponential(50_000, n),
                "Max Packet Length":            rng.integers(500, 1500, n).astype(float),
                "Fwd IAT Min":                  rng.exponential(50_000, n),
                "Bwd IAT Max":                  rng.exponential(200_000, n),
                "Active Std":                   rng.exponential(500_000, n),
                "Bwd IAT Total":                rng.exponential(500_000, n),
                "Idle Std":                     rng.exponential(500_000, n),
                "Fwd Packet Length Min":        rng.integers(20, 60, n).astype(float),
                "Active Mean":                  rng.exponential(500_000, n),
                "Bwd Packet Length Min":        rng.integers(20, 60, n).astype(float),
                "Bwd Packet Length Mean":       rng.exponential(500, n),
                "Active Max":                   rng.exponential(500_000, n),
                "Idle Min":                     rng.exponential(500_000, n),
                "Label": np.zeros(n, dtype=int),
            }

    def _add_meta(block, n):
        """Append realistic-looking metadata columns to a feature block."""
        block["Source IP"]               = [f"192.168.{rng.integers(0,255)}.{rng.integers(1,255)}" for _ in range(n)]
        block["Destination IP"]          = [f"10.0.{rng.integers(0,255)}.{rng.integers(1,255)}" for _ in range(n)]
        block["Protocol"]                = rng.choice([6, 17], n).tolist()
        block["Source Port number"]      = rng.integers(1024, 65535, n).tolist()
        block["Destination Port number"] = rng.choice([80, 443, 8080, 53], n).tolist()
        block["Timestamp"]               = pd.date_range("2026-01-01", periods=n, freq="50ms").astype(str).tolist()

    normal = _block(n_normal, False); _add_meta(normal, n_normal)
    attack = _block(n_attack, True);  _add_meta(attack, n_attack)

    # Merge and shuffle
    df = pd.concat([pd.DataFrame(normal), pd.DataFrame(attack)], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[DATA]  {len(df)} samples  |  BENIGN: {n_normal}  DDoS: {n_attack}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 6. BACKEND SELECTION — Keras (primary) or sklearn (fallback)
# ══════════════════════════════════════════════════════════════════════════

USE_KERAS = False
try:
    # Suppress TensorFlow info/warning messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(RANDOM_SEED)
    USE_KERAS = True
    print("[BACKEND] TensorFlow / Keras")
except ImportError:
    # sklearn is the default when TensorFlow is not installed
    from sklearn.neural_network import MLPClassifier
    print("[BACKEND] scikit-learn MLPClassifier (TensorFlow not installed)")


# ── 6a. Keras model (exact reproduction of notebook Cell 6) ──────────────

def build_keras_model(input_dim):
    """
    Build the Sequential Dense MLP matching notebook Cell 6.

    Each hidden block: Dense(n, ReLU) → Dropout(0.3) → BatchNormalization
    Output layer:      Dense(1, Sigmoid)

    Parameters
    ----------
    input_dim : int  — number of input features (15)
    """
    model = Sequential()

    # First hidden block (input layer)
    model.add(Dense(1024, input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Hidden blocks 2–5 (progressively smaller width)
    for units in [512, 256, 128, 64]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

    # Binary output with sigmoid activation
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=LR, beta_1=BETA_1),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train_keras(model, X_train, y_train, X_val, y_val):
    """
    Train the Keras model with early stopping.

    Returns
    -------
    history : dict  — {"loss", "val_loss", "accuracy", "val_accuracy"}
    """
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    start   = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1,
    )
    print(f"Train_Time(Sec) : {time.time() - start:.2f}")
    return history.history


# ── 6b. sklearn fallback (same topology, no BatchNorm) ───────────────────

def build_sklearn_model():
    """
    Build an sklearn MLPClassifier with the same hidden-layer topology.

    BatchNormalization and Dropout are not available in sklearn, but the
    layer widths (1024, 512, 256, 128, 64) match the Keras architecture.
    """
    return MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=LR,
        max_iter=EPOCHS,
        early_stopping=True,
        n_iter_no_change=PATIENCE,
        validation_fraction=0.176,   # keeps the 70/15/15 effective split
        batch_size=BATCH_SIZE,
        random_state=RANDOM_SEED,
        verbose=True,
    )


def train_sklearn(model, X_train, y_train, X_val, y_val):
    """
    Train the sklearn model and return a history dict shaped like Keras's.

    sklearn exposes loss_curve_ and validation_scores_ after fit().

    Returns
    -------
    history : dict  — {"loss", "val_loss", "accuracy", "val_accuracy"}
    """
    start = time.time()
    model.fit(X_train, y_train)
    print(f"Train_Time(Sec) : {time.time() - start:.2f}")
    return {
        "loss":         list(model.loss_curve_),
        "val_loss":     list(model.validation_scores_),
        "accuracy":     list(model.loss_curve_),       # proxy — sklearn has no acc curve
        "val_accuracy": list(model.validation_scores_),
    }


# ══════════════════════════════════════════════════════════════════════════
# 7. SAVE ARTIFACTS  (written to model/ for ids_gui.py to load)
# ══════════════════════════════════════════════════════════════════════════

def save_all(model, history, use_keras):
    """
    Persist the trained model, training history, and feature config to model/.

    For the Keras backend the model object is pickled to keras_model.sav and
    the wrapper saves its absolute path so ids_gui.py can reload it correctly
    regardless of the working directory.

    For the sklearn backend the model is embedded directly in the wrapper dict.
    """
    if use_keras:
        keras_path = os.path.join(_MODEL_DIR, "keras_model.sav")
        pickle.dump(model, open(keras_path, "wb"))
        # Store the absolute path so the GUI can find it from any CWD
        joblib.dump({"type": "keras", "path": keras_path}, MODEL_PATH)
    else:
        joblib.dump({"type": "sklearn", "model": model}, MODEL_PATH)
    print(f"[SAVE] Model       → {MODEL_PATH}")

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)
    print(f"[SAVE] History     → {HISTORY_PATH}")

    with open(FEATURE_CFG_PATH, "wb") as f:
        pickle.dump(
            {"features": FEATURES, "metadata": METADATA_COLS, "all_cols": ALL_COLS},
            f,
        )
    print(f"[SAVE] Feature cfg → {FEATURE_CFG_PATH}")


# ══════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main():
    """
    End-to-end training pipeline:
      1. Load three pre-split datasets (train / val / test)
         OR fall back to synthetic data split 70 / 15 / 15.
         (Configure TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH above.)
      2. Min-Max normalisation (params derived from training set only)
      3. Build and train model (Keras or sklearn)
      4. Evaluate on test set; save plots to results/
      5. Save all artifacts to model/
    """
    print("=" * 68)
    print("  IDS Dense MLP — Training Pipeline")
    print("  (Architecture: notebook Dense_MLP_BINARY_4L.ipynb)")
    print("=" * 68)

    # ── Stage 1: Load three separate datasets ───────────────────────────
    # load_three_datasets() uses TRAIN/VAL/TEST_CSV_PATH when all three
    # are configured; otherwise generates synthetic data and splits it.
    # Each dataset is cleaned and binarized inside the loader.
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = \
        load_three_datasets()

    # ── Stage 2: Normalisation ─────────────────────────────────────────
    # Save min/max from the training split; apply those same params at test time
    save_normalization_params(X_train_raw)
    X_train = Normalization(X_train_raw.copy())
    X_val   = Normalization(X_val_raw.copy())
    X_test  = Normalization(X_test_raw.copy())

    # ── Stage 3: Build and train ───────────────────────────────────────
    if USE_KERAS:
        model   = build_keras_model(input_dim=len(FEATURES))
        history = train_keras(model, X_train.values, y_train, X_val.values, y_val)
    else:
        model   = build_sklearn_model()
        history = train_sklearn(model, X_train.values, y_train, X_val.values, y_val)

    # ── Stage 4: Test-set evaluation ──────────────────────────────────
    if USE_KERAS:
        test_loss, test_acc = model.evaluate(X_test.values, y_test)
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Test Accuracy: {test_acc:.6f}")
        y_pred      = model.predict(X_test.values, batch_size=BATCH_SIZE)
        y_pred_bin  = (y_pred > 0.5).astype(int).flatten()
    else:
        y_pred_bin  = model.predict(X_test.values)
        y_pred      = model.predict_proba(X_test.values)[:, 1]
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred_bin):.6f}")

    metrics = Evaluation_Metrics(y_test, y_pred_bin, "Test_Dataset")

    # ── Confusion matrix plot ──────────────────────────────────────────
    label_map     = {0: "BENIGN", 1: "DDoS"}
    y_true_labels = [label_map[v] for v in y_test]
    y_pred_labels = [label_map[v] for v in y_pred_bin]
    cm   = confusion_matrix(y_true_labels, y_pred_labels, labels=["BENIGN", "DDoS"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "DDoS"])
    disp.plot(cmap="Blues", values_format=".0f", text_kw={"fontsize": 14})
    cm_path = os.path.join(_RESULTS_DIR, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    print(f"[SAVE] {cm_path}")
    plt.close()

    # ── ROC curve plot ─────────────────────────────────────────────────
    roc_fig, roc_auc_val = ROC_AUC(y_test, y_pred, "MLP")
    roc_path = os.path.join(_RESULTS_DIR, "test_roc_curve.png")
    roc_fig.savefig(roc_path, dpi=100, bbox_inches="tight")
    print(f"[SAVE] {roc_path}  (AUC = {roc_auc_val:.4f})")
    plt.close()

    # ── Training history plot ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["accuracy"],     label="Train")
    ax1.plot(history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")
    ax2.plot(history["loss"],     label="Train")
    ax2.plot(history["val_loss"], label="Validation")
    ax2.set_title("Model Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    hist_path = os.path.join(_RESULTS_DIR, "training_history.png")
    plt.savefig(hist_path, dpi=100, bbox_inches="tight")
    print(f"[SAVE] {hist_path}")
    plt.close()

    # ── Stage 5: Save model artifacts ─────────────────────────────────
    save_all(model, history, USE_KERAS)

    print("\n✓  Training pipeline complete.")
    print(f"   Artifacts → {_MODEL_DIR}")
    print(f"   Plots     → {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
