# Backend — IDS Dense MLP

This directory contains the two Python modules that form the core of the system.

---

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Full training pipeline: data loading, normalization, model building, evaluation, artifact export |
| `ids_gui.py` | Flask web application: live traffic simulation, CSV batch inference, server-side plots |

---

## train_model.py

### What it does

1. Loads data — either from the three pre-split CSV paths configured at the top, or from the built-in synthetic generator if those paths are unset.
2. Applies Min-Max normalization and saves `model/normalization_params.pkl`.
3. Detects whether TensorFlow is available and builds the appropriate model:
   - **Keras** — exact architecture from `Dense_MLP_BINARY_4L.ipynb`
   - **sklearn** — `MLPClassifier` with the same layer sizes as a fallback
4. Trains with early stopping (patience=10 on val_loss).
5. Evaluates on the test split and prints Sensitivity, Specificity, Precision, Accuracy, F1, Balanced Accuracy, and ROC-AUC.
6. Saves all artifacts to `../model/` and plots to `../results/`.

### Running

```bash
cd backend        # artifacts are written relative to this file's location
python train_model.py
```

### Configuration (top of the file)

```python
# Point these at your pre-split CIC-DDoS2019 CSVs.
# Leave as None to use the synthetic data generator instead.
TRAIN_CSV_PATH = None   # e.g. r"D:\Datasets\CICDDoS2019_train.csv"
VAL_CSV_PATH   = None
TEST_CSV_PATH  = None
```

### Output artifacts

| File | Contents |
|---|---|
| `model/saved_model.pkl` | joblib wrapper: `{"type": "keras"\|"sklearn", ...}` |
| `model/keras_model.sav` | Pickled Keras model (Keras backend only) |
| `model/normalization_params.pkl` | Per-feature `{"min": {...}, "max": {...}}` |
| `model/training_history.pkl` | Loss and accuracy curves |
| `model/feature_config.pkl` | Feature list and metadata |
| `results/test_confusion_matrix.png` | Confusion matrix |
| `results/test_roc_curve.png` | ROC curve |
| `results/training_history.png` | Loss/accuracy plot |

---

## ids_gui.py

### What it does

A single Flask application serving a Jinja2 single-page app with four views.

| Route | View |
|---|---|
| `/` | Live Traffic Monitor |
| `/upload` | CSV Upload & Batch Inference |
| `/summary` | Aggregate session summary |
| `/mlinfo` | Model architecture and training info |

### Running

```bash
# From the project root (so the template folder resolves correctly)
python backend/ids_gui.py
```

The app binds to `http://0.0.0.0:5000` by default.

### Key design points

**Feature contract** — `FEATURES` in `ids_gui.py` must stay identical to `FEATURES` in `train_model.py`.  Both consume the same 21-column list in the same order.  The app rejects uploaded CSVs that are missing any of these columns.

**Normalization** — `apply_normalization()` loads `model/normalization_params.pkl` and applies the training-set min/max.  Inference scaling must never be re-derived from the uploaded data.

**Live capture** — `LiveCapture._generate()` produces synthetic flow records that match the training distributions.  For production use, replace its body with a real capture backend (scapy, tshark) that emits the same 27 feature keys.

**Plots** — Confusion matrices and ROC curves are rendered server-side with matplotlib and returned as base-64 PNGs.  Live charts use Plotly.js with `Plotly.react()` for incremental updates.

**Saved results** — "Save & Analysis" writes `results/raw_live_traffic.csv` and `results/analyzed_live_traffic.csv`.

---

## Synthetic Data Generator

`generate_synthetic_data()` in `train_model.py` creates a CIC-DDoS2019-shaped dataset without requiring the real files.  Feature distributions are tuned to reflect known attack vs benign traffic patterns from the dataset:

- **DDoS traffic:** high Bwd Packets/s, elevated SYN and URG flag counts, low Init_Win_bytes_forward (0–8192), small backward packets, short inter-arrival times (Flow IAT Min and Fwd IAT Min ≈ exponential(500)), tight Bwd IAT Std, low Down/Up Ratio, small Active/Idle values.
- **Benign traffic:** moderate Bwd Packets/s, full Init_Win_bytes_forward (8000–65535), larger backward packets, longer inter-arrival times, higher Down/Up Ratio, larger Active Mean/Std/Max and Idle Std/Min.

The same distributions are mirrored in `LiveCapture._generate()` in `ids_gui.py`.

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
joblib
flask
tensorflow   # optional — enables Keras backend
```

Install with:

```bash
pip install -r ../requirements.txt
# Optional Keras backend:
pip install tensorflow
```
