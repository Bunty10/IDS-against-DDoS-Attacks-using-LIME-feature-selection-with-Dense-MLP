# IDS Dense MLP — Network Intrusion Detection System

A binary classifier that identifies network flows as **BENIGN (0)** or **DDoS (1)** using a Dense Multi-Layer Perceptron trained on the CIC-DDoS2019 dataset.  A Flask web GUI provides real-time (simulated) traffic monitoring and batch CSV analysis.

---

## Project Structure

```
IDS_Dense-MLP/
├── backend/
│   ├── train_model.py      # Training pipeline — builds and saves all model artifacts
│   ├── ids_gui.py          # Flask web application (live monitor + CSV upload)
│   └── README.md           # Backend-specific documentation
├── templates/
│   └── index.html          # Single-page application template (Jinja2 + Chart.js)
├── model/                  # Created automatically by train_model.py
│   ├── saved_model.pkl
│   ├── normalization_params.pkl
│   ├── training_history.pkl
│   └── feature_config.pkl
├── results/                # Created automatically; plots and exported CSVs
├── requirements.txt
└── README.md               # This file
```

---

## Feature Set

27 CIC-DDoS2019 flow features selected via **LIME** (Local Interpretable Model-agnostic Explanations) applied to a **Random Forest** classifier, retaining features whose median LIME importance exceeds the median threshold:

| Category | Features |
|---|---|
| TCP Flags | URG Flag Count, SYN Flag Count, CWE Flag Count, Fwd PSH Flags |
| Packet Size | Fwd Packet Length Min, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Max Packet Length, Packet Length Variance |
| Flow Rate | Bwd Packets/s, Down/Up Ratio, Total Backward Packets, Total Length of Bwd Packets |
| Inter-Arrival Time | Flow IAT Min, Fwd IAT Min, Bwd IAT Std, Bwd IAT Max, Bwd IAT Total |
| Active / Idle | Active Mean, Active Std, Active Max, Idle Std, Idle Min |
| Connection | Init_Win_bytes_forward, Bwd Header Length |

---

## Model Architecture

Dense MLP — Binary Classification (BENIGN vs DDoS)

```
Input (27 features)
  → Dense(1024, ReLU) → Dropout(0.3) → BatchNormalization
  → Dense(512,  ReLU) → Dropout(0.3) → BatchNormalization
  → Dense(256,  ReLU) → Dropout(0.3) → BatchNormalization
  → Dense(128,  ReLU) → Dropout(0.3) → BatchNormalization
  → Dense(64,   ReLU) → Dropout(0.3) → BatchNormalization
  → Dense(1, Sigmoid)
```

**Optimizer:** Adam (lr=0.001, β₁=0.999)  
**Loss:** BinaryCrossentropy  
**Batch size:** 1 024  
**Epochs:** 50 (early stopping, patience=10 on val_loss)  
**Normalization:** Min-Max (custom; handles zero-std columns)

The script selects the backend automatically: TensorFlow/Keras if installed, otherwise scikit-learn MLPClassifier with the same layer topology.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> TensorFlow is listed in `requirements.txt` as a comment. Install it separately to use the Keras backend:
> ```bash
> pip install tensorflow
> ```
> Without TensorFlow the pipeline falls back to scikit-learn automatically.

### 2. Train the model

```bash
cd backend
python train_model.py
```

By default the script uses a built-in synthetic data generator so it runs anywhere without an external dataset.  To train on the real **CIC-DDoS2019** data, open `train_model.py` and set the three CSV paths at the top of the configuration section:

```python
TRAIN_CSV_PATH = r"/path/to/CICDDoS2019_train.csv"
VAL_CSV_PATH   = r"/path/to/CICDDoS2019_val.csv"
TEST_CSV_PATH  = r"/path/to/CICDDoS2019_test.csv"
```

After training, all artifacts are written to `model/` and evaluation plots to `results/`.

### 3. Launch the web GUI

```bash
python backend/ids_gui.py
```

Open **http://localhost:5000** in a browser.

---

## Web GUI Pages

| Page | Description |
|---|---|
| **Live Traffic** | Simulated packet capture with animated Plotly chart and real-time metrics |
| **CSV Upload** | Upload a CIC-DDoS2019-format CSV for batch inference; shows confusion matrix and ROC curve |
| **Summary** | Aggregate statistics across both live and uploaded sessions |
| **ML Info** | Architecture diagram, training configuration, feature list, and training-history plot |

---

## Dataset

This project uses the **CIC-DDoS2019** dataset published by the Canadian Institute for Cybersecurity.

- Download: https://www.unb.ca/cic/datasets/ddos-2019.html
- The model expects a CSV with the 27 feature columns listed above plus a `Label` column containing `BENIGN` or the attack name.

---

## Evaluation Metrics

| Metric | Formula |
|---|---|
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Precision | TP / (TP + FP) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| F1 Score | 2 × Precision × Sensitivity / (Precision + Sensitivity) |
| Balanced Accuracy | (Sensitivity + Specificity) / 2 |

---

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn, matplotlib, joblib, flask
- TensorFlow ≥ 2.10 (optional — enables Keras backend)

See `requirements.txt` for pinned versions.

---

## License

This project is provided for academic and research purposes.  
Dataset: CIC-DDoS2019 — Canadian Institute for Cybersecurity, University of New Brunswick.
