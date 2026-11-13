#!/usr/bin/env python3
"""
============================================================================
 IDS DDoS Detection — Flask Web GUI  (v4 — 27-feature LIME set)
 ────────────────────────────────────
 Changes from v2:
   • All model artifacts now resolved from  <project_root>/model/
   • CSV saves now go to                    <project_root>/results/
   • chart_points NaN/Inf values sanitised before JSON serialisation
     (fixes "Unexpected token 'N'" SyntaxError in the browser)
   • Improved inline comments throughout

 Four-page single-page application (SPA):
   /          → Live Traffic Monitor   (simulated capture + animated chart)
   /upload    → CSV Upload & Analysis  (batch prediction + animated chart)
   /summary   → Aggregate summary of both sessions
   /mlinfo    → Model architecture, training config, training-history plot

 Served template:  <project_root>/templates/index.html
============================================================================
"""

import os, math, pickle, time, datetime, threading, io, base64, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for server-side rendering
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
# CONFIG / PATH RESOLUTION
# ══════════════════════════════════════════════════════════════════════════

# 27 CIC-DDoS2019 flow features selected via LIME (Random Forest,
# median importance threshold).
# This list MUST stay identical to the FEATURES list in train_model.py.
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

# Resolve the project root from this file's location so the app works
# regardless of the working directory it is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))   # …/backend
_ROOT = os.path.dirname(_HERE)                        # …/ (project root)


def _model(*parts):
    """Return an absolute path inside <project_root>/model/."""
    return os.path.join(_ROOT, "model", *parts)


def _results(*parts):
    """Return an absolute path inside <project_root>/results/."""
    return os.path.join(_ROOT, "results", *parts)


# Artifact paths — model/ directory holds everything produced by training
MODEL_PATH       = _model("saved_model.pkl")         # joblib wrapper (type + model)
NORM_PARAMS_PATH = _model("normalization_params.pkl") # Min-Max params for inference
HISTORY_PATH     = _model("training_history.pkl")     # loss/accuracy curves

# Template folder: try backend/templates first, fall back to root templates/
_tmpl = os.path.join(_HERE, "templates")
if not os.path.isdir(_tmpl):
    _tmpl = os.path.join(_ROOT, "templates")
TEMPLATE_FOLDER = _tmpl

# ── Light-theme matplotlib colour palette ────────────────────────────────
# These must be kept in sync with the CSS colour variables in index.html
BG      = "#F7F9FC"   # page background
CARD    = "#FFFFFF"   # card surface
PANEL   = "#EEF2F7"   # panel tint
BORDER  = "#D0DAE8"   # subtle dividers
NAVY    = "#1A3A5C"   # headings / primary dark
BLUE    = "#2563EB"   # primary accent
GREEN   = "#16A34A"   # BENIGN / success
RED     = "#DC2626"   # DDoS / danger
AMBER   = "#D97706"   # warning / highlight
TEAL    = "#0891B2"   # secondary accent
TXT     = "#1E293B"   # body text
TXT_DIM = "#64748B"   # muted text


# ══════════════════════════════════════════════════════════════════════════
# MODEL / ARTIFACT LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_model_and_params():
    """
    Load the trained model and normalisation parameters from model/.

    Returns
    -------
    model      : sklearn MLPClassifier or Keras Sequential
    norm_params: dict with 'min' and 'max' per feature, or None
    history    : training history dict, or None
    """
    wrapper = joblib.load(MODEL_PATH)
    if wrapper["type"] == "keras":
        # Keras models are stored as a separate pickle; the wrapper holds the path
        model = pickle.load(open(wrapper["path"], "rb"))
    else:
        # sklearn model is embedded directly in the wrapper dict
        model = wrapper["model"]

    norm_params = None
    if os.path.exists(NORM_PARAMS_PATH):
        with open(NORM_PARAMS_PATH, "rb") as f:
            norm_params = pickle.load(f)

    history = None
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)

    return model, norm_params, history


# ── Startup: load artifacts once at import time ───────────────────────────
print("[INIT] Script directory :", _HERE)
print("[INIT] Project root     :", _ROOT)
print("[INIT] Template folder  :", TEMPLATE_FOLDER)
print("[INIT] Model path       :", MODEL_PATH)
print("[INIT] Loading artifacts …")
MODEL, NORM_PARAMS, HISTORY = load_model_and_params()

# Determine backend type: sklearn exposes predict_proba; Keras does not
IS_SKLEARN = hasattr(MODEL, "predict_proba")
print(f"[INIT] Model loaded ({'sklearn' if IS_SKLEARN else 'keras'})")


# ══════════════════════════════════════════════════════════════════════════
# NORMALISATION (mirrors train_model.py exactly)
# ══════════════════════════════════════════════════════════════════════════

def apply_normalization(X, params):
    """
    Apply saved Min-Max normalisation to feature matrix X.

    Uses training-set min/max from params so that inference scaling is
    identical to training scaling.  Output is clipped to [0, 1].

    Parameters
    ----------
    X      : pd.DataFrame  — rows × 27-feature columns
    params : dict          — {"min": {col: float}, "max": {col: float}}
    """
    X = X.copy()
    for col in X.columns:
        cmin  = params["min"].get(col, 0)
        cmax  = params["max"].get(col, 1)
        denom = cmax - cmin
        if denom == 0:
            # Zero-range column: fill with 1 if max > 0, else 0 (matches notebook)
            X[col] = 1 if cmax > 0 else 0
        else:
            X[col] = (X[col] - cmin) / denom
    return X.clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════

def predict_df(df):
    """
    Run the model on a DataFrame that contains the 27 feature columns.

    Returns
    -------
    preds : np.ndarray[int]   — binary predictions (0 = BENIGN, 1 = DDoS)
    probs : np.ndarray[float] — probability scores for class 1 (DDoS)
    """
    X = df[FEATURES].copy()
    # Replace infinities and NaN with 0 before normalisation
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    if NORM_PARAMS:
        X = apply_normalization(X, NORM_PARAMS)

    if IS_SKLEARN:
        preds = MODEL.predict(X.values)
        probs = MODEL.predict_proba(X.values)[:, 1]
    else:
        # Keras: predict returns raw sigmoid probabilities
        probs = MODEL.predict(X.values, batch_size=1024, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)

    return preds.astype(int), probs.astype(float)


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS  (notebook Cell 5 formula)
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """
    Compute the six classification metrics used throughout the notebook.

    Returns a dict with: Sensitivity, Specificity, Precision, Accuracy,
    F1_Score, Balanced_Accuracy, TP, TN, FP, FN.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    Sn  = TP / (TP + FN) if (TP + FN) > 0 else 0   # Sensitivity / Recall
    Sp  = TN / (TN + FP) if (TN + FP) > 0 else 0   # Specificity
    Pr  = TP / (TP + FP) if (TP + FP) > 0 else 0   # Precision
    Acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    F1  = 2 * Pr * Sn / (Pr + Sn) if (Pr + Sn) > 0 else 0
    BA  = (Sn + Sp) / 2                             # Balanced Accuracy

    return {
        "Sensitivity": round(Sn, 6), "Specificity": round(Sp, 6),
        "Precision":   round(Pr, 6), "Accuracy":    round(Acc, 6),
        "F1_Score":    round(F1, 6), "Balanced_Accuracy": round(BA, 6),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
    }


# ══════════════════════════════════════════════════════════════════════════
# JSON-SAFE VALUE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _safe_float(v):
    """
    Convert v to a JSON-serialisable float.

    NaN and ±Inf are not valid JSON (bare 'NaN' causes a SyntaxError in
    browsers), so they are replaced with 0.0.  The result is also rounded
    to 2 decimal places to reduce payload size.
    """
    try:
        f = float(v)
        return 0.0 if not math.isfinite(f) else round(f, 2)
    except (TypeError, ValueError):
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# SERVER-SIDE MATPLOTLIB PLOTS  (rendered to base-64 PNG, light theme)
# ══════════════════════════════════════════════════════════════════════════

def _fig_b64(fig):
    """Render a matplotlib Figure to a base-64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_confusion_matrix_img(y_true, y_pred, title="Confusion Matrix"):
    """
    Render a BENIGN/DDoS confusion matrix as a light-themed PNG.

    Matches the notebook Cell 5 style (Blues colour map, integer format).
    """
    lmap   = {0: "BENIGN", 1: "DDoS"}
    y_t    = [lmap.get(int(v), "BENIGN") for v in y_true]
    y_p    = [lmap.get(int(v), "BENIGN") for v in y_pred]
    labels = ["BENIGN", "DDoS"]
    cm     = confusion_matrix(y_t, y_p, labels=labels)

    fig, ax = plt.subplots(figsize=(4, 3.2), facecolor=BG)
    ax.set_facecolor(CARD)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".0f",
              text_kw={"fontsize": 14}, ax=ax)
    ax.set_xlabel("Predicted", color=TXT_DIM, fontsize=9)
    ax.set_ylabel("Actual",    color=TXT_DIM, fontsize=9)
    ax.set_title(title, color=NAVY, fontsize=11, pad=10)
    ax.tick_params(colors=TXT_DIM, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    fig.tight_layout()
    return _fig_b64(fig)


def plot_roc_img(y_true, y_scores, title="MLP"):
    """
    Render a ROC curve as a light-themed PNG.

    Returns (base64_png_string, roc_auc_float).
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc      = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 3.2), facecolor=BG)
    ax.set_facecolor(CARD)
    ax.plot(fpr, tpr, color=BLUE, lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=TXT_DIM, lw=1)   # random classifier baseline
    ax.set_xlim([-0.03, 1.05]); ax.set_ylim([-0.03, 1.05])
    ax.set_xlabel("False Positive Rate", color=TXT_DIM, fontsize=9)
    ax.set_ylabel("True Positive Rate",  color=TXT_DIM, fontsize=9)
    ax.set_title("ROC Curve — " + title, color=NAVY, fontsize=11)
    ax.tick_params(colors=TXT_DIM, labelsize=9)
    ax.legend(fontsize=8, framealpha=0.6, facecolor=CARD,
              edgecolor=BORDER, labelcolor=TXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    fig.tight_layout()
    return _fig_b64(fig), round(roc_auc, 4)


def plot_training_history_img(history):
    """
    Render accuracy + loss training curves as a two-panel light-themed PNG.

    Called by the /mlinfo route when training_history.pkl is present.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    # Accuracy panel
    ax1.plot(history.get("accuracy",     []), color=GREEN, lw=1.8, label="Train")
    ax1.plot(history.get("val_accuracy", []), color=AMBER, lw=1.8, label="Validation")
    ax1.set_title("Model Accuracy", color=NAVY, fontsize=11)
    ax1.set_xlabel("Epoch", color=TXT_DIM, fontsize=9)
    ax1.set_ylabel("Accuracy", color=TXT_DIM, fontsize=9)
    ax1.tick_params(colors=TXT_DIM, labelsize=9)
    ax1.legend(fontsize=8, framealpha=0.6, facecolor=CARD,
               edgecolor=BORDER, labelcolor=TXT)

    # Loss panel
    ax2.plot(history.get("loss",     []), color=BLUE, lw=1.8, label="Train")
    ax2.plot(history.get("val_loss", []), color=RED,  lw=1.8, label="Validation")
    ax2.set_title("Model Loss", color=NAVY, fontsize=11)
    ax2.set_xlabel("Epoch", color=TXT_DIM, fontsize=9)
    ax2.set_ylabel("Loss",  color=TXT_DIM, fontsize=9)
    ax2.tick_params(colors=TXT_DIM, labelsize=9)
    ax2.legend(fontsize=8, framealpha=0.6, facecolor=CARD,
               edgecolor=BORDER, labelcolor=TXT)

    fig.tight_layout()
    return _fig_b64(fig)


# ══════════════════════════════════════════════════════════════════════════
# LIVE-CAPTURE ENGINE
# ══════════════════════════════════════════════════════════════════════════

class LiveCapture:
    """
    Thread-safe simulated network-traffic generator.

    Generates synthetic flow records at random intervals and runs the model
    on each one.  For production use, replace _generate() with a real
    packet-capture backend (e.g. scapy / tshark) that emits the same 15
    feature keys.

    Thread safety: all access to self.records goes through self._lock.
    """

    def __init__(self):
        self.records   = []        # list of row dicts accumulated during capture
        self._running  = False
        self._thread   = None
        self._lock     = threading.Lock()
        self._start_ts = None      # epoch seconds when the capture session began (t=0)

    # ── Synthetic flow generation ──────────────────────────────────────

    def _generate(self):
        """
        Produce one simulated network-flow record and classify it.

        35 % of records are generated as attack traffic; 65 % as benign.
        The feature distributions match those used in generate_synthetic_data()
        in train_model.py.

        Returns a dict with all 27 features + metadata + model predictions.
        """
        rng       = np.random.default_rng()
        is_attack = rng.random() < 0.35   # ~35 % attack rate

        if is_attack:
            # DDoS: SYN floods, high backward rate, small packets, tight IATs
            bwd_ps = float(rng.exponential(50_000))
            feat = {
                "URG Flag Count":              float(rng.integers(0, 3)),
                "SYN Flag Count":              float(rng.integers(2, 10)),
                "Bwd Packets/s":               bwd_ps,
                "CWE Flag Count":              float(rng.integers(0, 2)),
                "Total Length of Bwd Packets": float(rng.exponential(500)),
                "Down/Up Ratio":               float(rng.uniform(0, 1)),
                "Fwd PSH Flags":               float(rng.integers(0, 4)),
                "Init_Win_bytes_forward":       float(rng.integers(0, 8192)),
                "Bwd Header Length":            float(rng.integers(20, 60)),
                "Total Backward Packets":       float(rng.integers(0, 20)),
                "Bwd Packet Length Max":        float(rng.integers(0, 200)),
                "Flow IAT Min":                 float(rng.exponential(500)),
                "Bwd Packet Length Std":        float(rng.exponential(50)),
                "Bwd IAT Std":                  float(rng.exponential(1_000)),
                "Packet Length Variance":       float(rng.exponential(500)),
                "Max Packet Length":            float(rng.integers(0, 200)),
                "Fwd IAT Min":                  float(rng.exponential(500)),
                "Bwd IAT Max":                  float(rng.exponential(10_000)),
                "Active Std":                   float(rng.exponential(5_000)),
                "Bwd IAT Total":                float(rng.exponential(50_000)),
                "Idle Std":                     float(rng.exponential(5_000)),
                "Fwd Packet Length Min":        float(rng.integers(0, 20)),
                "Active Mean":                  float(rng.exponential(50_000)),
                "Bwd Packet Length Min":        float(rng.integers(0, 10)),
                "Bwd Packet Length Mean":       float(rng.exponential(50)),
                "Active Max":                   float(rng.exponential(50_000)),
                "Idle Min":                     float(rng.exponential(5_000)),
            }
        else:
            # Benign: moderate backward rate, larger packets, longer idle/active
            bwd_ps = float(rng.exponential(5_000))
            feat = {
                "URG Flag Count":              0.0,
                "SYN Flag Count":              float(rng.integers(0, 2)),
                "Bwd Packets/s":               bwd_ps,
                "CWE Flag Count":              0.0,
                "Total Length of Bwd Packets": float(rng.exponential(50_000)),
                "Down/Up Ratio":               float(rng.uniform(1, 10)),
                "Fwd PSH Flags":               float(rng.integers(0, 2)),
                "Init_Win_bytes_forward":       float(rng.integers(8000, 65535)),
                "Bwd Header Length":            float(rng.integers(20, 60)),
                "Total Backward Packets":       float(rng.integers(10, 100)),
                "Bwd Packet Length Max":        float(rng.integers(500, 1500)),
                "Flow IAT Min":                 float(rng.exponential(50_000)),
                "Bwd Packet Length Std":        float(rng.exponential(200)),
                "Bwd IAT Std":                  float(rng.exponential(100_000)),
                "Packet Length Variance":       float(rng.exponential(50_000)),
                "Max Packet Length":            float(rng.integers(500, 1500)),
                "Fwd IAT Min":                  float(rng.exponential(50_000)),
                "Bwd IAT Max":                  float(rng.exponential(200_000)),
                "Active Std":                   float(rng.exponential(500_000)),
                "Bwd IAT Total":                float(rng.exponential(500_000)),
                "Idle Std":                     float(rng.exponential(500_000)),
                "Fwd Packet Length Min":        float(rng.integers(20, 60)),
                "Active Mean":                  float(rng.exponential(500_000)),
                "Bwd Packet Length Min":        float(rng.integers(20, 60)),
                "Bwd Packet Length Mean":       float(rng.exponential(500)),
                "Active Max":                   float(rng.exponential(500_000)),
                "Idle Min":                     float(rng.exponential(500_000)),
            }

        # Build the full row: features + metadata + model output
        now = time.time()
        row = dict(feat)
        row["Source IP"]        = f"192.168.{rng.integers(0,255)}.{rng.integers(1,255)}"
        row["Destination IP"]   = f"10.0.{rng.integers(0,255)}.{rng.integers(1,255)}"
        row["Protocol"]         = int(rng.choice([6, 17]))   # 6=TCP, 17=UDP
        row["Source Port"]      = int(rng.integers(1024, 65535))
        row["Destination Port"] = int(rng.choice([80, 443, 8080, 53]))
        row["Timestamp"]        = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # ElapsedSec drives the x-axis on the chart (t=0 when capture started)
        row["ElapsedSec"]       = round(now - self._start_ts, 2) if self._start_ts else 0
        row["GroundTruth"]      = 1 if is_attack else 0

        preds, probs = predict_df(pd.DataFrame([feat]))
        row["Prediction"]  = int(preds[0])
        # Live traffic has no ground-truth labels — use NORMAL / ATTACK
        row["PredLabel"]   = "ATTACK" if preds[0] == 1 else "NORMAL"
        row["TruthLabel"]  = "ATTACK" if is_attack else "NORMAL"
        row["Confidence"]  = round(
            float(probs[0] if preds[0] == 1 else 1.0 - probs[0]) * 100, 1)
        return row

    def _loop(self):
        """Background thread: generate one record every 0.3–0.8 seconds."""
        import random
        while self._running:
            row = self._generate()
            with self._lock:
                self.records.append(row)
                # Cap the in-memory buffer to avoid unbounded growth
                if len(self.records) > 3000:
                    self.records = self.records[-3000:]
            time.sleep(random.uniform(0.3, 0.8))

    # ── Public control methods ─────────────────────────────────────────

    def start(self):
        """Start a fresh capture session (clears previous records)."""
        if self._running:
            return
        self._start_ts = time.time()
        self.records   = []
        self._running  = True
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the capture thread to stop after its current record."""
        self._running = False

    def resume(self):
        """
        Resume a previously stopped capture WITHOUT clearing accumulated records.

        Calling start() always resets to a fresh session; resume() picks up
        where stop() left off, preserving all previously captured rows.
        """
        if self._running:
            return   # already running — no-op
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get_since(self, since=0):
        """Return records from index `since` to the end (thread-safe)."""
        with self._lock:
            return self.records[since:]

    def get_all(self):
        """Return a snapshot of all accumulated records (thread-safe)."""
        with self._lock:
            return list(self.records)

    @property
    def is_running(self):
        return self._running


# ── Module-level singletons ───────────────────────────────────────────────
CAPTURE     = LiveCapture()
CSV_RESULTS = {"df": None}   # caches the last uploaded CSV DataFrame


# ══════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=None)
app.secret_key = "ids-ddos-v3"


# ── Page routes (server-side render, all use the same template) ───────────

@app.route("/")
def home():
    """Render the Live Traffic Monitor page."""
    return render_template("index.html", page="home")

@app.route("/upload")
def upload_page():
    """Render the Upload CSV Analysis page."""
    return render_template("index.html", page="upload")

@app.route("/summary")
def summary_page():
    """Render the Analysis Summary page."""
    return render_template("index.html", page="summary")

@app.route("/mlinfo")
def mlinfo_page():
    """Render the ML Information page with the training-history plot."""
    hist_img = plot_training_history_img(HISTORY) if HISTORY else None
    return render_template("index.html", page="mlinfo", history_img=hist_img)


# ── API: Live capture ─────────────────────────────────────────────────────

@app.route("/api/capture/start", methods=["POST"])
def api_capture_start():
    """Start (or restart) the live traffic capture."""
    CAPTURE.start()
    return jsonify({"status": "running"})


@app.route("/api/capture/stop", methods=["POST"])
def api_capture_stop():
    """Stop the live traffic capture."""
    CAPTURE.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/capture/resume", methods=["POST"])
def api_capture_resume():
    """Resume the live traffic capture without clearing accumulated records."""
    CAPTURE.resume()
    return jsonify({"status": "running"})


@app.route("/api/capture/poll")
def api_capture_poll():
    """
    Return new rows since the given `since` index.

    Query param: since (int, default 0)
    Response:    {rows, total, running}

    Each row includes ElapsedSec (chart x-axis) and Flow Bytes/s (y-axis)
    so the front-end can animate new points one at a time.
    """
    since = int(request.args.get("since", 0))
    rows  = CAPTURE.get_since(since)
    total = len(CAPTURE.get_all())
    return jsonify({"rows": rows, "total": total, "running": CAPTURE.is_running})


@app.route("/api/capture/save", methods=["POST"])
def api_capture_save():
    """
    Export the current capture session to two CSV files in results/.

    Writes:
      raw_live_traffic.csv      — raw feature + metadata columns
      analyzed_live_traffic.csv — same + Predicted_Label + Model_Confidence_%
    """
    records = CAPTURE.get_all()
    if not records:
        return jsonify({"error": "No data captured yet. Start capturing first."}), 400

    # Ensure the results/ directory exists (it should, but be defensive)
    os.makedirs(_ROOT + "/results", exist_ok=True)

    df = pd.DataFrame(records)

    # File 1: raw — exclude model-output columns
    raw_drop = {"Prediction", "PredLabel", "Confidence",
                "GroundTruth", "TruthLabel", "ElapsedSec"}
    raw_cols = [c for c in df.columns if c not in raw_drop]
    raw_path = _results("raw_live_traffic.csv")
    df[raw_cols].to_csv(raw_path, index=False)

    # File 2: analyzed — add human-readable prediction columns
    analyzed = df.copy()
    analyzed["Predicted_Label"]    = analyzed["PredLabel"]   # BENIGN / DDoS
    analyzed["Model_Confidence_%"] = analyzed["Confidence"]  # 0–100
    analyzed_path = _results("analyzed_live_traffic.csv")
    analyzed.to_csv(analyzed_path, index=False)

    return jsonify({
        "count":         len(records),
        "saved":         True,
        "raw_path":      raw_path,
        "analyzed_path": analyzed_path,
    })


# ── API: CSV Upload ───────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accept a CSV upload, run predictions, and return analysis results.

    Expected columns: the 15 FEATURES (required) + optionally 'Label'.
    If 'Label' is present, ground-truth metrics, ROC curve, and confusion
    matrices are included in the response.

    The chart_points array uses _safe_float() to convert Flow Bytes/s values
    so that NaN / ±Inf never produce invalid JSON (bare 'NaN' is not
    valid JSON and causes a SyntaxError in browsers).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file attached"}), 400

    try:
        df = pd.read_csv(request.files["file"])
    except Exception as e:
        return jsonify({"error": f"CSV read error: {e}"}), 400

    # Validate that all 15 required feature columns are present
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

    preds, probs = predict_df(df)
    df["Predicted"] = preds

    # Optional ground-truth column for metric computation
    has_label = "Label" in df.columns
    if has_label:
        # ── Multi-class re-labeling ─────────────────────────────────────────
        # CICDDoS2019 CSVs often contain many attack sub-classes
        # (DrDoS_LDAP, DrDoS_MSSQL, Syn, NTP, TFTP, etc.).
        # Strategy: whitelist BENIGN → 0; everything else → 1 (DDoS/Attack).
        # Using a BENIGN-whitelist (rather than an attack-whitelist) ensures
        # novel attack-class names are captured automatically, which also
        # prevents NaN values in the JSON response caused by a completely
        # BENIGN y_true when no attack names are recognised.
        lbl = df["Label"].astype(str).str.strip().str.lower()
        df["Label_bin"] = lbl.map(lambda x: 0 if x in ("0", "benign") else 1)
        n_attack_labels = int((df["Label_bin"] == 1).sum())
        n_benign_labels = int((df["Label_bin"] == 0).sum())
        print(f"[RELABEL] BENIGN: {n_benign_labels}  Attack (relabelled): {n_attack_labels}")

    CSV_RESULTS["df"] = df   # cache for the /summary endpoint

    # ── Flow Packets/s for chart y-axis ──────────────────────────────────
    # Prefer an explicit "Flow Packets/s" column when present in the CSV;
    # otherwise approximate: Flow Bytes/s ÷ Average Packet Size.
    if "Flow Packets/s" in df.columns:
        fps_series = df["Flow Packets/s"].copy()
    else:
        avg_pkt    = df["Average Packet Size"].replace(0, np.nan).fillna(1.0)
        fps_series = (df["Flow Bytes/s"] / avg_pkt)

    # ── Downsample chart points for large datasets (100 k+ rows) ────────
    # The browser chart displays at most MAX_CHART_PTS points; beyond that
    # we take every step-th row so the payload stays manageable.
    n           = len(df)
    MAX_CHART   = 5000
    if n > MAX_CHART:
        step    = max(1, n // MAX_CHART)
        indices = list(range(0, n, step))
    else:
        indices = list(range(n))

    # Optional metadata columns that uploaded CSVs may or may not contain
    has_src   = "Source IP"      in df.columns
    has_dst   = "Destination IP" in df.columns
    has_proto = "Protocol"       in df.columns
    has_ts    = "Timestamp"      in df.columns

    def _proto_str(v):
        """Convert numeric protocol code to human-readable string."""
        try:
            iv = int(float(v))
            return "TCP" if iv == 6 else ("UDP" if iv == 17 else str(iv))
        except (TypeError, ValueError):
            return str(v)

    chart_pts = []
    for i in indices:
        row = df.iloc[i]
        chart_pts.append({
            "x":     i,
            "y":     _safe_float(fps_series.iloc[i]),   # Flow Packets/s
            "label": int(preds[i]),
            # Ground-truth label: 0=BENIGN, 1=DDoS, -1=no ground truth in CSV
            "truth": int(df["Label_bin"].iloc[i]) if has_label else -1,
            # Hover-tooltip metadata — default to N/A if column absent
            "ts":    str(row["Timestamp"])       if has_ts    else f"idx:{i}",
            "src":   str(row["Source IP"])       if has_src   else "N/A",
            "dst":   str(row["Destination IP"])  if has_dst   else "N/A",
            "proto": _proto_str(row["Protocol"]) if has_proto else "N/A",
        })

    # ── Build the JSON response ──────────────────────────────────────────
    result = {
        "total":       n,
        "normal":      int((preds == 0).sum()),
        "attack":      int((preds == 1).sum()),
        "chart_points": chart_pts,
        "downsampled": n > MAX_CHART,   # front-end shows a notice when True
        "total_pts":   n,
    }

    if has_label:
        y_true = df["Label_bin"].values.astype(int)
        result["metrics"] = compute_metrics(y_true, preds)

        # ROC curve requires both classes to be present.
        # Guard against single-class datasets and protect against NaN values
        # in JSON (bare NaN is not valid JSON → "Unexpected token 'N'" error).
        try:
            if len(np.unique(y_true)) >= 2:
                roc_img, roc_val = plot_roc_img(y_true, probs, "MLP")
                result["roc_img"] = roc_img
                result["roc_auc"] = _safe_float(roc_val)   # NaN → 0.0
            else:
                result["roc_auc"] = 0.0   # single class → AUC undefined
        except Exception as e:
            print(f"[WARN] ROC computation failed: {e}")
            result["roc_auc"] = 0.0       # fallback; never leave bare NaN in JSON

        # Overall confusion matrix only
        result["cm_overall"] = plot_confusion_matrix_img(y_true, preds, "Overall")

        # ── Attack duration range ─────────────────────────────────────────
        # Span from the FIRST to the LAST DDoS ground-truth instance.
        # Sent as original DataFrame row indices (matching chart_points x values)
        # so the front-end can draw a single highlight rectangle.
        atk_indices = np.where(y_true == 1)[0]
        if len(atk_indices) > 0:
            result["attack_range"] = [int(atk_indices[0]), int(atk_indices[-1])]

    return jsonify(result)


# ── API: Summary ──────────────────────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    """
    Return aggregate statistics for both the live session and the last CSV.

    Response shape: {"live": {...} | null, "csv": {...} | null}
    Each sub-object has total, normal, attack, and (if ground truth is
    available) the full metric set plus ROC-AUC.
    """
    result = {"live": None, "csv": None}

    # ── Live-capture summary ──────────────────────────────────────────────
    # Live traffic has no ground-truth labels; only model predictions are
    # available.  Report counts only — no metrics, no confusion matrix.
    records = CAPTURE.get_all()
    if records:
        df     = pd.DataFrame(records)
        y_pred = df["Prediction"].values.astype(int)
        result["live"] = {
            "total":  len(df),
            "normal": int((y_pred == 0).sum()),   # NORMAL instances
            "attack": int((y_pred == 1).sum()),   # ATTACK instances
            # Label vocabulary tells the front-end which strings to display
            "labels": {"normal": "NORMAL", "attack": "ATTACK"},
        }

    # ── CSV upload summary ────────────────────────────────────────────────
    if CSV_RESULTS["df"] is not None:
        df     = CSV_RESULTS["df"]
        y_pred = df["Predicted"].values.astype(int)
        info   = {
            "total":  len(df),
            "normal": int((y_pred == 0).sum()),
            "attack": int((y_pred == 1).sum()),
            "labels": {"normal": "BENIGN", "attack": "DDoS"},
        }
        if "Label_bin" in df.columns:
            info.update(compute_metrics(df["Label_bin"].values.astype(int), y_pred))
            # Restore labels dict overwritten by compute_metrics update
            info["labels"] = {"normal": "BENIGN", "attack": "DDoS"}
        result["csv"] = info

    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[READY] IDS Web GUI (v3) → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
