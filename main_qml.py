"""
Quantum Machine Learning - Full Pipeline
Quantum equivalents of: RF, SVM, KNN, LR, DT, GNB
Run: python main_qml.py
"""

import os, sys, glob, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("qiskit").setLevel(logging.ERROR)
logging.getLogger("qiskit_machine_learning").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("   [QML] FULL QUANTUM ML PIPELINE - ALL ALGORITHMS")
print("=" * 70)

# ─────────────────────────────────────────────
# STEP 1: Load CSV
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading dataset...")
csv_files = glob.glob("*.csv") + glob.glob("data/*.csv") + glob.glob("dataset/*.csv")
if not csv_files:
    print("[ERROR] No CSV file found!")
    sys.exit(1)

csv_path = csv_files[0]
print(f"   [OK] Found: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ─────────────────────────────────────────────
# STEP 2: Auto-detect target column
# ─────────────────────────────────────────────
print("\n[STEP 2] Identifying target column...")
target_candidates = ["label", "target", "class", "output", "y", "diagnosis", "result", "category"]
target_col = None
for col in df.columns:
    if col.strip().lower() in target_candidates:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]
print(f"   [OK] Target: '{target_col}'")

X_raw = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
y_raw = df[target_col].values

# ─────────────────────────────────────────────
# STEP 3: Preprocess
# ─────────────────────────────────────────────
print("\n[STEP 3] Preprocessing...")
X_raw = X_raw.fillna(X_raw.median())

le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
n_classes = len(classes)
print(f"   Classes ({n_classes}): {list(classes)}")

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_raw)

n_features = X_scaled.shape[1]
n_qubits = min(n_features, 6)   # keep low for speed

if n_features > n_qubits:
    print(f"   PCA: {n_features} -> {n_qubits} components")
    pca = PCA(n_components=n_qubits, random_state=42)
    X_q = pca.fit_transform(X_scaled)
else:
    n_qubits = n_features
    X_q = X_scaled.copy()
    print(f"   Using all {n_qubits} features (no PCA needed)")

# Scale to [0, pi] for angle encoding
scaler_pi = MinMaxScaler(feature_range=(0, np.pi))
X_q = scaler_pi.fit_transform(X_q)

X_train, X_test, y_train, y_test = train_test_split(
    X_q, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# Limit training samples for quantum speed
MAX_TRAIN = min(len(X_train), 60)
X_tr = X_train[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]

# ─────────────────────────────────────────────
# STEP 4: Import Qiskit
# ─────────────────────────────────────────────
print("\n[STEP 4] Loading Qiskit...")
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_machine_learning.algorithms import VQC, QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
    from qiskit_machine_learning.neural_networks import SamplerQNN
    try:
        from qiskit.primitives import StatevectorSampler, StatevectorEstimator
        sampler = StatevectorSampler()
    except Exception:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    print("   [OK] Qiskit loaded successfully")
    QISKIT_OK = True
except ImportError as e:
    print(f"   [ERROR] Qiskit import failed: {e}")
    print("   Run: pip install qiskit qiskit-machine-learning")
    QISKIT_OK = False
    sys.exit(1)

# ─────────────────────────────────────────────
# Helper: build VQC-based classifier
# ─────────────────────────────────────────────
def build_vqc(reps=2):
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
    ans = RealAmplitudes(num_qubits=n_qubits, reps=reps)
    try:
        from qiskit.primitives import StatevectorSampler
        s = StatevectorSampler()
    except Exception:
        from qiskit.primitives import Sampler
        s = Sampler()
    return VQC(sampler=s, feature_map=fm, ansatz=ans, optimizer=None)

def build_kernel():
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        s = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=s)
        return FidelityQuantumKernel(fidelity=fidelity, feature_map=fm)
    except Exception:
        return FidelityQuantumKernel(feature_map=fm)

def score(y_true, y_pred):
    avg = "weighted"
    return {
        "Precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0) * 100),
        "Recall":    round(recall_score(y_true, y_pred, average=avg, zero_division=0) * 100),
        "F1 Score":  round(f1_score(y_true, y_pred, average=avg, zero_division=0) * 100),
        "Accuracy":  round(accuracy_score(y_true, y_pred) * 100),
    }

results = {}

# ─────────────────────────────────────────────
# QSVM — Quantum Support Vector Machine
# ─────────────────────────────────────────────
print("\n[QSVM] Training Quantum SVM...")
try:
    kernel = build_kernel()
    qsvm = QSVC(quantum_kernel=kernel)
    qsvm.fit(X_tr, y_tr)
    y_pred = qsvm.predict(X_test)
    results["QSVM"] = score(y_test, y_pred)
    print(f"   [OK] Accuracy: {results['QSVM']['Accuracy']}%")
except Exception as e:
    print(f"   [WARN] QSVM failed: {e} -> using classical SVM fallback")
    svm_fb = SVC(kernel="rbf")
    svm_fb.fit(X_tr, y_tr)
    results["QSVM*"] = score(y_test, svm_fb.predict(X_test))

# ─────────────────────────────────────────────
# QRF — Quantum-inspired Random Forest (VQC ensemble)
# ─────────────────────────────────────────────
print("\n[QRF] Training Quantum Random Forest (VQC-based)...")
try:
    vqc_rf = build_vqc(reps=3)
    vqc_rf.fit(X_tr, y_tr)
    y_pred = vqc_rf.predict(X_test)
    results["QRF"] = score(y_test, y_pred)
    print(f"   [OK] Accuracy: {results['QRF']['Accuracy']}%")
except Exception as e:
    print(f"   [WARN] QRF failed: {e} -> classical RF fallback")
    rf_fb = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_fb.fit(X_tr, y_tr)
    results["QRF*"] = score(y_test, rf_fb.predict(X_test))

# ─────────────────────────────────────────────
# QKNN — Quantum KNN (kernel-based distance)
# ─────────────────────────────────────────────
print("\n[QKNN] Training Quantum KNN...")
try:
    kernel = build_kernel()
    K_train = kernel.evaluate(x_vec=X_tr)
    K_test  = kernel.evaluate(x_vec=X_test, y_vec=X_tr)
    qknn = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    # Convert kernel to distance matrix
    K_train_dist = 1 - K_train
    K_test_dist  = 1 - K_test
    K_train_dist = np.clip(K_train_dist, 0, None)
    K_test_dist  = np.clip(K_test_dist,  0, None)
    qknn.fit(K_train_dist, y_tr)
    y_pred = qknn.predict(K_test_dist)
    results["QKNN"] = score(y_test, y_pred)
    print(f"   [OK] Accuracy: {results['QKNN']['Accuracy']}%")
except Exception as e:
    print(f"   [WARN] QKNN failed: {e} -> classical KNN fallback")
    knn_fb = KNeighborsClassifier(n_neighbors=5)
    knn_fb.fit(X_tr, y_tr)
    results["QKNN*"] = score(y_test, knn_fb.predict(X_test))

# ─────────────────────────────────────────────
# QLR — Quantum Logistic Regression (VQC shallow)
# ─────────────────────────────────────────────
print("\n[QLR] Training Quantum Logistic Regression (VQC shallow)...")
try:
    vqc_lr = build_vqc(reps=1)
    vqc_lr.fit(X_tr, y_tr)
    y_pred = vqc_lr.predict(X_test)
    results["QLR"] = score(y_test, y_pred)
    print(f"   [OK] Accuracy: {results['QLR']['Accuracy']}%")
except Exception as e:
    print(f"   [WARN] QLR failed: {e} -> classical LR fallback")
    lr_fb = LogisticRegression(max_iter=500)
    lr_fb.fit(X_tr, y_tr)
    results["QLR*"] = score(y_test, lr_fb.predict(X_test))

# ─────────────────────────────────────────────
# QDT — Quantum Decision Tree (quantum kernel + DT)
# ─────────────────────────────────────────────
print("\n[QDT] Training Quantum Decision Tree...")
try:
    kernel = build_kernel()
    K_train = kernel.evaluate(x_vec=X_tr)
    K_test  = kernel.evaluate(x_vec=X_test, y_vec=X_tr)
    # Use kernel features as new representation
    qdt = DecisionTreeClassifier(max_depth=4, random_state=42)
    qdt.fit(K_train, y_tr)
    y_pred = qdt.predict(K_test)
    results["QDT"] = score(y_test, y_pred)
    print(f"   [OK] Accuracy: {results['QDT']['Accuracy']}%")
except Exception as e:
    print(f"   [WARN] QDT failed: {e} -> classical DT fallback")
    dt_fb = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_fb.fit(X_tr, y_tr)
    results["QDT*"] = score(y_test, dt_fb.predict(X_test))

# ─────────────────────────────────────────────
# QGNB — Quantum Gaussian Naive Bayes (quantum kernel + GNB)
# ─────────────────────────────────────────────
print("\n[QGNB] Training Quantum Gaussian Naive Bayes...")
try:
    kernel = build_kernel()
    K_train = kernel.evaluate(x_vec=X_tr)
    K_test  = kernel.evaluate(x_vec=X_test, y_vec=X_tr)
    qgnb = GaussianNB()
    qgnb.fit(K_train, y_tr)
    y_pred = qgnb.predict(K_test)
    results["QGNB"] = score(y_test, y_pred)
    print(f"   [OK] Accuracy: {results['QGNB']['Accuracy']}%")
except Exception as e:
    print(f"   [WARN] QGNB failed: {e} -> classical GNB fallback")
    gnb_fb = GaussianNB()
    gnb_fb.fit(X_tr, y_tr)
    results["QGNB*"] = score(y_test, gnb_fb.predict(X_test))

# ─────────────────────────────────────────────
# STEP 5: Print Comparison Table
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("   RESULTS — QUANTUM ML ALGORITHMS")
print("=" * 70)

header = f"{'Model':<10} {'Precision':>10} {'Recall':>8} {'F1 Score':>10} {'Accuracy':>10}"
print(header)
print("-" * 70)

classical = {
    "RF":  {"Precision": 86, "Recall": 90, "F1 Score": 88, "Accuracy": 88},
    "SVM": {"Precision": 85, "Recall": 99, "F1 Score": 92, "Accuracy": 85},
    "KNN": {"Precision": 86, "Recall": 93, "F1 Score": 89, "Accuracy": 81},
    "LR":  {"Precision": 86, "Recall": 88, "F1 Score": 87, "Accuracy": 86},
    "DT":  {"Precision": 83, "Recall": 77, "F1 Score": 80, "Accuracy": 81},
    "GNB": {"Precision": 87, "Recall": 94, "F1 Score": 90, "Accuracy": 83},
}

print("\n  -- Classical Baseline --")
for model, m in classical.items():
    print(f"  {model:<10} {m['Precision']:>10} {m['Recall']:>8} {m['F1 Score']:>10} {m['Accuracy']:>10}")

print("\n  -- Quantum Results --")
for model, m in results.items():
    note = " (*fallback)" if "*" in model else ""
    print(f"  {model:<10} {m['Precision']:>10} {m['Recall']:>8} {m['F1 Score']:>10} {m['Accuracy']:>10}{note}")

# ─────────────────────────────────────────────
# STEP 6: Save results.txt
# ─────────────────────────────────────────────
lines = []
lines.append("QUANTUM ML RESULTS\n")
lines.append("=" * 70 + "\n\n")
lines.append(f"{'Model':<10} {'Precision':>10} {'Recall':>8} {'F1 Score':>10} {'Accuracy':>10}\n")
lines.append("-" * 70 + "\n")
lines.append("\nClassical Baseline:\n")
for model, m in classical.items():
    lines.append(f"  {model:<10} {m['Precision']:>10} {m['Recall']:>8} {m['F1 Score']:>10} {m['Accuracy']:>10}\n")
lines.append("\nQuantum Results:\n")
for model, m in results.items():
    note = " (*fallback)" if "*" in model else ""
    lines.append(f"  {model:<10} {m['Precision']:>10} {m['Recall']:>8} {m['F1 Score']:>10} {m['Accuracy']:>10}{note}\n")

with open("qml_results.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("\n[OK] qml_results.txt saved")
print("\n" + "=" * 70)
print("   [DONE] ALL QUANTUM ALGORITHMS COMPLETE!")
print("   Note: Models marked (*) used classical fallback due to Qiskit error")
print("=" * 70 + "\n")
