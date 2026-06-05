"""
Quantum Machine Learning Classification Pipeline
Using Qiskit & Qiskit Machine Learning
Run: python main.py
"""

import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

# Fix Windows encoding
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.svm import SVC

# ─────────────────────────────────────────────
# STEP 0: Banner
# ─────────────────────────────────────────────
print("=" * 60)
print("   [QML] QUANTUM MACHINE LEARNING PIPELINE (Qiskit)")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Auto-detect CSV
# ─────────────────────────────────────────────
print("\n[STEP 1] Detecting dataset...")

csv_files = glob.glob("*.csv") + glob.glob("data/*.csv") + glob.glob("dataset/*.csv")
if not csv_files:
    print("[ERROR] No CSV file found in project folder!")
    print("   Place your CSV file in the same folder as main.py and re-run.")
    sys.exit(1)

csv_path = csv_files[0]
print(f"   [OK] Found: {csv_path}")

df = pd.read_csv(csv_path)
print(f"   Shape  : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# STEP 2: Identify target column
# ─────────────────────────────────────────────
print("\n[STEP 2] Identifying target column...")

target_candidates = ["label", "target", "class", "output", "y",
                     "diagnosis", "result", "category"]
target_col = None
for col in df.columns:
    if col.strip().lower() in target_candidates:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]   # fallback: last column

print(f"   [OK] Target column : '{target_col}'")
print(f"   Class distribution:\n{df[target_col].value_counts().to_string()}")

X_raw = df.drop(columns=[target_col])
y_raw = df[target_col].values

# ─────────────────────────────────────────────
# STEP 3: Preprocessing
# ─────────────────────────────────────────────
print("\n[STEP 3] Preprocessing...")

X_raw = X_raw.select_dtypes(include=[np.number])
print(f"   Numeric features : {X_raw.shape[1]}")

null_count = X_raw.isnull().sum().sum()
if null_count > 0:
    print(f"   Found {null_count} missing values -> filling with column median")
    X_raw = X_raw.fillna(X_raw.median())
else:
    print("   No missing values [OK]")

le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
n_classes = len(classes)
print(f"   Classes ({n_classes}): {list(classes)}")

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_raw)

n_features = X_scaled.shape[1]
n_qubits = min(n_features, 8)
if n_features > n_qubits:
    print(f"   PCA: {n_features} features -> {n_qubits} components (qubit limit)")
    pca = PCA(n_components=n_qubits, random_state=42)
    X_final = pca.fit_transform(X_scaled)
    scaler2 = MinMaxScaler(feature_range=(0, np.pi))
    X_final = scaler2.fit_transform(X_final)
else:
    n_qubits = n_features
    scaler2 = MinMaxScaler(feature_range=(0, np.pi))
    X_final = scaler2.fit_transform(X_scaled)
    print(f"   No PCA needed -- using all {n_qubits} features as qubits")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 4: Build Quantum Circuit
# ─────────────────────────────────────────────
print("\n[STEP 4] Building Quantum Circuit...")

try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, VQC
    from qiskit.primitives import StatevectorSampler, StatevectorEstimator
    from scipy.optimize import minimize
    QISKIT_NEW = True
except ImportError:
    try:
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
        from qiskit.primitives import Sampler
        QISKIT_NEW = False
    except ImportError:
        print("[ERROR] Qiskit not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
ansatz = RealAmplitudes(num_qubits=n_qubits, reps=3)

from qiskit import QuantumCircuit
qc = QuantumCircuit(n_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

print(f"   Qubits       : {n_qubits}")
print(f"   Circuit depth: {qc.depth()}")
print(f"   Parameters   : {qc.num_parameters}")

# Save circuit diagram with utf-8 encoding
circuit_str = qc.draw(output="text").__str__()
with open("circuit.txt", "w", encoding="utf-8") as f:
    f.write("QUANTUM CIRCUIT DIAGRAM\n")
    f.write("=" * 60 + "\n")
    f.write(circuit_str)
print("   Circuit saved -> circuit.txt")

# ─────────────────────────────────────────────
# STEP 5: Train Quantum Model (VQC)
# ─────────────────────────────────────────────
print("\n[STEP 5] Training Quantum Classifier (VQC)...")
print("   (This may take a few minutes on simulator...)\n")

from qiskit_machine_learning.algorithms import VQC

try:
    from qiskit.primitives import StatevectorSampler
    sampler = StatevectorSampler()
except Exception:
    from qiskit.primitives import Sampler
    sampler = Sampler()

loss_history = []

def callback(weights, obj_func_eval):
    loss_history.append(obj_func_eval)
    if len(loss_history) % 10 == 0:
        print(f"   Iteration {len(loss_history):3d} | Loss: {obj_func_eval:.4f}")

MAX_TRAIN = min(len(X_train), 80)
X_tr = X_train[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=None,
    callback=callback,
)

QUANTUM_OK = False
try:
    vqc.fit(X_tr, y_tr)
    y_pred_q = vqc.predict(X_test)
    QUANTUM_OK = True
except Exception as e:
    print(f"   [WARN] VQC training error: {e}")
    print("   -> Falling back to QNN-based manual training...")

# ─────────────────────────────────────────────
# STEP 5b: Fallback — manual SamplerQNN training
# ─────────────────────────────────────────────
if not QUANTUM_OK:
    try:
        from qiskit.primitives import Sampler
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

        sampler_fb = Sampler()
        observable_params = ansatz.parameters
        input_params = feature_map.parameters

        def parity(x):
            return "{:b}".format(x).count("1") % 2

        qnn = SamplerQNN(
            circuit=qc,
            input_params=list(input_params),
            weight_params=list(observable_params),
            interpret=parity,
            output_shape=2,
            sampler=sampler_fb,
        )

        classifier = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=None,
            loss="cross_entropy",
            callback=callback,
        )
        classifier.fit(X_tr, y_tr)
        y_pred_q = classifier.predict(X_test)
        QUANTUM_OK = True
    except Exception as e2:
        print(f"   [WARN] QNN fallback also failed: {e2}")
        print("   -> Quantum simulation not possible in this env.")
        print("      Showing classical SVM results only.\n")
        QUANTUM_OK = False

# ─────────────────────────────────────────────
# STEP 6: Classical SVM Baseline
# ─────────────────────────────────────────────
print("\n[STEP 6] Training Classical SVM Baseline...")

svm = SVC(kernel="rbf", C=1.0, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm  = f1_score(y_test, y_pred_svm, average="weighted", zero_division=0)
print(f"   SVM Accuracy : {acc_svm:.4f}")
print(f"   SVM F1-Score : {f1_svm:.4f}")

# ─────────────────────────────────────────────
# STEP 7: Evaluate & Compare
# ─────────────────────────────────────────────
print("\n[STEP 7] Evaluation Results")
print("=" * 60)

results_lines = []
results_lines.append("QUANTUM ML CLASSIFICATION -- RESULTS\n")
results_lines.append("=" * 60 + "\n")

if QUANTUM_OK:
    acc_q  = accuracy_score(y_test, y_pred_q)
    prec_q = precision_score(y_test, y_pred_q, average="weighted", zero_division=0)
    rec_q  = recall_score(y_test, y_pred_q, average="weighted", zero_division=0)
    f1_q   = f1_score(y_test, y_pred_q, average="weighted", zero_division=0)
    cm_q   = confusion_matrix(y_test, y_pred_q)

    print("\n[QML] QUANTUM VQC:")
    print(f"   Accuracy  : {acc_q:.4f}")
    print(f"   Precision : {prec_q:.4f}")
    print(f"   Recall    : {rec_q:.4f}")
    print(f"   F1-Score  : {f1_q:.4f}")
    print("\n   Confusion Matrix:")
    print("   " + str(cm_q).replace("\n", "\n   "))
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred_q,
          target_names=[str(c) for c in classes], zero_division=0))

    results_lines.append("\n[QML] QUANTUM VQC RESULTS:\n")
    results_lines.append(f"  Accuracy  : {acc_q:.4f}\n")
    results_lines.append(f"  Precision : {prec_q:.4f}\n")
    results_lines.append(f"  Recall    : {rec_q:.4f}\n")
    results_lines.append(f"  F1-Score  : {f1_q:.4f}\n")
    results_lines.append(f"\n  Confusion Matrix:\n{cm_q}\n")
    results_lines.append("\n  Classification Report:\n")
    results_lines.append(classification_report(y_test, y_pred_q,
        target_names=[str(c) for c in classes], zero_division=0))

print("\n[SVM] CLASSICAL SVM BASELINE:")
print(f"   Accuracy  : {acc_svm:.4f}")
print(f"   F1-Score  : {f1_svm:.4f}")
print("\n   Confusion Matrix:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("   " + str(cm_svm).replace("\n", "\n   "))

results_lines.append("\n[SVM] CLASSICAL SVM BASELINE:\n")
results_lines.append(f"  Accuracy  : {acc_svm:.4f}\n")
results_lines.append(f"  F1-Score  : {f1_svm:.4f}\n")
results_lines.append(f"\n  Confusion Matrix:\n{cm_svm}\n")

if QUANTUM_OK:
    print("\n" + "-" * 60)
    print("[COMPARISON]")
    print(f"   QML Accuracy : {acc_q:.4f}  |  SVM Accuracy : {acc_svm:.4f}")
    diff = acc_q - acc_svm
    better = "QML" if diff > 0 else "SVM"
    print(f"   -> {better} wins by {abs(diff):.4f}")
    results_lines.append(f"\nCOMPARISON: QML={acc_q:.4f}  SVM={acc_svm:.4f}  -> {better} wins\n")

# ─────────────────────────────────────────────
# STEP 8: Save outputs
# ─────────────────────────────────────────────
print("\n[STEP 8] Saving outputs...")

with open("results.txt", "w", encoding="utf-8") as f:
    f.writelines(results_lines)
print("   [OK] results.txt saved")

if QUANTUM_OK:
    try:
        weights = vqc.weights
        np.save("model_weights.npy", weights)
        print("   [OK] model_weights.npy saved")
    except Exception:
        print("   [WARN] Could not save model weights")

import pickle
with open("svm_baseline.pkl", "wb") as f:
    pickle.dump(svm, f)
print("   [OK] svm_baseline.pkl saved")

print("\n" + "=" * 60)
print("   [DONE] PIPELINE COMPLETE!")
print("   Outputs: results.txt | circuit.txt | model_weights.npy")
print("=" * 60 + "\n")
