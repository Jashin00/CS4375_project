# ============================================================
# Step 3: Baseline Models (TF-IDF + Classical ML)
# Toxic Comment Classifier — Project Group
# ============================================================
# pip install pandas numpy scikit-learn matplotlib seaborn
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, hamming_loss,
    precision_score, recall_score, confusion_matrix, classification_report
)

# ── Config ──────────────────────────────────────────────────
DATA_DIR   = "outputs/preprocessed"
OUTPUT_DIR = "outputs/baseline"
MODEL_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# ── Load Data ────────────────────────────────────────────────
print("Loading preprocessed splits...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv").fillna("")
val_df   = pd.read_csv(f"{DATA_DIR}/val.csv").fillna("")
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv").fillna("")

X_train = train_df["clean_text_nosw"].values
X_val   = val_df["clean_text_nosw"].values
X_test  = test_df["clean_text_nosw"].values

y_train = train_df[LABELS].values
y_val   = val_df[LABELS].values
y_test  = test_df[LABELS].values

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")


# ── TF-IDF Vectorization ────────────────────────────────────
print("\nBuilding TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=50_000,        # Keep top 50k terms
    ngram_range=(1, 2),         # Unigrams + bigrams
    sublinear_tf=True,          # Apply log normalization
    strip_accents="unicode",
    analyzer="word",
    min_df=3,                   # Ignore very rare terms
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

print(f"  TF-IDF matrix shape: {X_train_tfidf.shape}")

# Save vectorizer for later use
with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("  Saved TF-IDF vectorizer")


# ── Evaluation Helper ────────────────────────────────────────
def evaluate(model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray) -> dict:
    """
    Compute and print all evaluation metrics for a multi-label classifier.
    Returns a dict of scores for comparison table.
    """
    roc_aucs = []
    for i, label in enumerate(LABELS):
        try:
            auc = roc_auc_score(y_true[:, i], y_proba[:, i])
        except Exception:
            auc = 0.5
        roc_aucs.append(auc)

    mean_auc   = np.mean(roc_aucs)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1   = f1_score(y_true, y_pred, average="micro", zero_division=0)
    h_loss     = hamming_loss(y_true, y_pred)
    precision  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall     = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'─'*55}")
    print(f"  MODEL: {model_name}")
    print(f"{'─'*55}")
    print(f"  Mean ROC-AUC   : {mean_auc:.4f}  ← main metric")
    print(f"  Macro F1-Score : {macro_f1:.4f}")
    print(f"  Micro F1-Score : {micro_f1:.4f}")
    print(f"  Hamming Loss   : {h_loss:.4f}  (lower = better)")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall   : {recall:.4f}")
    print(f"\n  Per-label ROC-AUC:")
    for label, auc in zip(LABELS, roc_aucs):
        bar = "█" * int(auc * 20)
        print(f"    {label:<15} {auc:.4f}  {bar}")

    return {
        "Model": model_name,
        "ROC-AUC": round(mean_auc, 4),
        "Macro F1": round(macro_f1, 4),
        "Micro F1": round(micro_f1, 4),
        "Hamming Loss": round(h_loss, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
    }


# ── Model 1: Logistic Regression ────────────────────────────
print("\n\n══ Training Logistic Regression ══")
lr_model = OneVsRestClassifier(
    LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",  # handles class imbalance
        random_state=42,
    ),
    n_jobs=-1,
)
lr_model.fit(X_train_tfidf, y_train)
lr_pred  = lr_model.predict(X_test_tfidf)
lr_proba = lr_model.predict_proba(X_test_tfidf)
lr_results = evaluate("Logistic Regression (TF-IDF)", y_test, lr_pred, lr_proba)

with open(f"{MODEL_DIR}/lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)


# ── Model 2: LinearSVC ───────────────────────────────────────
print("\n\n══ Training LinearSVC ══")
# Wrap with CalibratedClassifierCV to get probability estimates
svc_base = OneVsRestClassifier(
    CalibratedClassifierCV(
        LinearSVC(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    ),
    n_jobs=-1,
)
svc_base.fit(X_train_tfidf, y_train)
svc_pred  = svc_base.predict(X_test_tfidf)
svc_proba = svc_base.predict_proba(X_test_tfidf)
svc_results = evaluate("LinearSVC (TF-IDF)", y_test, svc_pred, svc_proba)

with open(f"{MODEL_DIR}/svc_model.pkl", "wb") as f:
    pickle.dump(svc_base, f)


# ── Model 3: Naive Bayes ─────────────────────────────────────
print("\n\n══ Training Naive Bayes ══")
from sklearn.preprocessing import MaxAbsScaler
# MultinomialNB requires non-negative values — use MaxAbsScaler
scaler = MaxAbsScaler()
X_train_nb = scaler.fit_transform(X_train_tfidf)
X_test_nb  = scaler.transform(X_test_tfidf)

nb_model = OneVsRestClassifier(MultinomialNB(alpha=0.1), n_jobs=-1)
nb_model.fit(X_train_nb, y_train)
nb_pred  = nb_model.predict(X_test_nb)
nb_proba = nb_model.predict_proba(X_test_nb)
nb_results = evaluate("Naive Bayes (TF-IDF)", y_test, nb_pred, nb_proba)

with open(f"{MODEL_DIR}/nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)


# ── Comparison Table ─────────────────────────────────────────
print("\n\n══ BASELINE COMPARISON TABLE ══")
results_df = pd.DataFrame([lr_results, svc_results, nb_results])
results_df = results_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/baseline_results.csv", index=False)


# ── Plot: ROC-AUC Comparison Bar Chart ───────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
models   = results_df["Model"].tolist()
auc_vals = results_df["ROC-AUC"].tolist()
colors   = ["#E94560", "#2980B9", "#F39C12"][:len(models)]

bars = ax.barh(models, auc_vals, color=colors, edgecolor="white", height=0.5)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("Mean ROC-AUC Score", fontsize=12)
ax.set_title("Baseline Model Comparison — ROC-AUC", fontsize=14, fontweight="bold")
ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
for bar, val in zip(bars, auc_vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/baseline_comparison.png")


# ── Plot: Per-Label ROC-AUC for best model ───────────────────
best_proba = lr_proba  # change if SVC is better
per_label_auc = []
for i, label in enumerate(LABELS):
    try:
        auc = roc_auc_score(y_test[:, i], best_proba[:, i])
    except Exception:
        auc = 0.5
    per_label_auc.append(auc)

fig, ax = plt.subplots(figsize=(9, 5))
colors_labels = ["#E94560", "#C0392B", "#F39C12", "#8E44AD", "#2980B9", "#16A085"]
bars = ax.bar(LABELS, per_label_auc, color=colors_labels, edgecolor="white")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("Per-Label ROC-AUC — Logistic Regression Baseline", fontsize=13, fontweight="bold")
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars, per_label_auc):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_label_auc_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/per_label_auc_baseline.png")

print("\n✅ Baseline training complete!")
print("   Next step: Run 04_bert.py for the advanced model")
