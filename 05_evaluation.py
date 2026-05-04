# ============================================================
# Step 5: Full Evaluation & Model Comparison
# Toxic Comment Classifier — Project Group
# ============================================================
# Run this AFTER both 03_baseline.py and 04_bert.py
# pip install matplotlib seaborn scikit-learn pandas
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, f1_score, hamming_loss,
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score
)

# ── Config ──────────────────────────────────────────────────
DATA_DIR   = "outputs/preprocessed"
BASE_DIR   = "outputs/baseline"
BERT_DIR   = "outputs/bert"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COLORS = ["#E94560", "#C0392B", "#F39C12", "#8E44AD", "#2980B9", "#16A085"]


# ── Load Test Data ───────────────────────────────────────────
print("Loading test data...")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv").fillna("")
y_test  = test_df[LABELS].values


# ── Load Baseline Predictions ────────────────────────────────
print("Loading baseline models and generating predictions...")

with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(f"{MODEL_DIR}/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open(f"{MODEL_DIR}/svc_model.pkl", "rb") as f:
    svc_model = pickle.load(f)

X_test_tfidf = tfidf.transform(test_df["clean_text_nosw"].values)

lr_proba  = lr_model.predict_proba(X_test_tfidf)
lr_pred   = lr_model.predict(X_test_tfidf)

svc_proba = svc_model.predict_proba(X_test_tfidf)
svc_pred  = svc_model.predict(X_test_tfidf)


# ── Load BERT Predictions ────────────────────────────────────
bert_per_label = pd.read_csv(f"{BERT_DIR}/bert_per_label_auc.csv")
print("BERT per-label AUC loaded from saved results.")

# If you want to regenerate BERT predictions, import 04_bert.py logic.
# Here we use the saved CSV for comparison plotting.


# ── Helper: Full Metrics ─────────────────────────────────────
def full_metrics(name, y_true, y_pred, y_proba):
    aucs = []
    for i in range(len(LABELS)):
        try:
            aucs.append(roc_auc_score(y_true[:, i], y_proba[:, i]))
        except Exception:
            aucs.append(0.5)
    return {
        "Model":        name,
        "ROC-AUC":      round(np.mean(aucs), 4),
        "Macro F1":     round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4),
        "Micro F1":     round(f1_score(y_true, y_pred, average="micro",  zero_division=0), 4),
        "Precision":    round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall":       round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Hamming Loss": round(hamming_loss(y_true, y_pred), 4),
        "per_label_auc": aucs,
    }

lr_m  = full_metrics("Logistic Regression", y_test, lr_pred,  lr_proba)
svc_m = full_metrics("LinearSVC",           y_test, svc_pred, svc_proba)

# Add BERT row from saved results
bert_auc_mean = bert_per_label["roc_auc"].mean()
bert_m = {
    "Model": "BERT (fine-tuned)",
    "ROC-AUC": round(bert_auc_mean, 4),
    "Macro F1": "see bert output",
    "Micro F1": "see bert output",
    "Precision": "—",
    "Recall": "—",
    "Hamming Loss": "—",
    "per_label_auc": bert_per_label["roc_auc"].tolist(),
}


# ── Table: Model Comparison ──────────────────────────────────
print("\n\n══ FULL MODEL COMPARISON ══")
compare_cols = ["Model", "ROC-AUC", "Macro F1", "Micro F1", "Precision", "Recall", "Hamming Loss"]
compare_df = pd.DataFrame([lr_m, svc_m, bert_m])[compare_cols]
print(compare_df.to_string(index=False))
compare_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/model_comparison.csv")


# ── Plot 1: Model Comparison Bar Chart ──────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
model_names = ["Logistic Regression", "LinearSVC", "BERT (fine-tuned)"]
auc_vals    = [lr_m["ROC-AUC"], svc_m["ROC-AUC"], bert_auc_mean]
bar_colors  = ["#2980B9", "#F39C12", "#E94560"]

bars = ax.bar(model_names, auc_vals, color=bar_colors, edgecolor="white", width=0.5)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Mean ROC-AUC", fontsize=12)
ax.set_title("Model Comparison — Mean ROC-AUC Across All 6 Labels", fontsize=13, fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")
for bar, val in zip(bars, auc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_model_comparison.png")


# ── Plot 2: Per-Label AUC Grouped Bar Chart ──────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(LABELS))
width = 0.25

ax.bar(x - width, lr_m["per_label_auc"],  width, label="Logistic Regression", color="#2980B9", edgecolor="white")
ax.bar(x,          svc_m["per_label_auc"], width, label="LinearSVC",           color="#F39C12", edgecolor="white")
ax.bar(x + width,  bert_m["per_label_auc"],width, label="BERT (fine-tuned)",   color="#E94560", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=20, ha="right")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("Per-Label ROC-AUC: All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_per_label_auc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_per_label_auc.png")


# ── Plot 3: Confusion Matrix per Label (Logistic Regression) ─
print("\nGenerating confusion matrices...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (label, color) in enumerate(zip(LABELS, COLORS)):
    cm = confusion_matrix(y_test[:, i], lr_pred[:, i])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"],
                cbar=False, linewidths=0.5)
    axes[i].set_title(f"{label.upper()}", fontsize=12, fontweight="bold", color=color)

fig.suptitle("Confusion Matrices — Logistic Regression (per label)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_confusion_matrices_lr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_confusion_matrices_lr.png")


# ── Plot 4: ROC Curves (Logistic Regression) ─────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for i, (label, color) in enumerate(zip(LABELS, COLORS)):
    fpr, tpr, _ = roc_curve(y_test[:, i], lr_proba[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{label} (AUC = {roc_auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Logistic Regression (per label)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_roc_curves_lr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_roc_curves_lr.png")


# ── Plot 5: Threshold Analysis for 'toxic' label ─────────────
print("\nThreshold analysis for 'toxic' label...")
thresholds = np.arange(0.1, 0.95, 0.05)
precisions, recalls, f1_scores = [], [], []

toxic_proba = lr_proba[:, 0]
toxic_true  = y_test[:, 0]

for thresh in thresholds:
    preds = (toxic_proba >= thresh).astype(int)
    precisions.append(precision_score(toxic_true, preds, zero_division=0))
    recalls.append(recall_score(toxic_true, preds, zero_division=0))
    f1_scores.append(f1_score(toxic_true, preds, zero_division=0))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precisions, "o-", color="#2980B9", label="Precision", linewidth=2)
ax.plot(thresholds, recalls,    "s-", color="#E94560", label="Recall",    linewidth=2)
ax.plot(thresholds, f1_scores,  "^-", color="#2ECC71", label="F1-Score",  linewidth=2)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Default threshold (0.5)")
ax.set_xlabel("Decision Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Threshold Analysis — 'Toxic' Label (Logistic Regression)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_threshold_analysis.png")


# ── Error Analysis: Show misclassified examples ──────────────
print("\n── Sample Misclassified Comments (Logistic Regression / Toxic label) ──")
toxic_pred_arr = lr_pred[:, 0]
toxic_true_arr = y_test[:, 0]

# False Negatives: toxic but predicted clean
fn_mask = (toxic_true_arr == 1) & (toxic_pred_arr == 0)
fn_df   = test_df[fn_mask][["comment_text"]].head(3)
print("\n[False Negatives — Toxic comment predicted as Clean]")
for _, row in fn_df.iterrows():
    print(f"  → {row['comment_text'][:150]}...\n")

# False Positives: clean but predicted toxic
fp_mask = (toxic_true_arr == 0) & (toxic_pred_arr == 1)
fp_df   = test_df[fp_mask][["comment_text"]].head(3)
print("\n[False Positives — Clean comment predicted as Toxic]")
for _, row in fp_df.iterrows():
    print(f"  → {row['comment_text'][:150]}...\n")

print("\n✅ Full evaluation complete! All plots saved to outputs/evaluation/")
print("   Next step: Run 06_interpretability.py for SHAP analysis")
