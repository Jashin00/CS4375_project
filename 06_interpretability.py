# ============================================================
# Step 6: Model Interpretability with SHAP
# Toxic Comment Classifier — Project Group
# ============================================================
# pip install shap pandas scikit-learn matplotlib
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────
DATA_DIR   = "outputs/preprocessed"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs/interpretability"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COLORS = ["#E94560", "#C0392B", "#F39C12", "#8E44AD", "#2980B9", "#16A085"]


# ── Load Data & Models ───────────────────────────────────────
print("Loading data and models...")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv").fillna("")

with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(f"{MODEL_DIR}/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

X_test_tfidf = tfidf.transform(test_df["clean_text_nosw"].values)
feature_names = tfidf.get_feature_names_out()

# Access individual estimators for SHAP (OneVsRest exposes .estimators_)
lr_estimators = lr_model.estimators_


# ── SHAP: Top Feature Importance per Label ───────────────────
print("\nComputing SHAP feature importance per label...")

# Use a background sample for SHAP linear explainer
background_sample = shap.sample(X_test_tfidf, 100, random_state=42)

for label_idx, (label, color) in enumerate(zip(LABELS, COLORS)):
    print(f"  Processing: {label}...")

    clf = lr_estimators[label_idx]

    # LinearExplainer works well with logistic regression + sparse input
    explainer = shap.LinearExplainer(clf, background_sample, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_tfidf[:500])  # Use first 500 for speed

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-20:][::-1]
    top_features = feature_names[top_idx]
    top_values   = mean_abs_shap[top_idx]

    # Plot top 20 features
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(20), top_values[::-1], color=color, edgecolor="white", alpha=0.85)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Top 20 Most Influential Words\nLabel: '{label.upper()}'",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: shap_{label}.png")


# ── Logistic Regression Coefficients (Alternative to SHAP) ───
print("\nGenerating LR coefficient importance plots...")

for label_idx, (label, color) in enumerate(zip(LABELS, COLORS)):
    clf = lr_estimators[label_idx]
    coefs = clf.coef_[0]

    # Top positive coefficients (most predictive of toxic)
    top_pos_idx = np.argsort(coefs)[-15:][::-1]
    top_neg_idx = np.argsort(coefs)[:15]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Positive coefficients → strong signal for the label
    axes[0].barh(range(15), coefs[top_pos_idx][::-1], color=color, edgecolor="white")
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels(feature_names[top_pos_idx][::-1], fontsize=9)
    axes[0].set_title(f"Words MOST associated with '{label}'", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("LR Coefficient", fontsize=10)

    # Negative coefficients → strong signal AGAINST the label (very non-toxic)
    axes[1].barh(range(15), coefs[top_neg_idx], color="#2980B9", edgecolor="white")
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels(feature_names[top_neg_idx], fontsize=9)
    axes[1].set_title(f"Words LEAST associated with '{label}'", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("LR Coefficient", fontsize=10)

    fig.suptitle(f"Logistic Regression Coefficients — Label: '{label.upper()}'",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/coef_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: coef_{label}.png")


# ── Explain Individual Predictions ───────────────────────────
print("\nExplaining individual predictions...")

sample_comments = [
    "You are absolutely wonderful, thanks for your help!",
    "I hate you, go kill yourself you stupid idiot",
    "This article seems biased against certain religious groups",
    "Great work on the edit, the article looks much better now",
]

print("\n── Per-comment prediction breakdown ──")
for comment in sample_comments:
    cleaned = comment.lower()
    vec = tfidf.transform([cleaned])

    print(f"\nComment: '{comment}'")
    print(f"{'Label':<18} {'Probability':>12} {'Prediction':>12}")
    print("─" * 45)
    for label_idx, label in enumerate(LABELS):
        clf = lr_estimators[label_idx]
        prob = clf.predict_proba(vec)[0][1]
        pred = "🚨 TOXIC" if prob >= 0.5 else "✅ Clean"
        print(f"  {label:<16} {prob:>10.4f} {pred:>12}")


# ── Summary SHAP Plot (all labels combined, 'toxic' only) ────
print("\nGenerating summary SHAP plot for 'toxic' label...")
clf_toxic    = lr_estimators[0]
explainer    = shap.LinearExplainer(clf_toxic, background_sample, feature_perturbation="interventional")
shap_values  = explainer.shap_values(X_test_tfidf[:200])

# Convert sparse to dense for SHAP summary plot
X_dense = X_test_tfidf[:200].toarray()

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_dense,
    feature_names=feature_names,
    max_display=20,
    plot_type="bar",
    show=False,
    color="#E94560",
)
plt.title("SHAP Summary — 'Toxic' Label (Top 20 Features)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary_toxic.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_summary_toxic.png")

print("\n✅ Interpretability analysis complete!")
print("   All outputs saved to outputs/interpretability/")
print("\n🎉 Project complete! You can now prepare your final presentation.")
