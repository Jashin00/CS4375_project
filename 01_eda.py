# ============================================================
# Step 1: Exploratory Data Analysis (EDA)
# Toxic Comment Classifier — Project Group
# ============================================================
# SETUP:
#   pip install pandas numpy matplotlib seaborn wordcloud kaggle
#
# DATASET:
#   Download from Kaggle: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
#   Place train.csv and test.csv inside a folder called: data/
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# ── Config ──────────────────────────────────────────────────
DATA_PATH  = "data/train.csv"
OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COLORS = ["#E94560", "#C0392B", "#F39C12", "#8E44AD", "#2980B9", "#16A085"]

# ── Load Data ───────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}\n")
print(df.head(3))

# ── Basic Stats ─────────────────────────────────────────────
print("\n── Label Distribution ──")
label_counts = df[LABELS].sum().sort_values(ascending=False)
label_pct    = (df[LABELS].mean() * 100).round(2)
print(pd.DataFrame({"Count": label_counts, "Percentage (%)": label_pct}))

total_clean = (df[LABELS].sum(axis=1) == 0).sum()
print(f"\nClean (no label) comments : {total_clean} ({total_clean/len(df)*100:.1f}%)")
print(f"At least 1 label          : {len(df) - total_clean} ({(len(df)-total_clean)/len(df)*100:.1f}%)")

# ── Plot 1: Label Distribution Bar Chart ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(LABELS, label_counts.values, color=COLORS, edgecolor="white", linewidth=0.8)
ax.set_title("Label Distribution — Toxic Comment Dataset", fontsize=15, fontweight="bold", pad=15)
ax.set_ylabel("Number of Comments", fontsize=12)
ax.set_xlabel("Toxicity Label", fontsize=12)
ax.set_facecolor("#1A1A2E")
fig.patch.set_facecolor("#1A1A2E")
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.title.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("#0F3460")
for bar, val in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="white")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 01_label_distribution.png")

# ── Plot 2: Label Co-occurrence Heatmap ─────────────────────
co_matrix = df[LABELS].T.dot(df[LABELS])
mask = np.zeros_like(co_matrix, dtype=bool)
np.fill_diagonal(mask, True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(co_matrix, annot=True, fmt="d", cmap="Reds",
            mask=mask, ax=ax, linewidths=0.5,
            cbar_kws={"label": "Co-occurrence Count"})
ax.set_title("Label Co-occurrence Heatmap\n(How often labels appear together)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_label_cooccurrence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_label_cooccurrence.png")

# ── Plot 3: Comment Length Distribution ─────────────────────
df["comment_length"] = df["comment_text"].str.len()
df["is_toxic"] = (df[LABELS].sum(axis=1) > 0).astype(int)

fig, ax = plt.subplots(figsize=(10, 5))
for label, color in [("Clean", "#2980B9"), ("Toxic", "#E94560")]:
    mask_val = 0 if label == "Clean" else 1
    subset = df[df["is_toxic"] == mask_val]["comment_length"]
    ax.hist(subset.clip(upper=2000), bins=60, alpha=0.6,
            label=f"{label} (n={len(subset):,})", color=color)
ax.set_title("Comment Length: Toxic vs. Clean", fontsize=14, fontweight="bold")
ax.set_xlabel("Comment Length (characters, capped at 2000)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_comment_length.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_comment_length.png")

# ── Plot 4: Word Clouds per Label ────────────────────────────
print("\nGenerating word clouds (this may take a moment)...")
for label, color in zip(LABELS, COLORS):
    texts = df[df[label] == 1]["comment_text"].dropna().str.cat(sep=" ")
    if not texts.strip():
        continue
    wc = WordCloud(width=800, height=400, background_color="black",
                   colormap="Reds", max_words=100).generate(texts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Most Frequent Words — '{label.upper()}' Comments",
                 fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor("#1A1A2E")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_wordcloud_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_wordcloud_{label}.png")

# ── Plot 5: Multi-label count per comment ────────────────────
df["label_count"] = df[LABELS].sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
counts = df["label_count"].value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values,
       color=["#2ECC71", "#E94560", "#F39C12", "#8E44AD", "#2980B9", "#16A085", "#C0392B"],
       edgecolor="white")
ax.set_title("Number of Labels per Comment", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Labels Assigned", fontsize=11)
ax.set_ylabel("Number of Comments", fontsize=11)
for i, v in enumerate(counts.values):
    ax.text(i, v + 20, f"{v:,}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_labels_per_comment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_labels_per_comment.png")

print("\n✅ EDA complete! All plots saved to outputs/eda/")
print("   Next step: Run 02_preprocessing.py")
