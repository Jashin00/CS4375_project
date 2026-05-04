# ============================================================
# Step 4: Advanced Model — Fine-tuned BERT
# Toxic Comment Classifier — Project Group
# ============================================================
# pip install torch transformers datasets scikit-learn pandas
#
# GPU strongly recommended. On Google Colab:
#   Runtime → Change runtime type → GPU (T4)
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss

# ── Config ──────────────────────────────────────────────────
DATA_DIR   = "outputs/preprocessed"
MODEL_DIR  = "models/bert"
OUTPUT_DIR = "outputs/bert"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS       = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_LABELS   = len(LABELS)
MODEL_NAME   = "bert-base-uncased"   # swap to "distilbert-base-uncased" if limited GPU
MAX_LEN      = 128                   # token length (reduce to 128 to save memory)
BATCH_SIZE   = 32                    # reduce to 8 if OOM error
EPOCHS       = 2
LR           = 2e-5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == "cpu":
    print("⚠  Warning: Training on CPU is very slow. Use Google Colab GPU for best results.")


# ── Dataset Class ────────────────────────────────────────────
class ToxicDataset(Dataset):
    """
    PyTorch Dataset for the Jigsaw Toxic Comment data.
    Tokenizes text with BERT tokenizer and returns tensors.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts     = df["clean_text"].fillna("").tolist()
        self.labels    = df[LABELS].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Load Data & Tokenizer ────────────────────────────────────
print("\nLoading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv").fillna("")
val_df   = pd.read_csv(f"{DATA_DIR}/val.csv").fillna("")
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv").fillna("")

print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

train_dataset = ToxicDataset(train_df, tokenizer, MAX_LEN)
val_dataset   = ToxicDataset(val_df,   tokenizer, MAX_LEN)
test_dataset  = ToxicDataset(test_df,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ── Model ────────────────────────────────────────────────────
print(f"\nLoading BERT model: {MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)
model = model.to(DEVICE)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps,
)

# Loss: BCEWithLogitsLoss — binary cross-entropy per label
# pos_weight boosts recall on rare classes (threat, identity_hate)
pos_weights = torch.tensor(
    [10.0, 100.0, 20.0, 300.0, 20.0, 100.0], dtype=torch.float
).to(DEVICE)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)


# ── Training Loop ────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Validation Loop ──────────────────────────────────────────
def evaluate_model(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    # Sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_logits))
    # Threshold at 0.5 for binary predictions
    all_preds = (all_probs >= 0.5).astype(int)

    # Compute metrics
    roc_aucs = []
    for i in range(NUM_LABELS):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except Exception:
            auc = 0.5
        roc_aucs.append(auc)

    return {
        "mean_roc_auc": np.mean(roc_aucs),
        "per_label_auc": dict(zip(LABELS, roc_aucs)),
        "macro_f1":  f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "micro_f1":  f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "hamming":   hamming_loss(all_labels, all_preds),
        "probs":     all_probs,
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ── Run Training ─────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f" Starting BERT Fine-Tuning")
print(f" Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
print(f"{'═'*55}\n")

history = {"train_loss": [], "val_auc": [], "val_f1": []}
best_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"── Epoch {epoch}/{EPOCHS} ──")

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
    val_metrics = evaluate_model(model, val_loader)

    history["train_loss"].append(train_loss)
    history["val_auc"].append(val_metrics["mean_roc_auc"])
    history["val_f1"].append(val_metrics["macro_f1"])

    print(f"  Train Loss     : {train_loss:.4f}")
    print(f"  Val ROC-AUC    : {val_metrics['mean_roc_auc']:.4f}")
    print(f"  Val Macro F1   : {val_metrics['macro_f1']:.4f}")
    print(f"  Val Hamming    : {val_metrics['hamming']:.4f}")
    print(f"  Per-label AUC  :")
    for label, auc in val_metrics["per_label_auc"].items():
        print(f"    {label:<15} {auc:.4f}")

    # Save best model
    if val_metrics["mean_roc_auc"] > best_auc:
        best_auc = val_metrics["mean_roc_auc"]
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"  ✅ New best model saved (AUC={best_auc:.4f})")
    print()


# ── Final Test Evaluation ────────────────────────────────────
print("\n══ FINAL TEST SET EVALUATION ══")
# Load best saved model
best_model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
best_model = best_model.to(DEVICE)

test_metrics = evaluate_model(best_model, test_loader)
print(f"  Test ROC-AUC  : {test_metrics['mean_roc_auc']:.4f}")
print(f"  Test Macro F1 : {test_metrics['macro_f1']:.4f}")
print(f"  Test Micro F1 : {test_metrics['micro_f1']:.4f}")
print(f"  Test Hamming  : {test_metrics['hamming']:.4f}")
print("\n  Per-label ROC-AUC:")
for label, auc in test_metrics["per_label_auc"].items():
    bar = "█" * int(auc * 20)
    print(f"    {label:<15} {auc:.4f}  {bar}")

# Save results
pd.DataFrame({
    "label": LABELS,
    "roc_auc": list(test_metrics["per_label_auc"].values())
}).to_csv(f"{OUTPUT_DIR}/bert_per_label_auc.csv", index=False)

pd.DataFrame(history).to_csv(f"{OUTPUT_DIR}/training_history.csv", index=False)


# ── Plot Training History ────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
epochs_x = range(1, EPOCHS + 1)

axes[0].plot(epochs_x, history["train_loss"], "o-", color="#E94560", linewidth=2, label="Train Loss")
axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
axes[0].legend()

axes[1].plot(epochs_x, history["val_auc"], "o-", color="#2ECC71", linewidth=2, label="Val ROC-AUC")
axes[1].plot(epochs_x, history["val_f1"],  "s-", color="#F39C12", linewidth=2, label="Val Macro F1")
axes[1].set_title("Validation Metrics", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/bert_training_history.png")

print("\n✅ BERT training complete!")
print("   Next step: Run 05_evaluation.py for full analysis")
