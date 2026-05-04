# ============================================================
# Step 2: Data Preprocessing
# Toxic Comment Classifier — Project Group
# ============================================================
# pip install pandas numpy scikit-learn nltk
# ============================================================

import pandas as pd
import numpy as np
import re
import os
import pickle

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ── Config ──────────────────────────────────────────────────
DATA_PATH  = "data/train.csv"
OUTPUT_DIR = "outputs/preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
STOP_WORDS = set(stopwords.words("english"))


# ── Text Cleaning Function ───────────────────────────────────
def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline for toxic comment detection.
    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove IP addresses
      4. Remove usernames (Wikipedia-style: [[User:name]])
      5. Remove special characters & punctuation
      6. Collapse extra whitespace
    Note: Stopwords are NOT removed — they matter for BERT context.
           Remove them only for TF-IDF baseline if needed.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove Wikipedia username patterns like [[User:SomeUser]]
    text = re.sub(r"\[\[user:[^\]]+\]\]", " ", text)

    # Remove IP addresses
    text = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}", " ", text)

    # Remove special characters — keep letters, digits, basic punctuation
    text = re.sub(r"[^a-z0-9\s\!\?\.\,\'\-]", " ", text)

    # Collapse repeated characters (e.g. "loooser" → "looser")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_no_stopwords(text: str) -> str:
    """Variant that also removes stopwords — used for TF-IDF baselines."""
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


# ── Load & Process ───────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")

print("\nCleaning text (standard)...")
df["clean_text"] = df["comment_text"].apply(clean_text)

print("Cleaning text (no stopwords for TF-IDF)...")
df["clean_text_nosw"] = df["comment_text"].apply(clean_text_no_stopwords)

# Add helper columns
df["comment_length"]    = df["comment_text"].str.len()
df["clean_length"]      = df["clean_text"].str.len()
df["label_count"]       = df[LABELS].sum(axis=1)
df["is_any_toxic"]      = (df["label_count"] > 0).astype(int)
df["word_count"]        = df["clean_text"].str.split().str.len()

# ── Train / Validation / Test Split ─────────────────────────
from sklearn.model_selection import train_test_split

print("\nSplitting into train / val / test (70 / 15 / 15)...")
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["is_any_toxic"])
val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["is_any_toxic"])

print(f"  Train : {len(train_df):,} rows")
print(f"  Val   : {len(val_df):,} rows")
print(f"  Test  : {len(test_df):,} rows")

# ── Save Splits ─────────────────────────────────────────────
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv",   index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
print(f"\nSaved splits to {OUTPUT_DIR}/")

# ── Quick Sanity Check ───────────────────────────────────────
print("\n── Sample cleaned text ──")
for i in range(3):
    row = df.iloc[i]
    print(f"\n[{i}] Original : {row['comment_text'][:120]}...")
    print(f"     Cleaned  : {row['clean_text'][:120]}...")

print("\n── Label stats in train set ──")
print(train_df[LABELS].sum())

print("\n✅ Preprocessing complete!")
print("   Next step: Run 03_baseline.py")
