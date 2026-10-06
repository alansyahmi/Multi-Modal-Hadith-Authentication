"""
Extract frozen AraBERT embeddings for the clean narrator cohort.

Zero Target Leakage:
  - Input text contains ONLY statements from earlier classical critics (2nd–4th century AH).
  - Uses the base pre-trained transformer `aubmindlab/bert-base-arabertv2` (uncontaminated by downstream labels).
  - Extracts both CLS token embedding and attention-weighted mean-pooled embedding.

Outputs:
  - `output/cleaned_cohort/clean_arabert_embeddings.npy` (335, 768) [Mean-pooled]
  - `output/cleaned_cohort/clean_arabert_cls_embeddings.npy` (335, 768) [CLS token]
"""

import os
import sys
import json
import types
from pathlib import Path
import numpy as np
import torch

# Bypass transformers version check
if "transformers.dependency_versions_check" not in sys.modules:
    _dvc = types.ModuleType("transformers.dependency_versions_check")
    _dvc.dep_version_check = lambda *a, **k: None
    sys.modules["transformers.dependency_versions_check"] = _dvc

from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
BATCH_SIZE = 16
MAX_LEN = 512

def main():
    print(f"=== Extracting Clean AraBERT Embeddings ===")
    print(f"Device: {DEVICE}")

    texts_path = ROOT / "output/cleaned_cohort/clean_texts.json"
    with open(texts_path, "r", encoding="utf-8") as f:
        texts_dict = json.load(f)

    n_identities = len(texts_dict)
    print(f"Loaded {n_identities} earlier-critic text entries.")

    # Map identity_id -> earlier_critic_text
    id_to_text = {}
    for entry in texts_dict.values():
        id_to_text[int(entry["identity_id"])] = entry["earlier_critic_text"]

    ordered_texts = []
    for i in range(n_identities):
        text = id_to_text[i].strip()
        if not text:
            text = "لا توجد ترجمة"
        ordered_texts.append(text)

    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    all_cls_embeddings = []
    all_mean_embeddings = []

    n_batches = (n_identities + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Extracting embeddings across {n_batches} batches (batch size {BATCH_SIZE}, max_length {MAX_LEN})...")

    with torch.no_grad():
        for b in range(n_batches):
            batch_texts = ordered_texts[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # (B, L, 768)
            attention_mask = inputs["attention_mask"]      # (B, L)

            # 1. CLS token embedding
            cls_emb = last_hidden_state[:, 0, :].cpu().numpy()
            all_cls_embeddings.append(cls_emb)

            # 2. Attention-weighted mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_emb = (sum_embeddings / sum_mask).cpu().numpy()
            all_mean_embeddings.append(mean_emb)

            if (b + 1) % 5 == 0 or (b + 1) == n_batches:
                print(f"  Batch {b + 1}/{n_batches} processed.")

    cls_arr = np.concatenate(all_cls_embeddings, axis=0).astype(np.float32)
    mean_arr = np.concatenate(all_mean_embeddings, axis=0).astype(np.float32)

    print(f"CLS embeddings shape: {cls_arr.shape}")
    print(f"Mean embeddings shape: {mean_arr.shape}")

    out_dir = ROOT / "output/cleaned_cohort"
    np.save(out_dir / "clean_arabert_cls_embeddings.npy", cls_arr)
    np.save(out_dir / "clean_arabert_mean_embeddings.npy", mean_arr)
    # Default clean_arabert_embeddings.npy to mean-pooled
    np.save(out_dir / "clean_arabert_embeddings.npy", mean_arr)

    print(f"Successfully saved clean embeddings to {out_dir}")

if __name__ == "__main__":
    main()
