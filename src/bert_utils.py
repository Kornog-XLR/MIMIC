"""Shared utilities for BERT training, finalization, and inference."""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.charlson_mapping import CATEGORIES

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def select_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Metrics callback for Trainer
# ---------------------------------------------------------------------------
def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    auprcs = []
    for i in range(len(CATEGORIES)):
        if labels[:, i].sum() > 0:
            auprcs.append(average_precision_score(labels[:, i], probs[:, i]))
    return {"macro_auprc": float(np.mean(auprcs)) if auprcs else 0.0}


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------
def optimize_thresholds(probs: np.ndarray, labels: np.ndarray, categories: list[str]) -> dict[str, float]:
    thresholds = {}
    for i, cat in enumerate(categories):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.05, 0.96, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[cat] = round(best_t, 2)
    return thresholds


# ---------------------------------------------------------------------------
# Full evaluation table
# ---------------------------------------------------------------------------
def full_evaluation(probs: np.ndarray, labels: np.ndarray, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    all_true, all_pred = [], []

    for i, cat in enumerate(CATEGORIES):
        t = thresholds[cat]
        y_true = labels[:, i]
        y_pred = (probs[:, i] >= t).astype(int)
        all_true.extend(y_true)
        all_pred.extend(y_pred)

        n_pos = int(y_true.sum())
        auroc = roc_auc_score(y_true, probs[:, i]) if 0 < n_pos < len(y_true) else 0.0
        auprc = average_precision_score(y_true, probs[:, i]) if n_pos > 0 else 0.0
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)

        rows.append({
            "category": cat, "threshold": t, "n_true": n_pos,
            "auroc": round(auroc, 4), "auprc": round(auprc, 4),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        })

    micro_p = precision_score(all_true, all_pred, zero_division=0)
    micro_r = recall_score(all_true, all_pred, zero_division=0)
    micro_f1 = f1_score(all_true, all_pred, zero_division=0)
    cat_rows = [r for r in rows if r["category"] in CATEGORIES]
    macro_f1 = float(np.mean([r["f1"] for r in cat_rows]))

    rows.append({"category": "micro_avg", "threshold": 0, "n_true": 0,
                 "auroc": 0, "auprc": 0,
                 "precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)})
    rows.append({"category": "macro_avg", "threshold": 0, "n_true": 0,
                 "auroc": round(float(np.mean([r["auroc"] for r in cat_rows])), 4),
                 "auprc": round(float(np.mean([r["auprc"] for r in cat_rows])), 4),
                 "precision": 0, "recall": 0, "f1": round(macro_f1, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Print metrics table
# ---------------------------------------------------------------------------
def print_metrics(metrics: pd.DataFrame):
    print(f"\n{'Catégorie':<15} {'Seuil':>6} {'AUROC':>7} {'AUPRC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 64)
    for _, r in metrics.iterrows():
        print(
            f"{r['category']:<15} {r['threshold']:>6.2f} {r['auroc']:>7.3f} {r['auprc']:>7.3f} "
            f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(pmh_path, labels_path, splits_path, split_name, n_samples=None, seed=42):
    pmh = pd.read_parquet(pmh_path, columns=["hadm_id", "pmh_text"])
    labels = pd.read_parquet(labels_path, columns=["hadm_id"] + CATEGORIES)
    splits = pd.read_parquet(splits_path)

    hadm_ids = set(splits.loc[splits["split"] == split_name, "hadm_id"])
    data = pmh.merge(labels, on="hadm_id")
    data = data[data["hadm_id"].isin(hadm_ids)].dropna(subset=["pmh_text"])

    if n_samples and n_samples < len(data):
        data = data.sample(n=n_samples, random_state=seed)

    return data


# ---------------------------------------------------------------------------
# Batch inference (probabilities)
# ---------------------------------------------------------------------------
def predict_probs(model, tokenizer, texts: list[str], max_length: int,
                  batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    model.to(device)
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, max_length=max_length, padding=True,
                truncation=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Save final model artifacts
# ---------------------------------------------------------------------------
def save_final(model, tokenizer, thresholds: dict, final_dir: str,
               checkpoint_path: str, epoch: int, step: int,
               base_model: str, max_length: int):
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    final_config = {
        "checkpoint": checkpoint_path,
        "epoch": epoch,
        "step": step,
        "base_model": base_model,
        "max_length": max_length,
        "finalized_at": datetime.now().isoformat(),
    }
    with open(os.path.join(final_dir, "final_config.json"), "w") as f:
        json.dump(final_config, f, indent=2)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def list_checkpoints(run_dir: str) -> list[dict]:
    """Return sorted list of checkpoint info dicts (step, epoch, path)."""
    ckpts = []
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        m = re.match(r"checkpoint-(\d+)", name)
        if m and os.path.isdir(path):
            step = int(m.group(1))
            epoch = None
            state_path = os.path.join(path, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                epoch = state.get("epoch")
            ckpts.append({"step": step, "epoch": epoch, "path": path, "name": name})
    ckpts.sort(key=lambda c: c["step"])
    return ckpts


def resolve_checkpoint(run_dir: str, choice: str) -> dict:
    """Resolve which checkpoint to use based on choice string.

    choice: 'best', 'last', or a checkpoint directory name like 'checkpoint-1262'.
    """
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* found in {run_dir}")

    if choice == "last":
        return ckpts[-1]

    if choice == "best":
        # Try to read best_model_checkpoint from the latest trainer_state.json
        latest = ckpts[-1]
        state_path = os.path.join(latest["path"], "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            best_path = state.get("best_model_checkpoint")
            if best_path:
                for c in ckpts:
                    if os.path.abspath(c["path"]) == os.path.abspath(best_path):
                        return c
        # Fallback: return last
        return ckpts[-1]

    # Explicit name
    for c in ckpts:
        if c["name"] == choice:
            return c
    raise FileNotFoundError(f"Checkpoint '{choice}' not found in {run_dir}")


def load_run_config(run_dir: str, args_base_model: str | None = None,
                    args_max_length: int | None = None) -> tuple[str, int]:
    """Load base_model and max_length from run_config.json or CLI args."""
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        base_model = cfg.get("model", args_base_model or "emilyalsentzer/Bio_ClinicalBERT")
        max_length = cfg.get("max_length", args_max_length or 256)
    else:
        base_model = args_base_model or "emilyalsentzer/Bio_ClinicalBERT"
        max_length = args_max_length or 256
    return base_model, max_length


def load_checkpoint_model_tokenizer(ckpt_path: str, base_model: str, num_labels: int):
    """Load model from checkpoint, tokenizer from checkpoint or base model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, num_labels=num_labels, problem_type="multi_label_classification",
    )
    # Try tokenizer from checkpoint, fallback to base
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer
