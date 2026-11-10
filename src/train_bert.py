"""Fine-tune a clinical BERT for multi-label Charlson classification."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bert_utils import (
    CATEGORIES,
    compute_metrics_fn,
    full_evaluation,
    load_split,
    optimize_thresholds,
    predict_probs,
    print_metrics,
    save_final,
    select_device,
)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharlsonDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Custom Trainer with optional pos_weight
# ---------------------------------------------------------------------------
class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight_tensor=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight_tensor = pos_weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        if self.pos_weight_tensor is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor.to(logits.device))
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Charlson classification.")
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--labels", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--splits-file", default="data/processed/splits.parquet")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", action="store_true", help="Weight BCE by inverse prevalence")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Trade compute for memory")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. Use --resume for auto-detect or --resume <path>")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_train = 200
        args.epochs = 1
        n_val = 100
    else:
        n_val = None

    run_name = args.run_name or f"bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"models/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Auto-detect resume checkpoint
    resume_path = None
    if args.resume == "auto":
        from src.bert_utils import list_checkpoints
        ckpts = list_checkpoints(output_dir)
        if ckpts:
            resume_path = ckpts[-1]["path"]
            print(f"Reprise depuis {resume_path}")
        else:
            print("Aucun checkpoint trouvé, entraînement depuis le début.")
    elif args.resume:
        resume_path = args.resume

    device = select_device()
    use_fp16 = device == "cuda"
    print(f"Device : {device} | fp16 : {use_fp16}")

    # --- Load data ---
    print("Chargement des données…")
    train_data = load_split(args.pmh, args.labels, args.splits_file, "train", args.n_train, args.seed)
    val_data = load_split(args.pmh, args.labels, args.splits_file, "val", n_val, args.seed)
    print(f"Train : {len(train_data)} | Val : {len(val_data)}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Truncation stats
    sample_texts = train_data["pmh_text"].head(min(2000, len(train_data))).tolist()
    sample_lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in sample_texts]
    n_trunc = sum(1 for l in sample_lengths if l > args.max_length)
    pct_trunc = 100.0 * n_trunc / len(sample_lengths)
    print(f"PMH tronquées (estimation) : {pct_trunc:.1f}%")

    # --- Datasets ---
    train_texts = train_data["pmh_text"].tolist()
    train_labels = train_data[CATEGORIES].values.astype(float)
    val_texts = val_data["pmh_text"].tolist()
    val_labels = val_data[CATEGORIES].values.astype(float)

    train_dataset = CharlsonDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = CharlsonDataset(val_texts, val_labels, tokenizer, args.max_length)

    # --- Pos weight ---
    pos_weight_tensor = None
    if args.pos_weight:
        pos_counts = train_labels.sum(axis=0)
        neg_counts = len(train_labels) - pos_counts
        weights = np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, 10.0)
        pos_weight_tensor = torch.tensor(weights, dtype=torch.float)
        print("Pos weights :", {cat: round(w, 2) for cat, w in zip(CATEGORIES, weights)})

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(CATEGORIES), problem_type="multi_label_classification",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Training ---
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_auprc",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        fp16=use_fp16,
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = WeightedBCETrainer(
        pos_weight_tensor=pos_weight_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_path)
    duration = time.time() - t0
    print(f"Entraînement terminé en {duration/60:.1f} min")

    # --- Finalize: val predictions, thresholds, metrics, save ---
    print("Prédiction sur val…")
    val_probs = predict_probs(trainer.model, tokenizer, val_texts, args.max_length,
                              args.batch_size * 2, device)

    thresholds = optimize_thresholds(val_probs, val_labels, CATEGORIES)
    print("Seuils optimisés :", thresholds)

    metrics = full_evaluation(val_probs, val_labels, thresholds)
    print_metrics(metrics)

    # Save probabilities
    import pandas as pd
    probs_df = pd.DataFrame(val_probs, columns=[f"{c}_prob" for c in CATEGORIES])
    probs_df["hadm_id"] = val_data["hadm_id"].values
    probs_path = f"data/processed/probs_bert_val_{run_name}.parquet"
    os.makedirs(os.path.dirname(probs_path), exist_ok=True)
    probs_df.to_parquet(probs_path, index=False)

    metrics_path = f"outputs/bert_metrics_val_{run_name}.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\nMétriques → {metrics_path}")

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    save_final(trainer.model, tokenizer, thresholds, final_dir,
               checkpoint_path=output_dir, epoch=args.epochs, step=0,
               base_model=args.model, max_length=args.max_length)

    # Also save run config at run root
    run_config = {
        "model": args.model,
        "run_name": run_name,
        "max_length": args.max_length,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "pos_weight": args.pos_weight,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "pct_truncated": round(pct_trunc, 2),
        "training_duration_s": round(duration, 1),
        "device": device,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nModèle finalisé dans {final_dir}")


if __name__ == "__main__":
    main()
