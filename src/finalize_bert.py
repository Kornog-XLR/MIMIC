"""Finalize a BERT run from its checkpoints: evaluate, optimize thresholds, save."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bert_utils import (
    CATEGORIES,
    full_evaluation,
    list_checkpoints,
    load_checkpoint_model_tokenizer,
    load_run_config,
    load_split,
    optimize_thresholds,
    predict_probs,
    print_metrics,
    save_final,
    select_device,
)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate_checkpoint(ckpt: dict, base_model: str, max_length: int,
                        val_texts: list[str], val_labels: np.ndarray,
                        device: str, batch_size: int):
    """Load a checkpoint, predict on val, compute metrics. Returns (probs, thresholds, metrics_df)."""
    model, tokenizer = load_checkpoint_model_tokenizer(
        ckpt["path"], base_model, num_labels=len(CATEGORIES),
    )
    probs = predict_probs(model, tokenizer, val_texts, max_length, batch_size, device)
    thresholds = optimize_thresholds(probs, val_labels, CATEGORIES)
    metrics = full_evaluation(probs, val_labels, thresholds)
    return probs, thresholds, metrics


def main():
    parser = argparse.ArgumentParser(description="Finalize a BERT run from checkpoints.")
    parser.add_argument("--run", required=True, help="Run name (subdirectory of models/)")
    parser.add_argument("--checkpoint", default="best",
                        help="best|last|<checkpoint dir name> (default: best)")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate all checkpoints and pick the best")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--labels", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--splits-file", default="data/processed/splits.parquet")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_dir = f"models/{args.run}"
    if not os.path.isdir(run_dir):
        print(f"Erreur : {run_dir} n'existe pas.")
        sys.exit(1)

    base_model, max_length = load_run_config(run_dir, args.base_model, args.max_length)
    print(f"Base model : {base_model} | max_length : {max_length}")

    # List checkpoints
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        print(f"Aucun checkpoint trouvé dans {run_dir}")
        sys.exit(1)

    print(f"\nCheckpoints trouvés ({len(ckpts)}) :")
    for c in ckpts:
        ep_str = f"epoch {c['epoch']:.1f}" if c["epoch"] is not None else "epoch ?"
        print(f"  {c['name']} — step {c['step']}, {ep_str}")

    device = select_device()
    print(f"\nDevice : {device}")

    # Load val data
    print("Chargement des données val…")
    val_data = load_split(args.pmh, args.labels, args.splits_file, "val")
    val_texts = val_data["pmh_text"].tolist()
    val_labels = val_data[CATEGORIES].values.astype(float)
    print(f"Val : {len(val_data)} notes")

    os.makedirs("outputs", exist_ok=True)

    if args.all_checkpoints:
        # Evaluate every checkpoint
        summary_rows = []
        best_auprc = -1
        best_ckpt = None
        best_probs = None
        best_thresholds = None
        best_metrics = None

        for ckpt in ckpts:
            ep_str = f"{ckpt['epoch']:.0f}" if ckpt['epoch'] is not None else "?"
            print(f"\n--- Évaluation {ckpt['name']} (epoch {ep_str}) ---")

            probs, thresholds, metrics = evaluate_checkpoint(
                ckpt, base_model, max_length, val_texts, val_labels, device, args.batch_size,
            )
            print_metrics(metrics)

            # Save per-epoch metrics
            ep_label = f"ep{ep_str}"
            metrics_path = f"outputs/bert_metrics_val_{args.run}_{ep_label}.csv"
            metrics.to_csv(metrics_path, index=False)

            # Extract summary
            macro_row = metrics[metrics["category"] == "macro_avg"].iloc[0]
            micro_row = metrics[metrics["category"] == "micro_avg"].iloc[0]
            macro_auprc = macro_row["auprc"]

            summary_rows.append({
                "checkpoint": ckpt["name"],
                "epoch": ckpt["epoch"],
                "step": ckpt["step"],
                "micro_f1": micro_row["f1"],
                "macro_f1": macro_row["f1"],
                "macro_auprc": macro_auprc,
                "macro_auroc": macro_row["auroc"],
            })

            if macro_auprc > best_auprc:
                best_auprc = macro_auprc
                best_ckpt = ckpt
                best_probs = probs
                best_thresholds = thresholds
                best_metrics = metrics

        # Save epoch comparison
        summary_df = pd.DataFrame(summary_rows)
        summary_path = f"outputs/bert_epochs_{args.run}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nComparaison des epochs → {summary_path}")
        print(summary_df.to_string(index=False))

        print(f"\nMeilleur checkpoint : {best_ckpt['name']} (macro AUPRC = {best_auprc:.4f})")
        chosen_ckpt = best_ckpt
        chosen_probs = best_probs
        chosen_thresholds = best_thresholds

    else:
        # Single checkpoint
        from src.bert_utils import resolve_checkpoint
        chosen_ckpt = resolve_checkpoint(run_dir, args.checkpoint)
        ep_str = f"{chosen_ckpt['epoch']:.0f}" if chosen_ckpt['epoch'] is not None else "?"
        print(f"\nCheckpoint sélectionné : {chosen_ckpt['name']} (epoch {ep_str})")

        chosen_probs, chosen_thresholds, metrics = evaluate_checkpoint(
            chosen_ckpt, base_model, max_length, val_texts, val_labels, device, args.batch_size,
        )
        print_metrics(metrics)

        ep_label = f"ep{ep_str}"
        metrics_path = f"outputs/bert_metrics_val_{args.run}_{ep_label}.csv"
        metrics.to_csv(metrics_path, index=False)
        print(f"\nMétriques → {metrics_path}")

    # Save val probabilities
    probs_df = pd.DataFrame(chosen_probs, columns=[f"{c}_prob" for c in CATEGORIES])
    probs_df["hadm_id"] = val_data["hadm_id"].values
    probs_path = f"data/processed/probs_bert_val_{args.run}.parquet"
    os.makedirs(os.path.dirname(probs_path), exist_ok=True)
    probs_df.to_parquet(probs_path, index=False)

    # Save final model
    model, tokenizer = load_checkpoint_model_tokenizer(
        chosen_ckpt["path"], base_model, num_labels=len(CATEGORIES),
    )
    final_dir = os.path.join(run_dir, "final")
    epoch_val = int(chosen_ckpt["epoch"]) if chosen_ckpt["epoch"] is not None else 0
    save_final(model, tokenizer, chosen_thresholds, final_dir,
               checkpoint_path=chosen_ckpt["path"], epoch=epoch_val,
               step=chosen_ckpt["step"], base_model=base_model, max_length=max_length)

    print(f"\nModèle finalisé dans {final_dir}")


if __name__ == "__main__":
    main()
