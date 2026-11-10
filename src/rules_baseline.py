"""Rule-based Charlson comorbidity prediction from PMH text using medspaCy."""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.charlson_mapping import CATEGORIES

# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

def load_lexicon(path: str = "config/charlson_lexicon.yaml") -> dict[str, list]:
    """Load lexicon YAML. Returns dict of category → list of terms/patterns."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Compile regex patterns from lexicon
# ---------------------------------------------------------------------------

def _compile_patterns(terms: list[str]) -> list[re.Pattern]:
    """Convert a list of terms (literals or regex:...) to compiled patterns."""
    patterns = []
    for term in terms:
        if term.startswith("regex:"):
            raw = term[len("regex:"):]
            patterns.append(re.compile(raw, re.IGNORECASE))
        else:
            # Literal: escape and add word boundaries
            escaped = re.escape(term)
            patterns.append(re.compile(r"\b" + escaped + r"\b", re.IGNORECASE))
    return patterns


def compile_lexicon(lexicon: dict) -> dict[str, list[re.Pattern]]:
    """Compile all lexicon entries to regex patterns."""
    return {cat: _compile_patterns(terms) for cat, terms in lexicon.items()}


# ---------------------------------------------------------------------------
# medspaCy pipeline
# ---------------------------------------------------------------------------

def build_nlp():
    """Build a medspaCy pipeline with ConText for negation detection."""
    import medspacy
    from medspacy.context import ConTextRule

    nlp = medspacy.load(enable=["medspacy_pyrush", "medspacy_context"])

    context = nlp.get_pipe("medspacy_context")

    # Add custom context rules
    custom_rules = [
        # Negation
        ConTextRule("no history of", "NEGATED_EXISTENCE", direction="FORWARD"),
        ConTextRule("no known", "NEGATED_EXISTENCE", direction="FORWARD"),
        ConTextRule("denies", "NEGATED_EXISTENCE", direction="FORWARD"),
        ConTextRule("negative for", "NEGATED_EXISTENCE", direction="FORWARD"),
        ConTextRule("without", "NEGATED_EXISTENCE", direction="FORWARD"),
        ConTextRule("not", "NEGATED_EXISTENCE", direction="FORWARD"),
        ConTextRule("never", "NEGATED_EXISTENCE", direction="FORWARD"),
        # Hypothetical / uncertain
        ConTextRule("r/o", "POSSIBLE_EXISTENCE", direction="FORWARD"),
        ConTextRule("rule out", "POSSIBLE_EXISTENCE", direction="FORWARD"),
        ConTextRule("possible", "POSSIBLE_EXISTENCE", direction="FORWARD"),
        ConTextRule("questionable", "POSSIBLE_EXISTENCE", direction="FORWARD"),
        ConTextRule("?", "POSSIBLE_EXISTENCE", direction="FORWARD"),
        # Family
        ConTextRule("family history of", "FAMILY", direction="FORWARD"),
        ConTextRule("family history", "FAMILY", direction="FORWARD"),
        ConTextRule("mother with", "FAMILY", direction="FORWARD"),
        ConTextRule("father with", "FAMILY", direction="FORWARD"),
        ConTextRule("brother with", "FAMILY", direction="FORWARD"),
        ConTextRule("sister with", "FAMILY", direction="FORWARD"),
        ConTextRule("maternal history of", "FAMILY", direction="FORWARD"),
        ConTextRule("paternal history of", "FAMILY", direction="FORWARD"),
        ConTextRule("FH", "FAMILY", direction="FORWARD"),
    ]
    context.add(custom_rules)

    return nlp


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------

def split_segments(text: str) -> list[str]:
    """Split PMH text into segments for local negation scope.

    If the text has newlines, split on newlines and semicolons.
    If single-line, split on commas and semicolons.
    """
    if not text or not text.strip():
        return []

    if "\n" in text:
        lines = text.split("\n")
        segments = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove list markers
            line = re.sub(r"^[-*•–—]\s*", "", line).strip()
            line = re.sub(r"^\d+[.)]\s*", "", line).strip()
            if not line:
                continue
            for part in line.split(";"):
                part = part.strip()
                if part:
                    segments.append(part)
        return segments if segments else [text.strip()]
    else:
        # Single-line: split on commas and semicolons
        segments = re.split(r"[;,]", text)
        return [s.strip() for s in segments if s.strip()] or [text.strip()]


# ---------------------------------------------------------------------------
# Negation / family / hypothetical detection via medspaCy
# ---------------------------------------------------------------------------

def _is_negated_context(nlp, segment: str) -> set[str]:
    """Return set of negation context types found in segment.

    Possible values: 'NEGATED', 'FAMILY', 'HYPOTHETICAL'.
    """
    doc = nlp(segment)
    contexts = set()
    for ent in doc.ents:
        if ent._.is_negated:
            contexts.add("NEGATED")
        if ent._.is_family:
            contexts.add("FAMILY")
        if getattr(ent._, "is_uncertain", False) or getattr(ent._, "is_hypothetical", False):
            contexts.add("HYPOTHETICAL")
    return contexts


def _check_segment_negation(nlp, segment: str) -> bool:
    """Check if the whole segment is under a negation/family/hypothetical scope.

    We process the segment with medspaCy and check if context modifiers
    apply. Returns True if the segment should be excluded.
    """
    doc = nlp(segment)

    # Check if there are negation/family/hypothetical modifiers
    for token in doc:
        # Check context modifiers from ConText
        pass

    # Simpler approach: check if the segment starts with a negation cue
    neg_cues = [
        r"^no\b", r"^not\b", r"^denies?\b", r"^negative\b", r"^without\b",
        r"^never\b", r"^no\s+history\b", r"^no\s+known\b", r"^none\b",
    ]
    hyp_cues = [r"^r/o\b", r"^rule\s+out\b", r"^possible\b", r"^\?\s*"]
    fam_cues = [
        r"^family\s+history\b", r"^FH\b", r"^mother\b", r"^father\b",
        r"^brother\b", r"^sister\b", r"^maternal\b", r"^paternal\b",
    ]

    seg_lower = segment.strip().lower()
    for cue in neg_cues:
        if re.match(cue, seg_lower):
            return True
    for cue in hyp_cues:
        if re.match(cue, seg_lower):
            return True
    for cue in fam_cues:
        if re.match(cue, seg_lower):
            return True

    return False


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict(
    text: str,
    compiled_lexicon: dict[str, list[re.Pattern]],
    nlp=None,
) -> tuple[dict[str, int], list[tuple[str, str, str]]]:
    """Predict 17 Charlson categories from PMH text.

    Args:
        text: PMH section text.
        compiled_lexicon: output of compile_lexicon().
        nlp: medspaCy pipeline (optional, for ConText-based negation).

    Returns:
        (flags dict {category: 0/1}, matched_terms list [(term, category, segment)])
    """
    flags = {cat: 0 for cat in CATEGORIES}
    matched_terms: list[tuple[str, str, str]] = []
    modifier_matches: dict[str, list[str]] = {
        "_diab_complications": [],
        "_liver_severe": [],
        "_metastatic": [],
    }

    segments = split_segments(text) if text else []

    for segment in segments:
        # Check if segment is negated/family/hypothetical
        is_excluded = _check_segment_negation(nlp, segment) if nlp else False
        if not is_excluded:
            # Simple regex-based negation fallback (when no nlp)
            if nlp is None:
                seg_lower = segment.strip().lower()
                if any(re.match(p, seg_lower) for p in [
                    r"^no\b", r"^not\b", r"^denies?\b", r"^negative\b",
                    r"^without\b", r"^never\b", r"^no\s+history\b",
                    r"^no\s+known\b", r"^none\b", r"^r/o\b",
                    r"^rule\s+out\b", r"^possible\b",
                    r"^family\s+history\b", r"^FH\b", r"^mother\b",
                    r"^father\b", r"^brother\b", r"^sister\b",
                ]):
                    is_excluded = True

        if is_excluded:
            continue

        # Match terms against segment
        for cat, patterns in compiled_lexicon.items():
            for pat in patterns:
                m = pat.search(segment)
                if m:
                    if cat.startswith("_"):
                        modifier_matches[cat].append(segment)
                    else:
                        flags[cat] = 1
                        matched_terms.append((m.group(), cat, segment))
                    break  # One match per category per segment is enough

    # --- Severity co-occurrence rules ---

    # Diabetes: diab + complication modifier → diabwc
    if flags.get("diab", 0) == 1 and modifier_matches["_diab_complications"]:
        flags["diabwc"] = 1

    # Liver: mld + severe modifier → msld
    if flags.get("mld", 0) == 1 and modifier_matches["_liver_severe"]:
        flags["msld"] = 1

    # Metastatic: if metastatic modifier found, set metacanc AND canc
    if modifier_matches["_metastatic"]:
        flags["metacanc"] = 1
        flags["canc"] = 1

    # Also: if metacanc directly matched, ensure canc is set
    if flags.get("metacanc", 0) == 1:
        flags["canc"] = 1

    return flags, matched_terms


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(preds: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Compute precision, recall, F1 per category + micro/macro."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    merged = preds.merge(labels, on="hadm_id", suffixes=("_pred", "_true"))

    rows = []
    all_y_true = []
    all_y_pred = []

    for cat in CATEGORIES:
        y_true = merged[f"{cat}_true"].values
        y_pred = merged[f"{cat}_pred"].values
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        n_pos = int(y_true.sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        rows.append({
            "category": cat, "n_true": n_pos,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        })

    # Micro F1
    all_y_true_arr = pd.array(all_y_true)
    all_y_pred_arr = pd.array(all_y_pred)
    micro_p = precision_score(all_y_true_arr, all_y_pred_arr, zero_division=0)
    micro_r = recall_score(all_y_true_arr, all_y_pred_arr, zero_division=0)
    micro_f1 = f1_score(all_y_true_arr, all_y_pred_arr, zero_division=0)
    rows.append({
        "category": "micro_avg", "n_true": sum(r["n_true"] for r in rows),
        "tp": 0, "fp": 0, "fn": 0,
        "precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4),
    })

    # Macro F1
    cat_rows = [r for r in rows if r["category"] not in ("micro_avg",)]
    macro_p = sum(r["precision"] for r in cat_rows) / len(cat_rows)
    macro_r = sum(r["recall"] for r in cat_rows) / len(cat_rows)
    macro_f1 = sum(r["f1"] for r in cat_rows) / len(cat_rows)
    rows.append({
        "category": "macro_avg", "n_true": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rule-based Charlson baseline.")
    parser.add_argument("--split", default="val", help="Split to evaluate on (default: val)")
    parser.add_argument("--lexicon", default="config/charlson_lexicon.yaml")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--labels", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--splits-file", default="data/processed/splits.parquet")
    parser.add_argument("--no-medspacy", action="store_true", help="Use regex-only negation (no medspaCy)")
    args = parser.parse_args()

    lexicon = load_lexicon(args.lexicon)
    compiled = compile_lexicon(lexicon)

    # Load medspaCy pipeline
    nlp = None
    if not args.no_medspacy:
        try:
            nlp = build_nlp()
            print("medspaCy pipeline chargé.")
        except ImportError:
            print("medspaCy non disponible, utilisation du fallback regex.")

    # Load data
    splits = pd.read_parquet(args.splits_file)
    split_hadm = set(splits.loc[splits["split"] == args.split, "hadm_id"])
    print(f"Split '{args.split}' : {len(split_hadm)} séjours")

    pmh = pd.read_parquet(args.pmh, columns=["hadm_id", "pmh_text"])
    pmh = pmh[pmh["hadm_id"].isin(split_hadm)]

    labels = pd.read_parquet(args.labels)
    labels = labels[labels["hadm_id"].isin(split_hadm)][["hadm_id"] + CATEGORIES]

    # Predict
    results = []
    for _, row in tqdm(pmh.iterrows(), total=len(pmh), desc="Predicting"):
        text = row["pmh_text"] if pd.notna(row["pmh_text"]) else ""
        flags, _ = predict(text, compiled, nlp)
        flags["hadm_id"] = row["hadm_id"]
        results.append(flags)

    preds = pd.DataFrame(results)

    # Save predictions
    out_path = f"data/processed/preds_rules_{args.split}.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    preds.to_parquet(out_path, index=False)
    print(f"Prédictions enregistrées dans {out_path}")

    # Evaluate
    metrics = evaluate(preds, labels)

    print(f"\n{'Catégorie':<15} {'N_true':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 67)
    for _, r in metrics.iterrows():
        print(
            f"{r['category']:<15} {r['n_true']:>7} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
            f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}"
        )

    metrics_path = f"outputs/rules_metrics_{args.split}.csv"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    print(f"\nMétriques enregistrées dans {metrics_path}")


if __name__ == "__main__":
    main()
