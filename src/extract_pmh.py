"""Extract Past Medical History sections from MIMIC-IV discharge notes."""

import argparse
import os
import re

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# En-têtes de début de PMH (insensible à la casse)
# ---------------------------------------------------------------------------
PMH_HEADERS = [
    "Past Medical History:",
    "PMH:",
    "Past Medical/Surgical History:",
    "Past Medical and Surgical History:",
    "Medical History:",
]

# ---------------------------------------------------------------------------
# En-têtes pouvant suivre la PMH (fin de section)
# ---------------------------------------------------------------------------
NEXT_SECTION_HEADERS = [
    "Past Surgical History:",
    "Social History:",
    "Family History:",
    "Physical Exam:",
    "Pertinent Results:",
    "Medications on Admission:",
    "Allergies:",
    "Brief Hospital Course:",
    "Discharge Medications:",
    "Discharge Diagnosis:",
    "Discharge Disposition:",
    "Review of Systems:",
    "History of Present Illness:",
    "Chief Complaint:",
]

# ---------------------------------------------------------------------------
# Regex pré-compilées
# ---------------------------------------------------------------------------
_pmh_pattern = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(re.escape(h) for h in PMH_HEADERS) + r")",
    re.IGNORECASE,
)

_known_next = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(re.escape(h) for h in NEXT_SECTION_HEADERS) + r")",
    re.IGNORECASE,
)

# En-tête générique : ligne commençant par des mots capitalisés suivis de ":"
_generic_header = re.compile(r"(?:^|\n)\s*[A-Z][A-Za-z]+(?:[ /\-&',()]+[A-Za-z]+)*:\s")

_multi_spaces = re.compile(r"[ \t]{2,}")


def extract_pmh(text: str) -> str | None:
    """Return the PMH section text, or None if absent."""
    if not isinstance(text, str):
        return None

    m = _pmh_pattern.search(text)
    if m is None:
        return None

    # Start of content: right after the matched header's ":"
    header_end = text.index(":", m.start()) + 1
    rest = text[header_end:]

    # Find the earliest next section header
    best = len(rest)

    for pat in (_known_next, _generic_header):
        hit = pat.search(rest)
        if hit is not None:
            # Skip if match is at position 0 (empty PMH → next header immediately)
            pos = hit.start()
            # Adjust: if the match starts with \n, the real position is after it
            if pos == 0 and rest[0] == "\n":
                # This is the very next header — PMH is empty between headers
                best = 0
                break
            if 0 < pos < best:
                best = pos

    pmh = rest[:best].strip()
    if not pmh:
        return None

    pmh = _multi_spaces.sub(" ", pmh)
    return pmh


def main():
    parser = argparse.ArgumentParser(description="Extract PMH from discharge notes.")
    parser.add_argument(
        "--input",
        default="data/raw/discharge.csv",
        help="Path to the discharge CSV (default: data/raw/discharge.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/interim/pmh.parquet",
        help="Output parquet path (default: data/interim/pmh.parquet)",
    )
    parser.add_argument(
        "--stats-output",
        default="outputs/pmh_stats.csv",
        help="Path for statistics CSV (default: outputs/pmh_stats.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N notes (for quick testing)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=20_000,
        help="Chunk size for CSV reading (default: 20000)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)

    chunks = []
    rows_read = 0

    reader = pd.read_csv(args.input, chunksize=args.chunksize)
    with tqdm(desc="Processing notes", unit=" notes") as pbar:
        for chunk in reader:
            if args.limit is not None:
                remaining = args.limit - rows_read
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining]

            chunk = chunk[["note_id", "subject_id", "hadm_id", "text"]].copy()
            chunk["pmh_text"] = chunk["text"].apply(extract_pmh)
            chunk["pmh_found"] = chunk["pmh_text"].notna()
            chunk["n_chars"] = chunk["pmh_text"].str.len().fillna(0).astype(int)
            chunk = chunk.drop(columns=["text"])
            chunks.append(chunk)

            rows_read += len(chunk)
            pbar.update(len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(args.output, index=False)

    # --- Aggregate statistics only ---
    n_notes = len(df)
    n_hadm = df["hadm_id"].nunique()
    n_found = df["pmh_found"].sum()
    pct_found = 100.0 * n_found / n_notes if n_notes else 0.0

    chars = df.loc[df["pmh_found"], "n_chars"]
    quantiles = chars.quantile([0, 0.25, 0.5, 0.75, 0.95, 1.0]) if len(chars) else pd.Series(dtype=float)
    n_short = int((chars < 20).sum()) if len(chars) else 0
    n_long = int((chars > 3000).sum()) if len(chars) else 0

    stats = {
        "n_notes": n_notes,
        "n_hadm_unique": n_hadm,
        "pct_pmh_found": round(pct_found, 2),
        "n_chars_min": int(quantiles.get(0, 0)),
        "n_chars_25": int(quantiles.get(0.25, 0)),
        "n_chars_median": int(quantiles.get(0.5, 0)),
        "n_chars_75": int(quantiles.get(0.75, 0)),
        "n_chars_95": int(quantiles.get(0.95, 0)),
        "n_chars_max": int(quantiles.get(1.0, 0)),
        "n_short_lt20": n_short,
        "n_long_gt3000": n_long,
    }

    print(f"Notes traitées      : {stats['n_notes']}")
    print(f"hadm_id uniques     : {stats['n_hadm_unique']}")
    print(f"PMH trouvées        : {n_found} ({stats['pct_pmh_found']} %)")
    print(f"n_chars min         : {stats['n_chars_min']}")
    print(f"n_chars 25 %        : {stats['n_chars_25']}")
    print(f"n_chars médiane     : {stats['n_chars_median']}")
    print(f"n_chars 75 %        : {stats['n_chars_75']}")
    print(f"n_chars 95 %        : {stats['n_chars_95']}")
    print(f"n_chars max         : {stats['n_chars_max']}")
    print(f"PMH < 20 chars      : {stats['n_short_lt20']}")
    print(f"PMH > 3000 chars    : {stats['n_long_gt3000']}")

    pd.DataFrame([stats]).to_csv(args.stats_output, index=False)
    print(f"\nStats enregistrées dans {args.stats_output}")
    print(f"Résultats enregistrés dans {args.output}")


if __name__ == "__main__":
    main()
