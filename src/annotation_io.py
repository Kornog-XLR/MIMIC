"""Save / load annotation logic for the gold set (no Streamlit dependency)."""

import os
from datetime import datetime

import pandas as pd

CATEGORIES = [
    "mi", "chf", "pvd", "cevd", "dementia", "cpd", "rheumd", "pud",
    "mld", "diab", "diabwc", "hp", "rend", "canc", "msld", "metacanc", "aids",
]

ANNOTATION_COLUMNS = ["hadm_id"] + CATEGORIES + ["incertain", "commentaire", "horodatage"]


def load_annotations(path: str) -> pd.DataFrame:
    """Load existing annotations CSV, or return an empty DataFrame."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path, dtype={"hadm_id": int, "commentaire": str})
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def save_annotation(
    path: str,
    hadm_id: int,
    flags: dict[str, int],
    incertain: bool,
    commentaire: str,
) -> None:
    """Append or update one annotation row."""
    df = load_annotations(path)

    row = {"hadm_id": hadm_id, "horodatage": datetime.now().isoformat()}
    for cat in CATEGORIES:
        row[cat] = flags.get(cat, 0)
    row["incertain"] = int(incertain)
    row["commentaire"] = commentaire

    if hadm_id in df["hadm_id"].values:
        idx = df.index[df["hadm_id"] == hadm_id][0]
        df.loc[idx] = row
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def find_resume_index(annotations: pd.DataFrame, gold: pd.DataFrame) -> int:
    """Return the 0-based index of the first unannotated stay in gold order."""
    if annotations.empty:
        return 0
    annotated = set(annotations["hadm_id"])
    for i, hid in enumerate(gold["hadm_id"]):
        if hid not in annotated:
            return i
    return len(gold)
