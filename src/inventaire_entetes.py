"""Recense les en-têtes de section dans les notes de discharge.csv."""

import re
import pandas as pd

df = pd.read_csv("data/raw/discharge.csv")

# Texte sur une seule ligne : en-têtes = mots capitalisés suivis de ":"
pattern = re.compile(r"(?<![a-zA-Z])([A-Z][A-Za-z]+(?:[ /\-&',()]+[A-Za-z]+)*:)")

header_counts: dict[str, int] = {}

for text in df["text"].dropna():
    found = set(match.strip() for match in pattern.findall(text))
    for h in found:
        header_counts[h] = header_counts.get(h, 0) + 1

result = (
    pd.DataFrame({"en_tete": list(header_counts.keys()), "nb_notes": list(header_counts.values())})
    .query("nb_notes >= 5")
    .sort_values("nb_notes", ascending=False)
    .reset_index(drop=True)
)

print(result.head(20).to_string(index=False))

