"""Agent LLM pour détecter le diabète dans les Past Medical History.

Lit le parquet PMH déjà extrait, envoie chaque texte au LLM,
et produit une table JSON {subject_id, diabete}.
"""

import argparse
import json
import sys

import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from tqdm import tqdm

from settings import API_KEY, ENDPOINT, MODEL


# -- Structured output --------------------------------------------------------

class DiabetesResult(BaseModel):
    diabete: bool


# -- Agent --------------------------------------------------------------------

INSTRUCTIONS = (
    "Tu es un assistant médical expert. "
    "On te donne la section 'Past Medical History' d'un compte-rendu hospitalier. "
    "Détermine si le patient est diabétique (type 1, type 2, ou diabète non précisé). "
    "Réponds uniquement par le champ JSON demandé."
)

agent = Agent(
    model=OpenAIModel(MODEL, base_url=ENDPOINT, api_key=API_KEY),
    system_prompt=INSTRUCTIONS,
    result_type=DiabetesResult,
)


def build_prompt(pmh_text: str) -> str:
    return (
        f"Voici la section Past Medical History du patient :\n\n"
        f"{pmh_text}\n\n"
        f"Question : ce patient est-il diabétique ?"
    )


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect diabetes from PMH via LLM agent.")
    parser.add_argument("--input", default="data/interim/pmh.parquet", help="PMH parquet file")
    parser.add_argument("--output", default="outputs/agent_diabetes.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N patients")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = df[df["pmh_text"].notna()].copy()

    # One row per patient (keep first PMH if multiple admissions)
    df = df.drop_duplicates(subset="subject_id", keep="first")

    if args.limit:
        df = df.head(args.limit)

    print(f"Patients à traiter : {len(df)}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM queries"):
        prompt = build_prompt(row["pmh_text"])
        try:
            resp = agent.run_sync(prompt)
            diabete = resp.data.diabete
        except Exception as e:
            print(f"  [ERREUR] subject_id={row['subject_id']}: {e}", file=sys.stderr)
            diabete = None

        pmh_out = row["pmh_text"] if diabete is True else None
        results.append({"subject_id": int(row["subject_id"]), "diabete": diabete, "pmh": pmh_out})

    # Save JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_true = sum(1 for r in results if r["diabete"] is True)
    n_false = sum(1 for r in results if r["diabete"] is False)
    n_err = sum(1 for r in results if r["diabete"] is None)
    print(f"\nRésultats : {n_true} diabétiques, {n_false} non-diabétiques, {n_err} erreurs")
    print(f"JSON enregistré dans {args.output}")


if __name__ == "__main__":
    main()
