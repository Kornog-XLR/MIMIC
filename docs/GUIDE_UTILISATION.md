# Guide d'utilisation - Pipeline NLP Charlson / MIMIC-IV

Ce guide explique comment installer et exécuter le pipeline sur une nouvelle machine.

---

## 1. Prerequis

- **Python >= 3.13**
- **uv** (gestionnaire de packages) : https://docs.astral.sh/uv/getting-started/installation/
- **Git**
- Acces aux donnees MIMIC-IV via PhysioNet (DUA signe)

---

## 2. Installation

```bash
# Cloner le depot
git clone <url-du-repo> MIMIC
cd MIMIC

# Installer les dependances (uv cree automatiquement un .venv)
uv sync
```

Cela installe toutes les dependances declarees dans `pyproject.toml` : pandas, torch, transformers, scikit-learn, streamlit, etc.

---

## 3. Preparer les donnees MIMIC-IV

Placer les fichiers CSV bruts dans `data/raw/` :

```
data/raw/
  discharge.csv       # Notes de sortie (table hosp/discharge)
  diagnoses_icd.csv   # Codes diagnostiques (table hosp/diagnoses_icd)
  patients.csv        # Donnees patients (table hosp/patients)
  admissions.csv      # Sejours (table hosp/admissions)
```

Ces fichiers proviennent de MIMIC-IV (PhysioNet). Ils ne sont jamais versiones dans Git.

---

## 4. Verifier que tout fonctionne

Lancer les tests (sur donnees fictives, pas besoin de MIMIC) :

```bash
uv run python -m pytest tests/ -v
```

200+ tests doivent passer.

---

## 5. Executer le pipeline etape par etape

### Etape 1 : Extraction des sections Past Medical History

**Script** : `src/extract_pmh.py`

**Ce qu'il fait** : Lit les notes de sortie (`discharge.csv`), detecte la section "Past Medical History" par regex, et extrait le texte de cette section pour chaque sejour.

**Sortie** : `data/interim/pmh.parquet`

```bash
uv run python src/extract_pmh.py
```

---

### Etape 2 : Construction des labels ICD (supervision distante)

**Scripts** : `src/labels_icd.py` + `src/charlson_mapping.py`

**Ce qu'ils font** :
- `charlson_mapping.py` contient le mapping de Quan 2005 (codes ICD-9/ICD-10 vers les 17 categories de Charlson) et le calcul vectorise du score.
- `labels_icd.py` charge les fichiers `diagnoses_icd.csv`, `patients.csv`, `admissions.csv`, calcule l'age a l'admission, et produit pour chaque sejour : 17 indicateurs binaires + score de Charlson.

**Sortie** : `data/processed/labels_charlson.parquet`

```bash
uv run python src/labels_icd.py
```

---

### Etape 3 : Decoupage train / val / test / gold

**Script** : `src/make_splits.py`

**Ce qu'il fait** : Decoupe les sejours en 4 splits au niveau patient (pas de fuite entre splits). Exclut les sejours sans PMH ou sans code ICD. Enrichit les splits en categories rares.

| Split | Taille approx. | Usage |
|-------|----------------|-------|
| train | ~20 000 | Entrainement BERT |
| val   | 2 000 | Optimisation des seuils |
| test  | 2 000 | Evaluation finale |
| gold  | 250 | Annotation manuelle |

**Sortie** : `data/processed/splits.parquet`

```bash
uv run python src/make_splits.py
```

---

### Etape 4 (optionnelle) : Annotation manuelle (gold standard)

**Scripts** :
- `src/prepare_gold.py` : Extrait les sejours du split gold dans un fichier pret a annoter (sans labels ICD, pour annotation en aveugle).
- `src/annotate_app.py` : Application Streamlit pour annoter les 17 categories case par case.
- `src/annotation_io.py` : Module de persistance CSV pour les annotations.

**Sorties** : `data/processed/gold_to_annotate.parquet`, `data/processed/gold_annotations.csv`

```bash
uv run python src/prepare_gold.py
uv run streamlit run src/annotate_app.py
```

Le guide d'annotation est dans `docs/guide_annotation.md`.

---

### Etape 5 : Baseline regles

**Script** : `src/rules_baseline.py`

**Ce qu'il fait** : Pour chaque sejour, segmente le texte PMH en items, detecte la negation, et matche un lexique medical (~200 termes definis dans `config/charlson_lexicon.yaml`). Applique des regles de co-occurrence pour la severite (ex : diabete + neuropathie = diabete avec complication).

**Sorties** : `data/processed/preds_rules_<split>.parquet`, `outputs/rules_metrics_<split>.csv`

```bash
uv run python src/rules_baseline.py --split val
uv run python src/rules_baseline.py --split test
```

---

### Etape 6 : Fine-tuning Bio_ClinicalBERT

Trois scripts a executer dans l'ordre :

#### 6a. Entrainement

**Script** : `src/train_bert.py`

**Ce qu'il fait** : Fine-tune Bio_ClinicalBERT en classification multi-label (17 sorties sigmoides, loss BCEWithLogitsLoss). Sauvegarde des checkpoints dans `models/<run_name>/`.

**Utilitaires partages** : `src/bert_utils.py` (selection device, optimisation seuils, evaluation, inference).

```bash
uv run python src/train_bert.py --model emilyalsentzer/Bio_ClinicalBERT --epochs 3 --lr 2e-5 --batch-size 4 --grad-accum 4 --max-length 128 --pos-weight --gradient-checkpointing --run-name bert_v2
```

> Sur une machine avec peu de RAM (8 Go), garder batch-size 4, grad-accum 4 et --gradient-checkpointing. L'entrainement dure 2-3h.
> Sur une machine avec GPU CUDA, augmenter batch-size (16 ou 32) et retirer --gradient-checkpointing.

#### 6b. Finalisation

**Script** : `src/finalize_bert.py`

**Ce qu'il fait** : Charge le meilleur checkpoint, predit sur val pour optimiser un seuil F1 par categorie, puis sauvegarde le modele final avec tokenizer et seuils dans `models/<run_name>/final/`.

```bash
uv run python src/finalize_bert.py --run bert_v2 --batch-size 4
```

#### 6c. Inference

**Script** : `src/predict_bert.py`

**Ce qu'il fait** : Charge le modele finalise, applique les seuils optimises, et produit les predictions binaires + probabilites.

**Sortie** : `data/processed/preds_bert_v2_<split>.parquet`

```bash
uv run python src/predict_bert.py --run bert_v2 --split test
```

---

### Etape 7 : Evaluation

**Script** : `src/evaluate.py`

**Ce qu'il fait** : Compare les methodes (rules, bert) contre la reference (labels ICD pour test, annotations manuelles pour gold). Calcule par categorie : precision, rappel, F1 avec IC 95% (bootstrap). Calcule aussi l'accord sur le score de Charlson (ICC, kappa pondere, Bland-Altman). Genere des figures.

**Sorties** : fichiers CSV dans `outputs/` et figures PNG dans `outputs/figures/`.

```bash
uv run python src/evaluate.py --split test --methods rules bert_v2
```

Pour evaluer sur le gold (necessite les annotations manuelles) :

```bash
uv run python src/evaluate.py --split gold --methods rules bert_v2
```

---

### Scripts utilitaires

| Script | Role |
|--------|------|
| `src/inventaire_entetes.py` | Inventorie les en-tetes de section dans les notes de sortie (exploration) |
| `src/inspect_errors.py` | Affiche les faux positifs/negatifs sur val/test pour analyse manuelle |
| `src/inspect_gold_errors.py` | Idem sur le split gold |

Ces scripts affichent du texte clinique brut. A utiliser uniquement en local.

```bash
uv run python src/inspect_errors.py --cat chf --type fp --split val
uv run python src/inspect_gold_errors.py --cat chf --method rules --type fn --n 20
```

---

## 6. Resumé des commandes (pipeline complet)

```bash
# 1. Installation
uv sync

# 2. Tests
uv run python -m pytest tests/ -v

# 3. Pipeline donnees
uv run python src/extract_pmh.py
uv run python src/labels_icd.py
uv run python src/make_splits.py

# 4. Baseline regles
uv run python src/rules_baseline.py --split val
uv run python src/rules_baseline.py --split test

# 5. BERT
uv run python src/train_bert.py --model emilyalsentzer/Bio_ClinicalBERT --epochs 3 --lr 2e-5 --batch-size 4 --grad-accum 4 --max-length 128 --pos-weight --gradient-checkpointing --run-name bert_v2
uv run python src/finalize_bert.py --run bert_v2 --batch-size 4
uv run python src/predict_bert.py --run bert_v2 --split test

# 6. Evaluation
uv run python src/evaluate.py --split test --methods rules bert_v2
```

---

## 7. Structure des dossiers

```
MIMIC/
├── config/charlson_lexicon.yaml   # Lexique medical (~200 termes)
├── data/
│   ├── raw/                       # Fichiers CSV MIMIC-IV (non versiones)
│   ├── interim/                   # Donnees intermediaires (pmh.parquet)
│   └── processed/                 # Labels, splits, predictions
├── models/                        # Checkpoints et modeles finalises
├── outputs/                       # Metriques CSV et figures PNG
├── src/                           # Code source (16 scripts)
├── tests/                         # Tests pytest (200+)
├── config/                        # Configuration (lexique)
└── docs/                          # Documentation
```
