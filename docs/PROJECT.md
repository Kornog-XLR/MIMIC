# Extraction automatique des comorbidités de Charlson à partir des notes de sortie MIMIC-IV

## Objectif

Ce projet de M2 construit un pipeline NLP complet pour extraire et classifier les 17 comorbidités de l'indice de Charlson à partir de la section **Past Medical History (PMH)** des notes de sortie de la base MIMIC-IV. Deux approches sont comparées :

1. **Baseline règles** : lexique médical + détection de négation
2. **Fine-tuning BERT clinique** : Bio_ClinicalBERT en classification multi-label

La supervision distante est assurée par les codes ICD-9/ICD-10 (mapping de Quan 2005). Un gold standard annoté manuellement permet de mesurer la qualité réelle des deux méthodes et le bruit de la supervision distante.

---

## Architecture du projet

```
MIMIC/
├── config/
│   └── charlson_lexicon.yaml      # Lexique pour le rules baseline (~200 termes)
├── data/
│   ├── raw/                        # Données brutes MIMIC-IV (DUA PhysioNet)
│   │   ├── discharge.csv
│   │   ├── diagnoses_icd.csv
│   │   ├── patients.csv
│   │   └── admissions.csv
│   ├── interim/
│   │   └── pmh.parquet             # Sections PMH extraites
│   └── processed/
│       ├── labels_charlson.parquet # Labels ICD (17 catégories binaires + score)
│       ├── splits.parquet          # Assignation train/val/test/gold
│       ├── gold_annotations.csv    # Annotations manuelles
│       ├── preds_rules_*.parquet   # Prédictions rules baseline
│       └── preds_bert_*.parquet    # Prédictions BERT
├── models/
│   └── <run_name>/
│       ├── checkpoint-*/           # Checkpoints HuggingFace
│       └── final/                  # Modèle finalisé (modèle + tokenizer + seuils)
├── outputs/
│   ├── *.csv                       # Tableaux de métriques
│   └── figures/                    # Graphiques (Bland-Altman, barres F1)
├── src/                            # Code source (détaillé ci-dessous)
├── tests/                          # Tests pytest (200+ tests, données fictives)
└── docs/
    ├── guide_annotation.md         # Guide pour l'annotation manuelle
    └── PROJECT.md                  # Ce fichier
```

---

## Les 17 catégories de Charlson

| Abréviation | Catégorie | Poids |
|---|---|---|
| mi | Infarctus du myocarde | 1 |
| chf | Insuffisance cardiaque congestive | 1 |
| pvd | Maladie vasculaire périphérique | 1 |
| cevd | Maladie cérébrovasculaire | 1 |
| dementia | Démence | 1 |
| cpd | Maladie pulmonaire chronique | 1 |
| rheumd | Maladie rhumatismale | 1 |
| pud | Ulcère gastro-duodénal | 1 |
| mld | Maladie hépatique légère | 1 |
| diab | Diabète sans complication | 1 |
| diabwc | Diabète avec complication | 2 |
| hp | Hémiplégie/paraplégie | 2 |
| rend | Maladie rénale | 2 |
| canc | Cancer (tumeur solide) | 2 |
| msld | Maladie hépatique sévère | 3 |
| metacanc | Cancer métastatique | 6 |
| aids | SIDA | 6 |

**Hiérarchie** (pour le calcul du score, pas pour les labels bruts) :
- diabwc (2) supprime diab (1)
- msld (3) supprime mld (1)
- metacanc (6) supprime canc (2)

---

## Pipeline détaillé

### Étape 1 : Extraction des sections PMH

**Script** : `src/extract_pmh.py`

Les notes de sortie MIMIC-IV sont stockées sous forme de texte continu sur une seule ligne (pas de retours à la ligne). Le script utilise des expressions régulières pour identifier les en-têtes de section ("Past Medical History:", "PMH:", "Medical History:", etc.) et extraire le texte entre l'en-tête PMH et l'en-tête de la section suivante.

**Entrée** : `data/raw/discharge.csv` (colonne `text`)
**Sortie** : `data/interim/pmh.parquet` (colonnes `hadm_id`, `pmh_text`, `pmh_found`)

**Commande** :
```bash
uv run python src/extract_pmh.py
```

### Étape 2 : Construction des labels ICD (supervision distante)

**Scripts** : `src/labels_icd.py`, `src/charlson_mapping.py`

Chaque séjour hospitalier (`hadm_id`) est associé à des codes diagnostiques ICD-9 et ICD-10 dans MIMIC-IV. Le mapping de Quan 2005 (étendu avec les codes ICD-10-CM spécifiques à MIMIC-IV) convertit ces codes en 17 indicateurs binaires.

`charlson_mapping.py` contient :
- Les tables de préfixes ICD-9 et ICD-10 pour chaque catégorie
- `vectorized_match()` : matching vectorisé avec `str.startswith(tuple)` + `groupby.max()` pour traiter ~330k séjours en quelques secondes
- `vectorized_score()` : calcul vectorisé du score de Charlson avec hiérarchie et pondérations

`labels_icd.py` :
- Charge `diagnoses_icd.csv`, `patients.csv`, `admissions.csv`
- Calcule l'âge à l'admission : `anchor_age + (year(admittime) - anchor_year)`
- Produit un DataFrame avec `hadm_id`, 17 colonnes binaires, `charlson_icd`, `charlson_icd_age`

**Sortie** : `data/processed/labels_charlson.parquet`

**Commande** :
```bash
uv run python src/labels_icd.py
```

### Étape 3 : Découpage train/val/test/gold

**Script** : `src/make_splits.py`

Le découpage est fait au niveau **patient** (pas de fuite de `subject_id` entre les splits). Le pool éligible exclut les séjours sans PMH, sans code ICD, avec un texte > 3000 caractères, et les doublons.

| Split | Taille | Particularité |
|---|---|---|
| gold | 200 random + 50 enrichis | Pour l'annotation manuelle, enrichi en catégories rares |
| test | 2000 | Évaluation finale |
| val | 2000 | Optimisation des seuils, sélection du modèle |
| train | 20000 + enrichissement | Entraînement BERT, min 300 positifs par catégorie rare |

La colonne `sampling` (random/enriched) dans `splits.parquet` permet de distinguer les séjours enrichis lors de l'évaluation sur le gold.

**Sortie** : `data/processed/splits.parquet`

**Commande** :
```bash
uv run python src/make_splits.py
```

### Étape 4 : Annotation manuelle (gold standard)

**Scripts** : `src/prepare_gold.py`, `src/annotate_app.py`, `src/annotation_io.py`

1. `prepare_gold.py` extrait les séjours du split gold, les mélange avec une graine fixe, et produit `gold_to_annotate.parquet` (sans labels ICD pour garantir l'annotation en aveugle)
2. `annotate_app.py` est une application Streamlit avec :
   - Affichage du texte PMH (un séjour à la fois)
   - 17 cases à cocher avec libellés français et infobulles
   - Case "incertain" et champ commentaire libre
   - Navigation avant/arrière et reprise automatique
3. `annotation_io.py` gère la persistance (CSV) sans dépendance Streamlit

**Sortie** : `data/processed/gold_annotations.csv`

**Commande** :
```bash
uv run streamlit run src/annotate_app.py
```

### Étape 5 : Baseline règles

**Scripts** : `src/rules_baseline.py`, `config/charlson_lexicon.yaml`

Le lexique YAML contient ~200 termes et patterns regex organisés par catégorie. Le script :

1. **Segmente** le texte PMH en items (par retour à la ligne, virgule ou point-virgule)
2. **Détecte la négation** au niveau de chaque segment (préfixes : "no", "denies", "negative for", "r/o", "family history of"...)
3. **Matche** les termes du lexique contre chaque segment non-nié
4. **Applique les règles de co-occurrence** pour la sévérité :
   - Diabète + complication (neuropathie, rétinopathie...) → `diabwc`
   - Maladie hépatique + marqueur sévère (ascite, varices...) → `msld`
   - Marqueur métastatique → `metacanc` + `canc`

Les termes ambigus ont été retirés du lexique après analyse des faux positifs : HD (hospital day), PD (Parkinson), PCP (primary care physician), ALL (mot "all"), tumor (bénin), malignancy (trop large), mets (sans "to").

**Sorties** :
- `data/processed/preds_rules_<split>.parquet`
- `outputs/rules_metrics_<split>.csv`

**Commande** :
```bash
uv run python src/rules_baseline.py --split val
uv run python src/rules_baseline.py --split test
```

### Étape 6 : Fine-tuning BERT

**Scripts** : `src/train_bert.py`, `src/bert_utils.py`, `src/finalize_bert.py`, `src/predict_bert.py`

#### Entraînement (`train_bert.py`)

- **Modèle** : `emilyalsentzer/Bio_ClinicalBERT` (BERT pré-entraîné sur des notes cliniques MIMIC-III)
- **Tâche** : classification multi-label (17 sorties sigmoïdes)
- **Loss** : `BCEWithLogitsLoss` avec `pos_weight` optionnel (ratio négatifs/positifs, clippé à [1, 10]) pour compenser le déséquilibre des classes
- **Dataset** : tokenisation à la volée (économie de RAM)
- **Sélection du modèle** : macro-AUPRC sur val
- **Optimisations mémoire** (8 Go RAM) : gradient accumulation, gradient checkpointing, `dataloader_pin_memory=False`, `dataloader_num_workers=0`
- **Resume** : `--resume` auto-détecte le dernier checkpoint

#### Finalisation (`finalize_bert.py`)

Transforme un checkpoint en modèle utilisable :
- Charge le checkpoint (choix : `best`, `last`, ou explicite)
- Prédit sur val, optimise un seuil par catégorie (grille F1, 0.05-0.95)
- Sauvegarde dans `models/<run>/final/` : modèle, tokenizer, `thresholds.json`, `final_config.json`
- Option `--all-checkpoints` : évalue chaque checkpoint et sélectionne le meilleur

#### Inférence (`predict_bert.py`)

Charge le modèle finalisé, applique les seuils optimisés, produit un parquet avec `hadm_id` + 17 probabilités + 17 prédictions binaires.

#### Utilitaires partagés (`bert_utils.py`)

Fonctions communes entre train, finalize et predict : `select_device`, `optimize_thresholds`, `full_evaluation`, `predict_probs`, `save_final`, `list_checkpoints`, `resolve_checkpoint`.

**Commandes** :
```bash
# Entraînement (optimisé 8 Go)
uv run python src/train_bert.py \
  --model emilyalsentzer/Bio_ClinicalBERT \
  --epochs 3 --lr 2e-5 --batch-size 4 --grad-accum 4 \
  --max-length 128 --pos-weight --gradient-checkpointing \
  --run-name bert_v1

# Finalisation
uv run python src/finalize_bert.py --run bert_v1 --batch-size 4

# Inférence
uv run python src/predict_bert.py --run bert_v1 --split test
```

### Étape 7 : Évaluation finale

**Scripts** : `src/evaluate.py`, `src/inspect_gold_errors.py`

#### Évaluation (`evaluate.py`)

Compare une ou plusieurs méthodes contre une référence (labels ICD pour test, annotations manuelles pour gold).

**Métriques de classification** :
- Précision, rappel, F1 par catégorie avec IC 95% par bootstrap (1000 rééchantillonnages)
- Micro et macro F1 avec IC
- AUROC et AUPRC pour BERT (quand les probabilités existent)

**Score de Charlson** :
- ICC (accord absolu) avec IC 95% analytique (pingouin)
- Kappa pondéré quadratique sur les classes de score (0, 1-2, 3-4, >=5)
- Bland-Altman : biais moyen et limites d'agrément, avec figure
- Pourcentage de scores exacts et à +-1 point

**Comparaison des méthodes** :
- Différence de macro F1 entre chaque paire avec IC 95% par bootstrap apparié

**Spécificités du gold** :
- Métriques globales (micro/macro) sur les séjours `sampling=random` uniquement
- Catégories rares (aids, hp, msld, metacanc, dementia, rheumd) sur random + enriched
- Analyse de sensibilité excluant les séjours marqués "incertain"
- Accord ICD vs annotation manuelle (P, R, F1, kappa de Cohen) : mesure du bruit de la supervision distante

**Sorties** :
- `outputs/eval_<split>_classification.csv`
- `outputs/eval_<split>_score.csv`
- `outputs/eval_<split>_comparisons.csv`
- `outputs/figures/f1_by_category_<split>.png`
- `outputs/figures/bland_altman_<split>_<method>.png`
- (gold) `outputs/eval_gold_icd_agreement.csv`, `outputs/eval_gold_sensitivity.csv`

**Commandes** :
```bash
uv run python src/evaluate.py --split test --methods rules bert_v1
uv run python src/evaluate.py --split gold --methods rules bert_v1
```

#### Inspection des erreurs (`inspect_gold_errors.py`, `inspect_errors.py`)

Scripts d'exploration manuelle des faux positifs et faux négatifs. Affichent le texte PMH avec les prédictions, annotations et labels ICD. **À exécuter par l'utilisateur uniquement** (affiche du texte clinique).

```bash
python src/inspect_gold_errors.py --cat chf --method rules --type fn --n 20
python src/inspect_errors.py --cat rend --type fp --split val
```

---

## Tests

Le projet contient **200+ tests pytest** organisés en 7 fichiers, tous sur données fictives (aucun accès aux données MIMIC) :

| Fichier | Tests | Couverture |
|---|---|---|
| `test_extract_pmh.py` | 13 | Extraction PMH : formats variés, cas limites |
| `test_labels_icd.py` | 60 | Mapping ICD-9/10/10-CM, score, hiérarchie, âge |
| `test_make_splits.py` | 11 | Non-chevauchement patients, tailles, enrichissement |
| `test_gold.py` | 13 | Préparation gold, I/O annotations, reprise |
| `test_rules_baseline.py` | 37 | Détection, négation, famille, sévérité |
| `test_bert.py` | 33 | Dataset, modèle, seuils, save/reload, checkpoints |
| `test_evaluate.py` | 33 | Métriques calculées à la main, bootstrap, ICC, kappa |

```bash
uv run python -m pytest tests/ -v
```

---

## Dépendances

Gérées avec `uv` (pyproject.toml) :

- **Données** : pandas, pyarrow
- **NLP règles** : pyyaml, medspacy (optionnel)
- **BERT** : torch, transformers, accelerate
- **Métriques** : scikit-learn, pingouin
- **Visualisation** : matplotlib
- **Annotation** : streamlit
- **Tests** : pytest

---

## Contraintes de données

Les données MIMIC-IV sont soumises à un Data Use Agreement (DUA) PhysioNet. En conséquence :
- Le dossier `data/` n'est jamais versionné (`.gitignore`)
- Aucun script n'affiche de texte clinique brut — uniquement des statistiques agrégées
- Les tests utilisent exclusivement des données fictives (`tests/fixtures/`)
- Les scripts d'inspection (`inspect_errors.py`, `inspect_gold_errors.py`) sont exécutés uniquement par l'utilisateur dans son terminal
