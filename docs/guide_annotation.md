# Guide d'annotation — Catégories de Charlson

## Principes généraux

- **Annoter d'après le texte seul** : ne pas consulter les codes ICD.
- **Négation** : "no history of …", "denies …", "r/o …" (rule out), "negative for" → **absent**.
- **Antécédent familial** : "family history of …", "mother with …" → **absent** (ne concerne pas le patient).
- **s/p (status post)** : "s/p CABG", "h/o MI" → **présent** (antécédent confirmé).
- **PMH vide / "none"** : aucune catégorie cochée.
- En cas de doute, cocher "incertain" et ajouter un commentaire.

---

## Catégories

### 1. mi — Infarctus du myocarde
**Définition** : antécédent d'infarctus du myocarde (nécrose myocardique).
**Positif** : MI, myocardial infarction, STEMI, NSTEMI, s/p MI, h/o MI, "heart attack".
**Négatif** : angine stable sans nécrose, "r/o MI".

### 2. chf — Insuffisance cardiaque
**Définition** : insuffisance cardiaque congestive, systolique ou diastolique.
**Positif** : CHF, HF, heart failure, HFrEF, HFpEF, "EF 30%", congestive heart failure, cardiomyopathy (dilatée, ischémique…).
**Négatif** : "diastolic dysfunction grade I" isolée sans diagnostic d'IC.
> **⚠ Point à trancher** : cocher si "diastolic dysfunction" est mentionnée sans diagnostic explicite d'IC ?

### 3. pvd — Maladie vasculaire périphérique
**Définition** : artériopathie périphérique, anévrisme aortique.
**Positif** : PVD, PAD, peripheral vascular disease, claudication, s/p fem-pop bypass, AAA (abdominal aortic aneurysm), aortic aneurysm.
**Négatif** : varices veineuses isolées.

### 4. cevd — Maladie cérébrovasculaire
**Définition** : antécédent d'AVC ou AIT.
**Positif** : CVA, stroke, TIA, carotid endarterectomy, "h/o stroke", "cerebrovascular accident".
**Négatif** : "carotid stenosis" asymptomatique sans AVC/AIT.
> **⚠ Point à trancher** : sténose carotidienne asymptomatique seule → inclure ou non ?

### 5. dementia — Démence
**Définition** : trouble cognitif majeur diagnostiqué.
**Positif** : dementia, Alzheimer's, vascular dementia, Lewy body dementia, "advanced cognitive decline".
**Négatif** : "mild cognitive impairment" (MCI), confusion aiguë (delirium).
> **⚠ Point à trancher** : MCI → exclure selon Charlson original, mais certains auteurs l'incluent.

### 6. cpd — Maladie pulmonaire chronique
**Définition** : BPCO, asthme persistant, ou autre pneumopathie chronique.
**Positif** : COPD, emphysema, chronic bronchitis, asthma (persistent), ILD, IPF, pulmonary fibrosis, "on home O2".
**Négatif** : asthme intermittent léger, pneumonie aiguë isolée.

### 7. rheumd — Maladie rhumatismale / du tissu conjonctif
**Définition** : connectivite ou rhumatisme inflammatoire systémique.
**Positif** : RA, rheumatoid arthritis, SLE, lupus, scleroderma, polymyositis, dermatomyositis, mixed connective tissue disease, polymyalgia rheumatica.
**Négatif** : arthrose (OA), goutte, fibromyalgie.

### 8. pud — Ulcère gastro-duodénal
**Définition** : antécédent d'ulcère gastrique ou duodénal.
**Positif** : PUD, peptic ulcer, gastric ulcer, duodenal ulcer, "s/p GI bleed from ulcer".
**Négatif** : gastrite, GERD, reflux sans ulcère documenté.

### 9. mld — Hépatopathie légère
**Définition** : hépatite chronique ou cirrhose compensée.
**Positif** : chronic hepatitis B/C, HCV, HBV, fatty liver, NASH, NAFLD, cirrhosis (sans complication), "elevated LFTs chronically".
**Négatif** : hépatite aiguë résolue, stéatose simple à l'imagerie sans diagnostic.
> **⚠ Point à trancher** : "NAFLD" isolé → inclure comme mld ?

### 10. diab — Diabète sans complication
**Définition** : diabète de type 1 ou 2 sans atteinte d'organe.
**Positif** : DM, DM2, DM1, diabetes mellitus, "on insulin", "on metformin", IDDM, NIDDM.
**Négatif** : prédiabète, diabète gestationnel résolu.
**Attention** : si une complication est mentionnée → cocher **diabwc** à la place (pas les deux, sauf doute).

### 11. diabwc — Diabète avec complication
**Définition** : diabète avec atteinte d'organe cible.
**Positif** : diabetic nephropathy, diabetic retinopathy, diabetic neuropathy, diabetic foot, "DM2 with neuropathy", "DM c/b nephropathy".
**Règle** : si diabwc est coché, ne pas cocher diab (la hiérarchie s'applique au score, mais pour l'annotation on coche ce qu'on voit dans le texte).
> **⚠ Point à trancher** : si le texte dit "DM2" ET "CKD" sans lien explicite → cocher diabwc ou seulement diab + rend ?

### 12. hp — Hémiplégie / paraplégie
**Définition** : paralysie d'un ou plusieurs membres.
**Positif** : hemiplegia, hemiparesis, paraplegia, quadriplegia, tetraplegia, "s/p spinal cord injury with paraplegia".
**Négatif** : faiblesse légère transitoire, parésie faciale isolée.

### 13. rend — Insuffisance rénale
**Définition** : insuffisance rénale chronique modérée à terminale.
**Positif** : CKD (stage 3–5), ESRD, dialysis, HD, PD, renal transplant, "Cr baseline 2.5", chronic renal failure/insufficiency.
**Négatif** : IRA résolue, CKD stage 1–2 isolée.
> **⚠ Point à trancher** : CKD stage 2 → exclure selon Quan ; CKD stage 3 → inclure. "CKD" sans stade précisé ?

### 14. canc — Cancer (tumeur solide, hémopathie maligne)
**Définition** : cancer actif ou traité dans les 5 dernières années, hors carcinome cutané non mélanome.
**Positif** : breast cancer, lung cancer, colon cancer, lymphoma, leukemia, "s/p chemotherapy for …", "h/o prostate cancer (___)", myeloma.
**Négatif** : BCC (basal cell carcinoma), SCC cutané, cancer guéri > 5 ans ("remote history of cancer in ___" avec date ancienne).
**Attention** : si métastases → cocher **aussi metacanc**.
> **⚠ Point à trancher** : "h/o cancer" sans date ni précision → inclure par défaut ?

### 15. msld — Hépatopathie modérée à sévère
**Définition** : cirrhose décompensée ou complications hépatiques sévères.
**Positif** : decompensated cirrhosis, esophageal varices, ascites (d'origine hépatique), hepatic encephalopathy, "MELD score", portal hypertension.
**Règle** : si msld coché, mld est redondant (la hiérarchie s'applique).

### 16. metacanc — Cancer métastatique
**Définition** : tumeur solide avec métastases à distance.
**Positif** : metastatic, stage IV, "mets to liver/bone/lung", carcinomatosis.
**Règle** : toujours cocher **canc** en même temps.

### 17. aids — SIDA
**Définition** : infection VIH au stade SIDA (infection opportuniste ou CD4 < 200).
**Positif** : AIDS, "HIV/AIDS", "HIV with PCP", "CD4 < 200", "h/o opportunistic infection".
**Négatif** : "HIV" seul, bien contrôlé sous ARV, "HIV with undetectable viral load" sans antécédent d'infection opportuniste.
> **⚠ Point à trancher** : "HIV" seul sans précision de stade → exclure par défaut (Charlson compte le SIDA, pas le VIH) ?

---

## Points à trancher (à compléter par l'annotateur)

| # | Question | Décision |
|---|----------|----------|
| 1 | Diastolic dysfunction sans diagnostic d'IC → chf ? | |
| 2 | Sténose carotidienne asymptomatique → cevd ? | |
| 3 | MCI → dementia ? | |
| 4 | NAFLD isolé → mld ? | |
| 5 | DM2 + CKD sans lien explicite → diabwc ? | |
| 6 | CKD sans stade → rend ? | |
| 7 | Cancer sans date → canc ? | |
| 8 | HIV seul sans stade SIDA → aids ? | |
