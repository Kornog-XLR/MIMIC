import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# --- 1. CHARGEMENT ET EXTRACTION ---
df = pd.read_csv("data/discharge_notes.csv")
texte_brut = df.loc[0, "text"]

# Extraction du texte entre "Past Medical History:" et "Social History:"
pattern_medical = r"Past Medical History:(.*?)Social History:"
historique = re.search(pattern_medical, texte_brut, re.DOTALL)
texte_a_predire = historique.group(1).strip() if historique else ""

# --- 2. CHARGEMENT DU MODÈLE ---
base_model_name = "emilyalsentzer/Bio_ClinicalBERT"
peft_model_id = "cja5553/Bio_ClinicalBERT_MIMIC_IV_ICU_stay_more_than_1_day_prediction_IA3_ti"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
model = PeftModel.from_pretrained(base_model, peft_model_id)

# --- 3. PRÉDICTION ---
# Mission A : Tokenize la variable contenant le texte pour générer des tenseurs PyTorch ("pt")
inputs = tokenizer(texte_a_predire, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = base_model(**inputs)
    logits = outputs.logits
    
    # Mission B : Applique la fonction softmax sur les logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Mission C : Récupère l'index de la classe ayant la plus forte probabilité
    predicted_class = torch.argmax(probabilities, dim=-1).item()

print(f"Probabilités : {probabilities}")
print(f"Classe prédite : {predicted_class}")