"""8 fake discharge notes for testing extract_pmh."""

# 1 — Format standard avec Past Medical History:
NOTE_STANDARD = (
    "Name: ___ Unit No: ___ Admission Date: ___ Discharge Date: ___\n"
    "Chief Complaint: Chest pain\n"
    "\n"
    "Past Medical History:\n"
    "- Hypertension\n"
    "- Type 2 diabetes mellitus\n"
    "- Hyperlipidemia\n"
    "\n"
    "Social History:\n"
    "Lives alone, former smoker.\n"
)

# 2 — Variante PMH:
NOTE_PMH_SHORT_HEADER = (
    "Name: ___ Admission Date: ___\n"
    "PMH:\n"
    "1. COPD\n"
    "2. Atrial fibrillation\n"
    "3. CKD stage III\n"
    "\n"
    "Family History:\n"
    "Mother with breast cancer.\n"
)

# 3 — Variante Past Medical/Surgical History:
NOTE_MED_SURG = (
    "Chief Complaint: Shortness of breath\n"
    "\n"
    "Past Medical/Surgical History:\n"
    "Appendectomy (___), cholecystectomy (___)\n"
    "Hypertension, gout\n"
    "\n"
    "Medications on Admission:\n"
    "Lisinopril 10 mg daily\n"
)

# 4 — Section absente
NOTE_NO_PMH = (
    "Name: ___ Unit No: ___\n"
    "Chief Complaint: Fall\n"
    "Social History:\n"
    "Retired, lives with spouse.\n"
    "Physical Exam:\n"
    "WNL\n"
)

# 5 — PMH en dernière section (fin de note)
NOTE_PMH_LAST = (
    "Name: ___\n"
    "Discharge Diagnosis: Pneumonia\n"
    "\n"
    "Past Medical History:\n"
    "- Asthma\n"
    "- Obesity\n"
    "- Depression\n"
)

# 6 — Liste numérotée
NOTE_NUMBERED_LIST = (
    "Chief Complaint: Syncope\n"
    "\n"
    "Past Medical History:\n"
    "1. Coronary artery disease s/p CABG ___\n"
    "2. Heart failure with reduced ejection fraction (EF 30%)\n"
    "3. Paroxysmal atrial fibrillation on warfarin\n"
    "4. Type 2 diabetes mellitus\n"
    "5. Chronic kidney disease stage IV\n"
    "\n"
    "Physical Exam:\n"
    "Vitals: T 98.6 HR 78 BP 130/80\n"
)

# 7 — PMH très courte : "none"
NOTE_PMH_NONE = (
    "Chief Complaint: Laceration\n"
    "\n"
    "Past Medical History:\n"
    "none\n"
    "\n"
    "Allergies:\n"
    "NKDA\n"
)

# 8 — PMH sur la même ligne que l'en-tête
NOTE_PMH_INLINE = (
    "Name: ___\n"
    "PMH: HTN, DM2, HLD\n"
    "Social History:\n"
    "Non-smoker.\n"
)

# 9 — En-tête PMH suivi directement d'un autre en-tête (PMH vide)
NOTE_PMH_EMPTY = (
    "Chief Complaint: Fever\n"
    "\n"
    "Past Medical History:\n"
    "Social History:\n"
    "Drinks socially.\n"
)

# 10 — Variante "Past Medical and Surgical History:"
NOTE_MED_AND_SURG = (
    "Chief Complaint: Abdominal pain\n"
    "\n"
    "Past Medical and Surgical History:\n"
    "- Diverticulosis\n"
    "- Hernia repair ___\n"
    "\n"
    "Brief Hospital Course:\n"
    "Patient was admitted for observation.\n"
)

# 11 — Variante "Medical History:"
NOTE_MEDICAL_HISTORY = (
    "Chief Complaint: Headache\n"
    "\n"
    "Medical History:\n"
    "Migraines, anxiety disorder\n"
    "\n"
    "Review of Systems:\n"
    "Negative except as above.\n"
)
