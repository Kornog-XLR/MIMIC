"""Streamlit app for blind annotation of gold-set PMH sections."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from src.annotation_io import (
    CATEGORIES,
    load_annotations,
    save_annotation,
    find_resume_index,
)

# ---------------------------------------------------------------------------
# Category labels and tooltips (French)
# ---------------------------------------------------------------------------
LABELS_FR = {
    "mi":        ("Infarctus du myocarde", "IDM, STEMI, NSTEMI, MI, infarctus, troponine positive avec nécrose"),
    "chf":       ("Insuffisance cardiaque", "IC, CHF, HF, FEVG abaissée, HFrEF, HFpEF, décompensation cardiaque"),
    "pvd":       ("Maladie vasculaire périphérique", "AOMI, PVD, claudication, pontage fémoral, anévrisme aortique"),
    "cevd":      ("Maladie cérébrovasculaire", "AVC, AIT, TIA, stroke, carotid stenosis, hémiplégie vasculaire"),
    "dementia":  ("Démence", "Alzheimer, démence vasculaire, démence à corps de Lewy, troubles cognitifs majeurs"),
    "cpd":       ("Maladie pulmonaire chronique", "BPCO, COPD, asthme persistant, emphysème, fibrose pulmonaire"),
    "rheumd":    ("Maladie rhumatismale", "PR, lupus, LED, polyarthrite rhumatoïde, sclérodermie, polymyosite"),
    "pud":       ("Ulcère gastro-duodénal", "UGD, PUD, ulcère gastrique, ulcère duodénal"),
    "mld":       ("Hépatopathie légère", "Hépatite B/C chronique, cirrhose compensée, stéatose, NASH"),
    "diab":      ("Diabète sans complication", "DT1, DT2, DM, diabète — SANS atteinte d'organe"),
    "diabwc":    ("Diabète avec complication", "Néphropathie, rétinopathie, neuropathie diabétique, pied diabétique"),
    "hp":        ("Hémiplégie / paraplégie", "Hémiplégie, paraplégie, tétraplégie, paralysie d'un membre"),
    "rend":      ("Insuffisance rénale", "IRC, CKD, dialyse, créatinine élevée chronique, greffe rénale"),
    "canc":      ("Cancer (solide, sans métastase)", "Tumeur solide traitée < 5 ans, lymphome, leucémie — hors peau non mélanome"),
    "msld":      ("Hépatopathie modérée à sévère", "Cirrhose décompensée, varices œsophagiennes, ascite, encéphalopathie hépatique"),
    "metacanc":  ("Cancer métastatique", "Métastases, cancer stade IV, carcinose péritonéale — cocher aussi canc"),
    "aids":      ("SIDA", "HIV/SIDA avec infection opportuniste, CD4 < 200, stade C"),
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLD_PATH = "data/processed/gold_to_annotate.parquet"
ANNOTATIONS_PATH = "data/processed/gold_annotations.csv"


def main():
    st.set_page_config(page_title="Annotation PMH – Charlson", layout="wide")
    st.title("Annotation des PMH — Catégories de Charlson")

    # Load data
    try:
        gold = pd.read_parquet(GOLD_PATH)
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {GOLD_PATH}. Lancez d'abord prepare_gold.py.")
        return

    n_total = len(gold)
    annotations = load_annotations(ANNOTATIONS_PATH)
    n_done = len(annotations)

    # Initialize session state
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = find_resume_index(annotations, gold)

    idx = st.session_state.current_idx

    # Progress
    st.progress(min(n_done / n_total, 1.0))
    st.caption(f"{n_done} / {n_total} annotés")

    if idx >= n_total:
        st.success("Toutes les PMH ont été annotées !")
        return

    # Current stay
    row = gold.iloc[idx]
    hadm_id = int(row["hadm_id"])
    pmh_text = row["pmh_text"] if pd.notna(row["pmh_text"]) else "(vide)"

    # Navigation
    col_prev, col_num, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Précédent", disabled=(idx == 0)):
            st.session_state.current_idx = max(0, idx - 1)
            st.rerun()
    with col_num:
        jump = st.number_input(
            "Aller à", min_value=1, max_value=n_total,
            value=idx + 1, step=1, label_visibility="collapsed",
        )
        if jump - 1 != idx:
            st.session_state.current_idx = jump - 1
            st.rerun()
    with col_next:
        st.write(f"**{idx + 1} / {n_total}**")

    st.divider()

    # PMH text
    st.subheader(f"PMH (séjour #{row['ordre']})")
    st.markdown(
        f'<div style="font-size:1.1em; line-height:1.6; white-space:pre-wrap; '
        f'background:#f8f9fa; padding:1em; border-radius:8px;">{pmh_text}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Load existing annotation for this hadm_id (if correcting)
    existing = annotations[annotations["hadm_id"] == hadm_id]
    has_existing = len(existing) > 0
    existing_row = existing.iloc[0] if has_existing else None

    # Checkboxes
    st.subheader("Catégories de Charlson")
    flags = {}
    cols = st.columns(3)
    for i, cat in enumerate(CATEGORIES):
        label, tooltip = LABELS_FR[cat]
        default = bool(existing_row[cat]) if has_existing else False
        with cols[i % 3]:
            flags[cat] = int(st.checkbox(f"**{cat}** — {label}", value=default, help=tooltip))

    st.divider()

    # Uncertain + comment
    col_inc, col_com = st.columns([1, 3])
    with col_inc:
        incertain = st.checkbox(
            "Incertain",
            value=bool(existing_row["incertain"]) if has_existing else False,
        )
    with col_com:
        commentaire = st.text_input(
            "Commentaire",
            value=str(existing_row["commentaire"]) if has_existing and pd.notna(existing_row["commentaire"]) else "",
        )

    # Save and advance
    if st.button("Enregistrer et suivant →", type="primary"):
        save_annotation(ANNOTATIONS_PATH, hadm_id, flags, incertain, commentaire)
        st.session_state.current_idx = min(idx + 1, n_total)
        st.rerun()


if __name__ == "__main__":
    main()
