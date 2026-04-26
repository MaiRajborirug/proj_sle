import json
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="SLE Screening | KMITL", layout="wide")

FEATURE_ORDER = [
    'Fever', 'ACL', 'SCL or DL', 'Oral Ulcer', 'Alopecia',
    'Joint involvement', 'Acute pericarditis',
    'Pleural or pericardial effusion', 'Proteinuria',
    'Delirium', 'Psychosis', 'Seizure',
    'Leukopenia', 'Thrombocytopenia', 'AIHA',
]


@st.cache_resource
def load_model():
    return joblib.load('model.joblib')


@st.cache_data
def load_criteria():
    with open('criteria.json') as f:
        return json.load(f)


def predict_with_ci(model, x):
    # pull the underlying RF from the calibrated wrapper
    rf = model.calibrated_classifiers_[0].estimator
    tree_probs = np.array([
        t.predict_proba(x.reshape(1, -1))[0, 1] for t in rf.estimators_
    ])
    p = tree_probs.mean()
    sd = tree_probs.std()
    return p, max(0.0, p - 1.96 * sd), min(1.0, p + 1.96 * sd)


# ── Load resources ────────────────────────────────────────────────────────────
model = load_model()
criteria = load_criteria()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("SLE Screening Tool")
st.caption("Based on 2019 EULAR/ACR criteria (limited feature set) · **Not a diagnosis** — for clinical decision support only.")

lang = st.sidebar.radio("Language / ภาษา", ["English", "ภาษาไทย"])
use_th = lang == "ภาษาไทย"

st.markdown("### Select present criteria / เลือกเกณฑ์ที่ตรวจพบ")

values = {}
cols = st.columns(2)

for i, c in enumerate(criteria):
    label = c["name_th"] if use_th else c["name_en"]
    desc = c["description_th"] if use_th else c["description_en"]
    domain = c["domain"]
    score = c["score"]

    with cols[i % 2]:
        with st.container(border=True):
            row = st.columns([6, 1])
            with row[0]:
                values[c["key"]] = st.checkbox(
                    f"**{label}**  `+{score} pts`",
                    key=c["key"],
                )
                st.caption(f"Domain: {domain}")
            with row[1]:
                with st.popover("ℹ️"):
                    st.markdown(f"**{c['name_en']}**  \n*{c['name_th']}*")
                    st.markdown(f"> {c['description_en']}")
                    st.markdown(f"> {c['description_th']}")
                    st.markdown(f"**Score if present:** +{score} pts (domain max applies)")

st.divider()

if st.button("Calculate SLE Probability", type="primary", use_container_width=True):
    x = np.array([int(values[k]) for k in FEATURE_ORDER], dtype=float)
    p, lo, hi = predict_with_ci(model, x)

    if p >= 0.7:
        color, icon = "#e74c3c", "🔴"
    elif p >= 0.3:
        color, icon = "#f39c12", "🟡"
    else:
        color, icon = "#27ae60", "🟢"

    st.markdown(
        f"<h2 style='color:{color}'>{icon} SLE Probability: {p:.1%}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(float(p))
    st.caption(f"95% confidence range: {lo:.0%} – {hi:.0%}")

    if p >= 0.7:
        st.error("High probability. Refer for full immunological workup (ANA, anti-dsDNA, complement levels).")
    elif p >= 0.3:
        st.warning("Intermediate probability. Consider specialist consultation.")
    else:
        st.success("Low probability based on selected criteria.")

    # Which criteria were checked
    checked = [c["name_en"] for c in criteria if values[c["key"]]]
    if checked:
        with st.expander("Selected criteria"):
            for name in checked:
                st.markdown(f"- {name}")

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is a screening aid only. "
    "The model uses a limited subset of EULAR/ACR 2019 criteria and excludes immunological markers. "
    "It does not replace clinical judgement or formal diagnostic workup."
)
