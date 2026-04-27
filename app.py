import base64
import json
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="SLE Screening | KMITL", layout="wide",
                   initial_sidebar_state="collapsed")

# ── i18n strings ──────────────────────────────────────────────────────────────
S = {
    "en": {
        "title":            "SLE Screening Tool",
        "screening_level":  "Screening level",
        "tests_required":   "Tests required",
        "validated_sens":   "Validated sensitivity",
        "select_header":    "Select present criteria",
        "select_one":       "Select at least one criterion to calculate probability.",
        "calc_btn":         "Calculate SLE Probability",
        "calc_eular_btn":   "Calculate EULAR/ACR Score",
        "ci_range":         "95% confidence range",
        "high_prob":        "High probability. Refer for full immunological workup (ANA, anti-dsDNA, complement levels).",
        "mid_prob":         "Intermediate probability. Consider specialist consultation.",
        "low_prob":         "Low probability based on selected criteria.",
        "selected":         "Selected criteria",
        "domain":           "Domain",
        "requires":         "Requires",
        "score_if_present": "Score if present",
        "eular_score":      "EULAR/ACR Score",
        "sle_met":          "SLE classification criteria met (score ≥ 10)",
        "sle_not_met":      "SLE classification criteria not met (score < 10)",
        "score_breakdown":  "Score by domain",
        "ana_label":        "ANA entry criterion",
        "ana_met":          "Met (ANA ≥ 1:80)",
        "ana_not_met":      "Not met — formal EULAR/ACR 2019 classification requires ANA ≥ 1:80",
        "disclaimer_suffix":"This tool does not replace clinical judgement or formal diagnostic workup.",
    },
    "th": {
        "title":            "ชุดคัดกรองโรคพุ่มพวง",
        "screening_level":  "ระดับการคัดกรอง",
        "tests_required":   "การตรวจที่ต้องการ",
        "validated_sens":   "ความไวที่ผ่านการตรวจสอบ",
        "select_header":    "เลือกเกณฑ์ที่ตรวจพบ",
        "select_one":       "กรุณาเลือกอย่างน้อย 1 เกณฑ์เพื่อคำนวณ",
        "calc_btn":         "คำนวณความน่าจะเป็นโรค SLE",
        "calc_eular_btn":   "คำนวณคะแนน EULAR/ACR",
        "ci_range":         "ช่วงความเชื่อมั่น 95%",
        "high_prob":        "ความน่าจะเป็นสูง ควรส่งตรวจภูมิคุ้มกัน (ANA, anti-dsDNA, ระดับ complement)",
        "mid_prob":         "ความน่าจะเป็นปานกลาง ควรปรึกษาแพทย์ผู้เชี่ยวชาญ",
        "low_prob":         "ความน่าจะเป็นต่ำตามเกณฑ์ที่เลือก",
        "selected":         "เกณฑ์ที่เลือก",
        "domain":           "หมวด",
        "requires":         "ต้องการ",
        "score_if_present": "คะแนนถ้าพบ",
        "eular_score":      "คะแนน EULAR/ACR",
        "sle_met":          "ตรงตามเกณฑ์ SLE (คะแนน ≥ 10)",
        "sle_not_met":      "ไม่ตรงตามเกณฑ์ SLE (คะแนน < 10)",
        "score_breakdown":  "คะแนนแยกตามหมวด",
        "ana_label":        "เกณฑ์เริ่มต้น ANA",
        "ana_met":          "ผ่าน (ANA ≥ 1:80)",
        "ana_not_met":      "ไม่ผ่าน — การจำแนกตาม EULAR/ACR 2019 ต้องการ ANA ≥ 1:80",
        "disclaimer_suffix":"เครื่องมือนี้ไม่ทดแทนการตัดสินใจทางคลินิกหรือการวินิจฉัยโรค",
    },
}

LEVEL_OPTS = {
    "en": ["At-home test", "Clinical test", "Full EULAR/ACR 2019"],
    "th": ["การตรวจที่บ้าน", "การตรวจทางคลินิก", "EULAR/ACR 2019 ครบชุด"],
}
LEVEL_KEYS = ["d9", "d5", "full"]

LEVEL_META = {
    "d9": {
        "model_file": "model_d9.joblib",
        "sens":  {"en": "0.905", "th": "0.905"},
        "tests": {"en": "Urine dipstick (~฿50–150)", "th": "การตรวจปัสสาวะ dipstick (~฿50–150)"},
        "disclaimer": {
            "en": "7-feature at-home model (sensitivity 0.905). Requires urine dipstick only. Excludes immunological markers (ANA, anti-dsDNA, complement) required by the full EULAR/ACR 2019 criteria.",
            "th": "แบบจำลอง 7 เกณฑ์สำหรับตรวจที่บ้าน (ความไว 0.905) ต้องการเพียง dipstick ปัสสาวะ ไม่รวมตัวชี้วัดทางภูมิคุ้มกัน (ANA, anti-dsDNA, complement) ตามเกณฑ์ EULAR/ACR 2019 ครบชุด",
        },
    },
    "d5": {
        "model_file": "model_d5.joblib",
        "sens":  {"en": "0.955", "th": "0.955"},
        "tests": {"en": "CBC, Direct Coombs, ECG, CXR/Echo (~฿900–2,950)", "th": "CBC, Coombs, ECG, CXR/Echo (~฿900–2,950)"},
        "disclaimer": {
            "en": "15-feature clinical model (sensitivity 0.955). Requires blood tests and imaging. Excludes immunological markers (ANA, anti-dsDNA, complement) required by the full EULAR/ACR 2019 criteria.",
            "th": "แบบจำลอง 15 เกณฑ์สำหรับการตรวจทางคลินิก (ความไว 0.955) ต้องการการตรวจเลือดและการตรวจภาพ ไม่รวมตัวชี้วัดทางภูมิคุ้มกันตามเกณฑ์ EULAR/ACR 2019 ครบชุด",
        },
    },
    "full": {
        "model_file": None,
        "sens":  {"en": "Rule-based (no model)", "th": "กฎ-based (ไม่มีแบบจำลอง)"},
        "tests": {"en": "Full immunology panel: ANA, anti-dsDNA, Anti-Sm, complement, APL, renal biopsy",
                  "th": "การตรวจภูมิคุ้มกันครบชุด: ANA, anti-dsDNA, Anti-Sm, complement, APL, ชิ้นเนื้อไต"},
        "disclaimer": {
            "en": "Full 2019 EULAR/ACR rule-based score (≥10 = SLE). Requires complete immunological workup.",
            "th": "คะแนนตามเกณฑ์ 2019 EULAR/ACR ครบชุด (≥10 = SLE) ต้องการการตรวจภูมิคุ้มกันครบชุด",
        },
    },
}

EULAR_DOMAINS = {
    "Constitutional":           {"Fever": 2},
    "Mucocutaneous":            {"ACL": 6, "SCL or DL": 4, "Oral Ulcer": 2, "Alopecia": 2},
    "Musculoskeletal":          {"Joint involvement": 6},
    "Serosal":                  {"Acute pericarditis": 6, "Pleural or pericardial effusion": 5},
    "Renal":                    {"Proteinuria": 4, "Renal class III or IV LN": 8, "Renal class II or V LN": 4},
    "Neuropsychiatric":         {"Delirium": 2, "Psychosis": 3, "Seizure": 5},
    "Hematologic":              {"Leukopenia": 3, "Thrombocytopenia": 4, "AIHA": 4},
    "Antiphospholipid Ab":      {"Antiphospholipid": 2},
    "Complement Proteins":      {"Low C3 or C4": 3, "Low C4 and C3": 4},
    "SLE-specific Antibodies":  {"Anti-dsDNA or Anti-Sm": 6},
}


@st.cache_resource
def load_model(level: str):
    return joblib.load(LEVEL_META[level]["model_file"])


@st.cache_data
def load_criteria():
    with open("criteria.json") as f:
        return json.load(f)


def predict_with_ci(model, x):
    fold_probs = np.array([
        cal_clf.predict_proba(x.reshape(1, -1))[0, 1]
        for cal_clf in model.calibrated_classifiers_
    ])
    p = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    sd = fold_probs.std()
    return p, max(0.0, p - 1.96 * sd), min(1.0, p + 1.96 * sd)


def format_prob(p):
    if p >= 0.99:
        return ">99%"
    if p <= 0.01:
        return "<1%"
    return f"{p:.1%}"


def compute_eular(values):
    total, breakdown = 0, {}
    for domain, items in EULAR_DOMAINS.items():
        pts = [v for k, v in items.items() if values.get(k, False)]
        domain_score = max(pts) if pts else 0
        breakdown[domain] = domain_score
        total += domain_score
    return total, breakdown


# ── Sidebar ───────────────────────────────────────────────────────────────────
all_criteria = load_criteria()

lang_choice = st.sidebar.radio("Language / ภาษา", ["English", "ภาษาไทย"])
lang = "th" if lang_choice == "ภาษาไทย" else "en"
t = S[lang]  # translation shortcut

st.sidebar.divider()
level_choice = st.sidebar.radio(
    t["screening_level"],
    LEVEL_OPTS[lang],
)
level = LEVEL_KEYS[LEVEL_OPTS[lang].index(level_choice)]
meta  = LEVEL_META[level]

st.sidebar.caption(f"**{t['tests_required']}:** {meta['tests'][lang]}")
if level != "full":
    st.sidebar.caption(f"**{t['validated_sens']}:** {meta['sens'][lang]}")

# ── Criteria filter ───────────────────────────────────────────────────────────
if level == "d9":
    criteria = [c for c in all_criteria if c["in_d9"]]
elif level == "d5":
    criteria = [c for c in all_criteria if c["in_d5"]]
else:
    criteria = [c for c in all_criteria if c["in_full"]]

entry_criterion = next((c for c in criteria if c.get("entry_criterion")), None)
scored_criteria  = [c for c in criteria if not c.get("entry_criterion")]
feature_order    = [c["key"] for c in scored_criteria]

# ── GitHub link (fixed top-right) ────────────────────────────────────────────
with open("icons/github.png", "rb") as _f:
    _gh_icon = base64.b64encode(_f.read()).decode()
st.markdown(
    f'<a href="https://github.com/MaiRajborirug/proj_sle" target="_blank" '
    f'style="position:fixed;bottom:20px;right:20px;z-index:9999;'
    f'background:#24292e;color:#fff;padding:6px 12px;border-radius:6px;'
    f'font-size:13px;font-weight:500;text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,0.3);">'
    f'<img src="data:image/png;base64,{_gh_icon}" style="height:16px;vertical-align:middle;margin-right:6px;"> GitHub</a>',
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
_, logo_col, qr_col, _ = st.columns([2, 5, 1, 2])
with logo_col:
    st.image("icons/md_kmitl.png", use_container_width=True)
with qr_col:
    st.image("icons/qr.png", use_container_width=True)

st.title(t["title"])
st.caption("Based on 2019 EULAR/ACR criteria · **Not a diagnosis** — for clinical decision support only.")
st.caption("version 1.2")

# ── ANA entry criterion (full EULAR only) ─────────────────────────────────────
ana_checked = False
if entry_criterion and level == "full":
    st.markdown(f"#### {t['ana_label']}")
    ana_checked = st.checkbox(
        f"**{entry_criterion['name_en']}**  *(required for formal classification)*",
        key="full_ANA",
    )
    if not ana_checked:
        st.warning(t["ana_not_met"])
    st.divider()

# ── Criteria form ─────────────────────────────────────────────────────────────
st.markdown(f"### {t['select_header']}")
values = {}
cols   = st.columns(2)

for i, c in enumerate(scored_criteria):
    label = c["name_th"] if lang == "th" else c["name_en"]
    test  = c.get("requires_test", "")
    pts_suffix = f"  `+{c['score']} pts`" if (level == "full" and c["score"]) else ""

    with cols[i % 2]:
        with st.container(border=True):
            icon_col, content_col, pop_col = st.columns([1, 6, 1])
            with icon_col:
                if c.get("icon"):
                    st.image(c["icon"], width=48)
            with content_col:
                values[c["key"]] = st.checkbox(
                    f"**{label}**{pts_suffix}",
                    key=f"{level}_{c['key']}",
                )
                caption = f"{t['domain']}: {c['domain']}"
                if test:
                    caption += f"  ·  {t['requires']}: {test}"
                st.caption(caption)
            with pop_col:
                with st.popover("•"):
                    st.markdown(f"**{c['name_en']}**  \n*{c['name_th']}*")
                    if test:
                        st.markdown(f"🔬 **{t['requires']}:** {test}")
                    st.markdown(f"> {c['description_en']}")
                    st.markdown(f"> {c['description_th']}")
                    if level == "full" and c["score"]:
                        st.markdown(f"**{t['score_if_present']}:** +{c['score']} pts (domain max applies)")
                    if c.get("med_image"):
                        st.image(c["med_image"])

st.divider()

# ── Inference / scoring ───────────────────────────────────────────────────────
n_selected = sum(values.values())
if n_selected == 0:
    st.info(t["select_one"])

if level == "full":
    btn_label = t["calc_eular_btn"]
else:
    btn_label = t["calc_btn"]

if st.button(btn_label, type="primary", use_container_width=True, disabled=(n_selected == 0)):

    if level == "full":
        score, breakdown = compute_eular(values)
        sle_classified = score >= 10
        color = "#e74c3c" if sle_classified else "#27ae60"
        icon  = "🔴" if sle_classified else "🟢"
        st.markdown(
            f"<h2 style='color:{color}'>{icon} {t['eular_score']}: {score}</h2>",
            unsafe_allow_html=True,
        )
        if not ana_checked:
            st.warning(t["ana_not_met"])
        if sle_classified:
            st.error(t["sle_met"])
        else:
            st.success(t["sle_not_met"])
        with st.expander(t["score_breakdown"]):
            for domain, pts in breakdown.items():
                if pts > 0:
                    st.markdown(f"- **{domain}**: +{pts}")
                else:
                    st.markdown(f"- {domain}: 0")

    else:
        model = load_model(level)
        x = np.array([int(values[k]) for k in feature_order], dtype=float)
        p, lo, hi = predict_with_ci(model, x)

        if p >= 0.7:
            color, icon = "#e74c3c", "🔴"
        elif p >= 0.3:
            color, icon = "#f39c12", "🟡"
        else:
            color, icon = "#27ae60", "🟢"

        st.markdown(
            f"<h2 style='color:{color}'>{icon} SLE Probability: {format_prob(p)}</h2>",
            unsafe_allow_html=True,
        )
        st.progress(float(p))
        st.caption(f"{t['ci_range']}: {format_prob(lo)} – {format_prob(hi)}")

        if p >= 0.7:
            st.error(t["high_prob"])
        elif p >= 0.3:
            st.warning(t["mid_prob"])
        else:
            st.success(t["low_prob"])

        checked = [c["name_en"] for c in scored_criteria if values[c["key"]]]
        if checked:
            with st.expander(t["selected"]):
                for name in checked:
                    st.markdown(f"- {name}")

st.divider()
st.caption(
    f"⚠️ **Disclaimer:** {meta['disclaimer'][lang]} "
    f"{t['disclaimer_suffix']}"
)
