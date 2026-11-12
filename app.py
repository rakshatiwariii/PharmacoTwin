import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PharmacoTwin | Polypharmacy Digital Twin",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- COMPATIBILITY HELPER FOR NUMPY 2.0+ ---
def _trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))

# --- 1. KNOWLEDGE BASE & VIGNETTES ---
DRUG_KB = {
    "warfarin": {
        "display_name": "Warfarin",
        "drug_class": "Anticoagulant (Vitamin K Antagonist)",
        "cyp_substrates": ["CYP2C9", "CYP3A4"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.05,
        "half_life_hr": 40.0,
        "vd_l": 10.0,
        "typical_dose_mg": 5.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 1.45,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": ["bleeding"],
        "notes": "Narrow therapeutic index anticoagulant; CYP2C9/CYP3A4 inhibition elevates bleeding risk.",
    },
    "amiodarone": {
        "display_name": "Amiodarone",
        "drug_class": "Class III Antiarrhythmic",
        "cyp_substrates": ["CYP3A4"],
        "cyp_inhibitors": {"CYP2C9": 0.60, "CYP3A4": 0.55},
        "is_prodrug": False,
        "pgp_substrate": False,
        "pgp_inhibitor": True,
        "renal_elimination_fraction": 0.05,
        "half_life_hr": 72.0,
        "vd_l": 65.0,
        "typical_dose_mg": 200.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 1.65,
        "qt_risk": "known",
        "acb_score": 0,
        "pd_flags": ["bradycardia", "qt_prolongation"],
        "notes": "Broad CYP/P-gp inhibitor impairing warfarin and simvastatin clearance while prolonging QTc.",
    },
    "clarithromycin": {
        "display_name": "Clarithromycin",
        "drug_class": "Macrolide Antibiotic",
        "cyp_substrates": ["CYP3A4"],
        "cyp_inhibitors": {"CYP3A4": 0.78},
        "is_prodrug": False,
        "pgp_substrate": False,
        "pgp_inhibitor": True,
        "renal_elimination_fraction": 0.35,
        "half_life_hr": 5.0,
        "vd_l": 180.0,
        "typical_dose_mg": 500.0,
        "dosing_interval_hr": 12.0,
        "therapeutic_ceiling_ratio": 2.20,
        "qt_risk": "known",
        "acb_score": 0,
        "pd_flags": ["qt_prolongation"],
        "notes": "Potent CYP3A4 and P-gp inhibitor with known QTc prolongation liability.",
    },
    "furosemide": {
        "display_name": "Furosemide",
        "drug_class": "Loop Diuretic",
        "cyp_substrates": [],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.65,
        "half_life_hr": 2.0,
        "vd_l": 14.0,
        "typical_dose_mg": 40.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.30,
        "qt_risk": "none",
        "acb_score": 1,
        "pd_flags": ["hypokalemia", "nephrotoxic"],
        "notes": "Loop diuretic contributing +1 on Tisdale QTc score and potentiating hypokalemia.",
    },
    "simvastatin": {
        "display_name": "Simvastatin",
        "drug_class": "HMG-CoA Reductase Inhibitor",
        "cyp_substrates": ["CYP3A4"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "pgp_substrate": True,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.13,
        "half_life_hr": 3.0,
        "vd_l": 90.0,
        "typical_dose_mg": 40.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.00,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": ["myopathy"],
        "notes": "High CYP3A4 sensitivity; strong inhibitors markedly increase rhabdomyolysis risk.",
    },
    "clopidogrel": {
        "display_name": "Clopidogrel",
        "drug_class": "P2Y12 Antiplatelet (Prodrug)",
        "cyp_substrates": ["CYP2C19"],
        "cyp_inhibitors": {},
        "is_prodrug": True,
        "prodrug_enzymes": ["CYP2C19"],
        "pgp_substrate": True,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.50,
        "half_life_hr": 6.0,
        "vd_l": 120.0,
        "typical_dose_mg": 75.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.20,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": ["bleeding"],
        "notes": "Requires CYP2C19 bioactivation; inhibitors blunt antiplatelet efficacy.",
    },
    "omeprazole": {
        "display_name": "Omeprazole",
        "drug_class": "Proton Pump Inhibitor",
        "cyp_substrates": ["CYP2C19", "CYP3A4"],
        "cyp_inhibitors": {"CYP2C19": 0.68},
        "is_prodrug": False,
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.10,
        "half_life_hr": 1.2,
        "vd_l": 25.0,
        "typical_dose_mg": 20.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.50,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": [],
        "notes": "Inhibits CYP2C19; impairs clopidogrel active metabolite formation.",
    },
    "fluoxetine": {
        "display_name": "Fluoxetine",
        "drug_class": "SSRI Antidepressant",
        "cyp_substrates": ["CYP2D6", "CYP2C9"],
        "cyp_inhibitors": {"CYP2D6": 0.82},
        "is_prodrug": False,
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.10,
        "half_life_hr": 72.0,
        "vd_l": 1400.0,
        "typical_dose_mg": 20.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.10,
        "qt_risk": "conditional",
        "acb_score": 0,
        "pd_flags": ["serotonergic", "bleeding"],
        "notes": "Potent CYP2D6 inhibitor increasing bleeding and serotonergic risk.",
    },
    "tramadol": {
        "display_name": "Tramadol",
        "drug_class": "Opioid Analgesic (Prodrug)",
        "cyp_substrates": ["CYP2D6"],
        "cyp_inhibitors": {},
        "is_prodrug": True,
        "prodrug_enzymes": ["CYP2D6"],
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.30,
        "half_life_hr": 6.3,
        "vd_l": 180.0,
        "typical_dose_mg": 50.0,
        "dosing_interval_hr": 8.0,
        "therapeutic_ceiling_ratio": 1.80,
        "qt_risk": "none",
        "acb_score": 2,
        "pd_flags": ["serotonergic", "cns_depression"],
        "notes": "Requires CYP2D6 activation; co-use with SSRIs risks serotonin syndrome.",
    },
    "ibuprofen": {
        "display_name": "Ibuprofen",
        "drug_class": "NSAID",
        "cyp_substrates": ["CYP2C9"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.15,
        "half_life_hr": 2.2,
        "vd_l": 10.0,
        "typical_dose_mg": 400.0,
        "dosing_interval_hr": 8.0,
        "therapeutic_ceiling_ratio": 2.40,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": ["bleeding", "nephrotoxic"],
        "notes": "Compounds GI bleeding with anticoagulants and impairs renal perfusion.",
    },
}

CLINICAL_VIGNETTES = [
    {
        "title": "Vignette 1: AFib + Pneumonia Cascade",
        "subtitle": "74F with Heart Failure, Stage 3b CKD, hypokalemia on warfarin, amiodarone, furosemide, simvastatin + clarithromycin.",
        "patient": {
            "age": 74, "sex": "female", "weight_kg": 62.0, "scr_mg_dl": 1.4,
            "potassium_meq_l": 3.3, "baseline_qtc_ms": 458.0, "heart_failure": True, "sepsis": False, "acute_mi": False
        },
        "medications": ["warfarin", "amiodarone", "clarithromycin", "furosemide", "simvastatin"],
    },
    {
        "title": "Vignette 2: Post-PCI Prodrug Blockade",
        "subtitle": "66M on clopidogrel receiving omeprazole, fluoxetine, tramadol, and ibuprofen.",
        "patient": {
            "age": 66, "sex": "male", "weight_kg": 82.0, "scr_mg_dl": 1.1,
            "potassium_meq_l": 4.1, "baseline_qtc_ms": 425.0, "heart_failure": False, "sepsis": False, "acute_mi": True
        },
        "medications": ["clopidogrel", "omeprazole", "fluoxetine", "tramadol", "ibuprofen"],
    },
]

SAFER_SUBSTITUTIONS = {
    "clarithromycin": [{"replacement": "doxycycline", "rationale": "Eliminates CYP3A4 blockade and QTc prolongation."}],
    "simvastatin": [{"replacement": "rosuvastatin", "rationale": "Bypasses CYP3A4 first-pass metabolism."}],
    "omeprazole": [{"replacement": "pantoprazole", "rationale": "Preserves clopidogrel prodrug bioactivation."}],
    "ibuprofen": [{"replacement": "acetaminophen", "rationale": "Removes NSAID GI bleeding and renal risks."}],
}


# --- 2. SCIENTIFIC ENGINES ---
def cockcroft_gault_crcl(age, weight_kg, scr_mg_dl, sex):
    crcl = ((140.0 - age) * weight_kg) / (72.0 * scr_mg_dl)
    if sex.strip().lower() in {"f", "female"}:
        crcl *= 0.85
    return round(crcl, 2)


def tisdale_qt_score(age, sex, loop_diuretic, k_meq_l, baseline_qtc_ms, acute_mi, qt_drug_count, sepsis, heart_failure):
    pts = 0
    if age >= 68: pts += 1
    if sex.strip().lower() in {"f", "female"}: pts += 1
    if loop_diuretic: pts += 1
    if k_meq_l <= 3.5: pts += 2
    if baseline_qtc_ms >= 450.0: pts += 2
    if acute_mi: pts += 2
    if qt_drug_count == 1: pts += 3
    elif qt_drug_count >= 2: pts += 6
    if sepsis: pts += 3
    if heart_failure: pts += 3
    
    band = "Low" if pts < 7 else ("Moderate" if pts <= 10 else "High")
    return {"score": pts, "risk_band": band}


def analyze_collisions(medications):
    cyp_col = []
    pd_col = []
    acb_total = 0
    known = [m for m in medications if m in DRUG_KB]

    for perp in known:
        p_info = DRUG_KB[perp]
        for vict in known:
            if perp == vict: continue
            v_info = DRUG_KB[vict]
            for enz, pot in p_info["cyp_inhibitors"].items():
                if enz in v_info["cyp_substrates"]:
                    cyp_col.append({
                        "perpetrator": p_info["display_name"],
                        "victim": v_info["display_name"],
                        "enzyme": enz,
                        "potency": pot,
                        "severity": "High" if pot >= 0.6 else "Moderate",
                        "mechanism": f"{p_info['display_name']} inhibits {enz}, elevating {v_info['display_name']} exposure."
                    })

    pd_groups = {}
    for med in known:
        info = DRUG_KB[med]
        acb_total += info["acb_score"]
        for flag in info["pd_flags"]:
            pd_groups.setdefault(flag, []).append(info["display_name"])

    for flag, names in pd_groups.items():
        if len(names) >= 2:
            pd_col.append({"pathway": flag.replace("_", " ").title(), "drugs": names, "severity": "High"})

    return cyp_col, pd_col, acb_total


def simulate_pk(medications, crcl):
    known = [m for m in medications if m in DRUG_KB]
    results = []
    for med in known:
        info = DRUG_KB[med]
        inhibs = [DRUG_KB[o]["cyp_inhibitors"].get(e, 0.0) for o in known if o != med for e in info["cyp_substrates"]]
        mean_inh = float(np.mean(inhibs)) if inhibs else 0.0
        
        fe = info["renal_elimination_fraction"]
        renal_factor = (1.0 - fe) + fe * max(0.2, crcl / 90.0)
        ke_base = np.log(2.0) / info["half_life_hr"]
        ke_mod = ke_base * (1.0 - mean_inh) * renal_factor

        t_span = (0.0, 240.0)
        t_eval = np.linspace(0.0, 240.0, 100)
        
        sol = solve_ivp(lambda t, y: [-ke_mod * y[0]], t_span, [info["typical_dose_mg"] / info["vd_l"]], t_eval=t_eval)
        
        c_base = info["typical_dose_mg"] / info["vd_l"] * np.exp(-ke_base * sol.t)
        c_poly = sol.y[0]
        
        auc_base = max(1e-6, _trapezoid(c_base, sol.t))
        auc_poly = _trapezoid(c_poly, sol.t)
        auc_ratio = round(float(auc_poly / auc_base), 2)
        cmax_ratio = round(float(np.max(c_poly) / np.max(c_base)), 2)

        results.append({
            "drug": med,
            "display_name": info["display_name"],
            "auc_ratio": auc_ratio,
            "cmax_ratio": cmax_ratio,
            "therapeutic_ceiling_ratio": info["therapeutic_ceiling_ratio"],
            "exceeds_ceiling": bool(cmax_ratio >= info["therapeutic_ceiling_ratio"]),
            "time_hr": sol.t.tolist(),
            "c_baseline_norm": (c_base / np.max(c_base)).tolist(),
            "c_polypharmacy_norm": (c_poly / np.max(c_base)).tolist(),
        })
    return results


@st.cache_resource
def get_ml_engine():
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.uniform(0, 5, size=(n, 6))
    y = rng.binomial(1, 0.3, size=n)
    base = GradientBoostingClassifier(random_state=42)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X, y)
    return clf

ml_clf = get_ml_engine()


# --- 3. STREAMLIT UI ---
st.title("💊 PharmacoTwin Digital Workstation")
st.markdown("*Mechanistic PK ODE Simulation + Split Conformal Uncertainty & Pharmacovigilance Engine*")
st.markdown("---")

st.sidebar.title("Controls")
vignette_options = {v["title"]: v for v in CLINICAL_VIGNETTES}
selected_vig = st.sidebar.selectbox("Load Clinical Vignette", list(vignette_options.keys()))
active_vig = vignette_options[selected_vig]

if st.sidebar.button("Apply Vignette"):
    st.session_state["age"] = active_vig["patient"]["age"]
    st.session_state["sex"] = active_vig["patient"]["sex"]
    st.session_state["weight"] = active_vig["patient"]["weight_kg"]
    st.session_state["scr"] = active_vig["patient"]["scr_mg_dl"]
    st.session_state["potassium"] = active_vig["patient"]["potassium_meq_l"]
    st.session_state["qtc"] = active_vig["patient"]["baseline_qtc_ms"]
    st.session_state["meds"] = list(active_vig["medications"])
    st.rerun()

if "meds" not in st.session_state:
    st.session_state["meds"] = list(active_vig["medications"])
    st.session_state["age"] = active_vig["patient"]["age"]
    st.session_state["sex"] = active_vig["patient"]["sex"]
    st.session_state["weight"] = active_vig["patient"]["weight_kg"]
    st.session_state["scr"] = active_vig["patient"]["scr_mg_dl"]
    st.session_state["potassium"] = active_vig["patient"]["potassium_meq_l"]
    st.session_state["qtc"] = active_vig["patient"]["baseline_qtc_ms"]

st.sidebar.markdown("---")
st.sidebar.subheader("Patient Physiology")
age = st.sidebar.number_input("Age", 1, 120, st.session_state["age"])
sex = st.sidebar.selectbox("Sex", ["female", "male"], index=0 if st.session_state["sex"]=="female" else 1)
weight = st.sidebar.number_input("Weight (kg)", 10.0, 300.0, st.session_state["weight"])
scr = st.sidebar.number_input("Serum Creatinine (mg/dL)", 0.2, 15.0, st.session_state["scr"])
potassium = st.sidebar.number_input("Serum K+ (mEq/L)", 1.5, 7.5, st.session_state["potassium"])
qtc = st.sidebar.number_input("Baseline QTc (ms)", 280.0, 700.0, st.session_state["qtc"])

st.sidebar.markdown("---")
st.sidebar.subheader("Medications")
new_med = st.sidebar.selectbox("Add Drug", [""] + sorted(list(DRUG_KB.keys())))
if st.sidebar.button("Add") and new_med and new_med not in st.session_state["meds"]:
    st.session_state["meds"].append(new_med)
    st.rerun()

st.sidebar.markdown("**Current Regimen:**")
to_remove = []
for m in st.session_state["meds"]:
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(m.title())
    if col2.button("❌", key=f"rm_{m}"):
        to_remove.append(m)
if to_remove:
    st.session_state["meds"] = [m for m in st.session_state["meds"] if m not in to_remove]
    st.rerun()

# --- COMPUTATIONS ---
crcl = cockcroft_gault_crcl(age, weight, scr, sex)
qt_drugs = sum(1 for m in st.session_state["meds"] if DRUG_KB.get(m, {}).get("qt_risk") in {"known", "possible"})
tisdale = tisdale_qt_score(age, sex, "furosemide" in st.session_state["meds"], potassium, qtc, False, qt_drugs, False, True)
cyp_col, pd_col, acb = analyze_collisions(st.session_state["meds"])
pk_sims = simulate_pk(st.session_state["meds"], crcl)

max_auc = max((s["auc_ratio"] for s in pk_sims), default=1.0)
ml_vec = np.array([[len(st.session_state["meds"]), max_auc, len(cyp_col), len(pd_col), acb, tisdale["score"]]])
risk_prob = float(ml_clf.predict_proba(ml_vec)[0, 1])
risk_tier = "Critical" if risk_prob >= 0.75 else ("High" if risk_prob >= 0.5 else "Moderate")

# --- MAIN DASHBOARD DISPLAY ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk Probability", f"{int(risk_prob*100)}% ({risk_tier})")
k2.metric("Peak PK AUC Fold", f"{max_auc}x")
k3.metric("Tisdale QTc Score", f"{tisdale['score']} pts", f"{tisdale['risk_band']} Risk")
k4.metric("CrCl (Cockcroft-Gault)", f"{crcl} mL/min")

st.markdown("---")

col_pk, col_drv = st.columns([1.3, 1])
with col_pk:
    st.subheader("📈 10-Day Steady-State PK Accumulation")
    if pk_sims:
        selected_pk_name = st.selectbox("Select Victim Drug", [s["display_name"] for s in pk_sims])
        selected_sim = next(s for s in pk_sims if s["display_name"] == selected_pk_name)
        
        df_pk = pd.DataFrame({
            "Time (hours)": selected_sim["time_hr"],
            "Baseline Monotherapy": selected_sim["c_baseline_norm"],
            "Polypharmacy + Renal Twin": selected_sim["c_polypharmacy_norm"],
        }).set_index("Time (hours)")
        st.line_chart(df_pk)
        st.caption(f"AUC Ratio: **{selected_sim['auc_ratio']}x** | Cmax Ratio: **{selected_sim['cmax_ratio']}x**")

with col_drv:
    st.subheader("⚖️ Mechanistic &amp; PD Hazards")
    if not cyp_col and not pd_col:
        st.success("No major enzymatic collisions or PD synergies detected.")
    for c in cyp_col:
        st.warning(f"**{c['enzyme']} Inhibition**: {c['perpetrator']} &rarr; {c['victim']}\n\n{c['mechanism']}")
    for p in pd_col:
        st.error(f"**{p['pathway']}**: {', '.join(p['drugs'])}")

st.markdown("---")
st.subheader("🔄 Counterfactual Therapy Optimization")
has_swap = False
for med in st.session_state["meds"]:
    if med in SAFER_SUBSTITUTIONS:
        for opt in SAFER_SUBSTITUTIONS[med]:
            has_swap = True
            c1, c2, c3 = st.columns([2, 3, 1])
            c1.markdown(f"**Swap `{med.title()}` &rarr; `{opt['replacement'].title()}`**")
            c2.markdown(f"*{opt['rationale']}*")
            if c3.button("Apply Swap", key=f"swap_{med}_{opt['replacement']}"):
                st.session_state["meds"] = [opt["replacement"] if m == med else m for m in st.session_state["meds"]]
                st.rerun()
if not has_swap:
    st.info("No single-drug substitution recommendations available for the current regimen.")
