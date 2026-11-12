import streamlit as st
import numpy as np
import pandas as pd

from pharmacotwin.data.pharmacology_kb import CLINICAL_VIGNETTES, DRUG_KB, SAFER_SUBSTITUTIONS
from pharmacotwin.engines.pk_simulator import (
    cockcroft_gault_crcl,
    classify_renal_stage,
    tisdale_qt_score,
    simulate_regimen_pk,
    analyze_cyp_and_pd_collisions,
)
from pharmacotwin.engines.conformal_ml import ConformalRiskEngine

st.set_page_config(
    page_title="PharmacoTwin | Polypharmacy Digital Twin",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_risk_engine():
    return ConformalRiskEngine(seed=42, n_samples=1500)

risk_engine = load_risk_engine()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("💊 PharmacoTwin")
st.sidebar.markdown("**Mechanistic PK & Conformal Risk Twin**")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Load Clinical Vignette")
vignette_options = {v["title"]: v for v in CLINICAL_VIGNETTES}
selected_vig_title = st.sidebar.selectbox("Select pre-set case", list(vignette_options.keys()))
active_vig = vignette_options[selected_vig_title]

if st.sidebar.button("Apply Selected Vignette", use_container_width=True):
    st.session_state["age"] = active_vig["patient"]["age"]
    st.session_state["sex"] = active_vig["patient"]["sex"]
    st.session_state["weight"] = active_vig["patient"]["weight_kg"]
    st.session_state["scr"] = active_vig["patient"]["scr_mg_dl"]
    st.session_state["potassium"] = active_vig["patient"]["potassium_meq_l"]
    st.session_state["qtc"] = active_vig["patient"]["baseline_qtc_ms"]
    st.session_state["hf"] = active_vig["patient"]["heart_failure"]
    st.session_state["mi"] = active_vig["patient"]["acute_mi"]
    st.session_state["sepsis"] = active_vig["patient"]["sepsis"]
    st.session_state["meds"] = list(active_vig["medications"])
    st.rerun()

# Initialize session state defaults if empty
if "age" not in st.session_state:
    st.session_state["age"] = active_vig["patient"]["age"]
    st.session_state["sex"] = active_vig["patient"]["sex"]
    st.session_state["weight"] = active_vig["patient"]["weight_kg"]
    st.session_state["scr"] = active_vig["patient"]["scr_mg_dl"]
    st.session_state["potassium"] = active_vig["patient"]["potassium_meq_l"]
    st.session_state["qtc"] = active_vig["patient"]["baseline_qtc_ms"]
    st.session_state["hf"] = active_vig["patient"]["heart_failure"]
    st.session_state["mi"] = active_vig["patient"]["acute_mi"]
    st.session_state["sepsis"] = active_vig["patient"]["sepsis"]
    st.session_state["meds"] = list(active_vig["medications"])

st.sidebar.markdown("---")
st.sidebar.subheader("2. Patient Physiology")
age = st.sidebar.number_input("Age (years)", 1, 120, st.session_state["age"])
sex = st.sidebar.selectbox("Sex", ["female", "male"], index=0 if st.session_state["sex"]=="female" else 1)
weight = st.sidebar.number_input("Weight (kg)", 15.0, 300.0, st.session_state["weight"])
scr = st.sidebar.number_input("Serum Creatinine (mg/dL)", 0.2, 15.0, st.session_state["scr"], step=0.1)
potassium = st.sidebar.number_input("Serum K+ (mEq/L)", 1.5, 7.5, st.session_state["potassium"], step=0.1)
qtc = st.sidebar.number_input("Baseline QTc (ms)", 280.0, 700.0, st.session_state["qtc"])

col_c1, col_c2, col_c3 = st.sidebar.columns(3)
hf = col_c1.checkbox("HF", value=st.session_state["hf"])
mi = col_c2.checkbox("AMI", value=st.session_state["mi"])
sepsis = col_c3.checkbox("Sepsis", value=st.session_state["sepsis"])

st.sidebar.markdown("---")
st.sidebar.subheader("3. Active Medications (Max 12)")
all_drug_keys = sorted(DRUG_KB.keys())
new_drug = st.sidebar.selectbox("Add Medication from Catalog", [""] + all_drug_keys)
if st.sidebar.button("Add Drug") and new_drug:
    if new_drug not in st.session_state["meds"] and len(st.session_state["meds"]) < 12:
        st.session_state["meds"].append(new_drug)
        st.rerun()

st.sidebar.markdown("**Current Regimen:**")
meds_to_remove = []
for m in st.session_state["meds"]:
    col_m1, col_m2 = st.sidebar.columns([3, 1])
    col_m1.text(DRUG_KB[m]["display_name"] if m in DRUG_KB else m)
    if col_m2.button("❌", key=f"rm_{m}"):
        meds_to_remove.append(m)

if meds_to_remove:
    st.session_state["meds"] = [m for m in st.session_state["meds"] if m not in meds_to_remove]
    st.rerun()

# --- MAIN DASHBOARD ---
st.title("💊 PharmacoTwin Digital Workstation")
st.markdown(
    "*Mechanistic PK ODE Simulation + Split Conformal Uncertainty Quantification + Pharmacovigilance Risk Engine*"
)

patient_data = {
    "age": age,
    "sex": sex,
    "weight_kg": weight,
    "scr_mg_dl": scr,
    "potassium_meq_l": potassium,
    "baseline_qtc_ms": qtc,
    "heart_failure": hf,
    "sepsis": sepsis,
    "acute_mi": mi,
}

if not st.session_state["meds"]:
    st.warning("Please add at least one medication in the sidebar to run the simulation.")
    st.stop()

# Run engine
engine_out = risk_engine.assess_and_optimize(patient_data, st.session_state["meds"])
conf = engine_out["conformal_assessment"]
phys = engine_out["physiology"]
sum_m = {
    "max_auc": max((s["auc_ratio"] for s in engine_out["pk_simulations"]), default=1.0),
    "cyp_count": len(engine_out["cyp_collisions"]),
    "pd_count": len(engine_out["pd_collisions"]),
    "acb": engine_out["acb_total"],
}

# KPI Metrics Row
k1, k2, k3, k4 = st.columns(4)
risk_color = "🔴" if conf["risk_tier"] in ["Critical", "High"] else "🟡" if conf["risk_tier"]=="Moderate" else "🟢"
k1.metric("Conformal Risk", f"{int(conf['risk_probability']*100)}% ({conf['risk_tier']})", f"{risk_color} 90% Set: {conf['prediction_set_90']}")
k2.metric("Peak PK AUC Fold", f"{sum_m['max_auc']}x", f"{sum_m['cyp_count']} CYP / {sum_m['pd_count']} PD hazards")
k3.metric("Tisdale QTc Score", f"{phys['tisdale']['score']} pts", f"{phys['tisdale']['risk_band']} Risk (ACB: {sum_m['acb']})")
k4.metric("Cockcroft-Gault CrCl", f"{phys['crcl']} mL/min", phys['renal_stage'])

st.markdown("---")

# PK Curve & Risk Drivers Row
col_pk, col_drv = st.columns([1.2, 1])

with col_pk:
    st.subheader("📈 10-Day Steady-State PK Accumulation")
    pk_options = {s["display_name"]: s for s in engine_out["pk_simulations"]}
    selected_pk_name = st.selectbox("Select Victim Drug", list(pk_options.keys()))
    selected_sim = pk_options[selected_pk_name]

    # Build Plotly-equivalent dataframe for Streamlit line chart
    df_pk = pd.DataFrame({
        "Time (hours)": selected_sim["time_hr"],
        "Baseline Monotherapy": selected_sim["c_baseline_norm"],
        "Polypharmacy + Renal Twin": selected_sim["c_polypharmacy_norm"],
    }).set_index("Time (hours)")

    st.line_chart(df_pk)
    st.caption(
        f"**{selected_sim['display_name']}**: Steady-state AUC ratio **{selected_sim['auc_ratio']}x** | "
        f"Cmax ratio **{selected_sim['cmax_ratio']}x** (Ceiling: {selected_sim['therapeutic_ceiling_ratio']}x)."
    )

with col_drv:
    st.subheader("⚖️ Risk Drivers & Conformal Coverage")
    st.markdown(f"**Model Calibration Benchmark:** ROC-AUC `{conf['model_benchmark']['roc_auc']}` | Coverage `{conf['model_benchmark']['empirical_coverage_90']*100:.1f}%`")
    
    for drv in conf["top_drivers"]:
        st.markdown(f"- **{drv['feature']}** (`{drv['value']}`): delta risk `{drv['delta_risk']:+.2f}`")

st.markdown("---")

# Counterfactual Optimization
st.subheader("🔄 Counterfactual Therapy Optimization")
cf_list = engine_out["counterfactuals"]
if not cf_list:
    st.info("No single-drug substitutions reduced risk further under current constraints.")
else:
    for idx, cf in enumerate(cf_list):
        c_col1, c_col2, c_col3, c_col4 = st.columns([1.5, 2, 1.2, 1])
        c_col1.markdown(f"**{cf['action_type']}**: `{cf['target_drug']}`")
        c_col2.markdown(f"*{cf['rationale']}*")
        c_col3.markdown(f"New Risk: **{int(cf['new_risk_probability']*100)}%** (`{cf['delta_risk_probability']:+}%`)")
        if c_col4.button("Apply Swap", key=f"apply_cf_{idx}"):
            # Replaced drug logic
            repl = cf.get("replacement_drug")
            target_key = next((k for k, v in DRUG_KB.items() if v["display_name"] == cf["target_drug"]), None)
            repl_key = next((k for k, v in DRUG_KB.items() if v["display_name"] == repl), None) if repl else None
            
            new_meds = []
            for m in st.session_state["meds"]:
                if m == target_key and repl_key:
                    new_meds.append(repl_key)
                elif m != target_key:
                    new_meds.append(m)
            st.session_state["meds"] = new_meds
            st.rerun()

st.markdown("---")

# Collisions Panel
col_cyp, col_pd = st.columns(2)
with col_cyp:
    st.subheader("🧬 Mechanistic CYP450 Collisions")
    if not engine_out["cyp_collisions"]:
        st.success("No significant CYP450 or P-gp inhibitory collisions detected.")
    else:
        for c in engine_out["cyp_collisions"]:
            st.warning(f"**{c['enzyme']}**: {c['perpetrator']} &rarr; {c['victim']} [{c['severity']}]\n\n{c['mechanism']}")

with col_pd:
    st.subheader("⚡ Pharmacodynamic Synergies")
    if not engine_out["pd_collisions"]:
        st.success("No cumulative pharmacodynamic hazards detected.")
    else:
        for p in engine_out["pd_collisions"]:
            st.error(f"**{p['pathway']}** [{p['severity']}] ({', ' .join(p['drugs'])})\n\n{p['clinical_effect']}")
