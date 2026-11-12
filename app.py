import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="PharmacoTwin | Polypharmacy Digital Twin",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


# --- 1. KNOWLEDGE BASE & VIGNETTES ---
DRUG_KB: Dict[str, Dict[str, Any]] = {
    "warfarin": {
        "display_name": "Warfarin",
        "drug_class": "Anticoagulant (Vitamin K Antagonist)",
        "cyp_substrates": ["CYP2C9", "CYP3A4"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "apixaban": {
        "display_name": "Apixaban",
        "drug_class": "Direct Oral Anticoagulant (Factor Xa)",
        "cyp_substrates": ["CYP3A4"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
        "pgp_substrate": True,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.27,
        "half_life_hr": 12.0,
        "vd_l": 21.0,
        "typical_dose_mg": 5.0,
        "dosing_interval_hr": 12.0,
        "therapeutic_ceiling_ratio": 1.80,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": ["bleeding"],
    },
    "amiodarone": {
        "display_name": "Amiodarone",
        "drug_class": "Class III Antiarrhythmic",
        "cyp_substrates": ["CYP3A4"],
        "cyp_inhibitors": {"CYP2C9": 0.60, "CYP2D6": 0.50, "CYP3A4": 0.55},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "clarithromycin": {
        "display_name": "Clarithromycin",
        "drug_class": "Macrolide Antibiotic",
        "cyp_substrates": ["CYP3A4"],
        "cyp_inhibitors": {"CYP3A4": 0.78},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "doxycycline": {
        "display_name": "Doxycycline",
        "drug_class": "Tetracycline Antibiotic",
        "cyp_substrates": [],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.35,
        "half_life_hr": 18.0,
        "vd_l": 50.0,
        "typical_dose_mg": 100.0,
        "dosing_interval_hr": 12.0,
        "therapeutic_ceiling_ratio": 2.50,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": [],
    },
    "furosemide": {
        "display_name": "Furosemide",
        "drug_class": "Loop Diuretic",
        "cyp_substrates": [],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "simvastatin": {
        "display_name": "Simvastatin",
        "drug_class": "HMG-CoA Reductase Inhibitor",
        "cyp_substrates": ["CYP3A4"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "rosuvastatin": {
        "display_name": "Rosuvastatin",
        "drug_class": "HMG-CoA Reductase Inhibitor",
        "cyp_substrates": ["CYP2C9"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.10,
        "half_life_hr": 19.0,
        "vd_l": 134.0,
        "typical_dose_mg": 10.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.50,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": ["myopathy"],
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
    },
    "omeprazole": {
        "display_name": "Omeprazole",
        "drug_class": "Proton Pump Inhibitor",
        "cyp_substrates": ["CYP2C19", "CYP3A4"],
        "cyp_inhibitors": {"CYP2C19": 0.68},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "pantoprazole": {
        "display_name": "Pantoprazole",
        "drug_class": "Proton Pump Inhibitor",
        "cyp_substrates": ["CYP2C19"],
        "cyp_inhibitors": {"CYP2C19": 0.08},
        "is_prodrug": False,
        "prodrug_enzymes": [],
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.10,
        "half_life_hr": 1.5,
        "vd_l": 15.0,
        "typical_dose_mg": 40.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.60,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": [],
    },
    "fluoxetine": {
        "display_name": "Fluoxetine",
        "drug_class": "SSRI Antidepressant",
        "cyp_substrates": ["CYP2D6", "CYP2C9"],
        "cyp_inhibitors": {"CYP2D6": 0.82, "CYP2C19": 0.40},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "sertraline": {
        "display_name": "Sertraline",
        "drug_class": "SSRI Antidepressant",
        "cyp_substrates": ["CYP2C19"],
        "cyp_inhibitors": {"CYP2D6": 0.18},
        "is_prodrug": False,
        "prodrug_enzymes": [],
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.05,
        "half_life_hr": 26.0,
        "vd_l": 1400.0,
        "typical_dose_mg": 50.0,
        "dosing_interval_hr": 24.0,
        "therapeutic_ceiling_ratio": 2.30,
        "qt_risk": "conditional",
        "acb_score": 0,
        "pd_flags": ["serotonergic", "bleeding"],
    },
    "tramadol": {
        "display_name": "Tramadol",
        "drug_class": "Opioid Analgesic (Prodrug)",
        "cyp_substrates": ["CYP2D6", "CYP3A4"],
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
    },
    "ibuprofen": {
        "display_name": "Ibuprofen",
        "drug_class": "NSAID",
        "cyp_substrates": ["CYP2C9"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
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
    },
    "acetaminophen": {
        "display_name": "Acetaminophen",
        "drug_class": "Non-NSAID Analgesic",
        "cyp_substrates": ["CYP2E1"],
        "cyp_inhibitors": {},
        "is_prodrug": False,
        "prodrug_enzymes": [],
        "pgp_substrate": False,
        "pgp_inhibitor": False,
        "renal_elimination_fraction": 0.05,
        "half_life_hr": 2.5,
        "vd_l": 65.0,
        "typical_dose_mg": 500.0,
        "dosing_interval_hr": 8.0,
        "therapeutic_ceiling_ratio": 2.40,
        "qt_risk": "none",
        "acb_score": 0,
        "pd_flags": [],
    },
}

CLINICAL_VIGNETTES = [
    {
        "title": "Vignette 1: AFib + Community Pneumonia Cascade",
        "patient": {
            "age": 74, "sex": "female", "weight_kg": 62.0, "scr_mg_dl": 1.4,
            "potassium_meq_l": 3.3, "baseline_qtc_ms": 458.0, "heart_failure": True, "sepsis": False, "acute_mi": False,
        },
        "medications": ["warfarin", "amiodarone", "clarithromycin", "furosemide", "simvastatin"],
    },
    {
        "title": "Vignette 2: Post-PCI Prodrug Blockade + NSAID",
        "patient": {
            "age": 66, "sex": "male", "weight_kg": 82.0, "scr_mg_dl": 1.1,
            "potassium_meq_l": 4.1, "baseline_qtc_ms": 425.0, "heart_failure": False, "sepsis": False, "acute_mi": True,
        },
        "medications": ["clopidogrel", "omeprazole", "fluoxetine", "tramadol", "ibuprofen"],
    },
]

SAFER_SUBSTITUTIONS = {
    "clarithromycin": [{"replacement": "doxycycline", "rationale": "Eliminates CYP3A4 blockade and additive QTc prolongation."}],
    "simvastatin": [{"replacement": "rosuvastatin", "rationale": "Bypasses CYP3A4 first-pass metabolism."}],
    "omeprazole": [{"replacement": "pantoprazole", "rationale": "Minimizes CYP2C19 inhibition so clopidogrel activation is preserved."}],
    "fluoxetine": [{"replacement": "sertraline", "rationale": "Reduces CYP2D6 blockade compared with fluoxetine."}],
    "ibuprofen": [{"replacement": "acetaminophen", "rationale": "Removes NSAID GI bleeding and renal stress."}],
    "warfarin": [{"replacement": "apixaban", "rationale": "Avoids CYP2C9 narrow-therapeutic-index exposure."}],
}

FEATURE_NAMES = [
    "Active Medications",
    "Max PK AUC Fold-Elevation",
    "Mean PK AUC Fold-Elevation",
    "CYP450 / P-gp Collisions",
    "Pharmacodynamic Overlaps",
    "Anticholinergic Burden (ACB)",
    "Tisdale QTc Risk Score",
    "Creatinine Clearance (mL/min)",
    "Patient Age (years)",
    "QT-Prolonging Drug Count",
]


# --- 2. MECHANISTIC PK, TISDALE & BCPNN ENGINES ---
def cockcroft_gault_crcl(age: float, weight_kg: float, scr_mg_dl: float, sex: str) -> float:
    crcl = ((140.0 - age) * weight_kg) / (72.0 * scr_mg_dl)
    if sex.strip().lower() in {"f", "female"}:
        crcl *= 0.85
    return round(crcl, 2)


def classify_renal_stage(crcl: float) -> str:
    if crcl >= 90.0:
        return "Normal / Stage 1 (>=90)"
    if crcl >= 60.0:
        return "Stage 2 (60-89)"
    if crcl >= 45.0:
        return "Stage 3a (45-59)"
    if crcl >= 30.0:
        return "Stage 3b (30-44)"
    if crcl >= 15.0:
        return "Stage 4 (15-29)"
    return "Stage 5 (<15)"


def tisdale_qt_score(
    age: int, sex: str, loop_diuretic: bool, k_meq_l: float, baseline_qtc_ms: float,
    acute_mi: bool, qt_drug_count: int, sepsis: bool, heart_failure: bool
) -> Dict[str, Any]:
    pts = 0
    if age >= 68:
        pts += 1
    if sex.strip().lower() in {"f", "female"}:
        pts += 1
    if loop_diuretic:
        pts += 1
    if k_meq_l <= 3.5:
        pts += 2
    if baseline_qtc_ms >= 450.0:
        pts += 2
    if acute_mi:
        pts += 2
    if qt_drug_count == 1:
        pts += 3
    elif qt_drug_count >= 2:
        pts += 6
    if sepsis:
        pts += 3
    if heart_failure:
        pts += 3
    band = "Low" if pts < 7 else ("Moderate" if pts <= 10 else "High")
    return {"score": pts, "risk_band": band}


def analyze_collisions(medications: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    cyp_col: List[Dict[str, Any]] = []
    known = [m for m in medications if m in DRUG_KB]

    for perp in known:
        p_info = DRUG_KB[perp]
        for vict in known:
            if perp == vict:
                continue
            v_info = DRUG_KB[vict]
            for enz, pot in p_info["cyp_inhibitors"].items():
                if enz in v_info["cyp_substrates"]:
                    is_prodrug_block = v_info["is_prodrug"] and enz in v_info["prodrug_enzymes"]
                    mech = (
                        f"{p_info['display_name']} inhibits {enz} ({int(pot * 100)}%), "
                        f"{'blocking prodrug activation of' if is_prodrug_block else 'elevating plasma exposure of'} {v_info['display_name']}."
                    )
                    cyp_col.append({
                        "perpetrator": p_info["display_name"],
                        "victim": v_info["display_name"],
                        "enzyme": enz,
                        "potency": pot,
                        "severity": "High" if pot >= 0.6 else "Moderate",
                        "mechanism": mech,
                    })
            if p_info["pgp_inhibitor"] and v_info["pgp_substrate"]:
                cyp_col.append({
                    "perpetrator": p_info["display_name"],
                    "victim": v_info["display_name"],
                    "enzyme": "P-gp",
                    "potency": 0.45,
                    "severity": "Moderate",
                    "mechanism": f"{p_info['display_name']} inhibits P-gp efflux, raising {v_info['display_name']} exposure.",
                })

    pd_groups: Dict[str, List[str]] = {}
    acb_total = 0
    for med in known:
        info = DRUG_KB[med]
        acb_total += info["acb_score"]
        for flag in info["pd_flags"]:
            pd_groups.setdefault(flag, []).append(info["display_name"])

    pd_col = [
        {"pathway": flag.replace("_", " ").title(), "drugs": names, "severity": "High"}
        for flag, names in pd_groups.items()
        if len(names) >= 2
    ]
    return cyp_col, pd_col, acb_total


def _integrate_multi_dose_ode(dose_mg: float, tau_hr: float, vd_l: float, ke_hr: float, n_days: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    t_all: List[float] = []
    c_all: List[float] = []
    c0 = 0.0
    total_doses = max(1, int((n_days * 24) / tau_hr))
    steps = max(4, int(tau_hr // 2))
    for d in range(total_doses):
        c_start = c0 + (dose_mg / vd_l)
        t0, t1 = d * tau_hr, (d + 1) * tau_hr
        t_eval = np.linspace(t0, t1, steps + 1)[:-1]
        sol = solve_ivp(lambda _t, y: [-ke_hr * y[0]], [t0, t1], [c_start], t_eval=t_eval)
        t_all.extend(sol.t.tolist())
        c_all.extend(sol.y[0].tolist())
        c0 = c_start * math.exp(-ke_hr * tau_hr)
    return np.array(t_all), np.array(c_all)


def simulate_regimen_pk(medications: List[str], crcl: float, n_days: int = 10) -> List[Dict[str, Any]]:
    known = [m for m in medications if m in DRUG_KB]
    results: List[Dict[str, Any]] = []
    for med in known:
        info = DRUG_KB[med]
        enz_inhibs = []
        for enz in info["cyp_substrates"]:
            surviving = 1.0
            for other in known:
                if other != med:
                    surviving *= 1.0 - DRUG_KB[other]["cyp_inhibitors"].get(enz, 0.0)
            enz_inhibs.append(1.0 - surviving)
        mean_inh = float(np.mean(enz_inhibs)) if enz_inhibs else 0.0
        if info["pgp_substrate"] and any(DRUG_KB[o]["pgp_inhibitor"] for o in known if o != med):
            mean_inh = min(0.90, mean_inh + 0.25 * (1.0 - mean_inh))
        mean_inh = min(0.90, max(0.0, mean_inh))

        fe = info["renal_elimination_fraction"]
        renal_factor = (1.0 - fe) + fe * max(0.20, min(1.0, crcl / 90.0))
        ke_base = math.log(2.0) / info["half_life_hr"]
        ke_mod = ke_base * renal_factor if info["is_prodrug"] else ke_base * (1.0 - mean_inh) * renal_factor

        t_arr, c_base = _integrate_multi_dose_ode(info["typical_dose_mg"], info["dosing_interval_hr"], info["vd_l"], ke_base, n_days)
        _, c_poly = _integrate_multi_dose_ode(info["typical_dose_mg"], info["dosing_interval_hr"], info["vd_l"], ke_mod, n_days)

        mask_24 = t_arr >= (n_days - 1) * 24.0
        auc_base = max(1e-9, _trapezoid(c_base[mask_24], t_arr[mask_24]))
        auc_poly = _trapezoid(c_poly[mask_24], t_arr[mask_24])
        cmax_base = max(1e-9, float(np.max(c_base[mask_24])))
        cmax_poly = float(np.max(c_poly[mask_24]))

        auc_ratio = round(auc_poly / auc_base, 3)
        cmax_ratio = round(cmax_poly / cmax_base, 3)
        ceiling = info["therapeutic_ceiling_ratio"]

        results.append({
            "drug": med,
            "display_name": info["display_name"],
            "auc_ratio": auc_ratio,
            "cmax_ratio": cmax_ratio,
            "therapeutic_ceiling_ratio": ceiling,
            "exceeds_ceiling": bool(cmax_ratio >= ceiling or auc_ratio >= ceiling),
            "time_hr": [round(float(x), 1) for x in t_arr],
            "c_baseline_norm": [round(float(y / cmax_base), 3) for y in c_base],
            "c_polypharmacy_norm": [round(float(y / cmax_base), 3) for y in c_poly],
            "ceiling_line": [ceiling] * len(t_arr),
        })
    results.sort(key=lambda r: (r["exceeds_ceiling"], r["auc_ratio"]), reverse=True)
    return results


def compute_disproportionality(a: int, b: int, c: int, d: int) -> Dict[str, Any]:
    n_total = a + b + c + d
    expected = ((a + b) * (a + c)) / max(1, n_total)
    ic = math.log2((a + 0.5) / (expected + 0.5))
    ic025 = ic - 3.3 * ((a + 0.5) ** -0.5) - 2.0 * ((a + 0.5) ** -1.5)
    aa, bb, cc, dd = (a + 0.5, b + 0.5, c + 0.5, d + 0.5) if min(a, b, c, d) == 0 else (float(a), float(b), float(c), float(d))
    ror = (aa * dd) / (bb * cc)
    se = math.sqrt((1.0 / aa) + (1.0 / bb) + (1.0 / cc) + (1.0 / dd))
    ror_low = math.exp(math.log(ror) - 1.96 * se)
    return {
        "a": a,
        "expected": round(expected, 2),
        "ic": round(ic, 3),
        "ic025": round(ic025, 3),
        "ror": round(ror, 2),
        "ror_ci_low": round(ror_low, 2),
        "signal_positive": bool(a >= 3 and ic025 > 0.0 and ror_low > 1.0),
    }


# --- 3. SPLIT CONFORMAL ML ENGINE ---
@st.cache_resource
def load_conformal_engine() -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    n = 2000
    n_drugs = rng.integers(1, 10, size=n)
    max_auc = rng.uniform(1.0, 4.0, size=n)
    mean_auc = np.clip(1.0 + (max_auc - 1.0) * rng.uniform(0.25, 0.85, size=n), 1.0, 3.5)
    cyp_col = rng.integers(0, 6, size=n)
    pd_col = rng.integers(0, 5, size=n)
    acb = rng.integers(0, 9, size=n)
    tisdale = rng.integers(0, 17, size=n)
    crcl = rng.uniform(15.0, 125.0, size=n)
    age = rng.uniform(22.0, 92.0, size=n)
    qt_cnt = rng.integers(0, 4, size=n)

    X = np.column_stack([n_drugs, max_auc, mean_auc, cyp_col, pd_col, acb, tisdale, crcl, age, qt_cnt])
    logit = (
        -4.1 + 0.22 * n_drugs + 1.05 * np.maximum(0.0, max_auc - 1.35) + 0.55 * cyp_col
        + 0.68 * pd_col + 0.24 * acb + 0.20 * np.maximum(0, tisdale - 4)
        - 0.028 * (crcl - 65.0) + 0.32 * qt_cnt + rng.normal(0.0, 0.45, size=n)
    )
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    base = GradientBoostingClassifier(n_estimators=90, max_depth=3, learning_rate=0.07, random_state=42)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)

    p_cal = clf.predict_proba(X_cal)[:, 1]
    scores_cal = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    n_cal = len(scores_cal)
    q_90 = float(np.quantile(scores_cal, min(1.0, math.ceil((n_cal + 1) * 0.90) / n_cal), method="higher"))

    p_test = clf.predict_proba(X_test)[:, 1]
    inc_pos = (1.0 - p_test) <= q_90
    inc_neg = p_test <= q_90
    cov_90 = float(np.mean(np.where(y_test == 1, inc_pos, inc_neg)))

    return {
        "clf": clf,
        "q_90": round(q_90, 4),
        "roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
        "brier": round(float(brier_score_loss(y_test, p_test)), 4),
        "coverage_90": round(cov_90, 4),
    }


def predict_conformal(engine: Dict[str, Any], vec: np.ndarray) -> Dict[str, Any]:
    clf = engine["clf"]
    q_90 = engine["q_90"]
    p = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])

    pred_set: List[str] = []
    if p <= q_90:
        pred_set.append("Low-Risk")
    if (1.0 - p) <= q_90:
        pred_set.append("High-Risk")
    if not pred_set:
        pred_set = ["Low-Risk" if p < 0.5 else "High-Risk"]

    tier = "Critical" if p >= 0.75 else ("High" if p >= 0.50 else ("Moderate" if p >= 0.25 else "Low"))

    healthy_ref = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 95.0, 45.0, 0.0])
    drivers = []
    for idx, name in enumerate(FEATURE_NAMES):
        cf = vec.copy()
        cf[idx] = healthy_ref[idx]
        p_cf = float(clf.predict_proba(cf.reshape(1, -1))[0, 1])
        delta = round(p - p_cf, 4)
        if abs(delta) >= 0.005:
            drivers.append({"feature": name, "value": round(float(vec[idx]), 2), "delta_risk": delta})
    drivers.sort(key=lambda d: abs(d["delta_risk"]), reverse=True)

    return {
        "risk_probability": round(p, 4),
        "risk_tier": tier,
        "prediction_set_90": pred_set,
        "is_uncertain_90": len(pred_set) > 1,
        "top_drivers": drivers[:5],
    }


# --- 4. LIVE OPENFDA & NLM RXNORM FETCHERS ---
@st.cache_data(ttl=3600, show_spinner=False)
def query_live_fda_and_rxnorm(drug_a: str, drug_b: str) -> Dict[str, Any]:
    da, db = drug_a.strip().upper(), drug_b.strip().upper()
    pair_expr = f'patient.drug.openfda.generic_name.exact:"{da}" AND patient.drug.openfda.generic_name.exact:"{db}"'
    out: Dict[str, Any] = {"drug_pair": [drug_a.lower(), drug_b.lower()], "rxcui": {}, "n_pair": 0, "reactions": [], "error": None}

    try:
        with httpx.Client(timeout=7.0) as client:
            for d in [drug_a.lower(), drug_b.lower()]:
                r_rx = client.get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json", params={"term": d, "maxEntries": 1})
                if r_rx.status_code == 200:
                    cands = r_rx.json().get("approximateGroup", {}).get("candidate", [])
                    if cands:
                        out["rxcui"][d] = cands[0].get("rxcui")

            r_tot = client.get("https://api.fda.gov/drug/event.json", params={"limit": 1})
            r_pair = client.get("https://api.fda.gov/drug/event.json", params={"search": pair_expr, "limit": 1})
            if r_pair.status_code == 404:
                return out
            r_tot.raise_for_status()
            r_pair.raise_for_status()

            n_total = int(r_tot.json()["meta"]["results"]["total"])
            n_pair = int(r_pair.json()["meta"]["results"]["total"])
            out["n_pair"] = n_pair

            if n_pair > 0:
                r_cnt = client.get(
                    "https://api.fda.gov/drug/event.json",
                    params={"search": pair_expr, "count": "patient.reaction.reactionmeddrapt.exact", "limit": 3},
                )
                if r_cnt.status_code == 200:
                    for item in r_cnt.json().get("results", [])[:3]:
                        pt = item.get("term", "")
                        if not pt:
                            continue
                        r_evt = client.get(
                            "https://api.fda.gov/drug/event.json",
                            params={"search": f'patient.reaction.reactionmeddrapt.exact:"{pt}"', "limit": 1},
                        )
                        r_both = client.get(
                            "https://api.fda.gov/drug/event.json",
                            params={"search": f'{pair_expr} AND patient.reaction.reactionmeddrapt.exact:"{pt}"', "limit": 1},
                        )
                        if r_evt.status_code == 200 and r_both.status_code == 200:
                            a_val = int(r_both.json()["meta"]["results"]["total"])
                            n_evt = int(r_evt.json()["meta"]["results"]["total"])
                            b_val = max(0, n_pair - a_val)
                            c_val = max(0, n_evt - a_val)
                            d_val = max(0, n_total - a_val - b_val - c_val)
                            if d_val > 0:
                                stats = compute_disproportionality(a_val, b_val, c_val, d_val)
                                out["reactions"].append({"reaction": pt, **stats})
    except Exception as exc:
        out["error"] = f"Live FDA/RxNorm query unavailable ({type(exc).__name__}); offline ODE & ML active."
    return out


# --- 5. STREAMLIT WORKSTATION UI ---
ml_engine = load_conformal_engine()

st.title("💊 PharmacoTwin Digital Workstation")
st.caption("Mechanistic PK ODE Simulation + Live OpenFDA BCPNN Pharmacovigilance + Split Conformal Risk Engine")
st.warning(
    "**Research Prototype Notice:** Combines 1-compartment PK approximations, synthetic conformal calibration benchmarks, "
    "and spontaneous openFDA FAERS reports. Not clinically validated for patient care."
)

# --- SAFE SESSION STATE INITIALIZATION ---
v0 = CLINICAL_VIGNETTES[0]
if "meds" not in st.session_state:
    st.session_state["meds"] = list(v0["medications"])
for k, val in v0["patient"].items():
    if k not in st.session_state:
        st.session_state[k] = val

# Sidebar Controls
st.sidebar.header("1. Clinical Vignettes")
vig_map = {v["title"]: v for v in CLINICAL_VIGNETTES}
chosen_vig = st.sidebar.selectbox("Select preset case", list(vig_map.keys()))
if st.sidebar.button("Apply Vignette", use_container_width=True):
    v = vig_map[chosen_vig]
    for k, val in v["patient"].items():
        st.session_state[k] = val
    st.session_state["meds"] = list(v["medications"])
    st.rerun()

st.sidebar.header("2. Patient Physiology")
age = st.sidebar.number_input("Age (years)", 1, 120, int(st.session_state.get("age", 74)))
sex = st.sidebar.selectbox("Sex", ["female", "male"], index=0 if st.session_state.get("sex", "female") == "female" else 1)
weight = st.sidebar.number_input("Weight (kg)", 15.0, 300.0, float(st.session_state.get("weight_kg", 62.0)))
scr = st.sidebar.number_input("Serum Creatinine (mg/dL)", 0.2, 15.0, float(st.session_state.get("scr_mg_dl", 1.4)), step=0.1)
potassium = st.sidebar.number_input("Serum K+ (mEq/L)", 1.5, 7.5, float(st.session_state.get("potassium_meq_l", 3.3)), step=0.1)
qtc = st.sidebar.number_input("Baseline QTc (ms)", 280.0, 700.0, float(st.session_state.get("baseline_qtc_ms", 458.0)))
c1, c2, c3 = st.sidebar.columns(3)
hf = c1.checkbox("HF", value=bool(st.session_state.get("heart_failure", True)))
mi = c2.checkbox("AMI", value=bool(st.session_state.get("acute_mi", False)))
sepsis = c3.checkbox("Sepsis", value=bool(st.session_state.get("sepsis", False)))

st.sidebar.header("3. Medications")
add_med = st.sidebar.selectbox("Add Drug", [""] + sorted(DRUG_KB.keys()))
if st.sidebar.button("Add Medication") and add_med and add_med not in st.session_state["meds"]:
    st.session_state["meds"].append(add_med)
    st.rerun()

rem_list = []
for m in st.session_state["meds"]:
    mc1, mc2 = st.sidebar.columns([3, 1])
    mc1.write(DRUG_KB[m]["display_name"] if m in DRUG_KB else m.title())
    if mc2.button("x", key=f"del_{m}"):
        rem_list.append(m)
if rem_list:
    st.session_state["meds"] = [m for m in st.session_state["meds"] if m not in rem_list]
    st.rerun()

live_fda_on = st.sidebar.checkbox("Enable Live OpenFDA & RxNorm", value=True)

# Run Core Computations
meds = st.session_state["meds"]
crcl = cockcroft_gault_crcl(age, weight, scr, sex)
qt_cnt = sum(1 for m in meds if DRUG_KB.get(m, {}).get("qt_risk") in {"known", "possible"})
tisdale = tisdale_qt_score(age, sex, "furosemide" in meds, potassium, qtc, mi, qt_cnt, sepsis, hf)
cyp_col, pd_col, acb_total = analyze_collisions(meds)
pk_sims = simulate_regimen_pk(meds, crcl, n_days=10)

max_auc = max((s["auc_ratio"] for s in pk_sims), default=1.0)
mean_auc = float(np.mean([s["auc_ratio"] for s in pk_sims])) if pk_sims else 1.0
feat_vec = np.array([len(meds), max_auc, mean_auc, len(cyp_col), len(pd_col), acb_total, tisdale["score"], crcl, age, qt_cnt], dtype=float)
conf = predict_conformal(ml_engine, feat_vec)

# Top KPI Cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Conformal Risk Score", f"{int(conf['risk_probability']*100)}% ({conf['risk_tier']})", f"90% Set: {{{', '.join(conf['prediction_set_90'])}}}")
k2.metric("Peak PK AUC Fold", f"{max_auc}x", f"{len(cyp_col)} CYP/P-gp | {len(pd_col)} PD")
k3.metric("Tisdale QTc Score", f"{tisdale['score']} pts", f"{tisdale['risk_band']} Risk (ACB: {acb_total})")
k4.metric("Cockcroft-Gault CrCl", f"{crcl} mL/min", classify_renal_stage(crcl))

st.markdown("---")

# Row 1: 10-Day Repeated-Dosing PK Curve + Split Conformal Uncertainty Card
col_pk, col_conf = st.columns([1.25, 1])

with col_pk:
    st.subheader("📈 10-Day Repeated-Dosing PK ODE Curve")
    if pk_sims:
        victim_choice = st.selectbox("Select Victim Drug", [s["display_name"] for s in pk_sims])
        sim = next(s for s in pk_sims if s["display_name"] == victim_choice)
        df_curve = pd.DataFrame({
            "Time (hr)": sim["time_hr"],
            "1. Baseline Monotherapy": sim["c_baseline_norm"],
            "2. Polypharmacy + Renal Twin": sim["c_polypharmacy_norm"],
            "3. Therapeutic Ceiling": sim["ceiling_line"],
        }).set_index("Time (hr)")
        st.line_chart(df_curve, color=["#94a3b8", "#38bdf8", "#ef4444"])
        st.caption(
            f"**{sim['display_name']}**: Steady-state AUC ratio **{sim['auc_ratio']}x** | "
            f"Cmax ratio **{sim['cmax_ratio']}x** vs. therapeutic ceiling **{sim['therapeutic_ceiling_ratio']}x**."
        )

with col_conf:
    st.subheader("⚖️ Split Conformal Uncertainty & Risk Drivers")
    if conf["is_uncertain_90"]:
        st.warning(f"**Ambiguous Prediction Set (90% Target):** `{{{', '.join(conf['prediction_set_90'])}}}` — Borderline case flagged for clinician review.")
    else:
        st.success(f"**Decisive Singleton Set (90% Target):** `{{{', '.join(conf['prediction_set_90'])}}}`")

    for drv in conf["top_drivers"]:
        st.write(f"- **{drv['feature']}** (`{drv['value']}`): `{drv['delta_risk']:+.1%}` risk attribution")

    st.caption(
        f"Synthetic Calibration Benchmark (n=2,000; 60/20/20 split): ROC-AUC **{ml_engine['roc_auc']}**, "
        f"Brier **{ml_engine['brier']}**, Empirical 90%-Target Coverage **{ml_engine['coverage_90']*100:.1f}%**."
    )

st.markdown("---")

# Row 2: Live OpenFDA FAERS (IC025) & NLM RxNorm + Custom Pair Probe
col_fda, col_coll = st.columns(2)

with col_fda:
    st.subheader("🌐 Live OpenFDA FAERS (IC025) & NLM RxNorm")
    probe_c1, probe_c2 = st.columns(2)
    default_a = cyp_col[0]["perpetrator"].lower() if cyp_col else (meds[0] if meds else "warfarin")
    default_b = cyp_col[0]["victim"].lower() if cyp_col else (meds[1] if len(meds) > 1 else "amiodarone")
    drug_a_in = probe_c1.text_input("Drug A", value=default_a)
    drug_b_in = probe_c2.text_input("Drug B", value=default_b)

    if live_fda_on and drug_a_in.strip() and drug_b_in.strip():
        fda_data = query_live_fda_and_rxnorm(drug_a_in, drug_b_in)
        if fda_data["error"]:
            st.info(fda_data["error"])
        else:
            rx_str = ", ".join(f"{k.upper()}: RxCUI {v}" for k, v in fda_data["rxcui"].items() if v)
            if rx_str:
                st.caption(f"**NLM RxNorm:** {rx_str}")
            st.write(f"**FAERS Co-Reports (`{drug_a_in.upper()} + {drug_b_in.upper()}`):** `{fda_data['n_pair']}`")
            if fda_data["reactions"]:
                st.dataframe(pd.DataFrame(fda_data["reactions"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No co-reported MedDRA reactions above zero returned for this pair.")
    else:
        st.caption("Toggle 'Enable Live OpenFDA & RxNorm' in the sidebar to query federal safety databases.")

with col_coll:
    st.subheader("🧬 Mechanistic CYP450 & PD Collisions")
    if not cyp_col and not pd_col:
        st.success("No major CYP450/P-gp collisions or PD overlaps detected.")
    for c in cyp_col:
        st.warning(f"**{c['enzyme']}**: {c['perpetrator']} → {c['victim']} [{c['severity']}]\n\n{c['mechanism']}")
    for p in pd_col:
        st.error(f"**{p['pathway']}** ({', '.join(p['drugs'])})")

st.markdown("---")

# Row 3: Counterfactual Optimization + JSON Audit Dossier Export
st.subheader("🔄 Counterfactual Therapy Optimization")
swaps_found = False
for med in meds:
    for opt in SAFER_SUBSTITUTIONS.get(med, []):
        repl = opt["replacement"]
        if repl in meds:
            continue
        swaps_found = True
        new_meds = [repl if m == med else m for m in meds]
        new_cyp, new_pd, new_acb = analyze_collisions(new_meds)
        new_pk = simulate_regimen_pk(new_meds, crcl, n_days=10)
        new_max = max((s["auc_ratio"] for s in new_pk), default=1.0)
        new_mean = float(np.mean([s["auc_ratio"] for s in new_pk])) if new_pk else 1.0
        new_qt = sum(1 for m in new_meds if DRUG_KB.get(m, {}).get("qt_risk") in {"known", "possible"})
        new_tis = tisdale_qt_score(age, sex, "furosemide" in new_meds, potassium, qtc, mi, new_qt, sepsis, hf)
        new_vec = np.array([len(new_meds), new_max, new_mean, len(new_cyp), len(new_pd), new_acb, new_tis["score"], crcl, age, new_qt], dtype=float)
        new_conf = predict_conformal(ml_engine, new_vec)
        delta_p = new_conf["risk_probability"] - conf["risk_probability"]

        sc1, sc2, sc3, sc4 = st.columns([1.8, 2.5, 1.4, 1.0])
        sc1.markdown(f"**Swap `{med.title()}` → `{repl.title()}`**")
        sc2.caption(opt["rationale"])
        sc3.write(f"Risk: **{int(new_conf['risk_probability']*100)}%** (`{delta_p:+.0%}`) | AUC: **{new_max}x**")
        if sc4.button("Apply Swap", key=f"swap_{med}_{repl}"):
            st.session_state["meds"] = new_meds
            st.rerun()

if not swaps_found:
    st.info("No single-drug substitutions reduced risk further under current parameters.")

st.markdown("---")

# Export JSON Dossier
audit_dossier = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "physiology": {"age": age, "sex": sex, "weight_kg": weight, "scr_mg_dl": scr, "potassium_meq_l": potassium, "baseline_qtc_ms": qtc, "crcl_ml_min": crcl, "tisdale": tisdale},
    "medications": meds,
    "conformal_assessment": conf,
    "model_calibration_benchmark": {k: v for k, v in ml_engine.items() if k != "clf"},
    "cyp_collisions": cyp_col,
    "pd_collisions": pd_col,
}
st.download_button(
    label="📥 Export Zero-PHI JSON Audit Dossier",
    data=json.dumps(audit_dossier, indent=2),
    file_name=f"pharmacotwin-dossier-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    mime="application/json",
)
