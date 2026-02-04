import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v21.0", page_icon="🔌", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    /* CAT Branding */
    .main-header {
        background: linear-gradient(90deg, #1A1A1A 0%, #2D2D2D 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #FFCD00;
    }
    .main-header h1 { color: #FFCD00 !important; margin: 0; }
    .main-header p { color: #CCCCCC !important; margin: 0; }
    
    .math-box { 
        background-color: #e8f4f8; 
        padding: 15px; 
        border-radius: 5px; 
        font-family: monospace; 
        border-left: 5px solid #17a2b8; 
        margin-bottom: 10px; 
    }
    .success-box { 
        background-color: #d4edda; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #28a745; 
        margin-bottom: 15px; 
    }
    .fail-box { 
        background-color: #f8d7da; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #dc3545; 
        margin-bottom: 15px; 
    }
    .warning-box { 
        background-color: #fff3cd; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #ffc107; 
        margin-bottom: 15px; 
    }
    .tier-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1976d2;
        margin-bottom: 15px;
        text-align: center;
    }
    .tier-label { font-size: 14px; color: #555; }
    .tier-value { font-size: 48px; font-weight: bold; color: #1976d2; }
    .tier-desc { font-size: 12px; color: #777; }
    
    .metric-value { font-size: 24px; font-weight: bold; }
    .metric-label { font-size: 12px; color: #555; text-transform: uppercase; }
    
    .validation-pass { color: #28a745; font-weight: bold; }
    .validation-fail { color: #dc3545; font-weight: bold; }
    .validation-warn { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. DATA LIBRARIES
# ==============================================================================

CAT_LIBRARY = {
    "XGC1900 (1.9 MW)":   {"mw": 1.9,   "xd": 0.16, "step_cap": 25.0, "voltage_kv": 0.48},
    "G3520FR (2.5 MW)":   {"mw": 2.5,   "xd": 0.16, "step_cap": 25.0, "voltage_kv": 0.48},
    "G3520K (2.4 MW)":    {"mw": 2.4,   "xd": 0.16, "step_cap": 25.0, "voltage_kv": 0.48},
    "CG260-16 (3.96 MW)": {"mw": 3.957, "xd": 0.15, "step_cap": 25.0, "voltage_kv": 4.16},
    "G20CM34 (9.76 MW)":  {"mw": 9.76,  "xd": 0.14, "step_cap": 20.0, "voltage_kv": 11.0},
    "Titan 130 (16.5 MW)":{"mw": 16.5,  "xd": 0.14, "step_cap": 15.0, "voltage_kv": 13.8},
    "Titan 250 (23.2 MW)":{"mw": 23.2,  "xd": 0.14, "step_cap": 15.0, "voltage_kv": 13.8},
    "Titan 350 (38.0 MW)":{"mw": 38.0,  "xd": 0.14, "step_cap": 15.0, "voltage_kv": 13.8}
}

# NEW: Transformer Library
TRANSFORMER_LIBRARY = {
    "1000 kVA": {"kva": 1000, "z_pct": 5.75, "x_r": 8, "mtbf": 200000, "mttr": 168},
    "1500 kVA": {"kva": 1500, "z_pct": 5.75, "x_r": 9, "mtbf": 200000, "mttr": 168},
    "2000 kVA": {"kva": 2000, "z_pct": 5.75, "x_r": 10, "mtbf": 200000, "mttr": 168},
    "2500 kVA": {"kva": 2500, "z_pct": 5.75, "x_r": 10, "mtbf": 200000, "mttr": 168},
    "3000 kVA": {"kva": 3000, "z_pct": 6.0, "x_r": 12, "mtbf": 200000, "mttr": 168},
    "3750 kVA": {"kva": 3750, "z_pct": 6.0, "x_r": 12, "mtbf": 200000, "mttr": 168},
}

# NEW: Switchgear Library
SWITCHGEAR_MV = {
    "15kV, 25kA": {"voltage_kv": 15, "kaic": 25, "continuous_a": 1200, "mtbf": 500000, "mttr": 24},
    "15kV, 40kA": {"voltage_kv": 15, "kaic": 40, "continuous_a": 2000, "mtbf": 500000, "mttr": 24},
    "15kV, 50kA": {"voltage_kv": 15, "kaic": 50, "continuous_a": 3000, "mtbf": 500000, "mttr": 24},
    "38kV, 25kA": {"voltage_kv": 38, "kaic": 25, "continuous_a": 1200, "mtbf": 500000, "mttr": 24},
    "38kV, 40kA": {"voltage_kv": 38, "kaic": 40, "continuous_a": 2000, "mtbf": 500000, "mttr": 24},
}

SWITCHGEAR_LV = {
    "480V, 65kA": {"voltage_v": 480, "kaic": 65, "continuous_a": 2000, "mtbf": 300000, "mttr": 8},
    "480V, 100kA": {"voltage_v": 480, "kaic": 100, "continuous_a": 3000, "mtbf": 300000, "mttr": 8},
    "480V, 150kA": {"voltage_v": 480, "kaic": 150, "continuous_a": 4000, "mtbf": 300000, "mttr": 8},
    "480V, 200kA": {"voltage_v": 480, "kaic": 200, "continuous_a": 5000, "mtbf": 300000, "mttr": 8},
}

# NEW: Tier Classification
TIER_LEVELS = {
    "IV": {"min_avail": 0.99995, "redundancy": "2N or 2(N+1)", "description": "Fault Tolerant", "downtime_hr_yr": 0.4},
    "III": {"min_avail": 0.99982, "redundancy": "N+1", "description": "Concurrently Maintainable", "downtime_hr_yr": 1.6},
    "II": {"min_avail": 0.99741, "redundancy": "N+1 Partial", "description": "Redundant Components", "downtime_hr_yr": 22},
    "I": {"min_avail": 0.99671, "redundancy": "N", "description": "Basic Infrastructure", "downtime_hr_yr": 28.8},
}

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

def calc_avail(mtbf, mttr):
    """Calculate availability from MTBF and MTTR."""
    if (mtbf + mttr) <= 0: return 0.0
    return mtbf / (mtbf + mttr)

def get_unavailability(mtbf, mttr):
    """Calculate unavailability from MTBF and MTTR."""
    if (mtbf + mttr) <= 0: return 1.0
    return mttr / (mtbf + mttr)

def rel_k_out_n(n_needed, n_total, p_unit):
    """Calculate k-out-of-n reliability using binomial distribution."""
    if n_total < n_needed: return 0.0
    prob = 0.0
    for k in range(n_needed, n_total + 1):
        prob += binom.pmf(k, n_total, p_unit)
    return prob

def get_n_for_reliability(n_needed, target_avail, p_unit_avail):
    """Optimize N (redundancy) to meet target reliability."""
    for added_redundancy in range(0, 50):
        n_total = n_needed + added_redundancy
        prob = rel_k_out_n(n_needed, n_total, p_unit_avail)
        if prob >= target_avail:
            return n_total, prob
    return n_needed + 50, 0.0

def calc_amps(mw, kv):
    """Calculate current in Amps from MW and kV."""
    if kv <= 0: return 0
    return (mw * 1e6) / (math.sqrt(3) * kv * 1000)

def calc_sc_ka_gen(mw_gen, xd, kv, n_gens):
    """Calculate short circuit contribution from generators."""
    if kv <= 0 or xd <= 0: return 0
    mva_gen = mw_gen / 0.8  # Assume 0.8 PF
    i_base = (mva_gen * 1e6) / (math.sqrt(3) * kv * 1000)
    i_sc_unit = i_base / xd
    return (i_sc_unit * n_gens) / 1000.0

def calc_sc_ka_xfmr(kva_xfmr, z_pct, kv_secondary, n_xfmrs):
    """Calculate short circuit contribution through transformers."""
    if kv_secondary <= 0 or z_pct <= 0: return 0
    mva_xfmr = kva_xfmr / 1000
    i_base = (mva_xfmr * 1e6) / (math.sqrt(3) * kv_secondary * 1000)
    i_sc_unit = i_base / (z_pct / 100)
    return (i_sc_unit * n_xfmrs) / 1000.0

def get_tier_level(availability):
    """
    Determine Tier level based on availability.
    Returns: (tier_name, tier_info)
    """
    for tier, info in TIER_LEVELS.items():
        if availability >= info['min_avail']:
            return tier, info
    return "Below I", {"min_avail": 0, "redundancy": "N", "description": "Below Standard", "downtime_hr_yr": ">28.8"}

def availability_to_nines(avail):
    """Convert availability to 'nines' notation."""
    if avail >= 0.999999:
        return "6 nines (99.9999%)"
    elif avail >= 0.99999:
        return "5 nines (99.999%)"
    elif avail >= 0.9999:
        return "4 nines (99.99%)"
    elif avail >= 0.999:
        return "3 nines (99.9%)"
    elif avail >= 0.99:
        return "2 nines (99%)"
    else:
        return f"{avail*100:.2f}%"

def avail_to_downtime(avail, hours_per_year=8760):
    """Convert availability to downtime hours per year."""
    return (1 - avail) * hours_per_year

# ==============================================================================
# 2. VALIDATION FUNCTIONS (NEW)
# ==============================================================================

def validate_short_circuit(sc_calculated_ka, equipment_kaic, equipment_name):
    """Validate short circuit current against equipment rating."""
    margin = (equipment_kaic - sc_calculated_ka) / equipment_kaic * 100
    
    if sc_calculated_ka > equipment_kaic:
        return {
            'status': 'FAIL',
            'message': f"⚠️ {equipment_name}: Isc {sc_calculated_ka:.1f} kA EXCEEDS rating {equipment_kaic} kA",
            'recommendation': "Increase equipment kAIC rating, add current-limiting reactors, or split buses",
            'margin': margin
        }
    elif margin < 20:
        return {
            'status': 'WARNING',
            'message': f"⚡ {equipment_name}: Isc {sc_calculated_ka:.1f} kA close to rating {equipment_kaic} kA ({margin:.0f}% margin)",
            'recommendation': "Consider higher rated equipment for future expansion",
            'margin': margin
        }
    else:
        return {
            'status': 'PASS',
            'message': f"✅ {equipment_name}: Isc {sc_calculated_ka:.1f} kA OK (rating {equipment_kaic} kA, {margin:.0f}% margin)",
            'recommendation': None,
            'margin': margin
        }

def validate_ampacity(current_calculated, equipment_rating, equipment_name):
    """Validate current against equipment ampacity."""
    margin = (equipment_rating - current_calculated) / equipment_rating * 100
    
    if current_calculated > equipment_rating:
        return {
            'status': 'FAIL',
            'message': f"⚠️ {equipment_name}: Current {current_calculated:.0f} A EXCEEDS rating {equipment_rating} A",
            'recommendation': "Increase bus rating, add parallel buses, or increase voltage level",
            'margin': margin
        }
    elif margin < 20:
        return {
            'status': 'WARNING',
            'message': f"⚡ {equipment_name}: Current {current_calculated:.0f} A close to rating {equipment_rating} A ({margin:.0f}% margin)",
            'recommendation': "Consider larger equipment for future growth",
            'margin': margin
        }
    else:
        return {
            'status': 'PASS',
            'message': f"✅ {equipment_name}: Current {current_calculated:.0f} A OK (rating {equipment_rating} A, {margin:.0f}% margin)",
            'recommendation': None,
            'margin': margin
        }

def select_transformer_size(load_mva):
    """Auto-select transformer size based on load."""
    for name, specs in TRANSFORMER_LIBRARY.items():
        if specs['kva'] / 1000 >= load_mva * 1.25:  # 25% margin
            return name, specs
    # Return largest if none fit
    return list(TRANSFORMER_LIBRARY.keys())[-1], list(TRANSFORMER_LIBRARY.values())[-1]

def select_mv_switchgear(voltage_kv, sc_ka):
    """Auto-select MV switchgear based on voltage and short circuit."""
    for name, specs in SWITCHGEAR_MV.items():
        if specs['voltage_kv'] >= voltage_kv and specs['kaic'] >= sc_ka * 1.25:
            return name, specs
    return list(SWITCHGEAR_MV.keys())[-1], list(SWITCHGEAR_MV.values())[-1]

def select_lv_switchgear(sc_ka):
    """Auto-select LV switchgear based on short circuit."""
    for name, specs in SWITCHGEAR_LV.items():
        if specs['kaic'] >= sc_ka * 1.25:
            return name, specs
    return list(SWITCHGEAR_LV.keys())[-1], list(SWITCHGEAR_LV.values())[-1]

# ==============================================================================
# 3. SIDEBAR INPUTS
# ==============================================================================

with st.sidebar:
    st.markdown("## ⚡ CAT Topology Designer")
    st.caption("v21.0 - With Validations & Tier Classification")
    
    with st.expander("📊 1. Project & Load", expanded=True):
        project_name = st.text_input("Project Name", "Data Center Project")
        p_it = st.number_input("IT Load (MW)", 10.0, 500.0, 100.0, step=10.0)
        target_avail_pct = st.number_input("Target Availability (%)", 99.0, 99.99999, 99.999, format="%.5f")
        target_avail = target_avail_pct / 100.0
        
        # Show target tier
        target_tier, _ = get_tier_level(target_avail)
        st.info(f"🎯 Target: Tier {target_tier}")
        
        volts_mode = st.selectbox("Voltage Selection", ["Auto-Calculate", "Manual"])
        manual_kv = 13.8
        if volts_mode == "Manual":
            manual_kv = st.number_input("Distribution Voltage (kV)", 4.16, 34.5, 13.8)

    with st.expander("🔧 2. Generation", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        gen_defaults = CAT_LIBRARY[gen_model]
        
        c1, c2 = st.columns(2)
        gen_xd = c1.number_input("X''d (pu)", 0.05, 0.5, gen_defaults['xd'], format="%.3f")
        gen_step_cap = c2.number_input("Step Cap (%)", 0.0, 100.0, gen_defaults['step_cap'])
        gen_specs = {"mw": gen_defaults['mw'], "xd": gen_xd, "step_cap": gen_step_cap, "voltage_kv": gen_defaults['voltage_kv']}
        
        st.caption("**Generator Reliability**")
        gen_mtbf = st.number_input("Gen MTBF (hours)", 1000, 100000, 43800, step=1000, 
                                   help="Typical: 40,000-50,000 hours")
        gen_mttr = st.number_input("Gen MTTR (hours)", 1, 500, 48, 
                                   help="Typical: 24-72 hours")

    with st.expander("🔋 3. BESS (Bridge Power)", expanded=True):
        enable_bess = st.checkbox("Enable BESS", value=True)
        
        if enable_bess:
            bess_inv_mw = st.number_input("BESS Inverter Unit (MW)", 0.5, 10.0, 3.8)
            bess_duration_min = st.number_input("Duration (minutes)", 1, 60, 10)
            
            st.caption("**BESS Reliability**")
            bess_mtbf = st.number_input("BESS MTBF (hours)", 1000, 100000, 50000)
            bess_mttr = st.number_input("BESS MTTR (hours)", 1, 200, 24)
        else:
            bess_inv_mw = 0
            bess_duration_min = 0
            bess_mtbf = 50000
            bess_mttr = 24

    with st.expander("⚡ 4. Transformers", expanded=False):
        xfmr_auto = st.checkbox("Auto-Select Transformers", value=True)
        
        if not xfmr_auto:
            xfmr_model = st.selectbox("Transformer Size", list(TRANSFORMER_LIBRARY.keys()))
            xfmr_specs = TRANSFORMER_LIBRARY[xfmr_model]
        
        lv_voltage = st.selectbox("LV Voltage", [480, 415, 400], help="Data center distribution voltage")

    with st.expander("🔌 5. Switchgear", expanded=False):
        swgr_auto = st.checkbox("Auto-Select Switchgear", value=True)
        
        if not swgr_auto:
            mv_swgr_model = st.selectbox("MV Switchgear", list(SWITCHGEAR_MV.keys()))
            mv_swgr_specs = SWITCHGEAR_MV[mv_swgr_model]
            
            lv_swgr_model = st.selectbox("LV Switchgear", list(SWITCHGEAR_LV.keys()))
            lv_swgr_specs = SWITCHGEAR_LV[lv_swgr_model]

    with st.expander("📈 6. Substation Reliability", expanded=False):
        st.caption("**Bus & Breaker**")
        bus_mtbf = st.number_input("Bus MTBF (hours)", 100000, 5000000, 1000000)
        bus_mttr = st.number_input("Bus MTTR (hours)", 1, 500, 12)
        cb_mtbf = st.number_input("Breaker MTBF (hours)", 50000, 1000000, 200000)
        cb_mttr = st.number_input("Breaker MTTR (hours)", 1, 100, 8)

    with st.expander("🔒 7. Equipment Limits", expanded=False):
        bus_amp_limit = st.number_input("MV Bus Amp Limit (A)", 1000, 6000, 3000)
        sc_limit_ka = st.number_input("MV Switchgear kAIC", 25, 100, 50)
        lv_sc_limit_ka = st.number_input("LV Switchgear kAIC", 50, 200, 100)

# ==============================================================================
# 4. CALCULATION ENGINE
# ==============================================================================

# --- STEP 1: LOAD & VOLTAGE ---
dc_aux = 15.0  # % auxiliaries
dist_loss = 1.5  # % distribution losses
parasitics = 3.0  # % parasitic loads

p_gross = (p_it * (1 + dc_aux/100)) / ((1 - dist_loss/100) * (1 - parasitics/100))

# Auto-select voltage based on current
if volts_mode == "Manual":
    calc_kv = manual_kv
else:
    raw_amps_13 = calc_amps(p_gross, 13.8)
    if raw_amps_13 > 4000:
        calc_kv = 34.5
    elif raw_amps_13 > 2000:
        calc_kv = 13.8
    else:
        calc_kv = 4.16

# --- STEP 2: GENERATOR FLEET ---
n_gen_needed = math.ceil(p_gross / gen_specs['mw'])
p_gen_unit_avail = calc_avail(gen_mtbf, gen_mttr)
u_gen_unit = get_unavailability(gen_mtbf, gen_mttr)

# --- STEP 3: TRANSFORMER SIZING ---
load_per_xfmr_mva = p_gross / 2  # Assume 2 transformers minimum for redundancy

if xfmr_auto:
    xfmr_model, xfmr_specs = select_transformer_size(load_per_xfmr_mva)
else:
    xfmr_specs = TRANSFORMER_LIBRARY[xfmr_model]

n_xfmr_needed = math.ceil(p_gross * 1000 / xfmr_specs['kva'])
n_xfmr_needed = max(2, n_xfmr_needed)  # Minimum 2 for redundancy

p_xfmr_unit_avail = calc_avail(xfmr_specs['mtbf'], xfmr_specs['mttr'])

# --- STEP 4: BESS SIZING ---
if enable_bess:
    bess_target_mw = p_gross
    n_bess_needed = math.ceil(bess_target_mw / bess_inv_mw)
    p_bess_unit_avail = calc_avail(bess_mtbf, bess_mttr)
    
    # Target very high reliability for BESS subsystem
    target_bess_subsys = 0.999999
    n_bess_total, bess_rel_actual = get_n_for_reliability(n_bess_needed, target_bess_subsys, p_bess_unit_avail)
    
    bess_installed_mw = n_bess_total * bess_inv_mw
    bess_energy_mwh = bess_installed_mw * (bess_duration_min / 60.0)
else:
    n_bess_needed = 0
    n_bess_total = 0
    bess_rel_actual = 1.0
    bess_installed_mw = 0
    bess_energy_mwh = 0

# --- STEP 5: SHORT CIRCUIT CALCULATIONS ---
# Generator contribution at MV bus
sc_gen_mv_ka = calc_sc_ka_gen(gen_specs['mw'], gen_specs['xd'], calc_kv, n_gen_needed + 2)  # N+2 for worst case

# Transformer contribution at LV bus
sc_xfmr_lv_ka = calc_sc_ka_xfmr(xfmr_specs['kva'], xfmr_specs['z_pct'], lv_voltage/1000, n_xfmr_needed)

# Total at LV (simplified - actual would need impedance network)
sc_total_lv_ka = sc_xfmr_lv_ka * 0.9  # Factor for cable impedance

# --- STEP 6: SWITCHGEAR SELECTION ---
if swgr_auto:
    mv_swgr_model, mv_swgr_specs = select_mv_switchgear(calc_kv, sc_gen_mv_ka)
    lv_swgr_model, lv_swgr_specs = select_lv_switchgear(sc_total_lv_ka)
else:
    mv_swgr_specs = SWITCHGEAR_MV[mv_swgr_model]
    lv_swgr_specs = SWITCHGEAR_LV[lv_swgr_model]

# --- STEP 7: CURRENT CALCULATIONS ---
mv_bus_current = calc_amps(p_gross, calc_kv)
lv_bus_current = calc_amps(p_gross, lv_voltage/1000)

# --- STEP 8: AVAILABILITY CALCULATION (COMPLETE MODEL) ---

# Component unavailabilities
u_bus = get_unavailability(bus_mtbf, bus_mttr)
u_cb = get_unavailability(cb_mtbf, cb_mttr)
u_xfmr = get_unavailability(xfmr_specs['mtbf'], xfmr_specs['mttr'])

# BaaH topology: Loss of access requires concurrent failures
u_access_baah = (u_cb * u_cb) + (u_bus * u_cb) + (u_cb * u_bus) + (u_bus * u_bus)
p_gen_effective_baah = 1.0 - (u_gen_unit + u_access_baah)

# Generator subsystem with optimization
target_gen_sys = target_avail / bess_rel_actual if bess_rel_actual > 0 else target_avail
n_gen_total, gen_sys_rel = get_n_for_reliability(n_gen_needed, target_gen_sys, p_gen_effective_baah)

# Transformer subsystem (N+1 redundancy)
n_xfmr_total = n_xfmr_needed + 1
xfmr_sys_rel = rel_k_out_n(n_xfmr_needed, n_xfmr_total, p_xfmr_unit_avail)

# Complete system availability (series model with BESS backup)
# A_total = A_gen × A_xfmr × A_swgr + (1 - A_gen × A_xfmr × A_swgr) × A_bess × P_switchover
p_swgr = calc_avail(mv_swgr_specs['mtbf'], mv_swgr_specs['mttr'])
a_primary = gen_sys_rel * xfmr_sys_rel * p_swgr
p_switchover = 0.9999  # Probability of successful transfer to BESS

if enable_bess:
    total_system_avail = a_primary + (1 - a_primary) * bess_rel_actual * p_switchover
else:
    total_system_avail = a_primary

# Get Tier classification
achieved_tier, tier_info = get_tier_level(total_system_avail)
target_tier, target_tier_info = get_tier_level(target_avail)

# ==============================================================================
# 5. MAIN DISPLAY
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1>🔌 CAT Topology Designer</h1>
    <p>Power System Design & Availability Analysis for Data Centers</p>
</div>
""", unsafe_allow_html=True)

st.caption(f"**Project:** {project_name} | **IT Load:** {p_it:.0f} MW | **Gross Load:** {p_gross:.1f} MW")

# --- TIER & AVAILABILITY SUMMARY ---
col_tier, col_avail, col_downtime = st.columns(3)

with col_tier:
    tier_color = "#28a745" if achieved_tier >= target_tier else "#dc3545"
    st.markdown(f"""
    <div class="tier-box" style="border-left-color: {tier_color};">
        <div class="tier-label">ACHIEVED TIER</div>
        <div class="tier-value" style="color: {tier_color};">Tier {achieved_tier}</div>
        <div class="tier-desc">{tier_info['description']}</div>
        <div class="tier-desc">Redundancy: {tier_info['redundancy']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_avail:
    avail_pct = total_system_avail * 100
    st.markdown(f"""
    <div class="tier-box">
        <div class="tier-label">SYSTEM AVAILABILITY</div>
        <div class="tier-value" style="font-size: 36px;">{avail_pct:.5f}%</div>
        <div class="tier-desc">{availability_to_nines(total_system_avail)}</div>
    </div>
    """, unsafe_allow_html=True)

with col_downtime:
    downtime_hr = avail_to_downtime(total_system_avail)
    downtime_min = downtime_hr * 60
    st.markdown(f"""
    <div class="tier-box">
        <div class="tier-label">EXPECTED DOWNTIME</div>
        <div class="tier-value" style="font-size: 36px;">{downtime_min:.1f}</div>
        <div class="tier-desc">minutes per year</div>
        <div class="tier-desc">({downtime_hr:.2f} hours/year)</div>
    </div>
    """, unsafe_allow_html=True)

# --- TARGET CHECK ---
if total_system_avail >= target_avail:
    st.markdown(f"""<div class="success-box">
        ✅ <b>TARGET MET!</b> System availability {total_system_avail*100:.5f}% meets target {target_avail*100:.5f}%
    </div>""", unsafe_allow_html=True)
else:
    gap = target_avail - total_system_avail
    st.markdown(f"""<div class="fail-box">
        ❌ <b>TARGET NOT MET.</b> Gap: {gap*100:.6f}% ({avail_to_downtime(target_avail) - downtime_hr:.2f} hours)
        <br>Consider: More generator redundancy, additional BESS, or higher-reliability components.
    </div>""", unsafe_allow_html=True)

# --- EQUIPMENT SUMMARY ---
st.markdown("### 📋 Equipment Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Voltage Level", f"{calc_kv} kV")
    st.metric("Generator Fleet", f"{n_gen_total} Units", f"N+{n_gen_total - n_gen_needed}")

with col2:
    st.metric("Transformers", f"{n_xfmr_total} × {xfmr_specs['kva']} kVA")
    st.metric("MV Switchgear", mv_swgr_model)

with col3:
    st.metric("LV Switchgear", lv_swgr_model)
    if enable_bess:
        st.metric("BESS Fleet", f"{n_bess_total} Units", f"N+{n_bess_total - n_bess_needed}")
    else:
        st.metric("BESS", "Disabled")

with col4:
    st.metric("MV Short Circuit", f"{sc_gen_mv_ka:.1f} kA")
    st.metric("LV Short Circuit", f"{sc_total_lv_ka:.1f} kA")

# --- VALIDATIONS ---
st.markdown("### ✅ Equipment Validations")

validations = []

# MV Short Circuit Validation
val_mv_sc = validate_short_circuit(sc_gen_mv_ka, mv_swgr_specs['kaic'], "MV Switchgear")
validations.append(val_mv_sc)

# LV Short Circuit Validation
val_lv_sc = validate_short_circuit(sc_total_lv_ka, lv_swgr_specs['kaic'], "LV Switchgear")
validations.append(val_lv_sc)

# MV Ampacity Validation
val_mv_amp = validate_ampacity(mv_bus_current, bus_amp_limit, "MV Bus")
validations.append(val_mv_amp)

# LV Ampacity Validation
val_lv_amp = validate_ampacity(lv_bus_current, lv_swgr_specs['continuous_a'], "LV Bus")
validations.append(val_lv_amp)

# Display validations
n_pass = sum(1 for v in validations if v['status'] == 'PASS')
n_warn = sum(1 for v in validations if v['status'] == 'WARNING')
n_fail = sum(1 for v in validations if v['status'] == 'FAIL')

col_v1, col_v2, col_v3 = st.columns(3)
col_v1.metric("✅ Passed", n_pass)
col_v2.metric("⚠️ Warnings", n_warn)
col_v3.metric("❌ Failed", n_fail)

for val in validations:
    if val['status'] == 'FAIL':
        st.markdown(f"<div class='fail-box'>{val['message']}<br><i>Recommendation: {val['recommendation']}</i></div>", 
                   unsafe_allow_html=True)
    elif val['status'] == 'WARNING':
        st.markdown(f"<div class='warning-box'>{val['message']}<br><i>Recommendation: {val['recommendation']}</i></div>", 
                   unsafe_allow_html=True)
    else:
        st.success(val['message'])

# --- CALCULATION BREAKDOWN ---
with st.expander("🧮 CALCULATION BREAKDOWN", expanded=False):
    st.markdown("### 1. Load Calculation")
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | IT Load | {p_it:.1f} MW |
    | Auxiliaries ({dc_aux}%) | +{p_it * dc_aux/100:.1f} MW |
    | Distribution Losses ({dist_loss}%) | Factor |
    | Parasitics ({parasitics}%) | Factor |
    | **Gross Load** | **{p_gross:.2f} MW** |
    """)
    
    st.markdown("### 2. Generator Subsystem")
    st.markdown(f"""
    * **Unit Availability:** $A_{{gen}} = \\frac{{{gen_mtbf}}}{{{gen_mtbf} + {gen_mttr}}} = {p_gen_unit_avail:.6f}$
    * **Access Unavailability (BaaH):** ${u_access_baah:.2e}$ (negligible)
    * **Effective Reliability:** ${p_gen_effective_baah:.6f}$
    * **Fleet:** {n_gen_needed} needed → {n_gen_total} installed (N+{n_gen_total - n_gen_needed})
    * **Subsystem Reliability:** ${gen_sys_rel:.8f}$
    """)
    
    st.markdown("### 3. Transformer Subsystem")
    st.markdown(f"""
    * **Unit Availability:** ${p_xfmr_unit_avail:.6f}$
    * **Fleet:** {n_xfmr_needed} needed → {n_xfmr_total} installed (N+1)
    * **Subsystem Reliability:** ${xfmr_sys_rel:.8f}$
    """)
    
    if enable_bess:
        st.markdown("### 4. BESS Subsystem")
        st.markdown(f"""
        * **Unit Availability:** ${p_bess_unit_avail:.6f}$
        * **Fleet:** {n_bess_needed} needed → {n_bess_total} installed
        * **Subsystem Reliability:** ${bess_rel_actual:.8f}$
        * **Installed Power:** {bess_installed_mw:.1f} MW
        * **Energy:** {bess_energy_mwh:.1f} MWh ({bess_duration_min} min)
        """)
    
    st.markdown("### 5. Total System Availability")
    st.markdown(f"""
    **Model:** Primary path with BESS backup
    
    $A_{{total}} = A_{{primary}} + (1 - A_{{primary}}) \\times A_{{BESS}} \\times P_{{switchover}}$
    
    Where:
    * $A_{{primary}} = A_{{gen}} \\times A_{{xfmr}} \\times A_{{swgr}} = {a_primary:.8f}$
    * $A_{{BESS}} = {bess_rel_actual:.8f}$
    * $P_{{switchover}} = {p_switchover}$
    
    **Result:** $A_{{total}} = {total_system_avail:.9f}$ = **{total_system_avail*100:.5f}%**
    """)

# --- ARCHITECTURE DIAGRAM ---
st.markdown("### 📐 Architecture Layout")

# Create improved Graphviz diagram
dot = graphviz.Digraph()
dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
dot.attr('node', fontname='Arial', fontsize='10')
dot.attr('edge', fontname='Arial', fontsize='8')

# Generators cluster
with dot.subgraph(name='cluster_gen') as gen:
    gen.attr(label=f'Generation ({n_gen_total} × {gen_specs["mw"]} MW)', style='dashed', color='darkgreen')
    n_show = min(4, n_gen_total)
    for i in range(n_show):
        gen.node(f'G{i}', f'G{i+1}\n{gen_specs["mw"]}MW', shape='circle', 
                style='filled', fillcolor='lightgreen', width='0.8')
    if n_gen_total > n_show:
        gen.node('Gmore', f'...+{n_gen_total - n_show}', shape='plaintext')

# MV Switchgear
dot.node('MV_BUS_A', f'MV Bus A ({calc_kv}kV)', shape='rect', width='4', height='0.2',
        style='filled', fillcolor='#333333', fontcolor='white')
dot.node('MV_BUS_B', f'MV Bus B ({calc_kv}kV)', shape='rect', width='4', height='0.2',
        style='filled', fillcolor='#333333', fontcolor='white')

# Transformers
with dot.subgraph(name='cluster_xfmr') as xfmr:
    xfmr.attr(label=f'Transformers ({n_xfmr_total} × {xfmr_specs["kva"]} kVA)', style='dashed', color='orange')
    n_show_x = min(3, n_xfmr_total)
    for i in range(n_show_x):
        xfmr.node(f'T{i}', f'T{i+1}\n{xfmr_specs["kva"]}kVA\n{xfmr_specs["z_pct"]}%Z', 
                 shape='box', style='rounded,filled', fillcolor='lightyellow')

# LV Switchgear
dot.node('LV_BUS', f'LV Bus ({lv_voltage}V)', shape='rect', width='5', height='0.2',
        style='filled', fillcolor='#666666', fontcolor='white')

# BESS
if enable_bess:
    dot.node('BESS', f'BESS\n{bess_installed_mw:.1f}MW\n{bess_energy_mwh:.1f}MWh', 
            shape='box3d', style='filled', fillcolor='lightblue')

# Data Center
dot.node('DC', f'Data Center\n{p_it} MW IT Load', shape='house', 
        style='filled', fillcolor='#e1bee7')

# Connections - Generators to MV Bus
for i in range(min(2, n_gen_total)):
    dot.edge(f'G{i}', 'MV_BUS_A', label='CB')
for i in range(2, min(4, n_gen_total)):
    dot.edge(f'G{i}', 'MV_BUS_B', label='CB')

# Bus tie
dot.edge('MV_BUS_A', 'MV_BUS_B', label='Tie', style='dashed', dir='both')

# Transformers
for i in range(min(3, n_xfmr_total)):
    if i % 2 == 0:
        dot.edge('MV_BUS_A', f'T{i}')
    else:
        dot.edge('MV_BUS_B', f'T{i}')
    dot.edge(f'T{i}', 'LV_BUS')

# BESS to LV
if enable_bess:
    dot.edge('BESS', 'LV_BUS', label='PCS')

# LV to DC
dot.edge('LV_BUS', 'DC', label='Feeders')

st.graphviz_chart(dot, use_container_width=True)

# --- BILL OF MATERIALS ---
st.markdown("### 📦 Bill of Materials (Summary)")

bom_data = [
    {"Item": "Generators", "Model": gen_model, "Quantity": n_gen_total, "Rating": f"{gen_specs['mw']} MW each"},
    {"Item": "MV Switchgear", "Model": mv_swgr_model, "Quantity": 2, "Rating": f"{mv_swgr_specs['kaic']} kAIC"},
    {"Item": "Transformers", "Model": xfmr_model, "Quantity": n_xfmr_total, "Rating": f"{xfmr_specs['kva']} kVA each"},
    {"Item": "LV Switchgear", "Model": lv_swgr_model, "Quantity": 2, "Rating": f"{lv_swgr_specs['kaic']} kAIC"},
]

if enable_bess:
    bom_data.append({"Item": "BESS Inverters", "Model": f"{bess_inv_mw} MW units", "Quantity": n_bess_total, 
                    "Rating": f"{bess_installed_mw:.1f} MW / {bess_energy_mwh:.1f} MWh total"})

df_bom = pd.DataFrame(bom_data)
st.dataframe(df_bom, use_container_width=True, hide_index=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>🔌 CAT Topology Designer v21.0</b></p>
    <p>Power System Design with Tier Classification, Equipment Validation & Availability Analysis</p>
    <p>Caterpillar Electric Power | 2026</p>
</div>
""", unsafe_allow_html=True)
