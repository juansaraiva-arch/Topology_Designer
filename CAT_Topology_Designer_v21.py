import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v22.0", page_icon="🔌", layout="wide")

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
    .info-box { 
        background-color: #e3f2fd; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #2196f3; 
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
    
    .voltage-path {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #ffb300;
        margin: 10px 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. DATA LIBRARIES - CORRECTED VOLTAGE LEVELS
# ==============================================================================

# Generator library with CORRECT terminal voltages
CAT_LIBRARY = {
    # Low Voltage Generators (can have step-up to MV)
    "XGC1900 (1.9 MW)": {
        "mw": 1.9,
        "available_voltages_kv": [0.48],  # Only LV
        "default_voltage_kv": 0.48,
        "xd": 0.16,
        "step_cap": 25.0,
        "type": "LV",
        "mtbf": 43800,
        "mttr": 48,
    },
    "G3520K (2.4 MW)": {
        "mw": 2.4,
        "available_voltages_kv": [0.48, 4.16, 13.8],  # LV or MV options
        "default_voltage_kv": 0.48,
        "xd": 0.16,
        "step_cap": 25.0,
        "type": "LV/MV",
        "mtbf": 43800,
        "mttr": 48,
    },
    "G3520FR (2.5 MW)": {
        "mw": 2.5,
        "available_voltages_kv": [0.48, 4.16, 13.8],  # LV or MV options
        "default_voltage_kv": 0.48,
        "xd": 0.16,
        "step_cap": 40.0,  # Fast Response
        "type": "LV/MV",
        "mtbf": 43800,
        "mttr": 48,
    },
    # Medium Voltage Generators
    "CG260-16 (3.96 MW)": {
        "mw": 3.96,
        "available_voltages_kv": [4.16, 11.0, 13.8],
        "default_voltage_kv": 11.0,
        "xd": 0.15,
        "step_cap": 25.0,
        "type": "MV",
        "mtbf": 50000,
        "mttr": 72,
    },
    "G20CM34 (9.76 MW)": {
        "mw": 9.76,
        "available_voltages_kv": [11.0, 13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.14,
        "step_cap": 20.0,
        "type": "MV",
        "mtbf": 50000,
        "mttr": 72,
    },
    "Titan 130 (16.5 MW)": {
        "mw": 16.5,
        "available_voltages_kv": [13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.14,
        "step_cap": 15.0,
        "type": "Gas Turbine",
        "mtbf": 40000,
        "mttr": 96,
    },
    "Titan 250 (23.2 MW)": {
        "mw": 23.2,
        "available_voltages_kv": [13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.14,
        "step_cap": 15.0,
        "type": "Gas Turbine",
        "mtbf": 40000,
        "mttr": 96,
    },
    "Titan 350 (38.0 MW)": {
        "mw": 38.0,
        "available_voltages_kv": [13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.14,
        "step_cap": 12.0,
        "type": "Gas Turbine",
        "mtbf": 40000,
        "mttr": 96,
    },
}

# Step-Up Transformer Library (Generator terminal to MV distribution)
STEP_UP_XFMR_LIBRARY = {
    "3 MVA 0.48/13.8kV": {"mva": 3.0, "primary_kv": 0.48, "secondary_kv": 13.8, "z_pct": 6.0, "mtbf": 200000, "mttr": 168},
    "5 MVA 0.48/13.8kV": {"mva": 5.0, "primary_kv": 0.48, "secondary_kv": 13.8, "z_pct": 6.5, "mtbf": 200000, "mttr": 168},
    "7.5 MVA 0.48/13.8kV": {"mva": 7.5, "primary_kv": 0.48, "secondary_kv": 13.8, "z_pct": 7.0, "mtbf": 200000, "mttr": 168},
    "5 MVA 4.16/13.8kV": {"mva": 5.0, "primary_kv": 4.16, "secondary_kv": 13.8, "z_pct": 6.0, "mtbf": 200000, "mttr": 168},
    "10 MVA 4.16/13.8kV": {"mva": 10.0, "primary_kv": 4.16, "secondary_kv": 13.8, "z_pct": 7.0, "mtbf": 200000, "mttr": 168},
    "15 MVA 13.8/34.5kV": {"mva": 15.0, "primary_kv": 13.8, "secondary_kv": 34.5, "z_pct": 8.0, "mtbf": 200000, "mttr": 168},
    "25 MVA 13.8/34.5kV": {"mva": 25.0, "primary_kv": 13.8, "secondary_kv": 34.5, "z_pct": 8.5, "mtbf": 200000, "mttr": 168},
    "50 MVA 13.8/34.5kV": {"mva": 50.0, "primary_kv": 13.8, "secondary_kv": 34.5, "z_pct": 9.0, "mtbf": 200000, "mttr": 168},
}

# Step-Down Transformer Library (Distribution to DC)
STEP_DOWN_XFMR_LIBRARY = {
    "1500 kVA 13.8/0.48kV": {"kva": 1500, "primary_kv": 13.8, "secondary_kv": 0.48, "z_pct": 5.75, "mtbf": 200000, "mttr": 168},
    "2000 kVA 13.8/0.48kV": {"kva": 2000, "primary_kv": 13.8, "secondary_kv": 0.48, "z_pct": 5.75, "mtbf": 200000, "mttr": 168},
    "2500 kVA 13.8/0.48kV": {"kva": 2500, "primary_kv": 13.8, "secondary_kv": 0.48, "z_pct": 5.75, "mtbf": 200000, "mttr": 168},
    "3000 kVA 13.8/0.48kV": {"kva": 3000, "primary_kv": 13.8, "secondary_kv": 0.48, "z_pct": 6.0, "mtbf": 200000, "mttr": 168},
    "3750 kVA 13.8/0.48kV": {"kva": 3750, "primary_kv": 13.8, "secondary_kv": 0.48, "z_pct": 6.0, "mtbf": 200000, "mttr": 168},
    "2000 kVA 34.5/0.48kV": {"kva": 2000, "primary_kv": 34.5, "secondary_kv": 0.48, "z_pct": 5.75, "mtbf": 200000, "mttr": 168},
    "2500 kVA 34.5/0.48kV": {"kva": 2500, "primary_kv": 34.5, "secondary_kv": 0.48, "z_pct": 5.75, "mtbf": 200000, "mttr": 168},
    "3000 kVA 34.5/0.48kV": {"kva": 3000, "primary_kv": 34.5, "secondary_kv": 0.48, "z_pct": 6.0, "mtbf": 200000, "mttr": 168},
}

# Switchgear Libraries
SWITCHGEAR_MV = {
    "5kV, 50kA": {"voltage_kv": 5, "kaic": 50, "continuous_a": 3000, "bil_kv": 60, "mtbf": 500000, "mttr": 24},
    "15kV, 25kA": {"voltage_kv": 15, "kaic": 25, "continuous_a": 1200, "bil_kv": 95, "mtbf": 500000, "mttr": 24},
    "15kV, 40kA": {"voltage_kv": 15, "kaic": 40, "continuous_a": 2000, "bil_kv": 95, "mtbf": 500000, "mttr": 24},
    "15kV, 50kA": {"voltage_kv": 15, "kaic": 50, "continuous_a": 3000, "bil_kv": 95, "mtbf": 500000, "mttr": 24},
    "15kV, 63kA": {"voltage_kv": 15, "kaic": 63, "continuous_a": 4000, "bil_kv": 95, "mtbf": 500000, "mttr": 24},
    "38kV, 25kA": {"voltage_kv": 38, "kaic": 25, "continuous_a": 1200, "bil_kv": 150, "mtbf": 500000, "mttr": 24},
    "38kV, 40kA": {"voltage_kv": 38, "kaic": 40, "continuous_a": 2000, "bil_kv": 150, "mtbf": 500000, "mttr": 24},
    "38kV, 50kA": {"voltage_kv": 38, "kaic": 50, "continuous_a": 3000, "bil_kv": 150, "mtbf": 500000, "mttr": 24},
}

SWITCHGEAR_LV = {
    "480V, 65kA": {"voltage_v": 480, "kaic": 65, "continuous_a": 3000, "mtbf": 300000, "mttr": 8},
    "480V, 100kA": {"voltage_v": 480, "kaic": 100, "continuous_a": 4000, "mtbf": 300000, "mttr": 8},
    "480V, 150kA": {"voltage_v": 480, "kaic": 150, "continuous_a": 5000, "mtbf": 300000, "mttr": 8},
    "480V, 200kA": {"voltage_v": 480, "kaic": 200, "continuous_a": 6000, "mtbf": 300000, "mttr": 8},
}

# Tier Classification
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
    if (mtbf + mttr) <= 0: return 0.0
    return mtbf / (mtbf + mttr)

def get_unavailability(mtbf, mttr):
    if (mtbf + mttr) <= 0: return 1.0
    return mttr / (mtbf + mttr)

def rel_k_out_n(n_needed, n_total, p_unit):
    if n_total < n_needed: return 0.0
    prob = 0.0
    for k in range(n_needed, n_total + 1):
        prob += binom.pmf(k, n_total, p_unit)
    return prob

def get_n_for_reliability(n_needed, target_avail, p_unit_avail):
    for added_redundancy in range(0, 50):
        n_total = n_needed + added_redundancy
        prob = rel_k_out_n(n_needed, n_total, p_unit_avail)
        if prob >= target_avail:
            return n_total, prob
    return n_needed + 50, 0.0

def calc_amps(mw, kv):
    if kv <= 0: return 0
    return (mw * 1e6) / (math.sqrt(3) * kv * 1000)

def calc_sc_ka(mva, z_pct, kv):
    """Calculate short circuit current from transformer/source."""
    if kv <= 0 or z_pct <= 0: return 0
    i_base = (mva * 1e6) / (math.sqrt(3) * kv * 1000)
    i_sc = i_base / (z_pct / 100)
    return i_sc / 1000

def calc_sc_ka_gen(mw_gen, xd, kv, n_gens):
    """Calculate short circuit contribution from generators."""
    if kv <= 0 or xd <= 0: return 0
    mva_gen = mw_gen / 0.8
    i_base = (mva_gen * 1e6) / (math.sqrt(3) * kv * 1000)
    i_sc_unit = i_base / xd
    return (i_sc_unit * n_gens) / 1000.0

def get_tier_level(availability):
    for tier, info in TIER_LEVELS.items():
        if availability >= info['min_avail']:
            return tier, info
    return "Below I", {"min_avail": 0, "redundancy": "N", "description": "Below Standard", "downtime_hr_yr": ">28.8"}

def availability_to_nines(avail):
    if avail >= 0.999999: return "6 nines (99.9999%)"
    elif avail >= 0.99999: return "5 nines (99.999%)"
    elif avail >= 0.9999: return "4 nines (99.99%)"
    elif avail >= 0.999: return "3 nines (99.9%)"
    elif avail >= 0.99: return "2 nines (99%)"
    else: return f"{avail*100:.2f}%"

def avail_to_downtime(avail, hours_per_year=8760):
    return (1 - avail) * hours_per_year

# ==============================================================================
# 2. VOLTAGE TOPOLOGY DESIGN FUNCTIONS
# ==============================================================================

def design_voltage_topology(p_gross_mw, gen_terminal_kv, n_gens, gen_xd, max_swgr_kaic=50, max_bus_amps=3000):
    """
    Design optimal voltage topology based on:
    - Generator terminal voltage
    - Ampacity limits
    - Short circuit limits
    
    Returns topology design with all voltage levels and transformer requirements.
    """
    topology = {
        'gen_terminal_kv': gen_terminal_kv,
        'needs_gen_step_up': False,
        'gen_step_up_ratio': None,
        'mv_bus_kv': gen_terminal_kv,
        'needs_distribution_step_up': False,
        'distribution_step_up_ratio': None,
        'distribution_kv': gen_terminal_kv,
        'dc_voltage_kv': 0.48,
        'issues': [],
        'recommendations': [],
    }
    
    # Calculate current at generator terminal voltage
    i_at_gen_voltage = calc_amps(p_gross_mw, gen_terminal_kv)
    
    # Calculate short circuit at generator terminal voltage
    sc_at_gen_voltage = calc_sc_ka_gen(p_gross_mw / n_gens, gen_xd, gen_terminal_kv, n_gens)
    
    # Store calculations
    topology['i_at_gen_voltage'] = i_at_gen_voltage
    topology['sc_at_gen_voltage'] = sc_at_gen_voltage
    
    # CASE 1: LV Generators (0.48 kV)
    if gen_terminal_kv < 1.0:
        # Always need step-up for distribution
        topology['needs_gen_step_up'] = True
        topology['gen_step_up_ratio'] = f"{gen_terminal_kv}/13.8 kV"
        topology['mv_bus_kv'] = 13.8
        topology['distribution_kv'] = 13.8
        
        # Recalculate at MV level
        i_at_mv = calc_amps(p_gross_mw, 13.8)
        # SC at MV is limited by transformer impedance
        topology['i_at_distribution'] = i_at_mv
        
        topology['recommendations'].append(f"LV generators require step-up transformers to 13.8 kV")
    
    # CASE 2: MV Generators (4.16, 11, 13.8 kV)
    else:
        # Check if current exceeds bus limits
        if i_at_gen_voltage > max_bus_amps:
            topology['issues'].append(f"Current {i_at_gen_voltage:.0f}A exceeds bus limit {max_bus_amps}A at {gen_terminal_kv}kV")
            
            # Need to step up to higher voltage
            if gen_terminal_kv <= 15:
                topology['needs_distribution_step_up'] = True
                topology['distribution_step_up_ratio'] = f"{gen_terminal_kv}/34.5 kV"
                topology['distribution_kv'] = 34.5
                
                i_at_34 = calc_amps(p_gross_mw, 34.5)
                topology['i_at_distribution'] = i_at_34
                topology['recommendations'].append(f"Step-up to 34.5 kV reduces current to {i_at_34:.0f}A")
        else:
            topology['i_at_distribution'] = i_at_gen_voltage
        
        # Check short circuit
        if sc_at_gen_voltage > max_swgr_kaic:
            topology['issues'].append(f"Short circuit {sc_at_gen_voltage:.1f}kA exceeds switchgear limit {max_swgr_kaic}kA")
            
            if not topology['needs_distribution_step_up']:
                # Step-up helps limit SC through transformer impedance
                topology['needs_distribution_step_up'] = True
                topology['distribution_step_up_ratio'] = f"{gen_terminal_kv}/34.5 kV"
                topology['distribution_kv'] = 34.5
                topology['recommendations'].append("Step-up transformer impedance will limit fault current")
        
        topology['mv_bus_kv'] = gen_terminal_kv
    
    return topology

def select_step_up_transformer(gen_mw, gen_kv, target_kv, n_gens_per_xfmr=1):
    """Select appropriate step-up transformer."""
    required_mva = gen_mw * n_gens_per_xfmr * 1.25  # 25% margin
    
    # Find matching transformer
    for name, specs in STEP_UP_XFMR_LIBRARY.items():
        if (abs(specs['primary_kv'] - gen_kv) < 0.5 and 
            abs(specs['secondary_kv'] - target_kv) < 1.0 and
            specs['mva'] >= required_mva):
            return name, specs
    
    # Return largest if no match
    return list(STEP_UP_XFMR_LIBRARY.keys())[-1], list(STEP_UP_XFMR_LIBRARY.values())[-1]

def select_step_down_transformer(load_mva, primary_kv, secondary_kv=0.48):
    """Select appropriate step-down transformer."""
    required_kva = load_mva * 1000 * 1.25  # 25% margin
    
    for name, specs in STEP_DOWN_XFMR_LIBRARY.items():
        if (abs(specs['primary_kv'] - primary_kv) < 1.0 and 
            abs(specs['secondary_kv'] - secondary_kv) < 0.05 and
            specs['kva'] >= required_kva):
            return name, specs
    
    return list(STEP_DOWN_XFMR_LIBRARY.keys())[-1], list(STEP_DOWN_XFMR_LIBRARY.values())[-1]

def select_switchgear(voltage_kv, required_kaic, required_amps):
    """Select appropriate switchgear."""
    library = SWITCHGEAR_MV if voltage_kv > 1 else SWITCHGEAR_LV
    
    for name, specs in library.items():
        voltage_key = 'voltage_kv' if 'voltage_kv' in specs else 'voltage_v'
        rated_v = specs[voltage_key]
        
        # Check voltage compatibility
        if voltage_kv > 1:  # MV
            if rated_v < voltage_kv:
                continue
        
        if specs['kaic'] >= required_kaic * 1.1 and specs['continuous_a'] >= required_amps * 1.1:
            return name, specs
    
    return list(library.keys())[-1], list(library.values())[-1]

# ==============================================================================
# 3. VALIDATION FUNCTIONS
# ==============================================================================

def validate_short_circuit(sc_calculated_ka, equipment_kaic, equipment_name):
    margin = (equipment_kaic - sc_calculated_ka) / equipment_kaic * 100 if equipment_kaic > 0 else -100
    
    if sc_calculated_ka > equipment_kaic:
        return {
            'status': 'FAIL',
            'message': f"⚠️ {equipment_name}: Isc {sc_calculated_ka:.1f}kA EXCEEDS rating {equipment_kaic}kA",
            'recommendation': "Increase kAIC rating, add reactors, or split buses",
            'margin': margin
        }
    elif margin < 20:
        return {
            'status': 'WARNING',
            'message': f"⚡ {equipment_name}: Isc {sc_calculated_ka:.1f}kA close to rating {equipment_kaic}kA ({margin:.0f}% margin)",
            'recommendation': "Consider higher rated equipment",
            'margin': margin
        }
    else:
        return {
            'status': 'PASS',
            'message': f"✅ {equipment_name}: Isc {sc_calculated_ka:.1f}kA OK (rating {equipment_kaic}kA, {margin:.0f}% margin)",
            'recommendation': None,
            'margin': margin
        }

def validate_ampacity(current_calculated, equipment_rating, equipment_name):
    margin = (equipment_rating - current_calculated) / equipment_rating * 100 if equipment_rating > 0 else -100
    
    if current_calculated > equipment_rating:
        return {
            'status': 'FAIL',
            'message': f"⚠️ {equipment_name}: Current {current_calculated:.0f}A EXCEEDS rating {equipment_rating}A",
            'recommendation': "Increase rating, add parallel paths, or increase voltage",
            'margin': margin
        }
    elif margin < 20:
        return {
            'status': 'WARNING',
            'message': f"⚡ {equipment_name}: Current {current_calculated:.0f}A close to rating {equipment_rating}A ({margin:.0f}% margin)",
            'recommendation': "Consider larger equipment",
            'margin': margin
        }
    else:
        return {
            'status': 'PASS',
            'message': f"✅ {equipment_name}: Current {current_calculated:.0f}A OK (rating {equipment_rating}A, {margin:.0f}% margin)",
            'recommendation': None,
            'margin': margin
        }

# ==============================================================================
# 4. SIDEBAR INPUTS
# ==============================================================================

with st.sidebar:
    st.markdown("## ⚡ CAT Topology Designer")
    st.caption("v22.0 - Correct Voltage Topology")
    
    with st.expander("📊 1. Project & Load", expanded=True):
        project_name = st.text_input("Project Name", "AI Data Center")
        p_it = st.number_input("IT Load (MW)", 10.0, 500.0, 100.0, step=10.0)
        target_avail_pct = st.number_input("Target Availability (%)", 99.0, 99.99999, 99.999, format="%.5f")
        target_avail = target_avail_pct / 100.0
        
        target_tier, _ = get_tier_level(target_avail)
        st.info(f"🎯 Target: Tier {target_tier}")

    with st.expander("🔧 2. Generation", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        gen_specs = CAT_LIBRARY[gen_model]
        
        st.caption(f"**Type:** {gen_specs['type']} | **Rating:** {gen_specs['mw']} MW")
        
        # Voltage selection based on available options
        available_voltages = gen_specs['available_voltages_kv']
        if len(available_voltages) > 1:
            voltage_options = [f"{v} kV" for v in available_voltages]
            selected_voltage_str = st.selectbox(
                "Generator Terminal Voltage",
                voltage_options,
                index=voltage_options.index(f"{gen_specs['default_voltage_kv']} kV"),
                help="Select generator output voltage"
            )
            gen_terminal_kv = float(selected_voltage_str.replace(" kV", ""))
        else:
            gen_terminal_kv = available_voltages[0]
            st.info(f"📌 Fixed terminal voltage: {gen_terminal_kv} kV")
        
        col1, col2 = st.columns(2)
        gen_xd = col1.number_input("X''d (pu)", 0.05, 0.5, gen_specs['xd'], format="%.3f")
        gen_step_cap = col2.number_input("Step Cap (%)", 0.0, 100.0, gen_specs['step_cap'])
        
        st.caption("**Reliability Parameters**")
        gen_mtbf = st.number_input("Gen MTBF (hours)", 1000, 100000, gen_specs['mtbf'])
        gen_mttr = st.number_input("Gen MTTR (hours)", 1, 500, gen_specs['mttr'])

    with st.expander("🔋 3. BESS (Bridge Power)", expanded=True):
        enable_bess = st.checkbox("Enable BESS", value=True)
        
        if enable_bess:
            bess_inv_mw = st.number_input("BESS Inverter Unit (MW)", 0.5, 10.0, 3.8)
            bess_duration_min = st.number_input("Duration (minutes)", 1, 60, 10)
            bess_mtbf = st.number_input("BESS MTBF (hours)", 1000, 100000, 50000)
            bess_mttr = st.number_input("BESS MTTR (hours)", 1, 200, 24)
        else:
            bess_inv_mw = 0
            bess_duration_min = 0
            bess_mtbf = 50000
            bess_mttr = 24

    with st.expander("⚡ 4. Voltage & Limits", expanded=False):
        st.caption("**Switchgear Limits**")
        max_mv_bus_amps = st.number_input("Max MV Bus Amps", 1000, 6000, 3000)
        max_mv_kaic = st.number_input("Max MV kAIC", 25, 100, 50)
        max_lv_kaic = st.number_input("Max LV kAIC", 50, 200, 100)
        
        st.caption("**DC Distribution**")
        dc_voltage = st.selectbox("DC Voltage (V)", [480, 415, 400])

    with st.expander("📈 5. Substation Reliability", expanded=False):
        bus_mtbf = st.number_input("Bus MTBF (hours)", 100000, 5000000, 1000000)
        bus_mttr = st.number_input("Bus MTTR (hours)", 1, 500, 12)
        cb_mtbf = st.number_input("Breaker MTBF (hours)", 50000, 1000000, 200000)
        cb_mttr = st.number_input("Breaker MTTR (hours)", 1, 100, 8)

# ==============================================================================
# 5. CALCULATION ENGINE
# ==============================================================================

# --- STEP 1: LOAD CALCULATION ---
dc_aux = 15.0
dist_loss = 1.5
parasitics = 3.0
p_gross = (p_it * (1 + dc_aux/100)) / ((1 - dist_loss/100) * (1 - parasitics/100))

# --- STEP 2: GENERATOR FLEET SIZING ---
n_gen_needed = math.ceil(p_gross / gen_specs['mw'])
p_gen_unit_avail = calc_avail(gen_mtbf, gen_mttr)

# --- STEP 3: VOLTAGE TOPOLOGY DESIGN ---
topology = design_voltage_topology(
    p_gross_mw=p_gross,
    gen_terminal_kv=gen_terminal_kv,
    n_gens=n_gen_needed + 2,  # Use N+2 for worst case SC
    gen_xd=gen_xd,
    max_swgr_kaic=max_mv_kaic,
    max_bus_amps=max_mv_bus_amps
)

# Determine MV bus voltage (after any step-up from generators)
mv_bus_kv = topology['mv_bus_kv']
distribution_kv = topology['distribution_kv']

# --- STEP 4: TRANSFORMER SELECTION ---

# Step-up transformers (if needed)
if topology['needs_gen_step_up']:
    # Group generators per transformer
    gens_per_step_up = 2 if gen_specs['mw'] < 3 else 1
    n_step_up_xfmr = math.ceil(n_gen_needed / gens_per_step_up)
    step_up_name, step_up_specs = select_step_up_transformer(
        gen_specs['mw'], gen_terminal_kv, mv_bus_kv, gens_per_step_up
    )
else:
    n_step_up_xfmr = 0
    step_up_name, step_up_specs = None, None

# Distribution step-up (13.8 to 34.5 if needed)
if topology['needs_distribution_step_up']:
    n_dist_step_up = math.ceil(p_gross / 25)  # ~25 MVA per transformer
    dist_step_up_name, dist_step_up_specs = select_step_up_transformer(
        p_gross / n_dist_step_up, mv_bus_kv, distribution_kv
    )
else:
    n_dist_step_up = 0
    dist_step_up_name, dist_step_up_specs = None, None

# Step-down transformers (to DC voltage)
load_per_step_down_mva = 3.0  # Typical unit substation size
n_step_down = math.ceil(p_gross / load_per_step_down_mva)
n_step_down = max(2, n_step_down)  # Minimum 2 for redundancy
step_down_name, step_down_specs = select_step_down_transformer(
    load_per_step_down_mva, distribution_kv, dc_voltage/1000
)

# --- STEP 5: SHORT CIRCUIT CALCULATIONS ---

# At generator terminal bus
sc_gen_terminal = calc_sc_ka_gen(gen_specs['mw'], gen_xd, gen_terminal_kv, n_gen_needed + 2)

# At MV bus (after step-up, limited by transformer Z)
if topology['needs_gen_step_up'] and step_up_specs:
    # SC is limited by step-up transformer impedance
    total_step_up_mva = n_step_up_xfmr * step_up_specs['mva']
    sc_mv_bus = calc_sc_ka(total_step_up_mva, step_up_specs['z_pct'], mv_bus_kv)
else:
    sc_mv_bus = sc_gen_terminal

# At distribution bus
if topology['needs_distribution_step_up'] and dist_step_up_specs:
    sc_distribution = calc_sc_ka(n_dist_step_up * dist_step_up_specs['mva'], 
                                  dist_step_up_specs['z_pct'], distribution_kv)
else:
    sc_distribution = sc_mv_bus

# At LV bus
sc_lv_bus = calc_sc_ka(n_step_down * step_down_specs['kva']/1000, 
                        step_down_specs['z_pct'], dc_voltage/1000)

# --- STEP 6: CURRENT CALCULATIONS ---
i_gen_terminal = calc_amps(p_gross, gen_terminal_kv)
i_mv_bus = calc_amps(p_gross, mv_bus_kv)
i_distribution = calc_amps(p_gross, distribution_kv)
i_lv_bus = calc_amps(p_gross, dc_voltage/1000)

# --- STEP 7: SWITCHGEAR SELECTION ---
mv_swgr_name, mv_swgr_specs = select_switchgear(mv_bus_kv, sc_mv_bus, i_mv_bus)
lv_swgr_name, lv_swgr_specs = select_switchgear(dc_voltage/1000, sc_lv_bus, i_lv_bus)

if topology['needs_distribution_step_up']:
    dist_swgr_name, dist_swgr_specs = select_switchgear(distribution_kv, sc_distribution, i_distribution)
else:
    dist_swgr_name, dist_swgr_specs = mv_swgr_name, mv_swgr_specs

# --- STEP 8: BESS SIZING ---
if enable_bess:
    bess_target_mw = p_gross
    n_bess_needed = math.ceil(bess_target_mw / bess_inv_mw)
    p_bess_unit_avail = calc_avail(bess_mtbf, bess_mttr)
    n_bess_total, bess_rel_actual = get_n_for_reliability(n_bess_needed, 0.999999, p_bess_unit_avail)
    bess_installed_mw = n_bess_total * bess_inv_mw
    bess_energy_mwh = bess_installed_mw * (bess_duration_min / 60.0)
else:
    n_bess_needed = n_bess_total = 0
    bess_rel_actual = 1.0
    bess_installed_mw = bess_energy_mwh = 0

# --- STEP 9: AVAILABILITY CALCULATION ---
u_bus = get_unavailability(bus_mtbf, bus_mttr)
u_cb = get_unavailability(cb_mtbf, cb_mttr)

# BaaH topology
u_access_baah = 2 * (u_cb ** 2) + 2 * (u_bus * u_cb)
u_gen_unit = get_unavailability(gen_mtbf, gen_mttr)
p_gen_effective = 1.0 - (u_gen_unit + u_access_baah)

# Generator subsystem
target_gen_sys = target_avail / bess_rel_actual if bess_rel_actual > 0 else target_avail
n_gen_total, gen_sys_rel = get_n_for_reliability(n_gen_needed, target_gen_sys, p_gen_effective)

# Transformer subsystem
if step_up_specs:
    p_step_up_avail = calc_avail(step_up_specs['mtbf'], step_up_specs['mttr'])
    step_up_sys_rel = rel_k_out_n(n_step_up_xfmr, n_step_up_xfmr + 1, p_step_up_avail)
else:
    step_up_sys_rel = 1.0

p_step_down_avail = calc_avail(step_down_specs['mtbf'], step_down_specs['mttr'])
step_down_sys_rel = rel_k_out_n(n_step_down, n_step_down + 1, p_step_down_avail)

# Switchgear
p_swgr = calc_avail(mv_swgr_specs['mtbf'], mv_swgr_specs['mttr'])

# Total
a_primary = gen_sys_rel * step_up_sys_rel * step_down_sys_rel * p_swgr
p_switchover = 0.9999

if enable_bess:
    total_system_avail = a_primary + (1 - a_primary) * bess_rel_actual * p_switchover
else:
    total_system_avail = a_primary

achieved_tier, tier_info = get_tier_level(total_system_avail)

# ==============================================================================
# 6. MAIN DISPLAY
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1>🔌 CAT Topology Designer</h1>
    <p>Power System Design with Correct Voltage Topology</p>
</div>
""", unsafe_allow_html=True)

st.caption(f"**Project:** {project_name} | **IT Load:** {p_it:.0f} MW | **Gross Load:** {p_gross:.1f} MW")

# --- VOLTAGE PATH VISUALIZATION ---
st.markdown("### ⚡ Voltage Topology Path")

# Build voltage path string
if topology['needs_gen_step_up'] and topology['needs_distribution_step_up']:
    path_str = f"""
    **Generator** ({gen_terminal_kv} kV) → **Step-Up Xfmr** → **MV Bus** ({mv_bus_kv} kV) → **Step-Up Xfmr** → **Distribution** ({distribution_kv} kV) → **Step-Down Xfmr** → **LV Bus** ({dc_voltage}V) → **Data Center**
    """
elif topology['needs_gen_step_up']:
    path_str = f"""
    **Generator** ({gen_terminal_kv} kV) → **Step-Up Xfmr** → **MV Bus** ({mv_bus_kv} kV) → **Step-Down Xfmr** → **LV Bus** ({dc_voltage}V) → **Data Center**
    """
elif topology['needs_distribution_step_up']:
    path_str = f"""
    **Generator** ({gen_terminal_kv} kV) → **MV Bus** ({mv_bus_kv} kV) → **Step-Up Xfmr** → **Distribution** ({distribution_kv} kV) → **Step-Down Xfmr** → **LV Bus** ({dc_voltage}V) → **Data Center**
    """
else:
    path_str = f"""
    **Generator** ({gen_terminal_kv} kV) → **MV Bus** ({mv_bus_kv} kV) → **Step-Down Xfmr** → **LV Bus** ({dc_voltage}V) → **Data Center**
    """

st.markdown(f'<div class="voltage-path">{path_str}</div>', unsafe_allow_html=True)

# Show topology issues and recommendations
if topology['issues']:
    for issue in topology['issues']:
        st.markdown(f'<div class="warning-box">⚠️ {issue}</div>', unsafe_allow_html=True)

if topology['recommendations']:
    for rec in topology['recommendations']:
        st.markdown(f'<div class="info-box">💡 {rec}</div>', unsafe_allow_html=True)

# --- TIER & AVAILABILITY ---
col_tier, col_avail, col_downtime = st.columns(3)

with col_tier:
    tier_color = "#28a745" if achieved_tier >= target_tier else "#dc3545"
    st.markdown(f"""
    <div class="tier-box" style="border-left-color: {tier_color};">
        <div class="tier-label">ACHIEVED TIER</div>
        <div class="tier-value" style="color: {tier_color};">Tier {achieved_tier}</div>
        <div class="tier-desc">{tier_info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_avail:
    st.markdown(f"""
    <div class="tier-box">
        <div class="tier-label">SYSTEM AVAILABILITY</div>
        <div class="tier-value" style="font-size: 36px;">{total_system_avail*100:.5f}%</div>
        <div class="tier-desc">{availability_to_nines(total_system_avail)}</div>
    </div>
    """, unsafe_allow_html=True)

with col_downtime:
    downtime_hr = avail_to_downtime(total_system_avail)
    st.markdown(f"""
    <div class="tier-box">
        <div class="tier-label">EXPECTED DOWNTIME</div>
        <div class="tier-value" style="font-size: 36px;">{downtime_hr*60:.1f}</div>
        <div class="tier-desc">minutes per year</div>
    </div>
    """, unsafe_allow_html=True)

# Target check
if total_system_avail >= target_avail:
    st.markdown(f'<div class="success-box">✅ <b>TARGET MET!</b> System availability meets target.</div>', 
               unsafe_allow_html=True)
else:
    st.markdown(f'<div class="fail-box">❌ <b>TARGET NOT MET.</b> Consider more redundancy or BESS.</div>', 
               unsafe_allow_html=True)

# --- EQUIPMENT SUMMARY ---
st.markdown("### 📋 Equipment Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Generation**")
    st.metric("Generators", f"{n_gen_total} × {gen_specs['mw']} MW", f"N+{n_gen_total - n_gen_needed}")
    st.metric("Terminal Voltage", f"{gen_terminal_kv} kV")
    if topology['needs_gen_step_up']:
        st.metric("Step-Up Xfmrs", f"{n_step_up_xfmr} × {step_up_specs['mva']} MVA")

with col2:
    st.markdown("**Distribution**")
    st.metric("MV Bus Voltage", f"{mv_bus_kv} kV")
    if topology['needs_distribution_step_up']:
        st.metric("Distribution Voltage", f"{distribution_kv} kV")
        st.metric("Dist Step-Up", f"{n_dist_step_up} × {dist_step_up_specs['mva']} MVA")
    st.metric("Step-Down Xfmrs", f"{n_step_down + 1} × {step_down_specs['kva']} kVA")

with col3:
    st.markdown("**Data Center**")
    st.metric("LV Voltage", f"{dc_voltage} V")
    st.metric("LV Switchgear", lv_swgr_name)
    if enable_bess:
        st.metric("BESS", f"{n_bess_total} × {bess_inv_mw} MW", f"{bess_energy_mwh:.1f} MWh")

# --- VALIDATIONS ---
st.markdown("### ✅ Equipment Validations")

validations = []

# MV Bus
validations.append(validate_short_circuit(sc_mv_bus, mv_swgr_specs['kaic'], f"MV Switchgear ({mv_bus_kv}kV)"))
validations.append(validate_ampacity(i_mv_bus, mv_swgr_specs['continuous_a'], f"MV Bus ({mv_bus_kv}kV)"))

# Distribution (if different)
if topology['needs_distribution_step_up']:
    validations.append(validate_short_circuit(sc_distribution, dist_swgr_specs['kaic'], f"Distribution Swgr ({distribution_kv}kV)"))
    validations.append(validate_ampacity(i_distribution, dist_swgr_specs['continuous_a'], f"Distribution Bus ({distribution_kv}kV)"))

# LV
validations.append(validate_short_circuit(sc_lv_bus, lv_swgr_specs['kaic'], f"LV Switchgear ({dc_voltage}V)"))
validations.append(validate_ampacity(i_lv_bus, lv_swgr_specs['continuous_a'], f"LV Bus ({dc_voltage}V)"))

# Display
n_pass = sum(1 for v in validations if v['status'] == 'PASS')
n_warn = sum(1 for v in validations if v['status'] == 'WARNING')
n_fail = sum(1 for v in validations if v['status'] == 'FAIL')

col_v1, col_v2, col_v3 = st.columns(3)
col_v1.metric("✅ Passed", n_pass)
col_v2.metric("⚠️ Warnings", n_warn)
col_v3.metric("❌ Failed", n_fail)

for val in validations:
    if val['status'] == 'FAIL':
        st.markdown(f"<div class='fail-box'>{val['message']}<br><i>{val['recommendation']}</i></div>", unsafe_allow_html=True)
    elif val['status'] == 'WARNING':
        st.markdown(f"<div class='warning-box'>{val['message']}<br><i>{val['recommendation']}</i></div>", unsafe_allow_html=True)
    else:
        st.success(val['message'])

# --- ARCHITECTURE DIAGRAM ---
st.markdown("### 📐 One-Line Diagram")

dot = graphviz.Digraph()
dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.0')
dot.attr('node', fontname='Arial', fontsize='10')
dot.attr('edge', fontname='Arial', fontsize='9')

# Generators cluster
with dot.subgraph(name='cluster_gen') as gen:
    gen.attr(label=f'Generation ({n_gen_total} × {gen_specs["mw"]} MW @ {gen_terminal_kv}kV)', 
             style='dashed', color='darkgreen')
    n_show = min(4, n_gen_total)
    for i in range(n_show):
        gen.node(f'G{i}', f'G{i+1}\n{gen_specs["mw"]}MW\n{gen_terminal_kv}kV', 
                shape='circle', style='filled', fillcolor='lightgreen', width='0.9')
    if n_gen_total > n_show:
        gen.node('Gmore', f'...+{n_gen_total - n_show}', shape='plaintext')

# Step-up transformers (if needed)
if topology['needs_gen_step_up']:
    with dot.subgraph(name='cluster_stepup') as su:
        su.attr(label=f'Step-Up Transformers ({gen_terminal_kv}/{mv_bus_kv}kV)', style='dashed', color='orange')
        n_show_su = min(3, n_step_up_xfmr)
        for i in range(n_show_su):
            su.node(f'SU{i}', f'T{i+1}\n{step_up_specs["mva"]}MVA\n{step_up_specs["z_pct"]}%Z',
                   shape='box', style='filled', fillcolor='lightyellow')
        if n_step_up_xfmr > n_show_su:
            su.node('SUmore', f'...+{n_step_up_xfmr - n_show_su}', shape='plaintext')

# MV Bus
dot.node('MV_BUS_A', f'MV Bus A\n({mv_bus_kv}kV)', shape='rect', width='4', height='0.3',
        style='filled', fillcolor='#333333', fontcolor='white')
dot.node('MV_BUS_B', f'MV Bus B\n({mv_bus_kv}kV)', shape='rect', width='4', height='0.3',
        style='filled', fillcolor='#333333', fontcolor='white')

# Distribution step-up (if needed)
if topology['needs_distribution_step_up']:
    with dot.subgraph(name='cluster_dist_su') as dsu:
        dsu.attr(label=f'Distribution Step-Up ({mv_bus_kv}/{distribution_kv}kV)', style='dashed', color='purple')
        for i in range(min(2, n_dist_step_up)):
            dsu.node(f'DSU{i}', f'T{i+1}\n{dist_step_up_specs["mva"]}MVA',
                    shape='box', style='filled', fillcolor='#e1bee7')
    
    # Distribution bus
    dot.node('DIST_BUS', f'Distribution Bus\n({distribution_kv}kV)', shape='rect', width='5', height='0.3',
            style='filled', fillcolor='#4a148c', fontcolor='white')

# Step-down transformers
with dot.subgraph(name='cluster_stepdown') as sd:
    primary_kv = distribution_kv if topology['needs_distribution_step_up'] else mv_bus_kv
    sd.attr(label=f'Step-Down Transformers ({primary_kv}kV/{dc_voltage}V)', style='dashed', color='blue')
    n_show_sd = min(3, n_step_down)
    for i in range(n_show_sd):
        sd.node(f'SD{i}', f'T{i+1}\n{step_down_specs["kva"]}kVA\n{step_down_specs["z_pct"]}%Z',
               shape='box', style='filled', fillcolor='lightblue')
    if n_step_down > n_show_sd:
        sd.node('SDmore', f'...+{n_step_down - n_show_sd}', shape='plaintext')

# BESS
if enable_bess:
    dot.node('BESS', f'BESS\n{bess_installed_mw:.1f}MW\n{bess_energy_mwh:.1f}MWh',
            shape='box3d', style='filled', fillcolor='#b2ebf2')

# LV Bus
dot.node('LV_BUS', f'LV Bus ({dc_voltage}V)', shape='rect', width='5', height='0.3',
        style='filled', fillcolor='#666666', fontcolor='white')

# Data Center
dot.node('DC', f'Data Center\n{p_it} MW IT Load', shape='house', style='filled', fillcolor='#f3e5f5')

# --- CONNECTIONS ---

# Generators to step-up or MV bus
if topology['needs_gen_step_up']:
    for i in range(min(2, n_gen_total)):
        dot.edge(f'G{i}', f'SU{min(i, n_show_su-1)}', label='CB')
    for i in range(min(n_show_su, n_step_up_xfmr)):
        if i % 2 == 0:
            dot.edge(f'SU{i}', 'MV_BUS_A')
        else:
            dot.edge(f'SU{i}', 'MV_BUS_B')
else:
    for i in range(min(2, n_gen_total)):
        dot.edge(f'G{i}', 'MV_BUS_A', label='CB')
    for i in range(2, min(4, n_gen_total)):
        dot.edge(f'G{i}', 'MV_BUS_B', label='CB')

# Bus tie
dot.edge('MV_BUS_A', 'MV_BUS_B', label='Tie', style='dashed', dir='both')

# MV to distribution or step-down
if topology['needs_distribution_step_up']:
    dot.edge('MV_BUS_A', 'DSU0')
    dot.edge('MV_BUS_B', 'DSU1' if n_dist_step_up > 1 else 'DSU0')
    for i in range(min(2, n_dist_step_up)):
        dot.edge(f'DSU{i}', 'DIST_BUS')
    for i in range(min(n_show_sd, n_step_down)):
        dot.edge('DIST_BUS', f'SD{i}')
else:
    for i in range(min(n_show_sd, n_step_down)):
        if i % 2 == 0:
            dot.edge('MV_BUS_A', f'SD{i}')
        else:
            dot.edge('MV_BUS_B', f'SD{i}')

# Step-down to LV
for i in range(min(n_show_sd, n_step_down)):
    dot.edge(f'SD{i}', 'LV_BUS')

# BESS to LV
if enable_bess:
    dot.edge('BESS', 'LV_BUS', label='PCS')

# LV to DC
dot.edge('LV_BUS', 'DC', label='Feeders')

st.graphviz_chart(dot, use_container_width=True)

# --- BOM ---
st.markdown("### 📦 Bill of Materials")

bom_data = [
    {"Item": "Generators", "Model": gen_model, "Qty": n_gen_total, 
     "Rating": f"{gen_specs['mw']} MW @ {gen_terminal_kv} kV"},
]

if topology['needs_gen_step_up']:
    bom_data.append({"Item": "Step-Up Transformers", "Model": step_up_name, "Qty": n_step_up_xfmr,
                     "Rating": f"{step_up_specs['mva']} MVA, {step_up_specs['z_pct']}%Z"})

bom_data.append({"Item": "MV Switchgear", "Model": mv_swgr_name, "Qty": 2,
                 "Rating": f"{mv_swgr_specs['kaic']} kAIC, {mv_swgr_specs['continuous_a']}A"})

if topology['needs_distribution_step_up']:
    bom_data.append({"Item": "Dist Step-Up Xfmrs", "Model": dist_step_up_name, "Qty": n_dist_step_up,
                     "Rating": f"{dist_step_up_specs['mva']} MVA"})
    bom_data.append({"Item": "HV Switchgear", "Model": dist_swgr_name, "Qty": 1,
                     "Rating": f"{dist_swgr_specs['kaic']} kAIC"})

bom_data.append({"Item": "Step-Down Transformers", "Model": step_down_name, "Qty": n_step_down + 1,
                 "Rating": f"{step_down_specs['kva']} kVA, {step_down_specs['z_pct']}%Z"})

bom_data.append({"Item": "LV Switchgear", "Model": lv_swgr_name, "Qty": 2,
                 "Rating": f"{lv_swgr_specs['kaic']} kAIC, {lv_swgr_specs['continuous_a']}A"})

if enable_bess:
    bom_data.append({"Item": "BESS Inverters", "Model": f"{bess_inv_mw} MW units", "Qty": n_bess_total,
                     "Rating": f"{bess_installed_mw:.1f} MW / {bess_energy_mwh:.1f} MWh"})

st.dataframe(pd.DataFrame(bom_data), use_container_width=True, hide_index=True)

# --- CALCULATION DETAILS ---
with st.expander("🧮 Calculation Details", expanded=False):
    st.markdown(f"""
    ### Load Calculation
    - IT Load: {p_it:.1f} MW
    - Auxiliaries ({dc_aux}%): +{p_it * dc_aux/100:.1f} MW
    - Gross Load: **{p_gross:.2f} MW**
    
    ### Voltage Topology Decision
    - Generator Terminal: {gen_terminal_kv} kV
    - Current at Gen Voltage: {i_gen_terminal:.0f} A
    - SC at Gen Voltage: {sc_gen_terminal:.1f} kA
    - Needs Gen Step-Up: {'Yes' if topology['needs_gen_step_up'] else 'No'}
    - Needs Dist Step-Up: {'Yes' if topology['needs_distribution_step_up'] else 'No'}
    - Final Distribution: {distribution_kv} kV
    
    ### Short Circuit Summary
    | Location | Voltage | Isc (kA) | Equipment | Rating (kA) | Status |
    |----------|---------|----------|-----------|-------------|--------|
    | MV Bus | {mv_bus_kv} kV | {sc_mv_bus:.1f} | {mv_swgr_name} | {mv_swgr_specs['kaic']} | {'✅' if sc_mv_bus < mv_swgr_specs['kaic'] else '❌'} |
    | LV Bus | {dc_voltage} V | {sc_lv_bus:.1f} | {lv_swgr_name} | {lv_swgr_specs['kaic']} | {'✅' if sc_lv_bus < lv_swgr_specs['kaic'] else '❌'} |
    
    ### Availability Calculation
    - Generator Subsystem: {gen_sys_rel:.6f}
    - Step-Up Xfmr Subsystem: {step_up_sys_rel:.6f}
    - Step-Down Xfmr Subsystem: {step_down_sys_rel:.6f}
    - Switchgear: {p_swgr:.6f}
    - Primary Path: {a_primary:.6f}
    - BESS Backup: {bess_rel_actual:.6f}
    - **Total System: {total_system_avail:.7f}**
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>🔌 CAT Topology Designer v22.0</b></p>
    <p>Correct Voltage Topology with Step-Up/Step-Down Transformer Logic</p>
    <p>Caterpillar Electric Power | 2026</p>
</div>
""", unsafe_allow_html=True)
