import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v23.0", page_icon="🔌", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1A1A1A 0%, #2D2D2D 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #FFCD00;
    }
    .main-header h1 { color: #FFCD00 !important; margin: 0; }
    .main-header p { color: #CCCCCC !important; margin: 0; }
    
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; margin-bottom: 15px; }
    .fail-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545; margin-bottom: 15px; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; margin-bottom: 15px; }
    .info-box { background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 5px solid #2196f3; margin-bottom: 15px; }
    
    .tier-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1976d2;
        margin-bottom: 15px;
        text-align: center;
    }
    .tier-label { font-size: 14px; color: #555; }
    .tier-value { font-size: 42px; font-weight: bold; color: #1976d2; }
    .tier-desc { font-size: 12px; color: #777; }
    
    .pod-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #4caf50;
        margin: 5px;
        text-align: center;
    }
    .pod-title { font-size: 14px; font-weight: bold; color: #2e7d32; }
    .pod-value { font-size: 18px; color: #1b5e20; }
    
    .topology-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. DATA LIBRARIES
# ==============================================================================

CAT_LIBRARY = {
    "G3516 (1.5 MW)": {
        "mw": 1.5,
        "available_voltages_kv": [0.48, 4.16, 13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.16,
        "step_cap": 25.0,
        "type": "High Speed Recip",
        "gens_per_pod_typical": 9,  # Typical configuration
        "mtbf": 43800,
        "mttr": 48,
    },
    "XGC1900 (1.9 MW)": {
        "mw": 1.9,
        "available_voltages_kv": [0.48, 4.16, 13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.16,
        "step_cap": 25.0,
        "type": "High Speed Recip",
        "gens_per_pod_typical": 7,
        "mtbf": 43800,
        "mttr": 48,
    },
    "G3520K (2.4 MW)": {
        "mw": 2.4,
        "available_voltages_kv": [0.48, 4.16, 13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.16,
        "step_cap": 25.0,
        "type": "High Speed Recip",
        "gens_per_pod_typical": 6,
        "mtbf": 43800,
        "mttr": 48,
    },
    "G3520FR (2.5 MW)": {
        "mw": 2.5,
        "available_voltages_kv": [0.48, 4.16, 13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.16,
        "step_cap": 40.0,
        "type": "High Speed Recip - Fast Response",
        "gens_per_pod_typical": 5,
        "mtbf": 43800,
        "mttr": 48,
    },
    "CG260-16 (3.96 MW)": {
        "mw": 3.96,
        "available_voltages_kv": [4.16, 11.0, 13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.15,
        "step_cap": 25.0,
        "type": "High Speed Recip",
        "gens_per_pod_typical": 4,
        "mtbf": 50000,
        "mttr": 72,
    },
    "G20CM34 (9.76 MW)": {
        "mw": 9.76,
        "available_voltages_kv": [11.0, 13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.14,
        "step_cap": 20.0,
        "type": "Medium Speed Recip",
        "gens_per_pod_typical": 2,
        "mtbf": 50000,
        "mttr": 72,
    },
    "Titan 130 (16.5 MW)": {
        "mw": 16.5,
        "available_voltages_kv": [13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.18,
        "step_cap": 15.0,
        "type": "Gas Turbine",
        "gens_per_pod_typical": 1,
        "mtbf": 40000,
        "mttr": 96,
    },
    "Titan 250 (23.2 MW)": {
        "mw": 23.2,
        "available_voltages_kv": [13.8],
        "default_voltage_kv": 13.8,
        "xd": 0.18,
        "step_cap": 15.0,
        "type": "Gas Turbine",
        "gens_per_pod_typical": 1,
        "mtbf": 40000,
        "mttr": 96,
    },
}

# Standard transformer sizes (MVA)
STEP_UP_XFMR_SIZES = [15, 20, 25, 30, 33.75, 40, 50, 63, 75]  # MVA

# Switchgear ratings
SWITCHGEAR_15KV = {
    "15kV, 25kA, 1200A": {"kaic": 25, "continuous_a": 1200},
    "15kV, 40kA, 2000A": {"kaic": 40, "continuous_a": 2000},
    "15kV, 50kA, 3000A": {"kaic": 50, "continuous_a": 3000},
    "15kV, 63kA, 4000A": {"kaic": 63, "continuous_a": 4000},
}

SWITCHGEAR_38KV = {
    "38kV, 25kA, 1200A": {"kaic": 25, "continuous_a": 1200},
    "38kV, 40kA, 2000A": {"kaic": 40, "continuous_a": 2000},
    "38kV, 50kA, 3000A": {"kaic": 50, "continuous_a": 3000},
}

TIER_LEVELS = {
    "IV": {"min_avail": 0.99995, "redundancy": "2N / 2(N+1)", "description": "Fault Tolerant"},
    "III": {"min_avail": 0.99982, "redundancy": "N+1", "description": "Concurrently Maintainable"},
    "II": {"min_avail": 0.99741, "redundancy": "N+1 Partial", "description": "Redundant Components"},
    "I": {"min_avail": 0.99671, "redundancy": "N", "description": "Basic Infrastructure"},
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

def calc_amps(mw, kv):
    if kv <= 0: return 0
    return (mw * 1e6) / (math.sqrt(3) * kv * 1000)

def calc_sc_ka_gen(mw_gen, xd, kv, n_gens):
    if kv <= 0 or xd <= 0: return 0
    mva_gen = mw_gen / 0.8
    i_base = (mva_gen * 1e6) / (math.sqrt(3) * kv * 1000)
    i_sc_unit = i_base / xd
    return (i_sc_unit * n_gens) / 1000.0

def get_tier_level(availability):
    for tier, info in TIER_LEVELS.items():
        if availability >= info['min_avail']:
            return tier, info
    return "Below I", {"min_avail": 0, "redundancy": "N", "description": "Below Standard"}

def avail_to_downtime_min(avail, hours_per_year=8760):
    return (1 - avail) * hours_per_year * 60

def select_transformer_size(required_mva):
    """Select standard transformer size >= required."""
    for size in STEP_UP_XFMR_SIZES:
        if size >= required_mva:
            return size
    return STEP_UP_XFMR_SIZES[-1]

def select_switchgear(voltage_kv, required_amps, required_kaic):
    """Select appropriate switchgear."""
    library = SWITCHGEAR_15KV if voltage_kv <= 15 else SWITCHGEAR_38KV
    
    for name, specs in library.items():
        if specs['continuous_a'] >= required_amps and specs['kaic'] >= required_kaic:
            return name, specs
    
    # Return largest if none fit
    return list(library.keys())[-1], list(library.values())[-1]

# ==============================================================================
# 2. POD TOPOLOGY DESIGN ALGORITHM
# ==============================================================================

def design_pod_topology(
    p_it_mw,
    gen_specs,
    gen_terminal_kv,
    redundancy_pct,
    target_avail,
    max_pod_bus_amps=2000,
    max_pod_gens=12,
    distribution_kv=34.5,
):
    """
    Design a Pod-based topology similar to CAT standard design.
    
    Architecture:
    - Generators at 13.8 kV
    - Generators grouped into Pods (A and B per transformer)
    - Each Pod pair feeds a step-up transformer (13.8/34.5 kV)
    - All transformers connect to 34.5 kV collector bus
    - 34.5 kV bus delivers to data center
    
    Returns complete topology design.
    """
    
    design = {
        'input': {
            'p_it_mw': p_it_mw,
            'gen_model': gen_specs,
            'gen_terminal_kv': gen_terminal_kv,
            'redundancy_pct': redundancy_pct,
            'target_avail': target_avail,
        },
        'pods': [],
        'transformers': [],
        'buses': {},
        'totals': {},
        'validations': [],
    }
    
    # === STEP 1: Calculate gross load ===
    dc_aux_pct = 15.0
    losses_pct = 3.0
    p_gross = p_it_mw * (1 + dc_aux_pct/100) / (1 - losses_pct/100)
    p_with_redundancy = p_gross * (1 + redundancy_pct/100)
    
    design['totals']['p_it_mw'] = p_it_mw
    design['totals']['p_gross_mw'] = p_gross
    design['totals']['p_with_redundancy_mw'] = p_with_redundancy
    
    # === STEP 2: Determine generators per pod ===
    gen_mw = gen_specs['mw']
    
    # Calculate max MW per pod based on bus ampacity limit
    max_mw_per_pod_ampacity = (max_pod_bus_amps * math.sqrt(3) * gen_terminal_kv) / 1000
    
    # Typical gens per pod from library
    gens_per_pod_typical = gen_specs.get('gens_per_pod_typical', 6)
    
    # Calculate based on ampacity
    gens_per_pod_ampacity = int(max_mw_per_pod_ampacity / gen_mw)
    
    # Use minimum of typical, ampacity limit, and max limit
    gens_per_pod = min(gens_per_pod_typical, gens_per_pod_ampacity, max_pod_gens)
    gens_per_pod = max(1, gens_per_pod)  # At least 1
    
    mw_per_pod = gens_per_pod * gen_mw
    
    design['pod_config'] = {
        'gens_per_pod': gens_per_pod,
        'mw_per_pod': mw_per_pod,
        'max_mw_ampacity': max_mw_per_pod_ampacity,
    }
    
    # === STEP 3: Calculate number of pods needed ===
    # Each transformer is fed by Pod A + Pod B
    mw_per_xfmr_pair = 2 * mw_per_pod  # Pod A + Pod B
    
    n_xfmr_needed = math.ceil(p_with_redundancy / mw_per_xfmr_pair)
    n_xfmr_needed = max(2, n_xfmr_needed)  # Minimum 2 for redundancy
    
    n_pods_total = n_xfmr_needed * 2  # Each transformer has Pod A and Pod B
    
    # Total generators
    n_gens_total = n_pods_total * gens_per_pod
    total_gen_capacity = n_gens_total * gen_mw
    
    design['totals']['n_transformers'] = n_xfmr_needed
    design['totals']['n_pods'] = n_pods_total
    design['totals']['n_gens'] = n_gens_total
    design['totals']['total_capacity_mw'] = total_gen_capacity
    design['totals']['actual_redundancy_pct'] = (total_gen_capacity / p_gross - 1) * 100
    
    # === STEP 4: Size transformers ===
    # Each transformer handles Pod A + Pod B
    xfmr_load_mva = mw_per_xfmr_pair / 0.85  # Assume 0.85 PF
    xfmr_size_mva = select_transformer_size(xfmr_load_mva * 1.25)  # 25% margin
    
    design['transformer_config'] = {
        'size_mva': xfmr_size_mva,
        'ratio': f"{gen_terminal_kv}/{distribution_kv} kV",
        'quantity': n_xfmr_needed,
        'impedance_pct': 8.0,  # Typical for this size
    }
    
    # === STEP 5: Create pod definitions ===
    for xfmr_idx in range(n_xfmr_needed):
        # Pod A
        pod_a = {
            'name': f"Pod {xfmr_idx + 1}A",
            'transformer': xfmr_idx + 1,
            'n_gens': gens_per_pod,
            'mw': mw_per_pod,
            'voltage_kv': gen_terminal_kv,
            'type': 'A',
        }
        design['pods'].append(pod_a)
        
        # Pod B
        pod_b = {
            'name': f"Pod {xfmr_idx + 1}B",
            'transformer': xfmr_idx + 1,
            'n_gens': gens_per_pod,
            'mw': mw_per_pod,
            'voltage_kv': gen_terminal_kv,
            'type': 'B',
        }
        design['pods'].append(pod_b)
        
        # Transformer
        xfmr = {
            'name': f"T{xfmr_idx + 1}",
            'size_mva': xfmr_size_mva,
            'primary_kv': gen_terminal_kv,
            'secondary_kv': distribution_kv,
            'pods': [f"Pod {xfmr_idx + 1}A", f"Pod {xfmr_idx + 1}B"],
            'load_mw': 2 * mw_per_pod,
        }
        design['transformers'].append(xfmr)
    
    # === STEP 6: Bus calculations ===
    
    # Pod bus (13.8 kV)
    pod_bus_current = calc_amps(mw_per_pod, gen_terminal_kv)
    pod_bus_sc = calc_sc_ka_gen(gen_mw, gen_specs['xd'], gen_terminal_kv, gens_per_pod)
    
    # Collector bus at 13.8 kV (where Pod A + Pod B meet before transformer)
    collector_current = calc_amps(2 * mw_per_pod, gen_terminal_kv)
    collector_sc = calc_sc_ka_gen(gen_mw, gen_specs['xd'], gen_terminal_kv, 2 * gens_per_pod)
    
    # Distribution bus at 34.5 kV
    dist_bus_current = calc_amps(p_with_redundancy, distribution_kv)
    # SC at 34.5 kV is limited by transformer impedance
    dist_bus_sc = (xfmr_size_mva * n_xfmr_needed) / (8.0/100) / (math.sqrt(3) * distribution_kv) * 1000 / 1000
    
    design['buses'] = {
        'pod_bus': {
            'voltage_kv': gen_terminal_kv,
            'current_a': pod_bus_current,
            'sc_ka': pod_bus_sc,
            'rating_a': max_pod_bus_amps,
        },
        'collector_bus': {
            'voltage_kv': gen_terminal_kv,
            'current_a': collector_current,
            'sc_ka': collector_sc,
        },
        'distribution_bus': {
            'voltage_kv': distribution_kv,
            'current_a': dist_bus_current,
            'sc_ka': dist_bus_sc,
        },
    }
    
    # === STEP 7: Select switchgear ===
    pod_swgr_name, pod_swgr_specs = select_switchgear(gen_terminal_kv, pod_bus_current * 1.25, pod_bus_sc * 1.25)
    dist_swgr_name, dist_swgr_specs = select_switchgear(distribution_kv, dist_bus_current * 1.25, dist_bus_sc * 1.25)
    
    design['switchgear'] = {
        'pod_level': {'name': pod_swgr_name, 'specs': pod_swgr_specs},
        'distribution_level': {'name': dist_swgr_name, 'specs': dist_swgr_specs},
    }
    
    # === STEP 8: Validations ===
    validations = []
    
    # Pod bus ampacity
    if pod_bus_current > max_pod_bus_amps:
        validations.append({
            'status': 'FAIL',
            'item': 'Pod Bus Ampacity',
            'message': f"Pod current {pod_bus_current:.0f}A > limit {max_pod_bus_amps}A",
            'recommendation': "Reduce generators per pod"
        })
    else:
        margin = (max_pod_bus_amps - pod_bus_current) / max_pod_bus_amps * 100
        validations.append({
            'status': 'PASS' if margin > 20 else 'WARNING',
            'item': 'Pod Bus Ampacity',
            'message': f"Pod current {pod_bus_current:.0f}A OK ({margin:.0f}% margin)",
            'recommendation': None if margin > 20 else "Consider larger bus"
        })
    
    # Pod switchgear SC
    if pod_bus_sc > pod_swgr_specs['kaic']:
        validations.append({
            'status': 'FAIL',
            'item': 'Pod Switchgear kAIC',
            'message': f"Pod Isc {pod_bus_sc:.1f}kA > rating {pod_swgr_specs['kaic']}kA",
            'recommendation': "Reduce generators per pod or use higher rated switchgear"
        })
    else:
        margin = (pod_swgr_specs['kaic'] - pod_bus_sc) / pod_swgr_specs['kaic'] * 100
        validations.append({
            'status': 'PASS' if margin > 20 else 'WARNING',
            'item': 'Pod Switchgear kAIC',
            'message': f"Pod Isc {pod_bus_sc:.1f}kA OK ({margin:.0f}% margin)",
            'recommendation': None
        })
    
    # Distribution switchgear
    if dist_bus_sc > dist_swgr_specs['kaic']:
        validations.append({
            'status': 'FAIL',
            'item': 'Distribution Switchgear kAIC',
            'message': f"Distribution Isc {dist_bus_sc:.1f}kA > rating {dist_swgr_specs['kaic']}kA",
            'recommendation': "Use higher rated 34.5kV switchgear"
        })
    else:
        validations.append({
            'status': 'PASS',
            'item': 'Distribution Switchgear kAIC',
            'message': f"Distribution Isc {dist_bus_sc:.1f}kA OK",
            'recommendation': None
        })
    
    design['validations'] = validations
    
    # === STEP 9: Availability calculation ===
    # Pod level availability
    p_gen_avail = calc_avail(gen_specs['mtbf'], gen_specs['mttr'])
    
    # Pod availability (k-out-of-n within pod)
    n_gens_needed_per_pod = gens_per_pod - 1  # N-1 redundancy within pod
    pod_avail = rel_k_out_n(n_gens_needed_per_pod, gens_per_pod, p_gen_avail)
    
    # Transformer pair availability (Pod A OR Pod B)
    # Either pod can feed the transformer
    xfmr_pair_avail = 1 - (1 - pod_avail) ** 2  # Parallel availability
    
    # Transformer availability
    xfmr_avail = 0.9999  # Typical transformer availability
    
    # System availability (need k transformers out of n)
    xfmrs_needed = math.ceil(p_gross / (2 * mw_per_pod))
    system_avail = rel_k_out_n(xfmrs_needed, n_xfmr_needed, xfmr_pair_avail * xfmr_avail)
    
    design['availability'] = {
        'gen_unit': p_gen_avail,
        'pod': pod_avail,
        'xfmr_pair': xfmr_pair_avail,
        'system': system_avail,
        'tier': get_tier_level(system_avail)[0],
        'downtime_min_yr': avail_to_downtime_min(system_avail),
    }
    
    return design


# ==============================================================================
# 3. DIAGRAM GENERATION
# ==============================================================================

def generate_topology_diagram(design):
    """Generate Graphviz diagram of the pod topology."""
    
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.4', ranksep='0.6')
    dot.attr('node', fontname='Arial', fontsize='9')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    n_xfmrs = design['totals']['n_transformers']
    gen_kv = design['input']['gen_terminal_kv']
    dist_kv = design['transformer_config']['ratio'].split('/')[1].replace(' kV', '')
    gens_per_pod = design['pod_config']['gens_per_pod']
    mw_per_pod = design['pod_config']['mw_per_pod']
    xfmr_mva = design['transformer_config']['size_mva']
    
    # Limit display for large systems
    max_display = min(5, n_xfmrs)
    
    # Distribution bus (34.5 kV) - at top
    dot.node('DIST_BUS_TOP', f'To Data Center\n{dist_kv}', 
            shape='rect', width='8', height='0.4',
            style='filled', fillcolor='#1a237e', fontcolor='white')
    
    # Main 34.5 kV collector bus
    dot.node('DIST_BUS', f'34.5 kV Collector Bus', 
            shape='rect', width='10', height='0.3',
            style='filled', fillcolor='#303f9f', fontcolor='white')
    
    dot.edge('DIST_BUS', 'DIST_BUS_TOP', label='To Load', style='bold')
    
    # Create transformer and pod structure
    for i in range(max_display):
        xfmr_name = f"T{i+1}"
        
        # Transformer
        dot.node(xfmr_name, f'{xfmr_name}\n{xfmr_mva} MVA\n{gen_kv}/{dist_kv}kV',
                shape='box', style='filled,rounded', fillcolor='#fff3e0', width='1.2')
        
        # Connect transformer to distribution bus
        dot.edge(xfmr_name, 'DIST_BUS')
        
        # Collector bus for this transformer pair (13.8 kV)
        collector_name = f"COL{i+1}"
        dot.node(collector_name, f'13.8kV', shape='rect', width='2.5', height='0.2',
                style='filled', fillcolor='#424242', fontcolor='white')
        
        dot.edge(collector_name, xfmr_name)
        
        # Pod A
        pod_a_name = f"POD{i+1}A"
        with dot.subgraph(name=f'cluster_pod{i+1}a') as pod:
            pod.attr(label=f'Pod {i+1}A\n{gens_per_pod}×Gen = {mw_per_pod:.1f}MW', 
                    style='dashed', color='#2e7d32')
            
            # Show a few generators
            n_show = min(3, gens_per_pod)
            for g in range(n_show):
                pod.node(f'G{i+1}A{g}', f'G\n{design["input"]["gen_model"]["mw"]}MW', 
                        shape='circle', style='filled', fillcolor='#c8e6c9', width='0.5')
            if gens_per_pod > n_show:
                pod.node(f'G{i+1}A_more', f'+{gens_per_pod - n_show}', shape='plaintext')
        
        # Pod B
        pod_b_name = f"POD{i+1}B"
        with dot.subgraph(name=f'cluster_pod{i+1}b') as pod:
            pod.attr(label=f'Pod {i+1}B\n{gens_per_pod}×Gen = {mw_per_pod:.1f}MW', 
                    style='dashed', color='#1565c0')
            
            n_show = min(3, gens_per_pod)
            for g in range(n_show):
                pod.node(f'G{i+1}B{g}', f'G\n{design["input"]["gen_model"]["mw"]}MW', 
                        shape='circle', style='filled', fillcolor='#bbdefb', width='0.5')
            if gens_per_pod > n_show:
                pod.node(f'G{i+1}B_more', f'+{gens_per_pod - n_show}', shape='plaintext')
        
        # Connect pods to collector
        dot.edge(f'G{i+1}A0', collector_name, label='CB')
        dot.edge(f'G{i+1}B0', collector_name, label='CB')
    
    # Show "more" indicator if there are more transformers
    if n_xfmrs > max_display:
        dot.node('MORE', f'... +{n_xfmrs - max_display} more\ntransformers', 
                shape='plaintext', fontsize='10')
        dot.edge('MORE', 'DIST_BUS', style='dashed')
    
    return dot


# ==============================================================================
# 4. SIDEBAR INPUTS
# ==============================================================================

with st.sidebar:
    st.markdown("## ⚡ CAT Topology Designer")
    st.caption("v23.0 - Pod-Based Architecture")
    
    with st.expander("📊 1. Project & Load", expanded=True):
        project_name = st.text_input("Project Name", "AI Data Center")
        p_it = st.number_input("IT Load (MW)", 10.0, 500.0, 120.0, step=10.0)
        redundancy_pct = st.number_input("Redundancy (%)", 0.0, 50.0, 15.0, step=5.0,
                                         help="Additional capacity above gross load")
        
        target_avail_pct = st.number_input("Target Availability (%)", 99.0, 99.99999, 99.99, format="%.4f")
        target_avail = target_avail_pct / 100.0
        
        target_tier, _ = get_tier_level(target_avail)
        st.info(f"🎯 Target: Tier {target_tier}")

    with st.expander("🔧 2. Generation", expanded=True):
        gen_model_name = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        gen_specs = CAT_LIBRARY[gen_model_name]
        
        st.caption(f"**Type:** {gen_specs['type']}")
        st.caption(f"**Rating:** {gen_specs['mw']} MW")
        st.caption(f"**Typical Gens/Pod:** {gen_specs['gens_per_pod_typical']}")
        
        # Voltage selection
        available_voltages = gen_specs['available_voltages_kv']
        voltage_options = [f"{v} kV" for v in available_voltages]
        default_idx = voltage_options.index(f"{gen_specs['default_voltage_kv']} kV")
        
        selected_voltage_str = st.selectbox("Generator Terminal Voltage", voltage_options, index=default_idx)
        gen_terminal_kv = float(selected_voltage_str.replace(" kV", ""))

    with st.expander("📦 3. Pod Configuration", expanded=True):
        st.caption("**Pod Bus Limits**")
        max_pod_bus_amps = st.number_input("Max Pod Bus Current (A)", 1000, 4000, 2000, step=200,
                                           help="Typically 2000A for 15kV switchgear")
        max_gens_per_pod = st.number_input("Max Generators per Pod", 1, 20, 12)
        
        st.caption("**Distribution Voltage**")
        distribution_kv = st.selectbox("Distribution to DC (kV)", [34.5, 23.0, 13.8], index=0,
                                       help="Voltage level for delivery to data center")

    with st.expander("📈 4. Reliability", expanded=False):
        gen_mtbf = st.number_input("Gen MTBF (hours)", 10000, 100000, gen_specs['mtbf'])
        gen_mttr = st.number_input("Gen MTTR (hours)", 1, 500, gen_specs['mttr'])
        
        # Override in specs
        gen_specs_modified = gen_specs.copy()
        gen_specs_modified['mtbf'] = gen_mtbf
        gen_specs_modified['mttr'] = gen_mttr

# ==============================================================================
# 5. MAIN CALCULATION
# ==============================================================================

# Run design algorithm
design = design_pod_topology(
    p_it_mw=p_it,
    gen_specs=gen_specs_modified,
    gen_terminal_kv=gen_terminal_kv,
    redundancy_pct=redundancy_pct,
    target_avail=target_avail,
    max_pod_bus_amps=max_pod_bus_amps,
    max_pod_gens=max_gens_per_pod,
    distribution_kv=distribution_kv,
)

# ==============================================================================
# 6. DISPLAY
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1>🔌 CAT Topology Designer</h1>
    <p>Pod-Based Power System Architecture for Data Centers</p>
</div>
""", unsafe_allow_html=True)

# === TOPOLOGY SUMMARY ===
st.markdown(f"""
<div class="topology-summary">
    <h3 style="margin:0; color: white;">📐 {project_name} - Topology Summary</h3>
    <p style="margin:5px 0; font-size: 18px;">
        <b>{design['totals']['n_gens']}</b> Generators in 
        <b>{design['totals']['n_pods']}</b> Pods feeding 
        <b>{design['totals']['n_transformers']}</b> Transformers @ 
        <b>{distribution_kv} kV</b>
    </p>
    <p style="margin:0; font-size: 14px; opacity: 0.9;">
        Total Capacity: {design['totals']['total_capacity_mw']:.1f} MW | 
        Actual Redundancy: {design['totals']['actual_redundancy_pct']:.1f}%
    </p>
</div>
""", unsafe_allow_html=True)

# === TIER & AVAILABILITY ===
col1, col2, col3, col4 = st.columns(4)

achieved_tier, tier_info = get_tier_level(design['availability']['system'])

with col1:
    tier_color = "#28a745" if achieved_tier >= target_tier else "#dc3545"
    st.markdown(f"""
    <div class="tier-box" style="border-left-color: {tier_color};">
        <div class="tier-label">ACHIEVED TIER</div>
        <div class="tier-value" style="color: {tier_color};">Tier {achieved_tier}</div>
        <div class="tier-desc">{tier_info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="tier-box">
        <div class="tier-label">AVAILABILITY</div>
        <div class="tier-value" style="font-size: 32px;">{design['availability']['system']*100:.4f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="tier-box">
        <div class="tier-label">DOWNTIME</div>
        <div class="tier-value" style="font-size: 32px;">{design['availability']['downtime_min_yr']:.1f}</div>
        <div class="tier-desc">minutes/year</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="tier-box">
        <div class="tier-label">TOTAL GENERATORS</div>
        <div class="tier-value" style="font-size: 32px;">{design['totals']['n_gens']}</div>
        <div class="tier-desc">{gen_model_name}</div>
    </div>
    """, unsafe_allow_html=True)

# Target check
if design['availability']['system'] >= target_avail:
    st.markdown(f'<div class="success-box">✅ <b>TARGET MET!</b> Availability {design["availability"]["system"]*100:.4f}% ≥ Target {target_avail*100:.4f}%</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="fail-box">❌ <b>TARGET NOT MET.</b> Need more redundancy or BESS backup.</div>', unsafe_allow_html=True)

# === POD CONFIGURATION ===
st.markdown("### 📦 Pod Configuration")

col_pod1, col_pod2, col_pod3, col_pod4 = st.columns(4)

with col_pod1:
    st.markdown(f"""
    <div class="pod-box">
        <div class="pod-title">Generators per Pod</div>
        <div class="pod-value">{design['pod_config']['gens_per_pod']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_pod2:
    st.markdown(f"""
    <div class="pod-box">
        <div class="pod-title">MW per Pod</div>
        <div class="pod-value">{design['pod_config']['mw_per_pod']:.1f} MW</div>
    </div>
    """, unsafe_allow_html=True)

with col_pod3:
    st.markdown(f"""
    <div class="pod-box">
        <div class="pod-title">Total Pods</div>
        <div class="pod-value">{design['totals']['n_pods']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_pod4:
    st.markdown(f"""
    <div class="pod-box">
        <div class="pod-title">Pod Bus Current</div>
        <div class="pod-value">{design['buses']['pod_bus']['current_a']:.0f} A</div>
    </div>
    """, unsafe_allow_html=True)

# === VALIDATIONS ===
st.markdown("### ✅ Equipment Validations")

n_pass = sum(1 for v in design['validations'] if v['status'] == 'PASS')
n_warn = sum(1 for v in design['validations'] if v['status'] == 'WARNING')
n_fail = sum(1 for v in design['validations'] if v['status'] == 'FAIL')

col_v1, col_v2, col_v3 = st.columns(3)
col_v1.metric("✅ Passed", n_pass)
col_v2.metric("⚠️ Warnings", n_warn)
col_v3.metric("❌ Failed", n_fail)

for val in design['validations']:
    if val['status'] == 'FAIL':
        st.markdown(f"<div class='fail-box'>❌ <b>{val['item']}:</b> {val['message']}<br><i>{val['recommendation']}</i></div>", unsafe_allow_html=True)
    elif val['status'] == 'WARNING':
        st.markdown(f"<div class='warning-box'>⚠️ <b>{val['item']}:</b> {val['message']}</div>", unsafe_allow_html=True)
    else:
        st.success(f"✅ **{val['item']}:** {val['message']}")

# === DIAGRAM ===
st.markdown("### 📐 One-Line Diagram")

diagram = generate_topology_diagram(design)
st.graphviz_chart(diagram, use_container_width=True)

# === DETAILED EQUIPMENT ===
st.markdown("### 📋 Equipment Summary")

col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.markdown("**Generation**")
    gen_data = {
        "Parameter": ["Generator Model", "Rating", "Terminal Voltage", "Total Units", "Total Capacity"],
        "Value": [
            gen_model_name,
            f"{gen_specs['mw']} MW",
            f"{gen_terminal_kv} kV",
            f"{design['totals']['n_gens']}",
            f"{design['totals']['total_capacity_mw']:.1f} MW"
        ]
    }
    st.dataframe(pd.DataFrame(gen_data), use_container_width=True, hide_index=True)
    
    st.markdown("**Transformers**")
    xfmr_data = {
        "Parameter": ["Quantity", "Size", "Ratio", "Impedance"],
        "Value": [
            f"{design['totals']['n_transformers']}",
            f"{design['transformer_config']['size_mva']} MVA",
            design['transformer_config']['ratio'],
            f"{design['transformer_config']['impedance_pct']}%"
        ]
    }
    st.dataframe(pd.DataFrame(xfmr_data), use_container_width=True, hide_index=True)

with col_eq2:
    st.markdown("**Switchgear**")
    swgr_data = {
        "Level": ["Pod (13.8 kV)", "Distribution (34.5 kV)"],
        "Rating": [
            design['switchgear']['pod_level']['name'],
            design['switchgear']['distribution_level']['name']
        ]
    }
    st.dataframe(pd.DataFrame(swgr_data), use_container_width=True, hide_index=True)
    
    st.markdown("**Bus Summary**")
    bus_data = {
        "Bus": ["Pod Bus", "Collector Bus", "Distribution Bus"],
        "Voltage": [
            f"{design['buses']['pod_bus']['voltage_kv']} kV",
            f"{design['buses']['collector_bus']['voltage_kv']} kV",
            f"{design['buses']['distribution_bus']['voltage_kv']} kV"
        ],
        "Current (A)": [
            f"{design['buses']['pod_bus']['current_a']:.0f}",
            f"{design['buses']['collector_bus']['current_a']:.0f}",
            f"{design['buses']['distribution_bus']['current_a']:.0f}"
        ],
        "Isc (kA)": [
            f"{design['buses']['pod_bus']['sc_ka']:.1f}",
            f"{design['buses']['collector_bus']['sc_ka']:.1f}",
            f"{design['buses']['distribution_bus']['sc_ka']:.1f}"
        ]
    }
    st.dataframe(pd.DataFrame(bus_data), use_container_width=True, hide_index=True)

# === BILL OF MATERIALS ===
st.markdown("### 📦 Bill of Materials")

bom = [
    {"Item": "Generators", "Model": gen_model_name, "Qty": design['totals']['n_gens'], 
     "Rating": f"{gen_specs['mw']} MW @ {gen_terminal_kv} kV"},
    {"Item": "15kV Switchgear Structures", "Model": design['switchgear']['pod_level']['name'], 
     "Qty": design['totals']['n_pods'] * (design['pod_config']['gens_per_pod'] + 2), 
     "Rating": "Pod level"},
    {"Item": "Step-Up Transformers", "Model": f"{design['transformer_config']['size_mva']} MVA", 
     "Qty": design['totals']['n_transformers'], 
     "Rating": design['transformer_config']['ratio']},
    {"Item": "35kV Switchgear Structures", "Model": design['switchgear']['distribution_level']['name'], 
     "Qty": design['totals']['n_transformers'] + 4,  # + bus sections and ties
     "Rating": "Distribution level"},
]

st.dataframe(pd.DataFrame(bom), use_container_width=True, hide_index=True)

# === CALCULATION DETAILS ===
with st.expander("🧮 Calculation Details", expanded=False):
    st.markdown(f"""
    ### Load Calculation
    - IT Load: {p_it:.1f} MW
    - Auxiliaries (15%): +{p_it * 0.15:.1f} MW
    - Losses (3%): Factor
    - **Gross Load: {design['totals']['p_gross_mw']:.2f} MW**
    - With {redundancy_pct}% redundancy: **{design['totals']['p_with_redundancy_mw']:.2f} MW**
    
    ### Pod Sizing Logic
    - Max MW per pod (ampacity): {design['pod_config']['max_mw_ampacity']:.1f} MW
    - Typical gens per pod: {gen_specs['gens_per_pod_typical']}
    - **Selected: {design['pod_config']['gens_per_pod']} gens/pod = {design['pod_config']['mw_per_pod']:.1f} MW/pod**
    
    ### Transformer Sizing
    - Each transformer fed by Pod A + Pod B = {2 * design['pod_config']['mw_per_pod']:.1f} MW
    - Required MVA: {2 * design['pod_config']['mw_per_pod'] / 0.85:.1f} MVA
    - Selected size: **{design['transformer_config']['size_mva']} MVA**
    - Number needed: **{design['totals']['n_transformers']}**
    
    ### Availability Model
    - Generator unit availability: {design['availability']['gen_unit']:.6f}
    - Pod availability (N-1): {design['availability']['pod']:.6f}
    - Transformer pair (Pod A || Pod B): {design['availability']['xfmr_pair']:.6f}
    - **System availability: {design['availability']['system']:.7f}**
    """)

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>🔌 CAT Topology Designer v23.0</b></p>
    <p>Pod-Based Architecture following CAT Standard Design</p>
    <p>Caterpillar Electric Power | 2026</p>
</div>
""", unsafe_allow_html=True)
