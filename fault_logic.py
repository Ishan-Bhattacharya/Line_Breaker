# In fault_logic.py

import pandas as pd
import numpy as np

# --- Threshold Calculation Logic ---

# MODIFIED: Function now accepts a DataFrame directly instead of a file path.
def calculate_current_threshold(df: pd.DataFrame, sigma_multiplier: int = 6) -> float:
    """
    Analyzes 'normal' data from a DataFrame to find a robust fault current threshold.

    Args:
        df (pd.DataFrame): The DataFrame containing the power grid data.
        sigma_multiplier (int): The number of standard deviations for the threshold.

    Returns:
        float: The calculated current threshold in Amps.
    """
    # The function no longer needs to read the CSV file.
    
    normal_df = df[df['label'] == 'normal']
    
    if normal_df.empty:
        print("Warning: No 'normal' data found in the DataFrame. Returning a default threshold.")
        return 500.0

    current_columns = [col for col in normal_df.columns if col.startswith('I_')]
    if not current_columns:
        print("Warning: No current columns ('I_...') found. Returning a default threshold.")
        return 500.0

    normal_currents_df = normal_df[current_columns]
    
    max_system_currents = normal_currents_df.max(axis=1)
    
    mean_peak_current = max_system_currents.mean()
    std_peak_current = max_system_currents.std()
    
    calculated_threshold = mean_peak_current + (sigma_multiplier * std_peak_current)
    
    return calculated_threshold


# --- Analysis Functions (These are unchanged) ---

def check_fault_status(system_state: dict, v_thresh: float, i_thresh: float):
    """Analyzes a system state snapshot based on provided thresholds."""
    faulted_buses, faulted_lines = [], []
    max_current, min_voltage = 0, 2.0
    for k, v in system_state.items():
        if k.startswith('V_'):
            if v < v_thresh: faulted_buses.append(k)
            if v < min_voltage: min_voltage = v
        elif k.startswith('I_'):
            if v > i_thresh: faulted_lines.append(k)
            if v > max_current: max_current = v
            
    status = "FAULT DETECTED" if (faulted_buses and faulted_lines) else "NORMAL"
    return {"status": status, "min_voltage": min_voltage, "max_current": max_current, "symptomatic_buses": faulted_buses, "symptomatic_lines": faulted_lines}


def determine_isolation_action(analysis_result: dict, system_state: dict):
    """Determines which line to open to isolate a fault."""
    if analysis_result["status"] != "FAULT DETECTED":
        return {"action": "NONE", "element_to_open": None}
    
    # This check prevents an error if no lines show fault symptoms
    if not analysis_result["symptomatic_lines"]:
        return {"action": "NONE", "element_to_open": None}

    faulted_line_name = max(
        analysis_result["symptomatic_lines"],
        key=lambda line: system_state[line]
    ).replace('I_', '')
    return {"action": "OPEN_LINE", "element_to_open": faulted_line_name}

