import shutil
import openpyxl
import os
import platform 
import subprocess 
from falcon_engine.utilities import print_slowly
from openpyxl.utils import get_column_letter
import pandas as pd

def run_mantis_formatter_pipeline(RUN_NAME, input_param_names, optimized_formulations): 
    print_slowly("\n\n--- FORMATTING OPTIMIZED FORMULATIONS FOR MANTIS ---")

    # Define source and destination file paths
    src = os.path.join("exp_templates", "5_comp_formulation_template.xlsx")
    dst = os.path.join("exp_templates", f"{RUN_NAME}_formulation_sheet.xlsx")
    
    # Copy the file
    shutil.copy(src, dst)
    print(f"Formulation template copied as {dst}")

    excel_param_order = ['IL_NP_ratio',	'(IL+HL)',	'HL_(IL+HL)', 'PEG_(Chol+PEG)',	'SORT_of_total']
    # Create a lookup: param name -> index in optimized_formulations
    param_index_lookup = {name: idx for idx, name in enumerate(input_param_names)}

    wb = openpyxl.load_workbook(dst)
    ws = wb["Formulations"] 
    reversed_x = optimized_formulations[0]
    opt_methods = optimized_formulations[2]

    #take user input for IL and HL names
    IL_name = input("Enter the name of your Ionizable Lipid used (IL)\n(Choose SM102, Dlin, or ALC0315): ")
    HL_name = input(
        "Enter the name of your Helper Lipid used (HL)\n"
        "(Choose from: DOTAP, DSPC, 18PG, DOPE, DDAB, 14PA, 18MP): "
    )
    SORT_name = input("Enter name of sort lipid")

    for i in range(len(reversed_x)): 
      for j, param_name in enumerate(excel_param_order):
        row_idx = 10 + j
        col_letter = get_column_letter(3 + i)
        value = reversed_x[i][param_index_lookup[param_name]]
        ws[f"{col_letter}{row_idx}"] = round(value, 3)
        ws[f"{col_letter}6"] = IL_name 
        ws[f"{col_letter}7"] = HL_name 
        ws[f"{col_letter}8"] = "Chol"
        ws[f"{col_letter}9"] = "DMG_PEG" 
        ws[f"{col_letter}10"] = SORT_name

        ws[f"{col_letter}3"] = i 
        ws[f"{col_letter}4"] =  opt_methods[i]

    wb.save(dst)
    print(f"Optimized formulations formatted for MANTIS and saved to {dst}")

    open_excel_file(dst)
    input("\nPlease edit the Excel file to fill in stock conentrations, etc.\nOnce you're done, save and close it, then press Enter to continue...")
    
    # format the file to mantis_ready_csv
    # load the file again 
    wb = openpyxl.load_workbook(dst, data_only = True)
    ws = wb["Formulations"] 
    raw = ws

    ROW_NAME = {
        "ionizable": 5,   # Excel row 6:   Ionizable Lipid Name
        "helper":    6,   # row 7: Helper Lipid Name
        "chol":      7,   # row 8: Cholesterol Lipid Name
        "peg":       8,   # row 9: PEG Lipid Name
        # "fifth":     9,   # row 10: 5th component name
    }
    ROW_CONC = {
        "ionizable": 78,  # row 79: Ionizable L. Conc.
        "helper":    79,  # row 80: Helper L. Conc.
        "chol":      80,  # row 81: Cholesterol Conc.
        "peg":       81,  # row 82: DMG-PEG Conc.
        # "fifth":     84,  # row 85: 5th component Conc.
    }
    ROW_VOL = {
        "ionizable": 90,  # row 91: SM102 solution (µL)
        "helper":    91,  # row 92: Helper lipid solution (µL)
        "chol":      92,  # row 93: Cholesterol solution (µL)
        "peg":       93,  # row 94: DMG-PEG solution (µL)
        "fifth":     95,  # row 96: 5th component TO ADD (µL)
        "ethanol":   96,  # row 97: Needed Ethanol Volume (µL)
    }

    # ←—— this is the one change
    num_cols = raw.shape[1]
    form_cols = [
        c for c in range(2, num_cols)
        if pd.notna(raw.iloc[ROW_NAME["ionizable"], c])
    ]

    records = []
    for c in form_cols:
        rec = {}
        for comp in ROW_NAME:
            name = raw.iloc[ROW_NAME[comp], c]
            conc = raw.iloc[ROW_CONC[comp],    c]
            vol  = raw.iloc[ROW_VOL[comp],     c]
            if pd.isna(name) or pd.isna(conc) or conc == 0:
                continue
            header = f"{conc:g}_mg_ml_{name}"
            rec[header] = vol

        eth = raw.iloc[ROW_VOL["ethanol"], c]
        if pd.notna(eth) and eth != 0:
            hv = math.floor(eth)
            lv = eth - hv
            rec["100%_EtOH_HV"] = hv
            rec["100%_EtOH_LV"] = lv

        records.append(rec)

    out = pd.DataFrame(records).fillna(0)

    # 6) Sort columns: ionizable, helper, fifth, chol, peg (by ascending concentration), then ethanol
    # Gather names by group
    names_by_group = {comp: set() for comp in ROW_NAME}
    for c in form_cols:
        for comp in ROW_NAME:
            nm = raw.iloc[ROW_NAME[comp], c]
            if pd.notna(nm):
                names_by_group[comp].add(nm)

    # Group headers and collect ethanol
    grouped = {comp: [] for comp in ['ionizable','helper','fifth','chol','peg']}
    eth_cols = []
    for col in out.columns:
        if col.startswith("100%_EtOH_"):
            eth_cols.append(col)
        else:
            conc_str, lipid = col.split("_mg_ml_", 1)
            conc = float(conc_str)
            for comp, names in names_by_group.items():
                if lipid in names:
                    grouped[comp].append((col, conc))
                    break

    # Build ordered column list
    ordered_cols = []
    for comp in ['ionizable','helper','fifth','chol','peg']:
        ordered_cols += [col for col, _ in sorted(grouped[comp], key=lambda x: x[1])]

    # Ethanol last: HV then LV
    for et in ["100%_EtOH_HV", "100%_EtOH_LV"]:
        if et in eth_cols:
            ordered_cols.append(et)

    out = out[ordered_cols]
    
    csv_path = f'exp_templates/{RUN_NAME}_mantis_ready.csv'
    # Save to CSV
    out.to_csv(csv_path, index = False)
    print(f"Mantis-ready CSV saved as {csv_path}")


def open_excel_file(filepath):
    system = platform.system()
    if system == "Windows":
        os.startfile(filepath)
    elif system == "Darwin":  # macOS
        subprocess.call(["open", filepath])
    elif system == "Linux":
        subprocess.call(["xdg-open", filepath])
    else:
        print("Unsupported OS. Please open the file manually:", filepath)
