import os
import pandas as pd
import numpy as np

CLINICAL_CSV_PATH = r"c:\project\majorprojectnew\HCC-TACE-Seg_Clinical_Data-V2.csv"

MRECIST_MAP = {
    1: "Complete Response (CR)",
    2: "Partial Response (PR)",
    3: "Stable Disease (SD)",
    4: "Progressive Disease (PD)"
}

EASL_MAP = {
    1: "Complete Response (CR)",
    2: "Partial Response (PR)",
    3: "Stable Disease (SD)",
    4: "Progressive Disease (PD)"
}

def load_clinical_dataset(csv_path=CLINICAL_CSV_PATH):
    """
    Loads and cleans the HCC-TACE clinical dataset covering HCC_001 to HCC_105.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinical CSV not found: {csv_path}")

    # Read CSV skipping data dictionary rows at the bottom
    df = pd.read_csv(csv_path)
    # Filter rows that have valid TCIA_ID starting with 'HCC_'
    df = df[df["TCIA_ID"].astype(str).str.startswith("HCC_")].copy()
    
    # Numeric conversions
    numeric_cols = [
        "Interval_BL", "Interval_FU", "TTP", "Death_1_StillAliveorLostToFU_0", 
        "Censored_0_progressed_1", "OS", "age", "Sex", "Smoking", "Alcohol",
        "fhx_can", "fhx_livc", "Diabetes", "Evidence_of_cirh", "Tr_Size",
        "Vascular invasion", "Metastasis", "Lymphnodes", "Portal Vein Thrombosis",
        "AFP", "CLIP_Score", "1_EASL", "1_RECIST", "1_mRECIST"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def get_patient_clinical_profile(patient_id, df=None):
    """
    Retrieves full clinical staging, etiology, pathology, chemotherapy,
    and post-TACE treatment response for a given patient.
    """
    if df is None:
        df = load_clinical_dataset()

    match = df[df["TCIA_ID"] == patient_id]
    if match.empty:
        return {}

    row = match.iloc[0]

    # Decode variables
    sex_str = "Male" if row.get("Sex") == 1 else ("Female" if row.get("Sex") == 2 else "Unknown")
    cirrhosis_str = "Yes (Cirrhotic)" if row.get("Evidence_of_cirh") == 1 else "No"
    vasc_inv = "Present (Vascular Invasion)" if row.get("Vascular invasion") == 1 else "Absent"
    pvt = "Present (PV Thrombosis)" if row.get("Portal Vein Thrombosis") == 1 else "Absent"
    mrecist_val = row.get("1_mRECIST")
    mrecist_str = MRECIST_MAP.get(int(mrecist_val), "N/A") if pd.notna(mrecist_val) else "N/A"
    
    chemo_regimen = str(row.get("chemotherapy", "Standard TACE Embolization"))
    if chemo_regimen in ["nan", "NA", ""]:
        chemo_regimen = "Standard TACE Embolization"

    profile = {
        "patient_id": patient_id,
        "age": int(row.get("age", 0)) if pd.notna(row.get("age")) else "N/A",
        "sex": sex_str,
        "hepatitis": str(row.get("hepatitis", "Unknown")),
        "cirrhosis": cirrhosis_str,
        "pathology": str(row.get("Pathology", "Unknown")),
        "cps_score": str(row.get("CPS", "A")), # Child-Pugh Score
        "bclc_stage": str(row.get("BCLC", "Stage-B")),
        "tnm_stage": str(row.get("TNM", "Stage-I")),
        "clip_stage": str(row.get("CLIP", "Stage_ 0-2")),
        "okuda_stage": str(row.get("Okuda", "Stage I")),
        "afp_ng_ml": float(row.get("AFP", 0.0)) if pd.notna(row.get("AFP")) else 0.0,
        "afp_group": str(row.get("AFP_group", "<400")),
        "tumor_size_cm": float(row.get("Tr_Size", 0.0)) if pd.notna(row.get("Tr_Size")) else 0.0,
        "tumor_nodularity": str(row.get("tumor_nodul", "multinodular")),
        "vascular_invasion": vasc_inv,
        "portal_vein_thrombosis": pvt,
        "t_involvement": str(row.get("T_involvment", "<= 50%")),
        "chemotherapy": chemo_regimen,
        "overall_survival_weeks": float(row.get("OS", 0.0)) if pd.notna(row.get("OS")) else 0.0,
        "time_to_progression_weeks": float(row.get("TTP", 0.0)) if pd.notna(row.get("TTP")) else 0.0,
        "mrecist_response": mrecist_str,
        "easl_response": EASL_MAP.get(int(row.get("1_EASL")), "N/A") if pd.notna(row.get("1_EASL")) else "N/A"
    }

    return profile

def get_cohort_summary_statistics(df=None):
    """
    Computes aggregate cohort statistics across all 105 HCC-TACE patients.
    """
    if df is None:
        df = load_clinical_dataset()

    total_pts = len(df)
    median_os = float(df["OS"].median()) if "OS" in df else 0.0
    median_ttp = float(df["TTP"].median()) if "TTP" in df else 0.0
    
    # BCLC distribution
    bclc_counts = df["BCLC"].value_counts().to_dict()
    # Child-Pugh distribution
    cps_counts = df["CPS"].value_counts().to_dict()
    # Hepatitis etiology
    hep_counts = df["hepatitis"].value_counts().to_dict()
    
    # mRECIST response breakdown
    mrecist_counts = {}
    if "1_mRECIST" in df:
        for val, count in df["1_mRECIST"].value_counts().items():
            try:
                name = MRECIST_MAP.get(int(val), f"Score {val}")
                mrecist_counts[name] = int(count)
            except:
                pass

    return {
        "total_patients": total_pts,
        "median_overall_survival_weeks": round(median_os, 1),
        "median_time_to_progression_weeks": round(median_ttp, 1),
        "bclc_distribution": bclc_counts,
        "child_pugh_distribution": cps_counts,
        "hepatitis_distribution": hep_counts,
        "mrecist_response_rates": mrecist_counts
    }

if __name__ == "__main__":
    df = load_clinical_dataset()
    print(f"Loaded {len(df)} clinical patient records.")
    p1 = get_patient_clinical_profile("HCC_001", df)
    print("Patient HCC_001 Profile:", p1)
    stats = get_cohort_summary_statistics(df)
    print("Cohort Stats:", stats)
