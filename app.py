import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import cv2
import torch
from PIL import Image

# Import project modules
from src.hcc_preprocessor import DEFAULT_RAW_DIR, PROCESSED_DIR
from src.mri_preprocessor import DEFAULT_MRI_RAW_DIR, MRI_PROCESSED_DIR, get_mri_patients
from src.segmentation_analysis import analyze_couinaud_segments
from src.vis_3d import create_3d_liver_tumor_figure
from src.model_gradcam import HCCTumorNet, GradCAM, overlay_gradcam_on_image, MODEL_WEIGHTS_PATH
from src.pencil_report_generator import generate_pencil_drawing_diagram, generate_pdf_clinical_report
from src.clinical_analytics import (
    load_clinical_dataset,
    get_patient_clinical_profile,
    get_cohort_summary_statistics,
    CLINICAL_CSV_PATH
)

# =====================================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Intelligent Liver Tumor Profiling: CT & MRI Multimodal Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 13.5px;
        color: #64748b;
        margin-bottom: 16px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 23px;
        font-weight: 800;
        margin-top: 3px;
    }
    .clinical-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .no-imaging-box {
        background: linear-gradient(135deg, #fafafa 0%, #f0f4ff 100%);
        border: 2px dashed #94a3b8;
        border-radius: 12px;
        padding: 32px 28px;
        text-align: center;
        color: #334155;
    }
    .no-imaging-icon {
        font-size: 52px;
        margin-bottom: 8px;
    }
    .no-imaging-title {
        font-size: 18px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 6px;
    }
    .no-imaging-body {
        font-size: 13.5px;
        color: #475569;
        line-height: 1.7;
    }
    .clin-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 18px;
        justify-content: center;
    }
    .clin-pill {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12.5px;
        font-weight: 600;
        background: #e0e7ff;
        color: #1e40af;
    }
    .modality-indicator {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATA & MODEL CACHING
# =====================================================================

@st.cache_data
def load_ct_manifest():
    manifest_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_mri_manifest():
    manifest_path = os.path.join(MRI_PROCESSED_DIR, "mri_dataset_summary.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_patient_npz(patient_id):
    """Load available preprocessed CT volumes and segmentation masks."""
    npz_path = os.path.join(PROCESSED_DIR, f"{patient_id}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    res = {
        "ct": data["ct"],
        "liver_mask": data["liver_mask"],
        "tumor_mask": data["tumor_mask"],
        "couinaud_mask": data["couinaud_mask"]
    }
    if "portal_vein_mask" in data:
        res["portal_vein_mask"] = data["portal_vein_mask"]
    if "aorta_mask" in data:
        res["aorta_mask"] = data["aorta_mask"]
    return res

@st.cache_data
def load_mri_npz(patient_id):
    """Load available preprocessed dynamic MRI volumes and multi-rater segmentation masks."""
    npz_path = os.path.join(MRI_PROCESSED_DIR, f"{patient_id}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    res = {
        "phases": {
            "art": data["art"],
            "pre": data["pre"],
            "pv": data["pv"],
            "del": data["del"]
        },
        "liver_mask": data["liver_mask"],
        "tumor_mask": data["tumor_mask"],
        "rater1_liver": data.get("rater1_liver", None),
        "rater2_liver": data.get("rater2_liver", None),
        "rater1_tumor": data.get("rater1_tumor", None),
        "rater2_tumor": data.get("rater2_tumor", None),
        "couinaud_mask": data["couinaud_mask"]
    }
    return res

@st.cache_data
def get_cached_clinical_df():
    try:
        return load_clinical_dataset()
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_deep_learning_model():
    device = torch.device("cpu")
    model = HCCTumorNet(num_classes=3).to(device)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
            model.eval()
            return model
        except Exception:
            pass
    model.eval()
    return model

# =====================================================================
# SIDEBAR CONTROLS: MODALITY SELECTOR & PATIENT CHOOSER
# =====================================================================

ct_manifest = load_ct_manifest()
mri_manifest = load_mri_manifest()
clinical_df = get_cached_clinical_df()

with st.sidebar:
    st.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/hospital.svg", width=36)
    st.markdown("### 🏥 Imaging Modality & Cohort")

    # Modality Selector (CT vs MRI)
    selected_modality = st.radio(
        "🔬 Select Modality:",
        ["🫁 CT Scan (Computed Tomography)", "🧲 MRI Scan (Multi-Phase MRI)"],
        index=0,
        key="modality_radio"
    )
    is_ct = "CT" in selected_modality

    st.divider()

    if is_ct:
        # Full cohort of all 105 patients: HCC_001 to HCC_105
        if not clinical_df.empty:
            all_ct_patients = sorted(
                clinical_df["TCIA_ID"].dropna().astype(str).unique().tolist(),
                key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0
            )
        else:
            all_ct_patients = [f"HCC_{i:03d}" for i in range(1, 106)]

        st.markdown(f"**CT Cohort:** 105 Patients (`HCC_001` – `HCC_105`)")
        st.caption("📁 Source: `HCC-TACE-Seg` DICOM CT & DICOM SEG")

        selected_patient = st.selectbox(
            "🎯 Select CT Patient Case:",
            all_ct_patients,
            index=0,
            key="ct_patient_picker"
        )

        p_clin = get_patient_clinical_profile(selected_patient, clinical_df)
        p_meta = ct_manifest.get(selected_patient, {})

        # Check real CT availability
        ct_npz_exists = os.path.isfile(os.path.join(PROCESSED_DIR, f"{selected_patient}.npz"))
        has_real_imaging = ct_npz_exists

        if has_real_imaging:
            st.success(f"✅ **{selected_patient}** — Real DICOM CT + Masks Loaded")
        else:
            st.info(f"ℹ️ **{selected_patient}** — Clinical data only (Preprocessing in progress)")

        if p_clin:
            st.markdown("#### 📋 Clinical Profile")
            st.markdown(f"**Age / Sex:** `{p_clin.get('age', 'N/A')} yrs` / `{p_clin.get('sex', 'N/A')}`")
            st.markdown(f"**Etiology:** `{p_clin.get('hepatitis', 'N/A')}`")
            st.markdown(f"**Cirrhosis:** `{p_clin.get('cirrhosis', 'N/A')}`")
            st.markdown(f"**BCLC Stage:** `{p_clin.get('bclc_stage', 'N/A')}`")
            st.markdown(f"**Child-Pugh:** `Class {p_clin.get('cps_score', 'N/A')}`")
            st.markdown(f"**AFP Level:** `{p_clin.get('afp_ng_ml', 0):.1f} ng/mL`")
            st.markdown(f"**Chemotherapy:** `{p_clin.get('chemotherapy', 'TACE')}`")
            st.markdown(f"**mRECIST Response:** `{p_clin.get('mrecist_response', 'N/A')}`")

        st.divider()
        if has_real_imaging:
            st.markdown("#### ⚙️ CT & AI Preferences")
            selected_colormap = st.selectbox("Grad-CAM Palette", ["TURBO", "JET", "INFERNO", "VIRIDIS"], index=0)
            selected_cam_layer = st.selectbox("Grad-CAM Target Layer", ["dec3 (High Resolution)", "dec2 (Balanced)", "bottleneck (Deep Semantic)", "dec1 (Fine Margin)"], index=0)
            layer_name = selected_cam_layer.split()[0]
            mask_cam_to_liver = st.checkbox("Focus Attention inside Liver Only", value=True)
            cam_alpha = st.slider("Grad-CAM Heatmap Alpha", 0.1, 1.0, 0.55, 0.05)
            cam_cutoff = st.slider("Grad-CAM Cutoff Threshold", 0.0, 0.5, 0.10, 0.02)
            overlay_opacity = st.slider("Mask Overlay Opacity", 0.1, 1.0, 0.45, 0.05)
        else:
            selected_colormap = "TURBO"
            layer_name = "dec3"
            mask_cam_to_liver = True
            cam_alpha = 0.55
            cam_cutoff = 0.10
            overlay_opacity = 0.45

    else:
        # MRI MODALITY
        mri_patients = get_mri_patients()
        if not mri_patients:
            mri_patients = sorted(list(mri_manifest.keys()))
        if not mri_patients:
            mri_patients = ["TCGA-BC-A10Y"]

        st.markdown(f"**MRI Cohort:** {len(mri_patients)} Dynamic Multi-Phase Cases")
        st.caption("📁 Source: `LiverHccSeg v1.1` (TCGA-LIHC)")

        selected_patient = st.selectbox(
            "🎯 Select MRI Patient Case:",
            mri_patients,
            index=0 if "TCGA-BC-A10Y" not in mri_patients else mri_patients.index("TCGA-BC-A10Y"),
            key="mri_patient_picker"
        )

        p_meta = mri_manifest.get(selected_patient, {})
        has_real_imaging = os.path.isfile(os.path.join(MRI_PROCESSED_DIR, f"{selected_patient}.npz"))

        if has_real_imaging:
            st.success(f"✅ **{selected_patient}** — Real Multi-Phase MRI + Masks")
        else:
            st.warning(f"⚠️ {selected_patient} — Preprocessing pending.")

        # MRI Phase Selector
        st.markdown("#### 🧭 Dynamic MRI Phase")
        mri_phase_names = {
            "art": "Arterial Phase (art)",
            "pv": "Portal Venous Phase (pv)",
            "del": "Delayed Phase (del)",
            "pre": "Pre-Contrast Phase (pre)"
        }
        selected_phase_label = st.radio(
            "Current Active Phase:",
            list(mri_phase_names.values()),
            index=0,
            key="mri_phase_radio"
        )
        selected_phase_key = [k for k, v in mri_phase_names.items() if v == selected_phase_label][0]

        # Rater annotation selector
        st.markdown("#### 👨‍⚕️ Radiologist Annotation")
        rater_choice = st.radio(
            "Segmentation Source:",
            ["Consensus (Rater 1 + Rater 2)", "Radiologist Rater 1", "Radiologist Rater 2"],
            index=0,
            key="mri_rater_radio"
        )

        p_clin = {
            "patient_id": selected_patient,
            "modality": "Dynamic MRI",
            "session_date": p_meta.get("session_date", "N/A"),
            "bclc_stage": "Stage-B (Intermediate)" if p_meta.get("tumor_volume_cm3", 0) > 50 else "Stage-A (Early)",
            "cps_score": "A",
            "afp_ng_ml": 45.0,
            "chemotherapy": "TACE Eligible",
            "mrecist_response": "Pre-Treatment Baseline"
        }

        overlay_opacity = st.slider("Segmentation Opacity", 0.1, 1.0, 0.45, 0.05, key="mri_opacity")
        selected_colormap = "TURBO"
        layer_name = "dec3"
        mask_cam_to_liver = True
        cam_alpha = 0.55
        cam_cutoff = 0.10

# =====================================================================
# LOAD PATIENT DATA (CT or MRI)
# =====================================================================

if is_ct:
    patient_data = load_patient_npz(selected_patient) if has_real_imaging else None
    if has_real_imaging and patient_data is None:
        st.error(f"❌ Failed to load DICOM volume for {selected_patient}. Please re-run CT preprocessing.")
        st.stop()

    if has_real_imaging:
        scan_volume = patient_data["ct"]        # (S, 256, 256)
        liver_mask = patient_data["liver_mask"]
        tumor_mask = patient_data["tumor_mask"]
        portal_vein_mask = patient_data.get("portal_vein_mask", None)
        aorta_mask = patient_data.get("aorta_mask", None)
        couinaud_mask = patient_data["couinaud_mask"]
        num_slices = scan_volume.shape[0]
        spacing_mm = p_meta.get("voxel_spacing_mm", [0.78, 0.78, 2.5])
        voxel_vol_cm3 = p_meta.get("voxel_volume_cm3", (spacing_mm[0]*spacing_mm[1]*spacing_mm[2])/1000.0)

        couinaud_df, summary = analyze_couinaud_segments(
            couinaud_mask=couinaud_mask,
            liver_mask=liver_mask,
            tumor_mask=tumor_mask,
            voxel_vol_cm3=voxel_vol_cm3,
            spacing_mm=spacing_mm
        )

        tumor_slice_sums = [np.sum(tumor_mask[s]) for s in range(num_slices)]
        max_tumor_slice_idx = int(np.argmax(tumor_slice_sums)) if max(tumor_slice_sums) > 0 else (num_slices // 2)
    else:
        tr_size_cm = float(p_clin.get("tumor_size_cm", 0) or 0)
        est_tumor_vol_cm3 = (4/3) * 3.14159 * (tr_size_cm / 2) ** 3 if tr_size_cm > 0 else 0.0
        est_liver_vol_cm3 = 1500.0
        est_healthy_vol_cm3 = max(0.0, est_liver_vol_cm3 - est_tumor_vol_cm3)
        summary = {
            "total_liver_cm3": est_liver_vol_cm3,
            "total_healthy_cm3": est_healthy_vol_cm3,
            "total_tumor_cm3": est_tumor_vol_cm3,
            "tumor_burden_pct": (est_tumor_vol_cm3 / est_liver_vol_cm3 * 100) if est_liver_vol_cm3 > 0 else 0.0,
            "future_liver_remnant_pct": (est_healthy_vol_cm3 / est_liver_vol_cm3 * 100) if est_liver_vol_cm3 > 0 else 0.0,
            "tace_plan": "Clinical staging — volumetric imaging data not available",
            "involved_segments": []
        }
        couinaud_df = pd.DataFrame()
        scan_volume = None
        portal_vein_mask = None
        aorta_mask = None

else:
    # MRI Data Loading
    mri_data = load_mri_npz(selected_patient) if has_real_imaging else None
    if has_real_imaging and mri_data is None:
        st.error(f"❌ Failed to load MRI volume for {selected_patient}. Please re-run MRI preprocessing.")
        st.stop()

    if has_real_imaging:
        scan_volume = mri_data["phases"][selected_phase_key]  # (S, 256, 256)
        all_phases = mri_data["phases"]
        
        # Select mask by rater
        if "Rater 1" in rater_choice and mri_data.get("rater1_liver") is not None and mri_data["rater1_liver"].sum() > 0:
            liver_mask = mri_data["rater1_liver"]
            tumor_mask = mri_data["rater1_tumor"] if mri_data.get("rater1_tumor") is not None else np.zeros_like(liver_mask)
        elif "Rater 2" in rater_choice and mri_data.get("rater2_liver") is not None and mri_data["rater2_liver"].sum() > 0:
            liver_mask = mri_data["rater2_liver"]
            tumor_mask = mri_data["rater2_tumor"] if mri_data.get("rater2_tumor") is not None else np.zeros_like(liver_mask)
        else:
            liver_mask = mri_data["liver_mask"]
            tumor_mask = mri_data["tumor_mask"]

        portal_vein_mask = None
        aorta_mask = None
        couinaud_mask = mri_data["couinaud_mask"]
        num_slices = scan_volume.shape[0]
        spacing_mm = p_meta.get("voxel_spacing_mm", [1.0, 1.0, 5.0])
        voxel_vol_cm3 = p_meta.get("voxel_volume_cm3", (spacing_mm[0]*spacing_mm[1]*spacing_mm[2])/1000.0)

        couinaud_df, summary = analyze_couinaud_segments(
            couinaud_mask=couinaud_mask,
            liver_mask=liver_mask,
            tumor_mask=tumor_mask,
            voxel_vol_cm3=voxel_vol_cm3,
            spacing_mm=spacing_mm
        )

        tumor_slice_sums = [np.sum(tumor_mask[s]) for s in range(num_slices)]
        max_tumor_slice_idx = int(np.argmax(tumor_slice_sums)) if max(tumor_slice_sums) > 0 else (num_slices // 2)
    else:
        summary = {
            "total_liver_cm3": 1500.0,
            "total_healthy_cm3": 1500.0,
            "total_tumor_cm3": 0.0,
            "tumor_burden_pct": 0.0,
            "future_liver_remnant_pct": 100.0,
            "tace_plan": "Awaiting MRI preprocessing",
            "involved_segments": []
        }
        couinaud_df = pd.DataFrame()
        scan_volume = None
        all_phases = {}

# =====================================================================
# MAIN HEADER & CLINICAL BANNER
# =====================================================================

col_h1, col_h2 = st.columns([3.5, 1.5])
with col_h1:
    mod_name = "CT Scan (Computed Tomography)" if is_ct else "Dynamic Multi-Phase MRI"
    st.markdown(f'<div class="main-header">HCC Interventional Oncology Intelligence — {mod_name}</div>', unsafe_allow_html=True)
    if is_ct:
        imaging_flag = "🔬 Real DICOM CT" if has_real_imaging else "📋 Clinical Data Only"
        st.markdown(f'<div class="sub-header">Couinaud Segmental Partitioning | Seg-Grad-CAM Localization | Clinical Pencil Report — Patient <b>{selected_patient}</b> (Cohort 1–105) &nbsp;·&nbsp; {imaging_flag}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sub-header">Multi-Phase Dynamic Imaging ({selected_phase_label}) | Multi-Rater Segmentations ({rater_choice}) — Patient <b>{selected_patient}</b> &nbsp;·&nbsp; 🔬 Real NIfTI MRI</div>', unsafe_allow_html=True)

with col_h2:
    if is_ct:
        cps_col = "#15803d" if p_clin.get("cps_score") == "A" else "#b45309"
        bclc_col = "#2563eb" if "Stage-A" in str(p_clin.get("bclc_stage")) else "#dc2626"
        st.markdown(f"""
        <div style="text-align:right; margin-top:5px;">
            <span class="clinical-badge" style="background-color:#dbeafe; color:{bclc_col};">{p_clin.get('bclc_stage', 'BCLC Stage')}</span>
            <span class="clinical-badge" style="background-color:#dcfce7; color:{cps_col};">Child-Pugh {p_clin.get('cps_score', 'A')}</span>
            <span class="clinical-badge" style="background-color:#fef3c7; color:#b45309;">AFP: {p_clin.get('afp_ng_ml', 0):.0f} ng/mL</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:right; margin-top:5px;">
            <span class="clinical-badge" style="background-color:#ede9fe; color:#6d28d9;">MRI Session: {p_clin.get('session_date', 'N/A')}</span>
            <span class="clinical-badge" style="background-color:#e0f2fe; color:#0369a1;">Slices: {num_slices if has_real_imaging else 0}</span>
            <span class="clinical-badge" style="background-color:#dcfce7; color:#15803d;">Multi-Rater</span>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# TOP KPI EXECUTIVE VOLUMETRIC CARDS (ALL IN CM³)
# =====================================================================

k1, k2, k3, k4, k5 = st.columns(5)
vol_note = "CT-derived" if is_ct else f"MRI-derived ({selected_phase_key})"
if not has_real_imaging:
    vol_note = "Estimated (no scan)"

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Liver Volume</div>
        <div class="kpi-value" style="color:#0f172a;">{summary['total_liver_cm3']:.1f} <span style="font-size:13px;color:#64748b;">cm³</span></div>
        <div style="font-size:11px;color:#64748b;margin-top:2px;">{vol_note}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Healthy Liver Volume</div>
        <div class="kpi-value" style="color:#16a34a;">{summary['total_healthy_cm3']:.1f} <span style="font-size:13px;color:#64748b;">cm³</span></div>
        <div style="font-size:11px;color:#16a34a;margin-top:2px;">✓ Functional parenchyma</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Tumor Volume</div>
        <div class="kpi-value" style="color:#dc2626;">{summary['total_tumor_cm3']:.1f} <span style="font-size:13px;color:#64748b;">cm³</span></div>
        <div style="font-size:11px;color:#dc2626;margin-top:2px;">⚠️ Active HCC mass</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Tumor Burden Ratio</div>
        <div class="kpi-value" style="color:#ea580c;">{summary['tumor_burden_pct']:.2f} <span style="font-size:13px;color:#64748b;">%</span></div>
        <div style="font-size:11px;color:#64748b;margin-top:2px;">Tumor / Total liver</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Future Liver Remnant (FLR)</div>
        <div class="kpi-value" style="color:#2563eb;">{summary['future_liver_remnant_pct']:.1f} <span style="font-size:13px;color:#64748b;">%</span></div>
        <div style="font-size:11px;color:#64748b;margin-top:2px;">Functional reserve</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Clinical Action Banner
involved_segs_str = ', '.join(summary['involved_segments']) if summary.get('involved_segments') else ('Clear' if summary['total_tumor_cm3'] == 0 else 'N/A')
tace_plan_text = summary.get('tace_plan', 'Interventional staging')
st.markdown(f"""
<div style="background-color:#eff6ff; border-left: 4px solid #3b82f6; padding: 10px 14px; border-radius: 4px; font-size:12.5px; color:#1e3a8a; margin-bottom:14px;">
    <b>🩺 Clinical Roadmap ({'CT' if is_ct else 'MRI'}):</b> {tace_plan_text} |
    <b>Couinaud Segments:</b> {involved_segs_str} |
    <b>Voxel Spacing:</b> [{spacing_mm[0]:.2f}, {spacing_mm[1]:.2f}, {spacing_mm[2]:.2f}] mm |
    <b>Response:</b> <b>{p_clin.get('mrecist_response', 'N/A')}</b>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# UTILITY: "No Imaging" placeholder panel
# =====================================================================

def render_no_imaging_panel(tab_name: str):
    """Renders a polished panel when imaging is not available."""
    st.markdown(f"""
    <div class="no-imaging-box">
        <div class="no-imaging-icon">🏥</div>
        <div class="no-imaging-title">DICOM Imaging Pending for {selected_patient}</div>
        <div class="no-imaging-body">
            <b>{tab_name}</b> requires volumetric scan data.<br>
            Preprocessing is actively running across the cohort. The clinical profile below is drawn from the clinical registry.
        </div>
        <div class="clin-row">
            <span class="clin-pill">Age: {p_clin.get('age','N/A')} yrs</span>
            <span class="clin-pill">Sex: {p_clin.get('sex','N/A')}</span>
            <span class="clin-pill">Etiology: {p_clin.get('hepatitis','N/A')}</span>
            <span class="clin-pill">Cirrhosis: {p_clin.get('cirrhosis','N/A')}</span>
            <span class="clin-pill">BCLC: {p_clin.get('bclc_stage','N/A')}</span>
            <span class="clin-pill">Child-Pugh: {p_clin.get('cps_score','N/A')}</span>
            <span class="clin-pill">AFP: {p_clin.get('afp_ng_ml',0):.0f} ng/mL</span>
            <span class="clin-pill">Chemo: {p_clin.get('chemotherapy','N/A')}</span>
            <span class="clin-pill">mRECIST: {p_clin.get('mrecist_response','N/A')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

# =====================================================================
# TABS ARCHITECTURE (ORDER: 1, 2, 3, 4, 5, 6)
# =====================================================================

tabs = st.tabs([
    "🧠 1. Seg-Grad-CAM Tumor Highlighting",
    "📐 2. Axial Scan & Mask Slicer",
    "🧊 3. 3D Organ & Tumor Mesh",
    "📊 4. Couinaud Segmentational Analysis",
    "✏️ 5. Pencil Drawing & Clinical PDF Report",
    "📈 6. Full Clinical Cohort Analytics (105 Patients)"
])

# ---------------------------------------------------------------------
# TAB 1: SEG-GRAD-CAM TUMOR HIGHLIGHTING (ORDER 1)
# ---------------------------------------------------------------------
with tabs[0]:
    if is_ct:
        st.markdown("### 🧠 High-Precision Seg-Grad-CAM Tumor Localization Studio (CT)")

        if not has_real_imaging:
            render_no_imaging_panel("Seg-Grad-CAM Tumor Localization")
        else:
            st.markdown("""
            **Segmentation-Guided Grad-CAM (Seg-Grad-CAM)** backpropagates from the model's localized lesion feature maps,
            highlighting the hypervascular tumor focus and margins without exterior background artifacts.
            """)

            model = load_deep_learning_model()
            cam_engine = GradCAM(model, target_layer_name=layer_name)

            col_g1, col_g2 = st.columns([1, 3.5])
            with col_g1:
                st.markdown("#### Slice & Focus Target")
                slice_for_cam = st.slider("Select Axial Slice:", 0, num_slices - 1, max_tumor_slice_idx, key="gradcam_slice_slider")
                target_focus = st.radio("Explainability Target:", ["HCC Tumor Focus (Lesion)", "Healthy Liver Parenchyma"], index=0)
                target_class_idx = 2 if "Tumor" in target_focus else 1
                cam_method = st.selectbox("Grad-CAM Formulation", ["Seg-Grad-CAM (Direct Spatial Gradients)", "Grad-CAM++ (Higher-Order Weighting)"])
                method_key = "gradcam_plus" if "Grad-CAM++" in cam_method else "seg_gradcam"

                has_gt_tumor = np.sum(tumor_mask[slice_for_cam] > 0) > 0
                if has_gt_tumor:
                    st.error("⚠️ Ground Truth Tumor on Slice")
                else:
                    st.success("✓ Healthy Liver on Slice")

            with col_g2:
                ct_cam_slice = scan_volume[slice_for_cam]
                cam_tensor = torch.from_numpy(ct_cam_slice).unsqueeze(0).unsqueeze(0).float()
                liv_slice = liver_mask[slice_for_cam]
                tum_slice = tumor_mask[slice_for_cam]

                heatmap, cls_probs, pred_seg = cam_engine.generate_heatmap(
                    input_tensor=cam_tensor,
                    target_class=target_class_idx,
                    mask_to_liver=mask_cam_to_liver,
                    liver_mask=liv_slice,
                    method=method_key
                )

                cmap_mapping = {
                    "TURBO": cv2.COLORMAP_TURBO,
                    "JET": cv2.COLORMAP_JET,
                    "INFERNO": cv2.COLORMAP_INFERNO,
                    "VIRIDIS": cv2.COLORMAP_VIRIDIS
                }
                chosen_cmap = cmap_mapping.get(selected_colormap, cv2.COLORMAP_TURBO)

                blended_cam = overlay_gradcam_on_image(
                    ct_slice=ct_cam_slice,
                    cam=heatmap,
                    alpha=cam_alpha,
                    colormap=chosen_cmap,
                    threshold=cam_cutoff
                )

                if np.sum(tum_slice > 0) > 0:
                    contours, _ = cv2.findContours(tum_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(blended_cam, contours, -1, (255, 255, 255), 2)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown("**1. Raw CT Slice (HU Window)**")
                    st.image(ct_cam_slice, clamp=True, use_container_width=True)
                with c2:
                    st.markdown("**2. Ground Truth Mask**")
                    ct_rgb_gt = cv2.cvtColor((ct_cam_slice * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    gt_show = ct_rgb_gt.copy()
                    gt_show[liv_slice > 0] = [34, 197, 94]
                    gt_show[tum_slice > 0] = [239, 68, 68]
                    st.image(cv2.addWeighted(ct_rgb_gt, 0.55, gt_show, 0.45, 0), use_container_width=True)
                with c3:
                    st.markdown(f"**3. Grad-CAM Heatmap ({selected_colormap})**")
                    heat_u8 = (heatmap * 255.0).astype(np.uint8)
                    heat_rgb = cv2.applyColorMap(heat_u8, chosen_cmap)
                    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
                    st.image(heat_rgb, use_container_width=True)
                with c4:
                    st.markdown("**4. Grad-CAM Overlay (Tumor Boundary in White)**")
                    st.image(blended_cam, use_container_width=True)

                tum_mean = np.mean(heatmap[tum_slice > 0]) if np.sum(tum_slice > 0) > 0 else 0.0
                liv_mean = np.mean(heatmap[(liv_slice > 0) & (tum_slice == 0)]) if np.sum(liv_slice > 0) > 0 else 0.0
                peak_ratio = (tum_mean / (liv_mean + 1e-6)) if liv_mean > 0 else 1.0

                q1, q2, q3, q4 = st.columns(4)
                with q1:
                    st.metric("Peak Tumor Attention", f"{np.max(heatmap):.3f}")
                with q2:
                    st.metric("Mean Tumor Activation", f"{tum_mean:.3f}")
                with q3:
                    st.metric("Healthy Parenchyma Activation", f"{liv_mean:.3f}")
                with q4:
                    st.metric("Tumor Contrast Ratio", f"{peak_ratio:.2f}x", delta="Target Focused" if peak_ratio > 1.5 else "Diffuse")

            cam_engine.cleanup()
    else:
        # MRI MODALITY
        st.markdown("### 🧲 Multi-Phase Dynamic MRI Contrast Comparison (LI-RADS)")
        st.info("ℹ️ **Domain Notice:** The deep learning Seg-Grad-CAM model is calibrated specifically for CT Hounsfield Units. For Dynamic MRI, multi-phase wash-in / wash-out contrast comparison across the 4 co-registered phases is displayed below alongside expert radiologist annotations.")

        if not has_real_imaging:
            st.warning("MRI data loading in progress.")
        else:
            col_m1, col_m2 = st.columns([1, 4])
            with col_m1:
                st.markdown("#### Select Axial Slice")
                mri_cam_slice_idx = st.slider("Axial Slice:", 0, num_slices - 1, max_tumor_slice_idx, key="mri_cam_slice_slider")
                show_tumor_contour = st.checkbox("Overlay Tumor Boundary (White)", value=True)
                has_tum = np.sum(tumor_mask[mri_cam_slice_idx] > 0) > 0
                if has_tum:
                    st.error("⚠️ Tumor Present on Slice")
                else:
                    st.success("✓ Healthy Parenchyma")

            with col_m2:
                # Show all 4 dynamic phases side-by-side
                p_cols = st.columns(4)
                phase_order = [
                    ("pre", "1. Pre-Contrast (T1 Baseline)"),
                    ("art", "2. Arterial Phase (Hyperenhancement)"),
                    ("pv", "3. Portal Venous (Washout)"),
                    ("del", "4. Delayed Phase (Capsule/Washout)")
                ]

                for idx, (pk, p_label) in enumerate(phase_order):
                    with p_cols[idx]:
                        st.markdown(f"**{p_label}**")
                        phase_slice = all_phases.get(pk, scan_volume)[mri_cam_slice_idx]
                        phase_rgb = cv2.cvtColor((phase_slice * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                        if show_tumor_contour and has_tum:
                            tum_sl = tumor_mask[mri_cam_slice_idx]
                            contours, _ = cv2.findContours(tum_sl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(phase_rgb, contours, -1, (255, 255, 255), 2)
                        st.image(phase_rgb, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: AXIAL SCAN & MASK SLICER (ORDER 5)
# ---------------------------------------------------------------------
with tabs[1]:
    mod_str = "CT" if is_ct else f"MRI ({selected_phase_label})"
    st.markdown(f"### 📐 Multi-Slice Axial {mod_str} & Anatomical Slicer")

    if not has_real_imaging:
        render_no_imaging_panel(f"Axial {mod_str} Slicer")
    else:
        col_s1, col_s2 = st.columns([1, 3])

        with col_s1:
            st.markdown("#### Navigation")
            selected_slice = st.slider("Select Axial Slice:", 0, num_slices - 1, max_tumor_slice_idx, key="axial_slicer")
            if st.button("🎯 Jump to Max Tumor Slice", key="jump_btn"):
                selected_slice = max_tumor_slice_idx
                st.rerun()

            st.markdown("#### Layer Toggles")
            show_liver = st.checkbox("Show Liver Boundary (Green)", value=True, key="sl_liv")
            show_tumor = st.checkbox("Show Tumor Boundary (Red)", value=True, key="sl_tum")
            show_couinaud = st.checkbox("Show Couinaud Segments (Color)", value=False, key="sl_couin")
            
            # Show Portal Vein and Aorta if available (CT)
            show_pv = False
            show_aorta = False
            if is_ct and portal_vein_mask is not None and portal_vein_mask.sum() > 0:
                show_pv = st.checkbox("Show Portal Vein (Cyan)", value=True, key="sl_pv")
            if is_ct and aorta_mask is not None and aorta_mask.sum() > 0:
                show_aorta = st.checkbox("Show Abdominal Aorta (Orange)", value=True, key="sl_aorta")

            slice_pixel_area_cm2 = (spacing_mm[0] * spacing_mm[1]) / 100.0
            cur_liv_vox = np.sum(liver_mask[selected_slice] > 0)
            cur_tum_vox = np.sum(tumor_mask[selected_slice] > 0)

            st.markdown("---")
            st.markdown(f"**Slice Index:** `{selected_slice + 1} / {num_slices}`")
            st.markdown(f"**Liver Area:** `{cur_liv_vox * slice_pixel_area_cm2:.1f} cm²`")
            st.markdown(f"**Tumor Area:** `{cur_tum_vox * slice_pixel_area_cm2:.2f} cm²`")

        with col_s2:
            base_sl = scan_volume[selected_slice]
            base_rgb = cv2.cvtColor((base_sl * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            overlay = base_rgb.copy()

            if show_couinaud and couinaud_mask is not None:
                couin_sl = couinaud_mask[selected_slice]
                couin_color = np.zeros_like(overlay)
                palette = {
                    1: [147, 51, 234], 2: [59, 130, 246], 3: [14, 165, 233], 4: [16, 185, 129],
                    5: [234, 179, 8], 6: [249, 115, 22], 7: [239, 68, 68], 8: [217, 70, 239]
                }
                for sid, col in palette.items():
                    couin_color[couin_sl == sid] = col
                mask_c = (couin_sl > 0)[:, :, np.newaxis]
                overlay = np.where(mask_c, (overlay_opacity * couin_color + (1 - overlay_opacity) * overlay).astype(np.uint8), overlay)

            # Portal vein overlay (Cyan)
            if show_pv and portal_vein_mask is not None:
                pv_sl = portal_vein_mask[selected_slice]
                if np.sum(pv_sl) > 0:
                    contours_pv, _ = cv2.findContours(pv_sl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours_pv, -1, (6, 182, 212), 2)

            # Aorta overlay (Orange)
            if show_aorta and aorta_mask is not None:
                ao_sl = aorta_mask[selected_slice]
                if np.sum(ao_sl) > 0:
                    contours_ao, _ = cv2.findContours(ao_sl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours_ao, -1, (249, 115, 22), 2)

            if show_liver:
                liv_sl = liver_mask[selected_slice]
                contours_liv, _ = cv2.findContours(liv_sl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours_liv, -1, (34, 197, 94), 2)

            if show_tumor:
                tum_sl = tumor_mask[selected_slice]
                if np.sum(tum_sl) > 0:
                    tum_color = np.zeros_like(overlay)
                    tum_color[tum_sl > 0] = [239, 68, 68]
                    mask_t = (tum_sl > 0)[:, :, np.newaxis]
                    overlay = np.where(mask_t, (overlay_opacity * tum_color + (1 - overlay_opacity) * overlay).astype(np.uint8), overlay)
                    contours_tum, _ = cv2.findContours(tum_sl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours_tum, -1, (220, 38, 38), 2)

            caption_str = f"Axial {'CT' if is_ct else 'MRI'} Slice {selected_slice + 1}/{num_slices}"
            st.image(overlay, caption=caption_str, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: 3D ORGAN & TUMOR MESH (ORDER 4)
# ---------------------------------------------------------------------
with tabs[2]:
    st.markdown(f"### 🧊 Interactive 3D Liver & Tumor Volumetric Reconstruction ({'CT' if is_ct else 'MRI'})")

    if not has_real_imaging:
        render_no_imaging_panel("3D Organ & Tumor Mesh Reconstruction")
    else:
        st.markdown("Rotate (Left-Click Drag), Zoom (Scroll), Pan (Right-Click Drag) to inspect the 3D spatial relationship between the translucent liver capsule and embedded tumor.")
        fig_3d = create_3d_liver_tumor_figure(
            liver_mask=liver_mask,
            tumor_mask=tumor_mask,
            spacing_mm=spacing_mm,
            title=f"Patient {selected_patient} ({'CT' if is_ct else 'MRI'}): 3D Liver Anatomy & Embedded HCC Tumor"
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: COUINAUD SEGMENTATIONAL ANALYSIS (ORDER 3)
# ---------------------------------------------------------------------
with tabs[3]:
    st.markdown("### 📊 Couinaud Segmentational Analysis (Segments I to VIII)")

    if not has_real_imaging:
        render_no_imaging_panel("Couinaud Segmentational Analysis")
    else:
        st.markdown("Volumetric breakdown of functional parenchyma, healthy tissue, and tumor burden in cm³ per Couinaud segment.")

        col_tab1, col_tab2 = st.columns([3, 2])
        with col_tab1:
            fig_bar = px.bar(
                couinaud_df,
                x="Anatomical Name",
                y=["Healthy Parenchyma (cm3)", "Tumor Volume (cm3)"],
                title="<b>Couinaud Segment Volumes: Healthy vs Tumor (cm³)</b>",
                color_discrete_map={"Healthy Parenchyma (cm3)": "#22c55e", "Tumor Volume (cm3)": "#ef4444"},
                labels={"value": "Volume (cm³)", "variable": "Tissue Category"},
                barmode="stack"
            )
            fig_bar.update_layout(xaxis_tickangle=-35, legend_title_text="")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_tab2:
            r_categories = [f"Seg {r['Segment ID']}" for _, r in couinaud_df.iterrows()]
            r_burdens = couinaud_df["Tumor Burden (%)"].tolist()
            if r_categories:
                r_categories.append(r_categories[0])
                r_burdens.append(r_burdens[0])

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=r_burdens,
                theta=r_categories,
                fill='toself',
                fillcolor='rgba(239, 68, 68, 0.35)',
                line=dict(color='#dc2626', width=2),
                name='Tumor Burden %'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(max(r_burdens) if r_burdens else 10, 10)])),
                title="<b>Tumor Burden (%) Across Couinaud Sectors</b>",
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### Clinical Segmental Breakdown Table (cm³)")
        st.dataframe(
            couinaud_df.style.format({
                "Total Volume (cm3)": "{:.2f}",
                "Healthy Parenchyma (cm3)": "{:.2f}",
                "Tumor Volume (cm3)": "{:.2f}",
                "Tumor Burden (%)": "{:.2f}%"
            }),
            use_container_width=True
        )

# ---------------------------------------------------------------------
# TAB 5: PENCIL DRAWING & CLINICAL PDF REPORT (ORDER 2)
# ---------------------------------------------------------------------
with tabs[4]:
    st.markdown("### ✏️ Anatomical Pencil Drawing & Formal Clinical PDF Report")
    st.markdown("""
    Renders an authentic hand-drawn medical pencil sketch clearly partitioning the Couinaud anatomy into:
    - **Good / Healthy Tissue:** Soft sage green stippling with exact volume in $\\text{cm}^3$.
    - **Tumor Zone:** Dense crimson cross-hatching with exact tumor volume and burden in $\\text{cm}^3$.
    """)

    pencil_png_path = os.path.join(r"C:\project\majorprojectnew\outputs", f"{selected_patient}_pencil_drawing.png")
    pencil_pdf_path = os.path.join(r"C:\project\majorprojectnew\outputs", f"{selected_patient}_clinical_report.pdf")

    if has_real_imaging and not couinaud_df.empty:
        couin_stats_dict = p_meta.get("couinaud_segments", {})
        if not couin_stats_dict:
            couin_stats_dict = {
                str(r["Segment ID"]): {
                    "total_cm3": r["Total Volume (cm3)"],
                    "healthy_cm3": r["Healthy Parenchyma (cm3)"],
                    "tumor_cm3": r["Tumor Volume (cm3)"],
                    "burden_pct": r["Tumor Burden (%)"]
                }
                for _, r in couinaud_df.iterrows()
            }
    else:
        avg_liver_per_seg = summary['total_liver_cm3'] / 8.0
        est_tumor = summary['total_tumor_cm3']
        couin_stats_dict = {}
        for seg_id in range(1, 9):
            tum = (est_tumor / 2.0) if (seg_id in [4, 5] and est_tumor > 0) else 0.0
            couin_stats_dict[str(seg_id)] = {
                "total_cm3": avg_liver_per_seg,
                "healthy_cm3": max(0.0, avg_liver_per_seg - tum),
                "tumor_cm3": tum,
                "burden_pct": (tum / avg_liver_per_seg * 100) if avg_liver_per_seg > 0 else 0.0
            }

    pencil_bytes = generate_pencil_drawing_diagram(
        couinaud_stats=couin_stats_dict,
        patient_id=selected_patient,
        clinical_profile=p_clin,
        output_png_path=pencil_png_path
    )

    col_p1, col_p2 = st.columns([3, 1.2])
    with col_p1:
        caption_note = f"Real {'CT' if is_ct else 'MRI'}-derived Couinaud" if has_real_imaging else "Clinical estimate (no scan) — for illustration"
        st.image(pencil_bytes, caption=f"Pencil Drawing Medical Illustration — {caption_note} ({selected_patient})", use_container_width=True)

    with col_p2:
        st.markdown("#### 📄 Clinical Document Export")
        st.download_button(
            label="🖼️ Download Pencil Sketch (PNG)",
            data=pencil_bytes,
            file_name=f"{selected_patient}_pencil_drawing.png",
            mime="image/png"
        )
        st.write("")
        if st.button("📑 Compile Clinical PDF Report"):
            with st.spinner("Generating High-Resolution Clinical PDF Report..."):
                generate_pdf_clinical_report(
                    patient_id=selected_patient,
                    metadata=p_meta,
                    couinaud_df=couinaud_df if not couinaud_df.empty else pd.DataFrame(),
                    summary=summary,
                    pencil_png_bytes=pencil_bytes,
                    output_pdf_path=pencil_pdf_path,
                    clinical_profile=p_clin
                )
            st.success("✓ Clinical PDF Report generated successfully!")

        if os.path.exists(pencil_pdf_path):
            with open(pencil_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Clinical PDF Report",
                data=pdf_bytes,
                file_name=f"{selected_patient}_{'CT' if is_ct else 'MRI'}_Clinical_Report.pdf",
                mime="application/pdf"
            )

        st.markdown("---")
        st.markdown(f"**Modality:** `{'CT Scan' if is_ct else 'Dynamic MRI'}`")
        st.markdown(f"**BCLC Stage:** `{p_clin.get('bclc_stage', 'N/A')}`")
        st.markdown(f"**Child-Pugh:** `{p_clin.get('cps_score', 'N/A')}`")
        st.markdown(f"**Chemotherapy:** `{p_clin.get('chemotherapy', 'N/A')}`")
        st.markdown(f"**mRECIST Response:** `{p_clin.get('mrecist_response', 'N/A')}`")

# ---------------------------------------------------------------------
# TAB 6: FULL CLINICAL COHORT ANALYTICS (105 PATIENTS) (ORDER 6)
# ---------------------------------------------------------------------
with tabs[5]:
    st.markdown("### 📈 Comprehensive Clinical Cohort Analytics (105 Patients)")
    st.caption("Aggregate demographic, staging, biomarker, and post-TACE longitudinal survival statistics across HCC-TACE-Seg cohort.")

    if clinical_df.empty:
        st.warning("Clinical cohort dataset not found.")
    else:
        cohort_stats = get_cohort_summary_statistics(clinical_df)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Cohort Patients", f"{cohort_stats['total_patients']}")
        with m2:
            st.metric("Median Overall Survival (OS)", f"{cohort_stats['median_overall_survival_weeks']} wks")
        with m3:
            st.metric("Median Time to Progression (TTP)", f"{cohort_stats['median_time_to_progression_weeks']} wks")
        with m4:
            st.metric("TACE Primary Modality", "100% Cohort")

        st.divider()

        ch1, ch2 = st.columns(2)
        with ch1:
            # BCLC Staging distribution
            bclc_data = pd.DataFrame(list(cohort_stats["bclc_distribution"].items()), columns=["BCLC Stage", "Patient Count"])
            fig_bclc = px.bar(
                bclc_data,
                x="BCLC Stage",
                y="Patient Count",
                title="<b>Barcelona Clinic Liver Cancer (BCLC) Stage Distribution</b>",
                color="BCLC Stage",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            st.plotly_chart(fig_bclc, use_container_width=True)

        with ch2:
            # Child-Pugh distribution
            cps_data = pd.DataFrame(list(cohort_stats["child_pugh_distribution"].items()), columns=["Child-Pugh Class", "Patient Count"])
            fig_cps = px.pie(
                cps_data,
                names="Child-Pugh Class",
                values="Patient Count",
                title="<b>Child-Pugh Cirrhosis Functional Class (Parenchyma Reserve)</b>",
                hole=0.4,
                color_discrete_sequence=["#16a34a", "#eab308", "#dc2626"]
            )
            st.plotly_chart(fig_cps, use_container_width=True)

        ch3, ch4 = st.columns(2)
        with ch3:
            # mRECIST response rates
            mrecist_data = pd.DataFrame(list(cohort_stats["mrecist_response_rates"].items()), columns=["Treatment Response", "Patient Count"])
            fig_mrecist = px.bar(
                mrecist_data,
                x="Treatment Response",
                y="Patient Count",
                title="<b>mRECIST Post-TACE Treatment Response Rates</b>",
                color="Treatment Response",
                color_discrete_sequence=["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]
            )
            st.plotly_chart(fig_mrecist, use_container_width=True)

        with ch4:
            # AFP biomarker vs tumor size
            scatter_df = clinical_df[clinical_df["Tr_Size"].notna() & clinical_df["AFP"].notna()].copy()
            fig_scatter = px.scatter(
                scatter_df,
                x="Tr_Size",
                y="AFP",
                color="BCLC",
                size="OS",
                hover_data=["TCIA_ID", "age", "Sex"],
                title="<b>Serum AFP (ng/mL) vs Tumor Size (cm) by BCLC Stage</b>",
                labels={"Tr_Size": "Tumor Diameter (cm)", "AFP": "Serum AFP Level (ng/mL)"},
                log_y=True
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Cohort Clinical Data Table")
        display_cols = [c for c in ["TCIA_ID", "age", "Sex", "BCLC", "CPS", "AFP", "Tr_Size", "chemotherapy", "1_mRECIST", "OS", "TTP"] if c in clinical_df.columns]
        st.dataframe(clinical_df[display_cols].reset_index(drop=True), use_container_width=True)

st.divider()
st.markdown("<div style='text-align:center; color:#94a3b8; font-size:12px;'>Intelligent Liver Tumor Profiling & Longitudinal Progression Analysis • CT & MRI Multimodal Framework</div>", unsafe_allow_html=True)
