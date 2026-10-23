import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy.ndimage import zoom

def create_3d_liver_tumor_figure(liver_mask, tumor_mask, spacing_mm, title="3D Liver & Tumor Volumetric Reconstruction"):
    """
    Creates an interactive 3D Plotly Mesh3D visualization of the Liver and Tumor.
    - Liver is rendered as a semi-transparent organ capsule (opacity ~0.22).
    - Tumor is rendered as a vibrant solid crimson mass nested inside the liver.
    """
    # Downsample slightly for ultra-fast 60FPS browser rendering if volume is large
    target_shape = (min(liver_mask.shape[0], 64), 64, 64)
    zoom_factors = [t / s for t, s in zip(target_shape, liver_mask.shape)]
    
    liv_down = zoom(liver_mask.astype(np.float32), zoom_factors, order=1) > 0.35
    tum_down = zoom(tumor_mask.astype(np.float32), zoom_factors, order=1) > 0.35
    
    # Scale physical spacing to match downsampling
    scaled_spacing = (
        spacing_mm[2] / zoom_factors[0],
        spacing_mm[1] / zoom_factors[1],
        spacing_mm[0] / zoom_factors[2]
    )
    
    fig = go.Figure()
    
    # 1. Liver 3D Mesh
    has_liver = np.sum(liv_down) > 20
    if has_liver:
        try:
            verts_liv, faces_liv, _, _ = measure.marching_cubes(liv_down, level=0.5, spacing=scaled_spacing)
            # Marching cubes coordinates: (z, y, x) -> map to (x, y, z) in mm
            fig.add_trace(go.Mesh3d(
                x=verts_liv[:, 2],
                y=verts_liv[:, 1],
                z=verts_liv[:, 0],
                i=faces_liv[:, 2],
                j=faces_liv[:, 1],
                k=faces_liv[:, 0],
                color='#8b5cf6', # Anatomical purple/slate translucent liver capsule
                opacity=0.22,
                name='Healthy Liver Capsule',
                showlegend=True,
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.4),
                lightposition=dict(x=100, y=200, z=500)
            ))
        except Exception as e:
            print(f"Marching cubes error for liver: {e}")

    # 2. Tumor 3D Mesh
    has_tumor = np.sum(tum_down) > 5
    if has_tumor:
        try:
            verts_tum, faces_tum, _, _ = measure.marching_cubes(tum_down, level=0.5, spacing=scaled_spacing)
            fig.add_trace(go.Mesh3d(
                x=verts_tum[:, 2],
                y=verts_tum[:, 1],
                z=verts_tum[:, 0],
                i=faces_tum[:, 2],
                j=faces_tum[:, 1],
                k=faces_tum[:, 0],
                color='#ef4444', # Solid Vibrant Crimson for Tumor
                opacity=0.95,
                name='HCC Tumor Focus',
                showlegend=True,
                lighting=dict(ambient=0.7, diffuse=0.9, specular=0.6, roughness=0.2),
                lightposition=dict(x=100, y=200, z=500)
            ))
        except Exception as e:
            print(f"Marching cubes error for tumor: {e}")

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:#94a3b8;'>Interactive 3D Mesh — Translucent Liver with Embedded HCC Tumor (Scale: mm)</span>",
            font=dict(color="#f8fafc", size=15)
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        scene=dict(
            xaxis=dict(title='X (mm) [Right - Left]', backgroundcolor="#1e293b", gridcolor="#334155", showbackground=True, color="#94a3b8"),
            yaxis=dict(title='Y (mm) [Ant - Post]', backgroundcolor="#1e293b", gridcolor="#334155", showbackground=True, color="#94a3b8"),
            zaxis=dict(title='Z (mm) [Inf - Sup]', backgroundcolor="#1e293b", gridcolor="#334155", showbackground=True, color="#94a3b8"),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        legend=dict(
            x=0.02, y=0.95,
            font=dict(color="#f8fafc", size=11),
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig
