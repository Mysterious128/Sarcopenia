"""
L3 Body Composition — Sarcopenia screening from abdominal CT.

    pip install streamlit torch numpy matplotlib pydicom nibabel dicom2nifti monai
    streamlit run app.py

Folder layout:
    app.py
    models/            attunet_best.pt, unet_best.pt, swinunetr_best.pt
    assets/            hero_l3_ct.jpg, sarcopenia_case_study.jpg, demo.mp4 (optional)
"""

import io
import os
import zipfile
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
ASSET_DIR = HERE / "assets"

# Sex-specific L3 skeletal muscle reference values (Derstine et al., Sci Rep, 2018)
CUTOFF = {"Male": 144.3, "Female": 92.2}
ATTEN_REF = {"Male": 38.5, "Female": 34.3}

# Measurements within this margin of the cut-off are reported as indeterminate,
# because the tool's own error is of comparable magnitude.
MARGIN = 5.0

st.set_page_config(
    page_title="SarcoScan AI: Automated L3 Sarcopenia Screening",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# Design Tokens & Dark Clinical Theme Styling
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg-dark: #090E17;
  --surface-dark: #111A28;
  --surface-card: #162438;
  --surface-hover: #1E2E46;
  --border-line: #21344D;
  --border-glow: #0E6BA8;
  
  --text-main: #F1F5F9;
  --text-sub: #94A3B8;
  --text-muted: #64748B;
  
  --blue-accent: #00B4D8;
  --blue-bright: #38BDF8;
  --blue-deep: #0E6BA8;
  --navy-dark: #0F172A;
  
  --alert-red: #EF4444;
  --warn-amber: #F59E0B;
  --ok-green: #10B981;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  background-color: var(--bg-dark) !important;
  color: var(--text-main) !important;
}

.stApp {
  background-color: var(--bg-dark);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { 
  padding: 1.5rem 1.5rem !important; 
  max-width: 100% !important; 
  position: relative !important; 
}

/* ---------- Navigation Header Bar ---------- */
.header-bar {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  padding: 0 0 1rem 0; /* Remove left/top padding to align with content, keep bottom padding */
  padding-right: 400px; /* Reserve space for the tabs on the right */
  background: transparent;
  border: none;
  border-bottom: 1px solid var(--border-line); /* Only light line on bottom */
  border-radius: 0;
  margin-bottom: 1.5rem;
  box-shadow: none;
  min-height: 72px;
}

.brand-container {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.brand-logo {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, var(--blue-accent), var(--blue-deep));
  border-radius: 50%;
  display: grid;
  place-items: center;
  box-shadow: 0 0 12px rgba(0, 180, 216, 0.5);
}

.brand-logo::after {
  content: '';
  width: 12px;
  height: 12px;
  background: var(--bg-dark);
  border-radius: 50%;
}

.brand-title {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-main);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.brand-title span {
  color: var(--blue-accent);
}

.status-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--blue-bright);
  background: rgba(14, 107, 168, 0.2);
  border: 1px solid rgba(56, 189, 248, 0.3);
  padding: 0.35rem 0.75rem;
  border-radius: 4px;
  margin-left: 1rem;
}

/* ---------- Tabs Styling ---------- */
div[data-testid="stTabs"] {
  margin-top: -5.5rem;
  position: relative;
  z-index: 100;
}

div[data-testid="stTabs"] > div:first-child {
  background: transparent;
  border: none;
  padding-right: 2rem;
  width: 100%;
}

div[data-baseweb="tab-list"] {
  justify-content: flex-end; /* Align tabs to the right */
  gap: 0.5rem;
  background-color: transparent;
  padding: 0;
  border: none;
}

div[data-baseweb="tab-panel"] {
  margin-top: 2rem; /* Push panels down below the header */
}

div[data-baseweb="tab"] {
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 600;
  font-size: 1.3rem;
  color: var(--text-main) !important;
  background-color: transparent !important;
  border-radius: 6px;
  padding: 0.55rem 1.2rem !important;
  border: none !important;
  transition: all 0.2s ease;
}

div[data-baseweb="tab"]:hover {
  color: var(--text-main) !important;
  background-color: var(--surface-card) !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
  color: #FFFFFF !important;
  background: linear-gradient(135deg, var(--blue-deep), #063959) !important;
  box-shadow: 0 2px 10px rgba(14, 107, 168, 0.4);
}

/* ---------- Cards & Panels ---------- */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  display: flex;
  flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
  flex: 1;
  display: flex;
  flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .clinical-card {
  flex: 1;
  height: 100%;
}

.clinical-card {
  background: var(--surface-dark);
  border: 1px solid var(--border-line);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  transition: border-color 0.2s ease;
}

.clinical-card:hover {
  border-color: var(--border-glow);
}

.hero-banner {
  background: transparent;
  border: none;
  padding: 1rem 0;
  margin-bottom: 2.5rem;
  position: relative;
}

.hero-headline {
  font-size: 2.8rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: #FFFFFF;
  margin-bottom: 1.2rem;
}

.hero-subtitle {
  font-size: 1.25rem;
  color: var(--text-sub);
  max-width: 68ch;
  line-height: 1.6;
}

/* ---------- Readouts (IBM Plex Mono) ---------- */
.mono-readout {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--blue-bright);
  letter-spacing: -0.03em;
}

.mono-unit {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.9rem;
  color: var(--text-muted);
  font-weight: 400;
  margin-left: 0.2rem;
}

.mono-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-sub);
  margin-bottom: 0.35rem;
}

/* ---------- Verdict Badge ---------- */
.verdict-box {
  padding: 1.1rem 1.4rem;
  border-radius: 6px;
  border-left: 4px solid;
  background: var(--surface-card);
  margin-bottom: 1.25rem;
}

.verdict-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-sub);
}

.verdict-val {
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-top: 0.25rem;
}

.v-low { border-left-color: var(--alert-red); }
.v-low .verdict-val { color: var(--alert-red); }

.v-ind { border-left-color: var(--warn-amber); }
.v-ind .verdict-val { color: var(--warn-amber); }

.v-ok  { border-left-color: var(--ok-green); }
.v-ok  .verdict-val { color: var(--ok-green); }

/* ---------- Step Numbers ---------- */
.step-hdr {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  margin: 1.8rem 0 1rem;
}

.step-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  font-weight: 700;
  color: #FFFFFF;
  background: var(--blue-deep);
  width: 1.75rem;
  height: 1.75rem;
  display: grid;
  place-items: center;
  border-radius: 4px;
  flex: none;
}

.step-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-main);
  margin: 0;
}

.step-hint {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-left: auto;
}

/* ---------- Scale HTML Bar ---------- */
.scale-wrap { margin: 1.5rem 0 0.5rem; }
.scale {
  position: relative;
  height: 44px;
  background: #152233;
  border: 1px solid var(--border-line);
  border-radius: 4px;
  overflow: hidden;
}
.scale .band {
  position: absolute; top: 0; bottom: 0;
  background: rgba(245, 158, 11, 0.22);
  border-left: 1px dashed rgba(245, 158, 11, 0.6);
  border-right: 1px dashed rgba(245, 158, 11, 0.6);
}
.scale .cut { position: absolute; top: 0; bottom: 0; width: 2px; background: #FFFFFF; }
.scale .pin {
  position: absolute; top: -4px; bottom: -4px; width: 4px; background: var(--blue-bright);
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
}
.scale-ax {
  display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem; color: var(--text-sub); margin-top: 0.4rem;
}
.scale-key {
  font-size: 0.78rem; color: var(--text-sub); margin-top: 0.6rem; display: flex; gap: 1.5rem; flex-wrap: wrap;
}
.scale-key i { display: inline-block; width: 10px; height: 10px; margin-right: 0.4rem; border-radius: 2px; }

/* Streamlit Button & Widget Overrides */
div.stButton > button {
  background: linear-gradient(135deg, var(--blue-deep), #063959);
  color: #FFFFFF;
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 4px;
  font-weight: 600;
  padding: 0.6rem 1.5rem;
  transition: all 0.2s ease;
}
div.stButton > button:hover {
  background: linear-gradient(135deg, var(--blue-accent), var(--blue-deep));
  border-color: var(--blue-bright);
  box-shadow: 0 0 12px rgba(0, 180, 216, 0.4);
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface-dark);
  border: 1px dashed var(--border-line);
  border-radius: 6px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def step_hdr(n, title, hint=""):
    st.markdown(
        f"<div class='step-hdr'><span class='step-num'>{n}</span>"
        f"<h3 class='step-title'>{title}</h3><span class='step-hint'>{hint}</span></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# PyTorch Architectures
# ─────────────────────────────────────────────────────────────
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.d1, self.d2, self.d3 = block(1, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2])
        self.bott = block(ch[2], ch[3])
        self.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); self.c3 = block(ch[3], ch[2])
        self.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); self.c2 = block(ch[2], ch[1])
        self.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); self.c1 = block(ch[1], ch[0])
        self.out = nn.Conv2d(ch[0], 1, 1); self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.pool(d1)); d3 = self.d3(self.pool(d2))
        b = self.bott(self.pool(d3))
        x = self.c3(torch.cat([self.u3(b), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)


class AttentionGate(nn.Module):
    def __init__(self, g, xc, inter):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(g, inter, 1), nn.BatchNorm2d(inter))
        self.wx = nn.Sequential(nn.Conv2d(xc, inter, 1), nn.BatchNorm2d(inter))
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.wg(g) + self.wx(x)))


class AttUNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.d1, self.d2, self.d3 = block(1, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2])
        self.bott = block(ch[2], ch[3])
        self.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.a3 = AttentionGate(ch[2], ch[2], ch[2] // 2); self.c3 = block(ch[3], ch[2])
        self.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.a2 = AttentionGate(ch[1], ch[1], ch[1] // 2); self.c2 = block(ch[2], ch[1])
        self.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.a1 = AttentionGate(ch[0], ch[0], ch[0] // 2); self.c1 = block(ch[1], ch[0])
        self.out = nn.Conv2d(ch[0], 1, 1); self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.pool(d1)); d3 = self.d3(self.pool(d2))
        b = self.bott(self.pool(d3))
        g3 = self.u3(b); x = self.c3(torch.cat([g3, self.a3(g3, d3)], 1))
        g2 = self.u2(x); x = self.c2(torch.cat([g2, self.a2(g2, d2)], 1))
        g1 = self.u1(x); x = self.c1(torch.cat([g1, self.a1(g1, d1)], 1))
        return self.out(x)


@st.cache_resource(show_spinner=False)
def load_models():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    specs = [
        ("Attention U-Net", lambda: AttUNet(), "attunet_best.pt"),
        ("U-Net",           lambda: UNet(),    "unet_best.pt"),
        ("SwinUNETR",       lambda: SwinUNETR(in_channels=1, out_channels=1,
                                              feature_size=48, spatial_dims=2),
                            "swinunetr_best.pt"),
    ]
    found = []
    for name, build, fname in specs:
        p = MODEL_DIR / fname
        if p.exists():
            net = build()
            net.load_state_dict(torch.load(p, map_location=dev))
            found.append((name, net.to(dev).eval()))
    return found, dev


# ─────────────────────────────────────────────────────────────
# Medical Imaging Helpers
# ─────────────────────────────────────────────────────────────
def window(hu, level=40, width=400):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def segment(hu, models, dev, thr):
    x = torch.from_numpy(window(hu).astype(np.float32))[None, None].to(dev)
    x = torch.nn.functional.interpolate(x, (256, 256), mode="bilinear", align_corners=False)
    with torch.no_grad():
        p = torch.stack([torch.sigmoid(n(x)) for _, n in models]).mean(0)
        p = torch.nn.functional.interpolate(p, hu.shape, mode="bilinear", align_corners=False)
    return p.cpu().numpy()[0, 0] > thr


def rotate_array(arr, deg):
    k = -(deg // 90) % 4
    return np.rot90(arr, k) if k != 0 else arr


def create_pulsing_muscle_gif(hu, mask, fps=12, rotation=0):
    """Generates an animated GIF with sinusoidal red pulsing over the L3 CT slice."""
    import imageio
    from scipy.ndimage import binary_dilation

    hu_rot = rotate_array(hu, rotation)
    mask_rot = rotate_array(mask, rotation)

    disp = window(hu_rot)  # 0..1 float
    base_u8 = (disp * 255).astype(np.uint8)
    base_rgb = np.stack([base_u8] * 3, axis=-1)

    # Muscle boundary for glowing edge contour
    edge = binary_dilation(mask_rot, iterations=2) & ~mask_rot

    frames = []
    # 14 frames smooth breathing pulsation
    alphas = [0.30 + 0.52 * (0.5 + 0.5 * np.sin(2 * np.pi * i / 14)) for i in range(14)]

    for a in alphas:
        fr = base_rgb.copy()
        # Vivid red fill [240, 25, 40]
        fr[mask_rot] = (fr[mask_rot] * (1.0 - a) + np.array([240, 25, 40]) * a).astype(np.uint8)
        # Bright neon red edge contour [255, 75, 85]
        edge_a = min(1.0, a * 1.35)
        fr[edge] = (fr[edge] * (1.0 - edge_a) + np.array([255, 75, 85]) * edge_a).astype(np.uint8)
        frames.append(fr)

    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=1.0 / fps, loop=0)
    return buf.getvalue()


def create_progressive_muscle_gif(hu, mask, fps=14, rotation=0):
    return create_anatomical_progressive_gif(hu, mask, fps, rotation)


def create_anatomical_progressive_gif(hu, mask, fps=12, rotation=270):
    """
    Generates an animated GIF with anatomical region-based progressive mask rendering:
    1. Starts with two posterior paraspinal/psoas muscles
    2. Wraps smoothly around the outer abdominal wall
    3. Holds full delineated muscle mask
    """
    import imageio
    from scipy.ndimage import binary_dilation, label, distance_transform_edt

    hu_rot = rotate_array(hu, rotation)
    mask_rot = rotate_array(mask, rotation)

    disp = window(hu_rot)  # 0..1 float
    base_u8 = (disp * 255).astype(np.uint8)
    base_rgb = np.stack([base_u8] * 3, axis=-1)

    if not mask_rot.any():
        buf = io.BytesIO()
        imageio.mimsave(buf, [base_rgb], format="GIF", duration=1.0)
        return buf.getvalue()

    edge_outer = binary_dilation(mask_rot, iterations=2) & ~mask_rot

    # Region-based anatomical priority scoring
    y_idx, x_idx = np.where(mask_rot)
    y_min, y_max = y_idx.min(), y_idx.max()
    y_span = max(1.0, float(y_max - y_min))

    labeled_mask, num_features = label(mask_rot)
    priority_map = np.zeros_like(mask_rot, dtype=np.float32)

    if num_features >= 2:
        comp_info = []
        for c in range(1, num_features + 1):
            cy = np.where(labeled_mask == c)[0].mean()
            comp_info.append((cy, c))
        comp_info.sort(key=lambda item: item[0])

        mid = max(1, len(comp_info) // 2)
        post_comps = {c for _, c in comp_info[:mid]}

        for c in range(1, num_features + 1):
            c_mask = (labeled_mask == c)
            d_c = distance_transform_edt(c_mask)
            max_d_c = max(1.0, d_c.max())
            if c in post_comps:
                priority_map[c_mask] = 0.45 * (d_c[c_mask] / max_d_c)
            else:
                priority_map[c_mask] = 0.45 + 0.55 * (d_c[c_mask] / max_d_c)
    else:
        dist = distance_transform_edt(mask_rot)
        max_d = max(1.0, dist.max())
        norm_y = (y_idx - y_min) / y_span
        priority_map[mask_rot] = 0.5 * norm_y + 0.5 * (dist[mask_rot] / max_d)
        if priority_map[mask_rot].max() > 0:
            priority_map[mask_rot] /= priority_map[mask_rot].max()

    frames = []
    num_fill_frames = 18
    hold_frames = 6

    for f in range(num_fill_frames):
        fr = base_rgb.copy()
        thresh = (f + 1) / num_fill_frames

        active_fill = mask_rot & (priority_map <= thresh)
        leading_edge = mask_rot & (np.abs(priority_map - thresh) <= 0.08)

        # 1. Base outer border guide
        fr[edge_outer] = (fr[edge_outer] * 0.35 + np.array([255, 60, 60]) * 0.65).astype(np.uint8)

        # 2. Filled muscle region so far (translucent red)
        alpha_fill = 0.58
        fr[active_fill] = (fr[active_fill] * (1.0 - alpha_fill) + np.array([239, 44, 44]) * alpha_fill).astype(np.uint8)

        # 3. Leading laser scanline tracer
        alpha_lead = 0.95
        fr[leading_edge] = (fr[leading_edge] * (1.0 - alpha_lead) + np.array([255, 160, 160]) * alpha_lead).astype(np.uint8)

        frames.append(fr)

    # Phase 3: Full illuminated mask hold
    for h in range(hold_frames):
        fr = base_rgb.copy()
        pulse = 0.55 + 0.12 * np.sin(np.pi * h / hold_frames)
        fr[mask_rot] = (fr[mask_rot] * (1.0 - pulse) + np.array([239, 44, 44]) * pulse).astype(np.uint8)
        fr[edge_outer] = (fr[edge_outer] * 0.2 + np.array([255, 80, 80]) * 0.8).astype(np.uint8)
        frames.append(fr)

    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=1.0 / fps, loop=0)
    return buf.getvalue()


def read_dcm(b):
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(b), force=True)
    hu = ds.pixel_array.astype(np.float32)
    hu = hu * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    px = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else None
    return hu, px


def check_slice_plausible(ds, hu):
    problems = []
    o = getattr(ds, "ImageOrientationPatient", None)
    if o and not (abs(float(o[2])) < 0.1 and abs(float(o[5])) < 0.1):
        problems.append("This is not an axial slice. L3 measurement requires an axial cross-section.")

    desc = f"{getattr(ds,'SeriesDescription','')} {getattr(ds,'BodyPartExamined','')}".lower()
    if any(k in desc for k in ("chest", "thorax", "lung", "mediast")) and \
       not any(k in desc for k in ("abdo", "pelvis")):
        problems.append(
            f"This slice is labelled '{getattr(ds,'SeriesDescription','?').strip()}' "
            "(chest). L3 lies lower in the lumbar region."
        )

    lung_frac = float((hu < -400).sum()) / hu.size
    body_frac = float((hu > -200).sum()) / hu.size
    if body_frac > 0 and lung_frac / max(body_frac, 1e-6) > 0.25:
        problems.append("Large air-filled regions detected, consistent with chest/lung slice.")

    return problems


def run_full_study(zip_bytes, say):
    import nibabel as nib
    import pydicom
    from collections import defaultdict

    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw"; raw.mkdir()
    dcm = tmp / "dcm"; dcm.mkdir()

    say("Unpacking study…")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(raw)

    say("Selecting best axial CT series…")
    SKIP = ('.txt', '.csv', '.gif', '.jse', '.cm', '.exe', '.dll', '.ini',
            '.js', '.htm', '.html', '.bat', '.inf', '.ico', '.bmp', '.jpg', '.png', '.pdf')
    series = defaultdict(list)
    for p in raw.rglob("*"):
        if not p.is_file() or p.suffix.lower() in SKIP:
            continue
        if any(s in str(p).lower() for s in ("dcmvwr", "colormap", "misc")):
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True,
                                 specific_tags=["Modality", "SeriesInstanceUID",
                                                "SeriesDescription", "SliceThickness",
                                                "ImageOrientationPatient"])
            if getattr(ds, "Modality", "") != "CT":
                continue
            series[str(getattr(ds, "SeriesInstanceUID", "?"))].append((p, ds))
        except Exception:
            continue

    if not series:
        raise ValueError("No CT series found in archive.")

    def is_axial(ds):
        o = getattr(ds, "ImageOrientationPatient", None)
        return bool(o) and abs(float(o[2])) < 0.1 and abs(float(o[5])) < 0.1

    def score(ds, n):
        d = str(getattr(ds, "SeriesDescription", "")).lower()
        s = 5 if is_axial(ds) else -20
        if any(k in d for k in ("abdo", "abdomen", "pelvis")): s += 4
        if any(k in d for k in ("b30", "b20", "b40", "soft", "routine")): s += 3
        if any(k in d for k in ("b70", "bone", "lung", "topogram", "protocol",
                                "mediast", "coronal", "sag")): s -= 6
        try:
            t = float(getattr(ds, "SliceThickness", 99))
            s += 2 if 1.0 <= t <= 3.0 else (1 if t <= 5.0 else 0)
        except Exception:
            pass
        return s + min(n, 400) / 200

    best, best_sc, best_ds = None, -1e9, None
    for uid, items in series.items():
        sc = score(items[0][1], len(items))
        if sc > best_sc:
            best_sc, best, best_ds = sc, items, items[0][1]

    if not is_axial(best_ds):
        raise ValueError("No axial series found in this CT study.")
    if len(best) < 20:
        raise ValueError(f"Series has only {len(best)} slices — too few for L3 localization.")

    say(f"Selected: {getattr(best_ds,'SeriesDescription','?')} ({len(best)} slices)")
    for i, (p, _) in enumerate(best):
        (dcm / f"{i:05d}.dcm").write_bytes(p.read_bytes())

    say("Locating L3 vertebra via TotalSegmentator…")
    seg = tmp / "seg"
    subprocess.run(["TotalSegmentator", "-i", str(dcm), "-o", str(seg), "--fast",
                    "--nr_thr_saving", "1", "--roi_subset", "vertebrae_L3"],
                   capture_output=True, text=True, check=True)
    prof = nib.load(str(seg / "vertebrae_L3.nii.gz")).get_fdata().sum(axis=(0, 1))
    if prof.sum() == 0:
        raise ValueError("No L3 vertebra detected in CT volume.")
    l3 = int(round(np.average(np.arange(len(prof)), weights=prof)))

    say("Extracting L3 cross-sectional slice…")
    import dicom2nifti
    nii = tmp / "nii"; nii.mkdir()
    dicom2nifti.convert_directory(str(dcm), str(nii), compression=True, reorient=True)
    ct = nib.load(sorted(nii.glob("*.nii.gz"))[0])

    if ct.shape[2] != len(best):
        raise ValueError(f"Slice mismatch: volume has {ct.shape[2]} slices vs {len(best)} DICOMs.")

    return (np.asarray(ct.dataobj[:, :, l3], dtype=np.float32),
            float(ct.header.get_zooms()[0]), l3, ct.shape[2])


def scale_html(area, cut, lo=40, hi=230):
    pct = lambda v: max(0, min(100, (v - lo) / (hi - lo) * 100))
    return f"""
<div class='scale-wrap'>
  <div class='scale'>
    <div class='band' style='left:{pct(cut-MARGIN)}%; width:{pct(cut+MARGIN)-pct(cut-MARGIN)}%'></div>
    <div class='cut' style='left:{pct(cut)}%'></div>
    <div class='pin' style='left:{pct(area)}%'></div>
  </div>
  <div class='scale-ax'><span>{lo} cm²</span><span>Reference Cutoff: {cut} cm²</span><span>{hi} cm²</span></div>
  <div class='scale-key'>
    <span><i style='background:var(--blue-bright)'></i>Patient Measurement: <strong>{area:.1f} cm²</strong></span>
    <span><i style='background:rgba(245,158,11,0.6)'></i>Indeterminate Zone (±{MARGIN:.0f} cm²)</span>
    <span><i style='background:#FFFFFF'></i>Sex-Specific Cutoff</span>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────
# Main Application Initialization
# ─────────────────────────────────────────────────────────────
models, dev = load_models()

# Top Header Bar
st.markdown(f"""
<div class='header-bar'>
  <div style='display:flex; flex-direction:column; gap:0.4rem;'>
    <div class='brand-container'>
      <div class='brand-logo'></div>
      <div class='brand-title'>Sarcopenia Analytics</div>
    </div>
    <div style='display:flex; align-items:center; gap:0.5rem; margin-left:2.85rem;'>
      <div class='status-badge' style='margin:0; padding:0.2rem 0.5rem; font-size:0.65rem;'>Deep Ensemble · {dev.upper()}</div>
      <div class='status-badge' style='margin:0; padding:0.2rem 0.5rem; font-size:0.65rem; background:rgba(16,185,129,0.15); color:#10B981; border-color:rgba(16,185,129,0.3);'>Clinical Research</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not models:
    st.error(f"No trained model weights found in `{MODEL_DIR}`. Please add `attunet_best.pt` or `unet_best.pt`.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Navigation Tabs
# ─────────────────────────────────────────────────────────────
tab_home, tab_about, tab_demo, tab_scan = st.tabs([
    "🏠 Home",
    "ℹ️ About Us",
    "🎥 Demo",
    "🔬 Scan / Upload"
])

# =============================================================
# TAB 1: HOME PAGE & CASE STUDY
# =============================================================
with tab_home:
    col_hero_left, col_hero_right = st.columns([1, 1.25])
    
    with col_hero_left:
        st.markdown("""
        <div class='hero-banner'>
            <div class='hero-headline'>AUTOMATED L3<br><span style='color:var(--blue-bright);'>SARCOPENIA ANALYTICS</span></div>
            <div class='hero-subtitle'>
                AI imaging software for precise L3 body composition analytics. Clinically validated, 
                automated sarcopenia detection & muscle attenuation profiling from routine abdominal CT scans.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metric Highlight Pills
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("""
            <div class='clinical-card' style='text-align:center; height: 110px; display: flex; flex-direction: column; justify-content: center; padding: 1rem;'>
                <div class='mono-label'>Landmark</div>
                <div class='mono-readout'>L3 <span class='mono-unit'>Vertebra</span></div>
                <div style='font-size:0.78rem; color:var(--text-muted); margin-top:0.2rem;'>Gold standard slice</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class='clinical-card' style='text-align:center; height: 110px; display: flex; flex-direction: column; justify-content: center; padding: 1rem;'>
                <div class='mono-label'>HU Range</div>
                <div class='mono-readout'>-29 <span class='mono-unit'>to</span> +150</div>
                <div style='font-size:0.78rem; color:var(--text-muted); margin-top:0.2rem;'>Skeletal muscle HU</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown("""
            <div class='clinical-card' style='text-align:center; height: 110px; display: flex; flex-direction: column; justify-content: center; padding: 1rem;'>
                <div class='mono-label'>Validation</div>
                <div class='mono-readout'>141 <span class='mono-unit'>Studies</span></div>
                <div style='font-size:0.78rem; color:var(--text-muted); margin-top:0.2rem;'>NHS Trust dataset</div>
            </div>
            """, unsafe_allow_html=True)

        # New Sarcopenia Information Section
        st.markdown("""
        <div style='margin-top: 1.5rem; margin-bottom: 1rem;'>
            <h3 style='font-size: 1.5rem; font-weight: 600; color: #FFFFFF;'>Why Sarcopenia Matters</h3>
            <p style='font-size: 1.05rem; color: var(--text-sub); line-height: 1.6; max-width: 900px;'>
                Sarcopenia—the severe depletion of skeletal muscle mass and quality—is a critical, often overlooked comorbidity. 
                It is an independent predictor of chemotherapy toxicity, surgical complications, and poor overall survival in oncology. 
                Automated L3 screening transforms routine abdominal CT scans into powerful prognostic tools, enabling early nutritional and physical interventions before clinical decline.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_hero_right:
        hero_img_path = ASSET_DIR / "hero_l3_ct.jpg"
        if hero_img_path.exists():
            st.image(str(hero_img_path), width="stretch", caption="L3 Lumbar Vertebra Skeletal Muscle AI Segmentation")
        else:
            st.markdown("""
            <div class='clinical-card' style='height:300px; display:grid; place-items:center; text-align:center;'>
                <div style='color:var(--blue-bright); font-size:2rem;'>🧬</div>
                <div style='color:var(--text-sub); margin-top:0.5rem;'>L3 Vertebra Cross-Section View</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='min-height:25vh;'></div><hr style='border:0; height:1px; background:var(--border-line); margin:2.5rem 0;'>", unsafe_allow_html=True)
    
    # ── CASE STUDY SECTION ─────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; margin-bottom:2rem;'>
        <div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.78rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--blue-bright);'>CLINICAL CASE STUDY</div>
        <h2 style='font-size:1.85rem; font-weight:700; color:#FFFFFF; margin-top:0.2rem;'>L3-Level Sarcopenia Workflow & Diagnostic Rationale</h2>
        <p style='color:var(--text-sub); max-width:700px; margin:0.5rem auto 0;'>
            Sarcopenia is the progressive loss of skeletal muscle mass and strength. Measuring muscle area at the 3rd lumbar (L3) vertebra provides a clinically validated proxy for whole-body muscle mass.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    cs_img_path = ASSET_DIR / "sarcopenia_case_study.jpg"
    if cs_img_path.exists():
        st.image(str(cs_img_path), width="stretch", caption="Axial CT Body Composition Comparison at L3 Level: Normal Muscle Mass vs. Sarcopenia with Fat Infiltration")
        
    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    
    cs_col1, cs_col2, cs_col3 = st.columns(3)
    
    with cs_col1:
        st.markdown("""
        <div class='clinical-card'>
            <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.75rem;'>
                <span style='color:var(--blue-bright); font-size:1.2rem;'>📐</span>
                <h4 style='margin:0; font-size:1.05rem; font-weight:600; color:#FFFFFF;'>Anatomical Muscle Groups</h4>
            </div>
            <p style='font-size:0.88rem; color:var(--text-sub); line-height:1.6;'>
                At the 3rd lumbar (L3) vertebra, the AI ensemble automatically segments four primary skeletal muscle groups:
            </p>
            <ul style='font-size:0.85rem; color:var(--text-sub); padding-left:1.2rem; line-height:1.6; margin-bottom:0;'>
                <li><strong>Psoas Major:</strong> Bilateral anterior muscle group along the lumbar spine.</li>
                <li><strong>Paraspinal Group:</strong> Erector spinae & quadratus lumborum (posterior).</li>
                <li><strong>Rectus Abdominis:</strong> Anterior abdominal wall muscle pair.</li>
                <li><strong>Lateral Abdominal Wall:</strong> Transverse abdominis, internal & external obliques.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with cs_col2:
        st.markdown("""
        <div class='clinical-card'>
            <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.75rem;'>
                <span style='color:var(--blue-bright); font-size:1.2rem;'>📊</span>
                <h4 style='margin:0; font-size:1.05rem; font-weight:600; color:#FFFFFF;'>Radiodensity & Quality (HU)</h4>
            </div>
            <p style='font-size:0.88rem; color:var(--text-sub); line-height:1.6;'>
                Quantity alone is insufficient — muscle attenuation in Hounsfield Units (HU) measures <strong>myosteatosis</strong> (fat infiltration).
            </p>
            <ul style='font-size:0.85rem; color:var(--text-sub); padding-left:1.2rem; line-height:1.6; margin-bottom:0;'>
                <li><strong>HU Windowing Range:</strong> Standard muscle attenuation window (-29 to +150 HU).</li>
                <li><strong>Healthy Muscle Quality:</strong> High attenuation (> 38.5 HU Male, > 34.3 HU Female).</li>
                <li><strong>Myosteatosis Detection:</strong> Low HU (< 30 HU) indicates lipid droplet accumulation.</li>
                <li><strong>Functional Prognosis:</strong> Low attenuation predicts weakness even with normal volume.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with cs_col3:
        st.markdown("""
        <div class='clinical-card'>
            <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.75rem;'>
                <span style='color:var(--blue-bright); font-size:1.2rem;'>🎯</span>
                <h4 style='margin:0; font-size:1.05rem; font-weight:600; color:#FFFFFF;'>Prognostic Value</h4>
            </div>
            <p style='font-size:0.88rem; color:var(--text-sub); line-height:1.6;'>
                Automated L3 sarcopenia quantification impacts patient outcomes across clinical specialties:
            </p>
            <ul style='font-size:0.85rem; color:var(--text-sub); padding-left:1.2rem; line-height:1.6; margin-bottom:0;'>
                <li><strong>Oncology Outcomes:</strong> Predicts chemotherapy toxicity, tolerance & overall survival.</li>
                <li><strong>Surgical Risk:</strong> Identifies high-risk surgical candidates for post-op complications.</li>
                <li><strong>Geriatric Screening:</strong> Enables early opportunistic screening during routine CTs.</li>
                <li><strong>Metabolic Health:</strong> Correlates with frailty index, insulin resistance & quality.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Workflow Pipeline Cards
    st.markdown("""
    <div style='margin-top:2.5rem; margin-bottom:1.2rem;'>
        <h3 style='font-size:1.35rem; font-weight:600; color:#FFFFFF;'>Automated 3-Stage AI Processing Pipeline</h3>
    </div>
    """, unsafe_allow_html=True)
    
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class='clinical-card' style='height:100%; min-height:175px; box-sizing:border-box;'>
            <div class='mono-label' style='color:var(--blue-bright);'>STAGE 1 · QC & SERIES SELECTION</div>
            <h4 style='margin:0.4rem 0 0.6rem; color:#FFFFFF; font-size:1rem;'>Axial CT Soft-Tissue Screening</h4>
            <p style='font-size:0.85rem; color:var(--text-sub); line-height:1.5;'>
                Evaluates DICOM headers for axial slice orientation, soft-tissue reconstruction kernel (e.g. B30/B40), and slice thickness (1–3mm) to discard non-diagnostic series.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class='clinical-card' style='height:100%; min-height:175px; box-sizing:border-box;'>
            <div class='mono-label' style='color:var(--blue-bright);'>STAGE 2 · SPINE LOCALIZATION</div>
            <h4 style='margin:0.4rem 0 0.6rem; color:#FFFFFF; font-size:1rem;'>L3 Vertebral Slice Identification</h4>
            <p style='font-size:0.85rem; color:var(--text-sub); line-height:1.5;'>
                Runs 3D anatomical volume segmentation via TotalSegmentator to pinpoint the exact mid-vertebral center slice of L3 automatically across hundreds of CT slices.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class='clinical-card' style='height:100%; min-height:175px; box-sizing:border-box;'>
            <div class='mono-label' style='color:var(--blue-bright);'>STAGE 3 · ENSEMBLE AI SEGMENTATION</div>
            <h4 style='margin:0.4rem 0 0.6rem; color:#FFFFFF; font-size:1rem;'>Muscle Area & Sarcopenia Stratification</h4>
            <p style='font-size:0.85rem; color:var(--text-sub); line-height:1.5;'>
                Applies an ensemble of Attention U-Net, U-Net, and SwinUNETR models to delineate muscle boundaries, calculates total area (cm²), and compares against sex-specific cutoffs.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# TAB 2: ABOUT US
# =============================================================
with tab_about:
    ab_left, ab_right = st.columns([1.2, 1])
    
    with ab_left:
        st.markdown("""
        <div class='clinical-card'>
            <h3 style='font-size:1.4rem; font-weight:700; color:#FFFFFF; margin-bottom:0.8rem;'>Mission & Research Rationale</h3>
            <p style='font-size:0.92rem; color:var(--text-sub); line-height:1.75;'>
                <strong>SarcoScan AI</strong> was built to transform routine diagnostic abdominal CT scans into opportunistic screening tools for sarcopenia and body composition assessment. 
            </p>
            <p style='font-size:0.92rem; color:var(--text-sub); line-height:1.75;'>
                While manual contouring of L3 muscle cross-sections on PACS software takes 10–15 minutes per patient, our deep learning ensemble delivers clinically precise skeletal muscle area (SMA) and mean attenuation (HU) in under 3 seconds.
            </p>
        </div>
        
        <div class='clinical-card'>
            <h3 style='font-size:1.25rem; font-weight:600; color:#FFFFFF; margin-bottom:0.8rem;'>Clinical Dataset & Model Validation</h3>
            <p style='font-size:0.9rem; color:var(--text-sub); line-height:1.6;'>
                Our models were trained and validated on <strong>141 anonymised CT studies</strong> (423 image–mask pairs) obtained from the Northern Care Alliance NHS Foundation Trust.
            </p>
            <ul style='font-size:0.88rem; color:var(--text-sub); padding-left:1.2rem; line-height:1.7;'>
                <li><strong>Ensemble Architecture:</strong> Combines 2D Attention U-Net, standard UNet, and Vision Transformer (SwinUNETR) architectures.</li>
                <li><strong>Reference Cutoffs:</strong> Derstine et al. (Sci Rep, 2018) healthy adult normative values:
                    <ul style='margin-top:0.3rem;'>
                        <li>Male L3 Cutoff: <strong style='font-family:"IBM Plex Mono"; color:var(--blue-bright);'>144.3 cm²</strong> (Mean HU: 38.5)</li>
                        <li>Female L3 Cutoff: <strong style='font-family:"IBM Plex Mono"; color:var(--blue-bright);'>92.2 cm²</strong> (Mean HU: 34.3)</li>
                    </ul>
                </li>
            </ul>
        </div>

        <div class='clinical-card' style='margin-bottom:0;'>
            <h3 style='font-size:1.2rem; font-weight:600; color:#FFFFFF; margin-bottom:0.8rem;'>Cohort Demographics & Acquisition Robustness</h3>
            <p style='font-size:0.88rem; color:var(--text-sub); line-height:1.6; margin-bottom:0.8rem;'>
                Evaluated under rigorous 5-fold cross-validation across 141 successfully processed patient studies (88.1% automated pipeline completion rate across routine clinical CTs).
            </p>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-bottom:0.9rem;'>
                <div style='background:rgba(255,255,255,0.03); border:1px solid var(--border-line); border-radius:4px; padding:0.6rem;'>
                    <div class='mono-label' style='color:var(--blue-bright);'>MALE COHORT (n=68)</div>
                    <div style='font-size:0.88rem; color:#FFFFFF; font-weight:600;'>Mean SMA: 137.5 cm²</div>
                    <div style='font-size:0.74rem; color:var(--text-muted);'>SD ±22.0 · Range 86.7–190.6</div>
                </div>
                <div style='background:rgba(255,255,255,0.03); border:1px solid var(--border-line); border-radius:4px; padding:0.6rem;'>
                    <div class='mono-label' style='color:#10B981;'>FEMALE COHORT (n=60)</div>
                    <div style='font-size:0.88rem; color:#FFFFFF; font-weight:600;'>Mean SMA: 97.8 cm²</div>
                    <div style='font-size:0.74rem; color:var(--text-muted);'>SD ±14.2 · Range 71.5–131.4</div>
                </div>
            </div>
            <div class='mono-label' style='color:var(--blue-bright); margin-bottom:0.4rem;'>PROTOCOL INVARIANCE & STABILITY (CHAPTER 5)</div>
            <ul style='font-size:0.82rem; color:var(--text-sub); padding-left:1.2rem; line-height:1.65; margin:0;'>
                <li><strong>Contrast Phase Shift:</strong> Contrast-enhanced CTs (n=113) produced mean attenuation of 31.3 HU vs 22.1 HU in unenhanced CTs (n=28) — confirming a significant 9.2 HU shift that requires phase-aware diagnostic interpretation.</li>
                <li><strong>Slice Thickness Tolerance:</strong> Routine 2.0–3.0 mm slices achieved Dice 0.949 (MAE 2.56 cm²), demonstrating reliable clinical segmentation without requiring specialized thin-slice (&lt; 1.5 mm) protocols.</li>
                <li><strong>In-Plane Resolution:</strong> Consistently robust across fine (&lt; 0.75 mm, Dice 0.944), medium (0.75–0.85 mm, Dice 0.945), and coarse (&gt; 0.85 mm, Dice 0.938) pixel spacings.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with ab_right:
        st.markdown("""
<div style='background:var(--surface-dark); border:1px solid var(--border-line); border-left:3px solid #00B4D8; border-radius:6px; padding:1rem 1rem 1rem 1.2rem; margin-bottom:0.6rem;'>
    <h3 style='font-size:1.15rem; font-weight:600; color:#FFFFFF; margin:0 0 1rem 0;'>Model Architecture Specifications</h3>
    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
        <div>
            <div class='mono-label' style='color:var(--blue-bright);'>MODEL 01 · PYTORCH</div>
            <div style='font-size:1rem; font-weight:700; color:#FFFFFF; margin-top:0.2rem;'>Attention U-Net</div>
        </div>
        <div style='text-align:right;'>
            <div style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; font-weight:700; color:var(--blue-bright);'>1.95M</div>
            <div style='font-size:0.72rem; color:var(--text-muted);'>parameters</div>
        </div>
    </div>
    <p style='font-size:0.82rem; color:var(--text-sub); margin-top:0.55rem; line-height:1.5;'>
        Encoder–decoder with soft <strong>attention gates</strong> at skip connections. Suppresses irrelevant visceral organs while focusing on muscle borders.
    </p>
    <div style='display:grid; grid-template-columns: repeat(3, 1fr); gap:0.4rem; margin:0.5rem 0; background:rgba(0,180,216,0.06); padding:0.45rem 0.6rem; border-radius:4px; border:1px solid rgba(0,180,216,0.15);'>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>5-Fold Dice:</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>0.9382</strong></div>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>Area MAE:</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>3.20 cm²</strong></div>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>HD95 (mm):</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>2.67 mm</strong></div>
    </div>
    <div style='display:flex; flex-wrap:wrap; gap:0.4rem;'>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(0,180,216,0.12); color:var(--blue-bright); border:1px solid rgba(0,180,216,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>Conv2D ×12</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(0,180,216,0.12); color:var(--blue-bright); border:1px solid rgba(0,180,216,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>AttentionGate ×3</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(0,180,216,0.12); color:var(--blue-bright); border:1px solid rgba(0,180,216,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>Bias: -1.64 cm²</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(0,180,216,0.12); color:var(--blue-bright); border:1px solid rgba(0,180,216,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>7.5 MB</span>
    </div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div style='background:var(--surface-dark); border:1px solid var(--border-line); border-left:3px solid #10B981; border-radius:6px; padding:1rem 1rem 1rem 1.2rem; margin-bottom:0.6rem;'>
    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
        <div>
            <div class='mono-label' style='color:#10B981;'>MODEL 02 · PYTORCH</div>
            <div style='font-size:1rem; font-weight:700; color:#FFFFFF; margin-top:0.2rem;'>Standard U-Net</div>
        </div>
        <div style='text-align:right;'>
            <div style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; font-weight:700; color:#10B981;'>1.93M</div>
            <div style='font-size:0.72rem; color:var(--text-muted);'>parameters</div>
        </div>
    </div>
    <p style='font-size:0.82rem; color:var(--text-sub); margin-top:0.55rem; line-height:1.5;'>
        Classic encoder–decoder with <strong>dense skip connections</strong>. Delivers high-frequency spatial detail, but vulnerable to isolated severe boundary delineation errors.
    </p>
    <div style='display:grid; grid-template-columns: repeat(3, 1fr); gap:0.4rem; margin:0.5rem 0; background:rgba(16,185,129,0.06); padding:0.45rem 0.6rem; border-radius:4px; border:1px solid rgba(16,185,129,0.15);'>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>5-Fold Dice:</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>0.9388</strong></div>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>Area MAE:</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>3.29 cm²</strong></div>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>HD95 (mm):</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>2.93 mm</strong></div>
    </div>
    <div style='display:flex; flex-wrap:wrap; gap:0.4rem;'>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(16,185,129,0.12); color:#10B981; border:1px solid rgba(16,185,129,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>Conv2D ×10</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(16,185,129,0.12); color:#10B981; border:1px solid rgba(16,185,129,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>MaxPool2D ×3</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(16,185,129,0.12); color:#10B981; border:1px solid rgba(16,185,129,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>Bias: -2.33 cm²</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(16,185,129,0.12); color:#10B981; border:1px solid rgba(16,185,129,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>7.4 MB</span>
    </div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div style='background:var(--surface-dark); border:1px solid var(--border-line); border-left:3px solid #A78BFA; border-radius:6px; padding:1rem 1rem 1rem 1.2rem; margin-bottom:0.6rem;'>
    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
        <div>
            <div class='mono-label' style='color:#A78BFA;'>MODEL 03 · MONAI / VIT</div>
            <div style='font-size:1rem; font-weight:700; color:#FFFFFF; margin-top:0.2rem;'>SwinUNETR</div>
        </div>
        <div style='text-align:right;'>
            <div style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; font-weight:700; color:#A78BFA;'>25.14M</div>
            <div style='font-size:0.72rem; color:var(--text-muted);'>parameters</div>
        </div>
    </div>
    <p style='font-size:0.82rem; color:var(--text-sub); margin-top:0.55rem; line-height:1.5;'>
        Swin Transformer backbone. <strong>Shifted window self-attention</strong> captures long-range spatial context across the entire slice; exhibits slight over-measurement bias (+1.53 cm²).
    </p>
    <div style='display:grid; grid-template-columns: repeat(3, 1fr); gap:0.4rem; margin:0.5rem 0; background:rgba(167,139,250,0.06); padding:0.45rem 0.6rem; border-radius:4px; border:1px solid rgba(167,139,250,0.15);'>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>5-Fold Dice:</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>0.9331</strong></div>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>Area MAE:</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>4.06 cm²</strong></div>
        <div><span style='font-size:0.7rem; color:var(--text-muted);'>HD95 (mm):</span> <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#FFFFFF;'>3.11 mm</strong></div>
    </div>
    <div style='display:flex; flex-wrap:wrap; gap:0.4rem;'>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(167,139,250,0.12); color:#A78BFA; border:1px solid rgba(167,139,250,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>Swin Blocks ×12</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(167,139,250,0.12); color:#A78BFA; border:1px solid rgba(167,139,250,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>feature_size=48</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(167,139,250,0.12); color:#A78BFA; border:1px solid rgba(167,139,250,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>Bias: +1.53 cm²</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.68rem; background:rgba(167,139,250,0.12); color:#A78BFA; border:1px solid rgba(167,139,250,0.25); padding:0.15rem 0.5rem; border-radius:3px;'>96.1 MB</span>
    </div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div style='background:var(--surface-dark); border:1px solid var(--border-line); border-left:3px solid #00E5FF; border-radius:6px; padding:1.15rem 1.1rem 1.25rem 1.25rem; margin-bottom:0;'>
    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
        <div>
            <div class='mono-label' style='color:#00E5FF;'>ENSEMBLE · PYTORCH &amp; MONAI</div>
            <div style='font-size:1.02rem; font-weight:700; color:#FFFFFF; margin-top:0.2rem;'>Tri-Model Probability Fusion</div>
        </div>
        <div style='text-align:right;'>
            <div style='font-family:IBM Plex Mono,monospace; font-size:0.85rem; font-weight:700; color:#00E5FF;'>29.02M</div>
            <div style='font-size:0.72rem; color:var(--text-muted);'>parameters · 111 MB</div>
        </div>
    </div>
    <p style='font-size:0.82rem; color:var(--text-sub); margin-top:0.55rem; line-height:1.5; margin-bottom:0.6rem;'>
        Averages sigmoid probability maps across all 3 models. Suppresses architecture-specific boundary outliers, smooths fascial interface uncertainty, and neutralizes systematic bias across diverse anatomy.
    </p>
    <div style='background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.18); border-radius:4px; padding:0.45rem 0.65rem; margin-bottom:0.55rem;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span style='font-size:0.73rem; color:var(--text-sub);'>Worst-Case Outlier (HD95 Max):</span>
            <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#00E5FF;'>28.0 mm (halved vs U-Net 59 mm)</strong>
        </div>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-top:0.25rem;'>
            <span style='font-size:0.73rem; color:var(--text-sub);'>Diagnostic Classification:</span>
            <strong style='font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#10B981;'>Sens 1.000 · Spec 0.941 · AUROC 1.0</strong>
        </div>
    </div>
    <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:0.4rem; margin-bottom:0.5rem; background:rgba(0,229,255,0.06); padding:0.45rem 0.6rem; border-radius:4px; border:1px solid rgba(0,229,255,0.18);'>
        <div><span style='font-size:0.68rem; color:var(--text-muted);'>5-Fold Dice:</span><br><strong style='font-family:IBM Plex Mono,monospace; font-size:0.82rem; color:#00E5FF;'>0.9433</strong></div>
        <div><span style='font-size:0.68rem; color:var(--text-muted);'>Pooled IoU:</span><br><strong style='font-family:IBM Plex Mono,monospace; font-size:0.82rem; color:#00E5FF;'>0.8937</strong></div>
        <div><span style='font-size:0.68rem; color:var(--text-muted);'>Area MAE:</span><br><strong style='font-family:IBM Plex Mono,monospace; font-size:0.82rem; color:#00E5FF;'>2.74 cm²</strong></div>
        <div><span style='font-size:0.68rem; color:var(--text-muted);'>HD95 Median:</span><br><strong style='font-family:IBM Plex Mono,monospace; font-size:0.82rem; color:#00E5FF;'>1.81 mm</strong></div>
    </div>
    <div style='display:flex; flex-wrap:nowrap; gap:0.35rem;'>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.62rem; white-space:nowrap; background:rgba(0,229,255,0.12); color:#00E5FF; border:1px solid rgba(0,229,255,0.25); padding:0.12rem 0.35rem; border-radius:3px;'>Mean Probability Averaging</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.62rem; white-space:nowrap; background:rgba(0,229,255,0.12); color:#00E5FF; border:1px solid rgba(0,229,255,0.25); padding:0.12rem 0.35rem; border-radius:3px;'>Zero False Negatives</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.62rem; white-space:nowrap; background:rgba(0,229,255,0.12); color:#00E5FF; border:1px solid rgba(0,229,255,0.25); padding:0.12rem 0.35rem; border-radius:3px;'>Bias: -1.08 cm²</span>
        <span style='font-family:IBM Plex Mono,monospace; font-size:0.62rem; white-space:nowrap; background:rgba(0,229,255,0.12); color:#00E5FF; border:1px solid rgba(0,229,255,0.25); padding:0.12rem 0.35rem; border-radius:3px;'>&lt; 3s Latency</span>
    </div>
</div>
        """, unsafe_allow_html=True)

    st.markdown("""
<div class='clinical-card' style='border-left:4px solid var(--border-glow); margin-top:0.75rem; margin-bottom:0;'>
    <div class='mono-label' style='color:var(--blue-bright);'>REGULATORY & RESEARCH DISCLAIMER</div>
    <p style='font-size:0.83rem; color:var(--text-sub); margin-top:0.35rem; line-height:1.5; margin-bottom:0;'>
        SarcoScan AI is intended strictly for research and clinical demonstration purposes. Full diagnostic PACS deployment requires institutional user authentication, audit logging, role-based access controls, and CE/FDA medical device software accreditation.
    </p>
</div>
    """, unsafe_allow_html=True)


# =============================================================
# TAB 3: DEMO & VIDEO WALKTHROUGH
# =============================================================
with tab_demo:
    st.markdown("""
<div style='margin-bottom:1.5rem;'>
    <h2 style='font-size:1.6rem; font-weight:700; color:#FFFFFF;'>Interactive Video Walkthrough & Clinician Guide</h2>
    <p style='color:var(--text-sub); font-size:0.95rem;'>Watch how SarcoScan AI processes DICOM archives, automatically extracts the L3 vertebra slice, and generates instant sarcopenia metrics.</p>
</div>
    """, unsafe_allow_html=True)
    
    vid_col1, vid_col2 = st.columns([1.6, 1])
    
    with vid_col1:
        vid_file = next((p for p in [ASSET_DIR / "demo.mp4", ASSET_DIR / "demo.mov", ASSET_DIR / "demo.webm"] if p.exists()), None)
        if vid_file:
            st.video(str(vid_file))
        else:
            st.markdown("""
<div class='clinical-card' style='padding:3rem 2rem; text-align:center; background:#0D1726; border:1px dashed var(--blue-deep);'>
    <div style='font-size:2.5rem; color:var(--blue-bright); margin-bottom:0.8rem;'>🎬</div>
    <h4 style='color:#FFFFFF; margin-bottom:0.4rem;'>Walkthrough Video Ready</h4>
    <p style='color:var(--text-sub); font-size:0.88rem; max-width:450px; margin:0 auto 1.2rem;'>
        To view your custom screen recording, place your video file at:
        <br><code style='font-family:"IBM Plex Mono",monospace; color:var(--blue-bright); background:rgba(0,0,0,0.4); padding:0.2rem 0.5rem; border-radius:4px;'>assets/demo.mp4</code>
    </p>
</div>
            """, unsafe_allow_html=True)
            
    with vid_col2:
        with st.container(border=True):
            st.markdown("""
<h4 style='color:#FFFFFF; margin-bottom:0.8rem; font-size:1.05rem;'>Quick Start Instructions</h4>

<div style='margin-bottom:1rem;'>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:var(--blue-bright); font-weight:700;'>STEP 01 · INPUT SCAN</div>
    <div style='font-size:0.86rem; color:var(--text-sub); margin-top:0.2rem;'>
        Navigate to the <strong>Scan / Upload</strong> tab. Select patient sex and upload either a full CT ZIP archive or a pre-extracted L3 DICOM/Numpy slice.
    </div>
</div>

<div style='margin-bottom:1rem;'>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:var(--blue-bright); font-weight:700;'>STEP 02 · AI INFERENCE</div>
    <div style='font-size:0.86rem; color:var(--text-sub); margin-top:0.2rem;'>
        The deep ensemble automatically normalizes Hounsfield Units (-29 to +150 HU window), computes model predictions, and displays the muscle overlay.
    </div>
</div>

<div>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:var(--blue-bright); font-weight:700;'>STEP 03 · REPORT GENERATION</div>
    <div style='font-size:0.86rem; color:var(--text-sub); margin-top:0.2rem;'>
        Review total skeletal muscle area (cm²), attenuation (HU), and diagnostic cutoff status. Download the structured CSV clinical report.
    </div>
</div>
            """, unsafe_allow_html=True)


# =============================================================
# TAB 4: SCAN / UPLOAD & ANALYSIS WORKFLOW (RADIOLOGICAL DASHBOARD)
# =============================================================
with tab_scan:
    # Read values from session_state for Advanced Research Settings expander
    thr = st.session_state.get("adv_thr", 0.50)
    px_manual = st.session_state.get("adv_px", 0.74)

    # Check for uploaded file from Screen 1 central uploader
    up = st.session_state.get("empty_file_uploader", None)

    # ── SCREEN 1: UNUPLOADED / UPLOAD STATE (NO FILE UPLOADED YET) ──
    if not up:
        # Top Header Banner (Only visible when no file is uploaded)
        st.markdown("""
        <div style='background:var(--surface-dark); border:1px solid var(--border-line); border-left:4px solid var(--blue-bright); border-radius:8px; padding:0.6rem 1rem; margin-bottom:1rem;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <div class='mono-label' style='color:var(--blue-bright); font-weight:700;'>SARCOSCAN AI</div>
                    <h3 style='font-size:1.15rem; font-weight:700; color:#FFFFFF; margin:0.1rem 0;'>SarcoScan AI: Automated L3 Sarcopenia Screening</h3>
                </div>
                <div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.7rem; color:var(--text-muted);'>
                    Single-Screen High-Density Dashboard
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='max-width:700px; margin:1.5rem auto 1rem auto; text-align:center;'>
            <p style='color:var(--text-sub); font-size:0.92rem; line-height:1.6; margin:0;'>
                Select patient biological sex and upload an abdominal CT study archive (.zip) or a single L3 cross-section (.dcm / .npy) to initiate deep learning sarcopenia assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c_left, c_center, c_right = st.columns([0.15, 1, 0.15])
        with c_center:
            with st.container(border=True):
                st.markdown("""
                <div style='text-align:center; padding:0.4rem 0 0.8rem 0;'>
                    <h3 style='color:#FFFFFF; font-size:1.2rem; font-weight:700; margin-bottom:0.3rem;'>
                        Diagnostic Scan Upload &amp; Patient Setup
                    </h3>
                    <p style='font-size:0.83rem; color:var(--text-sub); margin:0;'>
                        Configure patient biological sex and drag &amp; drop CT study file below.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.72rem; color:var(--blue-bright); font-weight:700; margin-bottom:0.25rem; text-align:center;'>1. PATIENT BIOLOGICAL SEX</div>", unsafe_allow_html=True)
                sex = st.radio(
                    "Patient Biological Sex",
                    ["Male", "Female"],
                    horizontal=True,
                    key="empty_sex_radio"
                )

                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
                st.markdown("<div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.72rem; color:var(--blue-bright); font-weight:700; margin-bottom:0.25rem; text-align:center;'>2. UPLOAD DIAGNOSTIC SCAN</div>", unsafe_allow_html=True)

                up = st.file_uploader(
                    "Upload a full CT study (.zip) or a single L3 slice (.dcm / .npy)",
                    type=["zip", "dcm", "npy"],
                    key="empty_file_uploader",
                    help="Drag and drop or browse DICOM archive (.zip) or single L3 slice (.dcm / .npy)"
                )

    # Retrieve sex choice from session_state
    sex = st.session_state.get("empty_sex_radio", "Male")

    # ── PROCESS FILE WHEN UPLOADED ─────────────────────────────
    hu = px = l3 = n_sl = None
    npy_alert = False

    if up:
        if up.name.endswith(".zip"):
            slot = st.empty()
            try:
                with st.spinner("Processing CT volume..."):
                    hu, px, l3, n_sl = run_full_study(up.read(), lambda m: slot.info(m))
                slot.empty()
            except FileNotFoundError:
                slot.error("TotalSegmentator is not installed. Please upload a single L3 slice DICOM/Numpy file.")
            except subprocess.CalledProcessError as e:
                slot.error("Vertebral segmentation failed on this CT volume.")
                st.code((e.stderr or "")[-700:])
            except ValueError as e:
                slot.error(str(e))
        elif up.name.endswith(".npy"):
            npy_alert = True
            hu, px = np.load(io.BytesIO(up.read())).astype(np.float32), px_manual
        else:
            import pydicom
            raw_bytes = up.read()
            ds_check = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
            hu, got_px = read_dcm(raw_bytes)
            px = got_px or px_manual
            issues = check_slice_plausible(ds_check, hu)
            if issues:
                for msg in issues:
                    st.error(msg)
                hu = None

    # ── SCREEN 2: RESULTS DASHBOARD (SINGLE-PAGE LAYOUT AT TOP) ──────
    if hu is not None and px:
        mask = segment(hu, models, dev, thr)
        n_px = int(mask.sum())
        area = n_px * px * px / 100.0
        atten = float(hu[mask].mean()) if n_px else float("nan")
        cut = CUTOFF[sex]; gap = area - cut
        ind = abs(gap) <= MARGIN

        cls, txt = ("v-ind", "Indeterminate Margin") if ind else \
                   (("v-low", "Low Muscle Mass (Sarcopenia)") if gap < 0 else ("v-ok", "Within Normal Limits"))
        status_color = "#F59E0B" if ind else ("#EF4444" if gap < 0 else "#10B981")

        # NPY Alert Banner if applicable
        if npy_alert:
            st.warning("⚠️ NumPy file detected (lacks DICOM metadata). Using default pixel spacing (0.74 mm/px). Adjust in Advanced Settings if known.")

        col_left, col_right = st.columns([0.65, 0.35])

        # ── LEFT COLUMN (65% WIDTH): IMAGE VIEWER ─────────────────
        with col_left:
            st.markdown("""
            <div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.76rem; font-weight:700; color:#FFFFFF; padding-bottom:0.4rem;'>
                L3 CT ANATOMICAL VIEWER &amp; AI SEGMENTATION
            </div>
            """, unsafe_allow_html=True)

            img_c1, img_c2 = st.columns(2)

            with img_c1:
                disp_rot = window(rotate_array(hu, 270))
                fig_ct, ax_ct = plt.subplots(figsize=(4.5, 4.5), facecolor="#090E17")
                ax_ct.imshow(disp_rot, cmap="gray")
                ax_ct.set_title("Raw Axial L3 Slice", fontsize=9.5, color="#F1F5F9", pad=6, fontweight="600")
                ax_ct.axis("off")
                plt.tight_layout(pad=0.2)
                st.pyplot(fig_ct, use_container_width=True)
                plt.close(fig_ct)

            with img_c2:
                disp_rot = window(rotate_array(hu, 270))
                mask_rot = rotate_array(mask, 270)

                fig_seg, ax_seg = plt.subplots(figsize=(4.5, 4.5), facecolor="#090E17")
                ax_seg.imshow(disp_rot, cmap="gray")

                # Solid red overlay for segmented muscle mass (no animation or progressive highlighting)
                overlay = np.zeros((*mask_rot.shape, 4), dtype=np.float32)
                overlay[mask_rot] = [0.937, 0.173, 0.173, 0.55]  # Solid red with 55% opacity

                ax_seg.imshow(overlay)
                ax_seg.set_title("AI Segmented Muscle Overlay", fontsize=9.5, color="#F1F5F9", pad=6, fontweight="600")
                ax_seg.axis("off")
                plt.tight_layout(pad=0.2)
                st.pyplot(fig_seg, use_container_width=True)
                plt.close(fig_seg)

            # Plain-English Anatomical Explanation directly underneath images
            st.markdown("""
            <div class='clinical-card' style='margin-top:0.6rem; margin-bottom:0.4rem; padding:0.85rem 1rem;'>
                <div class='mono-label' style='color:#EF4444;'>DELINEATED ANATOMICAL MUSCLE GROUPS</div>
                <ul style='font-size:0.82rem; color:var(--text-sub); padding-left:1.2rem; line-height:1.55; margin:0.4rem 0 0 0;'>
                    <li><strong>Psoas Major:</strong> Bilateral anterior lumbar spine bodies.</li>
                    <li><strong>Paraspinal Group:</strong> Erector spinae &amp; quadratus lumborum.</li>
                    <li><strong>Rectus Abdominis:</strong> Anterior abdominal wall midline.</li>
                    <li><strong>Lateral Wall:</strong> Transverse abdominis &amp; obliques.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # ── RIGHT COLUMN (35% WIDTH): CLINICAL DIAGNOSTIC PANEL ───
        with col_right:
            # 1. Diagnostic Outcome Banner at Top Right
            st.markdown(f"""
            <div class='verdict-box {cls}' style='margin-bottom:0.7rem;'>
              <div class='verdict-label'>Diagnostic Outcome</div>
              <div class='verdict-val'>{txt}</div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Key Metrics Grid
            st.markdown(f"""
            <div class='clinical-card' style='border-left:4px solid {status_color}; padding:0.85rem 1rem; margin-bottom:0.7rem;'>
                <div style='display:grid; grid-template-columns: 1fr 1fr; gap:0.5rem;'>
                    <div>
                        <div class='mono-label'>Skeletal Muscle Area</div>
                        <div class='mono-readout' style='font-size:1.1rem;'>{area:.1f}<span class='mono-unit'> cm²</span></div>
                        <div style='font-size:0.72rem; color:var(--text-sub);'>{gap:+.1f} cm² vs cutoff</div>
                    </div>
                    <div>
                        <div class='mono-label'>Mean Attenuation</div>
                        <div class='mono-readout' style='font-size:1.1rem;'>{atten:.1f}<span class='mono-unit'> HU</span></div>
                        <div style='font-size:0.72rem; color:var(--text-sub);'>{atten - ATTEN_REF[sex]:+.1f} HU vs ref</div>
                    </div>
                    <div>
                        <div class='mono-label'>Pixel Spacing</div>
                        <div class='mono-readout' style='font-size:1.1rem;'>{px:.3f}<span class='mono-unit'> mm</span></div>
                        <div style='font-size:0.72rem; color:var(--text-sub);'>{n_px:,} muscle px</div>
                    </div>
                    <div>
                        <div class='mono-label'>Reference Cutoff</div>
                        <div class='mono-readout' style='font-size:1.1rem;'>{cut}<span class='mono-unit'> cm²</span></div>
                        <div style='font-size:0.72rem; color:var(--text-sub);'>{sex} normative</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3. Visual Gauge / Threshold Bar
            st.markdown(scale_html(area, cut), unsafe_allow_html=True)

            if ind:
                st.warning(f"Measurement falls within ±{MARGIN:.0f} cm² margin of error near the reference cutoff.")

            # 4. Plain-English Clinical Summary Box
            if gap < 0 and not ind:
                summary_text = (
                    f"Patient muscle area is {area:.1f} cm², falling {abs(gap):.1f} cm² below the "
                    f"{sex.lower()} reference cut-off of {cut} cm², indicating significant skeletal muscle depletion (sarcopenia)."
                )
            elif gap >= 0 and not ind:
                summary_text = (
                    f"Patient muscle area is {area:.1f} cm², measuring {abs(gap):.1f} cm² above the "
                    f"{sex.lower()} reference cut-off of {cut} cm², confirming skeletal muscle mass within normal reference limits."
                )
            else:
                summary_text = (
                    f"Patient muscle area is {area:.1f} cm², falling within the ±{MARGIN:.0f} cm² margin of error "
                    f"around the {sex.lower()} reference cut-off ({cut} cm²). Clinical correlation recommended."
                )

            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid var(--border-line); border-radius:6px; padding:0.75rem 0.9rem; margin-top:0.6rem; margin-bottom:0.7rem;'>
                <div class='mono-label' style='color:var(--blue-bright); margin-bottom:0.25rem;'>CLINICAL SUMMARY</div>
                <p style='font-size:0.83rem; color:var(--text-sub); line-height:1.55; margin:0;'>
                    {summary_text}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 5. Download Button at Bottom of Column
            csv_data = (
                "sex,muscle_area_cm2,cutoff_cm2,difference_cm2,attenuation_HU,reference_HU,"
                "muscle_pixels,pixel_spacing_mm,threshold,outcome\n"
                f"{sex},{area:.2f},{cut},{gap:+.2f},{atten:.2f},{ATTEN_REF[sex]},"
                f"{n_px},{px:.4f},{thr},"
                f"{'indeterminate' if ind else ('low' if gap < 0 else 'normal')}\n"
            )
            st.download_button(
                "📥 Download Clinical Report (CSV)",
                data=csv_data,
                file_name=f"sarcoscan_l3_sarcopenia_report_{sex.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ── 4. ADVANCED RESEARCH SETTINGS (EXPANDER AT VERY BOTTOM) ───
    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    with st.expander("Advanced Research Settings", expanded=False):
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.select_slider(
                "AI Segmentation Threshold (0.5)",
                [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
                value=0.50,
                key="adv_thr"
            )
        with exp_col2:
            st.number_input(
                "Pixel Spacing Manual Override (mm/pixel)",
                0.30, 2.00, 0.74, 0.01,
                key="adv_px"
            )

