"""
========================================================================================
Autonomous Satellite & Drone Flood Mapping with Hyper-Local Emergency Alert System
Disaster Decision Support & Automated Alert Pipeline — Streamlit Command Center
Pondicherry University Karaikal Campus · Smart India Hackathon 2026

Run command:
    pip install -r requirements.txt && streamlit run app.py
========================================================================================
"""

import io
import time
import random
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import folium
from streamlit_folium import st_folium
import requests
from io import BytesIO

# ----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Karaikal Flood Command Center · SIH 2026",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
:root {
  --bg: #0b1120;
  --panel: #111a2e;
  --line: #1e293b;
  --txt: #e2e8f0;
  --mut: #94a3b8;
  --em: #10b981;
  --am: #f59e0b;
  --ro: #f43f5e;
  --sky: #38bdf8;
}
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: radial-gradient(circle at 20% 0%, #0f1d33 0%, var(--bg) 55%); color: var(--txt); }
section[data-testid="stSidebar"] { background: #070d1a; border-right: 1px solid var(--line); }
h1, h2, h3 { letter-spacing: -0.02em; }
.mono { font-family: 'JetBrains Mono', monospace; }
.hdr {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.6rem;
  margin-bottom: 1rem;
}
.hdr .t { font-size: 1.5rem; font-weight: 700; }
.hdr .s { color: var(--mut); font-size: 0.8rem; font-family: 'JetBrains Mono'; }
.live {
  display: inline-flex;
  gap: 0.4rem;
  align-items: center;
  color: var(--em);
  font-family: 'JetBrains Mono';
  font-size: 0.75rem;
}
.live:before {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--em);
  box-shadow: 0 0 10px var(--em);
  animation: p 1.2s infinite;
}
@keyframes p { 50% { opacity: 0.25; } }
.card {
  background: linear-gradient(180deg, #121c31, #0d1527);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  height: 100%;
}
.card .l { color: var(--mut); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; font-family: 'JetBrains Mono'; }
.card .v { font-size: 1.85rem; font-weight: 700; margin: 0.2rem 0; }
.card .d { font-size: 0.75rem; color: var(--mut); }
.card.ro { border-color: #7f1d2d; box-shadow: 0 0 24px -8px var(--ro); } .card.ro .v { color: var(--ro); }
.card.am { border-color: #78501a; } .card.am .v { color: var(--am); }
.card.em .v { color: var(--em); } 
.card.sk .v { color: var(--sky); }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; }
.tag {
  display: inline-block;
  padding: 0.1rem 0.5rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-family: 'JetBrains Mono';
  margin-right: 0.3rem;
}
.tag.ro { background: #3b0d18; color: #fda4af; }
.tag.am { background: #3a2606; color: #fcd34d; }
.tag.em { background: #052e22; color: #6ee7b7; }
.stButton>button { border-radius: 8px; border: 1px solid var(--line); background: #132036; color: var(--txt); }
.broadcast .stButton>button {
  width: 100%;
  padding: 1rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  font-size: 1rem;
  background: linear-gradient(90deg, #be123c, #f43f5e);
  border: none;
  color: white;
  box-shadow: 0 0 30px -6px var(--ro);
}
.broadcast .stButton>button:hover { filter: brightness(1.15); }
div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 0.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# GEOSPATIAL & RESPONSE TEAMS CONFIGURATION (Karaikal & PU Campus)
# ----------------------------------------------------------------------------
KARAIKAL = (10.9254, 79.8380)
CAMPUS = (10.9555, 79.8222)

FLOOD_POLYGONS = [
    {
        "name": "Arasalar River Mouth Floodplain",
        "depth": 1.4,
        "polygon": [[79.842, 10.905], [79.858, 10.903], [79.860, 10.918], [79.848, 10.924], [79.838, 10.916]],
    },
    {
        "name": "Karaikalmedu Coastal Fishing Village",
        "depth": 1.1,
        "polygon": [[79.852, 10.928], [79.864, 10.926], [79.866, 10.945], [79.855, 10.948]],
    },
    {
        "name": "PU Campus Low-lying Hostel Zone",
        "depth": 0.6,
        "polygon": [[79.816, 10.950], [79.826, 10.949], [79.828, 10.957], [79.818, 10.959]],
    },
    {
        "name": "Thirunallar Road Paddy Belt",
        "depth": 0.9,
        "polygon": [[79.800, 10.920], [79.818, 10.915], [79.822, 10.932], [79.805, 10.936]],
    },
    {
        "name": "Kottucherry Drainage Basin",
        "depth": 0.8,
        "polygon": [[79.825, 10.965], [79.842, 10.962], [79.846, 10.978], [79.830, 10.982]],
    },
]

DRONES = [
    {"id": "Drone-01", "lat": 10.912, "lon": 79.850, "status": "ONLINE", "info": "Inundation detected at 82% · Arasalar mouth", "bat": 71},
    {"id": "Drone-02", "lat": 10.953, "lon": 79.821, "status": "ONLINE", "info": "Hostel block perimeter · 0.6m standing water", "bat": 58},
    {"id": "Drone-03", "lat": 10.937, "lon": 79.859, "status": "ONLINE", "info": "Karaikalmedu shoreline · surge front advancing", "bat": 44},
    {"id": "Drone-04", "lat": 10.970, "lon": 79.836, "status": "OFFLINE", "info": "Link lost · RTH initiated (wind 62 km/h)", "bat": 12},
]

HELP_REQUESTS = [
    {"id": "SOS-1142", "lat": 10.931, "lon": 79.861, "info": "Family of 6 on rooftop · Karaikalmedu", "prio": "CRITICAL"},
    {"id": "SOS-1147", "lat": 10.909, "lon": 79.846, "info": "Elderly patient needs dialysis transport · Kovilpathu", "prio": "CRITICAL"},
    {"id": "SOS-1151", "lat": 10.956, "lon": 79.819, "info": "32 students stranded · PU Boys Hostel Block C", "prio": "HIGH"},
    {"id": "SOS-1153", "lat": 10.925, "lon": 79.808, "info": "Cattle shed collapse · Thirunallar Road", "prio": "HIGH"},
]

TEAM_MEMBERS = [
    {
        "name": "Member 1 (Team Lead)",
        "role": "Disaster Commander & Mapping",
        "phone": "8590693129",
        "telegram_id": "",
        "sector": "Karaikal GH Sector",
        "lat": 10.9254,
        "lon": 79.8380,
    },
    {
        "name": "Member 2",
        "role": "Field Operations & Logistics",
        "phone": "9497812151",
        "telegram_id": "",
        "sector": "Kottucherry Bypass",
        "lat": 10.9480,
        "lon": 79.8220,
    },
]

ZONES = {
    "Zone A · PU Campus Main Block": 1850,
    "Zone B · PU Hostels (Boys/Girls)": 1240,
    "Zone C · Karaikalmedu Fishing Village": 3600,
    "Zone D · Arasalar River Mouth / Kovilpathu": 2900,
    "Zone E · Thirunallar Road Settlements": 2150,
    "Zone F · Karaikal Port & Kottucherry": 1700,
}

TEMPLATES = {
    "— Custom —": "",
    "Flash Flood Warning": "⚠️ FLASH FLOOD WARNING – Karaikal District Admin: Water levels rising rapidly near your area. Move to higher floors immediately. Do not cross flowing water.",
    "Evacuation Order": "🚨 EVACUATE NOW – Proceed to the nearest cyclone shelter (Govt. Hr. Sec. School, Karaikalmedu / PU Karaikal Auditorium). Carry ID, medicines and drinking water.",
    "Storm Surge Alert": "🌊 STORM SURGE ALERT – Fishermen of Karaikalmedu & Pattinacherry must NOT venture to sea. Surge of 1.5m expected at high tide (18:40 IST).",
    "Campus Lockdown": "🏫 PU KARAIKAL CAMPUS – Hostel residents shift to Main Block 1st floor. Mess services relocated. Helpline: 1077.",
    "All Clear": "✅ ALL CLEAR – Water receding in your zone. Avoid fallen power lines. Boil drinking water before use.",
}

DATES = ["Oct 4, 2026", "Oct 5, 2026", "Oct 6, 2026", "Oct 7, 2026", "Oct 8, 2026"]
DATE_LEVEL = {d: v for d, v in zip(DATES, [0.15, 0.35, 0.62, 0.88, 0.74])}

# ----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault(
    "history",
    [
        {"Time (IST)": "20:41:07", "Zones": "Zone C", "Channels": "C-DOT CB, Telegram", "Recipients": 3600, "Message": "STORM SURGE ALERT – Fishermen must not venture to sea…", "Status": "DELIVERED"},
        {"Time (IST)": "19:58:32", "Zones": "Zone B", "Channels": "Telegram", "Recipients": 1240, "Message": "PU KARAIKAL CAMPUS – Hostel residents shift to Main Block…", "Status": "DELIVERED"},
    ],
)
ss.setdefault("reached", 12450)

# ----------------------------------------------------------------------------
# UI HELPERS & CV PIPELINE
# ----------------------------------------------------------------------------
def header(title, sub):
    now = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    st.markdown(
        f'<div class="hdr"><div><div class="t">{title}</div><div class="s">{sub}</div></div>'
        f'<div style="text-align:right"><span class="live">LIVE FEED</span><div class="s">{now} IST · Cyclone "MONTHA-II" Cat-2</div></div></div>',
        unsafe_allow_html=True,
    )

def card(col, label, value, desc, tone):
    col.markdown(
        f'<div class="card {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="d">{desc}</div></div>',
        unsafe_allow_html=True,
    )

def get_font(size):
    for f in ["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()

def analyze_flood_imagery(pil_img: Image.Image, kernel_sz=5, custom_hsv=None):
    """Real OpenCV HSV Segmentation & Morphological Masking"""
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if custom_hsv:
        hue_lo, hue_hi, sat_min = custom_hsv
        lower_custom = np.array([hue_lo, sat_min, 40])
        upper_custom = np.array([hue_hi, 255, 240])
        combined_mask = cv2.inRange(hsv, lower_custom, upper_custom)
    else:
        # Turbid / muddy water
        mask_mud = cv2.inRange(hsv, np.array([10, 30, 40]), np.array([30, 210, 200]))
        # Standing / reflective blue water
        mask_blue = cv2.inRange(hsv, np.array([85, 20, 30]), np.array([135, 255, 220]))
        combined_mask = cv2.bitwise_or(mask_mud, mask_blue)

    k = max(3, kernel_sz | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = clean_mask.shape[0] * clean_mask.shape[1]
    flood_pixels = np.count_nonzero(clean_mask)
    inundation_pct = (flood_pixels / max(1, total_pixels)) * 100.0

    overlay = img_bgr.copy()
    overlay[clean_mask > 0] = [235, 120, 10]  # Amber-cyan highlight
    blended = cv2.addWeighted(img_bgr, 0.65, overlay, 0.35, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return clean_mask, blended_rgb, inundation_pct

# ----------------------------------------------------------------------------
# PROCEDURAL DRONE GENERATOR (EDGE TELEMETRY SIMULATOR)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def drone_gif(seed, kernel, thresh, hue_lo, hue_hi, morph_open, morph_close, label):
    rng = np.random.default_rng(seed)
    W, H, N = 480, 270, 14
    base = rng.normal(0, 1, (H // 6, W // 6))
    terrain = (
        np.array(
            Image.fromarray(((base - base.min()) / np.ptp(base) * 255).astype(np.uint8))
            .resize((W * 2, H * 2), Image.BICUBIC)
            .filter(ImageFilter.GaussianBlur(6))
        )
        / 255.0
    )
    font, small = get_font(13), get_font(11)
    water_thr = 0.35 + (thresh - 128) / 600
    acc = np.clip(
        78 + kernel * 0.9 - abs(thresh - 140) * 0.05 + (2 if morph_open else 0) + (2 if morph_close else 0) - abs((hue_hi - hue_lo) - 60) * 0.04,
        60,
        97.8,
    )
    depth = 0.6 + (seed % 3) * 0.35
    frames = []
    for i in range(N):
        ox, oy = int(i * 6), int(i * 3)
        t = terrain[oy : oy + H, ox : ox + W]
        water = t < water_thr
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., 0] = 60 + t * 90
        img[..., 1] = 85 + t * 110
        img[..., 2] = 45 + t * 60
        ripple = 0.5 + 0.5 * np.sin(np.linspace(0, 12, W)[None, :] + i * 0.8 + np.linspace(0, 6, H)[:, None])
        img[water] = np.stack([70 + ripple[water] * 20, 95 + ripple[water] * 25, 105 + ripple[water] * 30], -1)
        img += rng.normal(0, 6, img.shape)
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        ys, xs = np.where(water)
        if len(xs):
            pad = max(2, 14 - kernel)
            x0, y0 = max(int(np.percentile(xs, 8)) - pad, 4), max(int(np.percentile(ys, 8)) - pad, 22)
            x1, y1 = min(int(np.percentile(xs, 92)) + pad, W - 4), min(int(np.percentile(ys, 92)) + pad, H - 24)
            d.rectangle([x0, y0, x1, y1], outline=(16, 185, 129), width=2)
            for cx, cy in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                d.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=(16, 185, 129))
            tag = f"[WATER_DETECTED: {acc + rng.normal(0, .3):.1f}% ACCURACY]"
            d.rectangle([x0, y0 - 18, x0 + len(tag) * 8 + 6, y0], fill=(6, 40, 30))
            d.text((x0 + 3, y0 - 16), tag, fill=(110, 231, 183), font=font)
            tag2 = f"[DEPTH_EST: {depth + rng.normal(0, .03):.2f}m]"
            d.rectangle([x0, y1, x0 + len(tag2) * 8 + 6, y1 + 18], fill=(60, 38, 6))
            d.text((x0 + 3, y1 + 2), tag2, fill=(252, 211, 77), font=font)
        # HUD Overlays
        d.text((8, 5), f"{label}  ● REC", fill=(244, 63, 94) if i % 2 else (226, 232, 240), font=small)
        d.text((W - 170, 5), "ALT 120m  SPD 8.4m/s", fill=(226, 232, 240), font=small)
        d.text((8, H - 16), f"LAT {10.91 + seed*0.01:.4f}N  LON {79.84 + seed*0.005:.4f}E", fill=(148, 163, 184), font=small)
        d.text((W - 120, H - 16), f"FRAME {1000 + i + seed*37}", fill=(148, 163, 184), font=small)
        d.line([(W // 2 - 10, H // 2), (W // 2 + 10, H // 2)], fill=(226, 232, 240))
        d.line([(W // 2, H // 2 - 10), (W // 2, H // 2 + 10)], fill=(226, 232, 240))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=120, loop=0)
    return buf.getvalue(), acc, depth, float(np.mean(terrain[:H, :W] < water_thr) * 100)

def offline_frame(label):
    arr = np.random.default_rng(4).integers(0, 50, (270, 480)).astype(np.uint8)
    im = Image.merge("RGB", [Image.fromarray(arr)] * 3)
    d = ImageDraw.Draw(im)
    d.text((150, 120), "SIGNAL LOST · RTH", fill=(244, 63, 94), font=get_font(18))
    d.text((8, 5), label, fill=(148, 163, 184), font=get_font(11))
    return im

# ----------------------------------------------------------------------------
# SATELLITE (SAR & OPTICAL) SYNTHESIS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def sar_terrain(size=420):
    rng = np.random.default_rng(7)
    b = rng.normal(0, 1, (size // 10, size // 10))
    t = (
        np.array(
            Image.fromarray(((b - b.min()) / np.ptp(b) * 255).astype(np.uint8))
            .resize((size, size), Image.BICUBIC)
            .filter(ImageFilter.GaussianBlur(5))
        )
        / 255.0
    )
    xx = np.linspace(0, 1, size)[None, :].repeat(size, 0)
    t = t * 0.65 + (1 - xx) * 0.35
    river = np.exp(-((np.linspace(0, 1, size)[:, None] - 0.62 - 0.05 * np.sin(xx * 9)) ** 2) / 0.0012)
    t = t - river * 0.35
    return t

def sar_image(level, mode):
    t = sar_terrain()
    size = t.shape[0]
    sea = np.linspace(0, 1, size)[None, :].repeat(size, 0) > 0.86
    water = (t < (0.18 + level * 0.32)) | sea
    rng = np.random.default_rng(int(level * 100))
    speckle = rng.gamma(4, 0.25, t.shape)
    if mode == "Grayscale backscatter (VV)":
        land = np.clip(0.45 + t * 0.5, 0, 1) * speckle
        img = np.where(water, 0.04 * speckle, land)
        rgb = np.stack([img] * 3, -1)
    else:
        rgb = np.zeros((size, size, 3))
        rgb[..., 0] = np.clip(0.25 + t * 0.6, 0, 1) * speckle * 0.8
        rgb[..., 1] = np.clip(0.2 + t * 0.4, 0, 1) * speckle * 0.6
        rgb[..., 2] = np.clip(0.1 + t * 0.2, 0, 1) * speckle * 0.4
        rgb[water & ~sea] = [0.05, 0.75, 0.95]
        rgb[sea] = [0.02, 0.15, 0.35]
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    edge = Image.fromarray((water & ~sea).astype(np.uint8) * 255).filter(ImageFilter.FIND_EDGES)
    e = np.array(edge) > 0
    a = np.array(im)
    a[e] = [244, 63, 94]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    d.rectangle([int(size * 0.22), int(size * 0.18), int(size * 0.24), int(size * 0.2)], fill=(16, 185, 129))
    d.text((int(size * 0.25), int(size * 0.17)), "PU KARAIKAL", fill=(16, 185, 129), font=get_font(11))
    d.text((int(size * 0.55), int(size * 0.6) - 16), "ARASALAR", fill=(250, 250, 250), font=get_font(10))
    d.text((8, size - 16), "S1A IW GRDH · VV · 10m", fill=(200, 200, 200), font=get_font(10))
    flooded = float(np.mean(water & ~sea) * 100)
    return im, flooded

def optical_gif(level):
    t = sar_terrain()
    size = t.shape[0]
    base = np.zeros((size, size, 3))
    base[..., 0] = 60 + t * 80
    base[..., 1] = 90 + t * 90
    base[..., 2] = 50 + t * 40
    rng = np.random.default_rng(3)
    c = rng.normal(0, 1, (size // 14, size // 7))
    cloud = (
        np.array(
            Image.fromarray(((c - c.min()) / np.ptp(c) * 255).astype(np.uint8))
            .resize((size * 2, size), Image.BICUBIC)
            .filter(ImageFilter.GaussianBlur(10))
        )
        / 255.0
    )
    frames = []
    for i in range(12):
        sh = cloud[:, i * 14 : i * 14 + size]
        alpha = np.clip(0.82 + sh * 0.3, 0, 1)[..., None]
        img = base * (1 - alpha) + 245 * alpha
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((8, 8), "SENTINEL-2 MSI · TRUE COLOUR", fill=(80, 90, 110), font=get_font(11))
        d.text((8, size - 18), "CLOUD COVER 98.6% · NO USABLE PIXELS", fill=(190, 18, 60), font=get_font(11))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=160, loop=0)
    return buf.getvalue()

# ----------------------------------------------------------------------------
# DISPATCH API INTEGRATIONS
# ----------------------------------------------------------------------------
def dispatch_team_telegram_alerts(bot_token: str, team_members: list, flood_pct: float) -> dict:
    if not bot_token or not bot_token.strip():
        return {"status": "skipped", "details": "Operating in Live Radio Simulation Mode"}

    clean_token = bot_token.strip()
    delivered_count = 0
    depth = 0.3 + (flood_pct / 45.0)

    for member in team_members:
        chat_id = member.get("telegram_id", "").strip()
        if not chat_id:
            continue
        msg = (
            f"🚨 *KARAIKAL EMERGENCY DISASTER ALERT*\n\n"
            f"Officer: *{member['name']}* [{member['role']}]\n"
            f"Sector: `{member['sector']}`\n"
            f"⚠️ Hazard: *{flood_pct:.1f}% Inundation* (Est. Depth: *{depth:.1f}m*)\n"
            f"Action: Mobilize rescue protocol. Evacuate low-lying areas.\n"
            f"Control Room: Dial 1077 / 112."
        )
        url = f"https://api.telegram.org/bot{clean_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        try:
            res = requests.post(url, json=payload, timeout=5)
            if res.status_code == 200:
                delivered_count += 1
        except Exception:
            pass

    return {"status": "success", "count": delivered_count}

def build_emergency_card(member_name: str, flood_pct: float, role: str, phone: str) -> str:
    depth = 0.3 + (flood_pct / 45.0)
    return (
        f"KKL-EMERGENCY: {member_name} [{role} | Ph: {phone}]. "
        f"AI Alert: Inundation {flood_pct:.1f}%, est depth {depth:.1f}m. "
        f"Activate protocol. Helpline: 1077."
    )

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ FLOOD·MAP KKL")
    st.caption("Autonomous Satellite & Drone Flood Mapping · Hyper-Local Alert System")
    page = st.radio(
        "Navigation",
        [
            "🗺️ Command Center",
            "🚁 Live Drone Streams & CV Engine",
            "📡 Satellite Analytics (Sentinel-1 SAR)",
            "📢 Hyper-Local Alert Gateway",
            "👥 First Responder Directory",
            "⚙️ Campus & Regional Settings",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<span class="tag ro">RED ALERT</span><span class="tag am">IMD: Cyclone</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mono" style="font-size:.75rem;color:#94a3b8;margin-top:.6rem">'
        "Karaikal District, U.T. of Puducherry<br>Rainfall (24h): 214 mm<br>Tide: 1.9 m ↑ (HT 18:40)<br>Wind: 62 km/h NE</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("##### 📱 Live Alert Gateway Config")
    telegram_bot_token = st.text_input(
        "Telegram Bot Token (Optional)",
        type="password",
        help="Paste bot token from @BotFather. Leave blank for Simulation Mode.",
    )
    primary_chat_id = st.text_input("Commander Chat ID (Optional)", help="User ID from @userinfobot to receive live push alerts.")
    if primary_chat_id:
        TEAM_MEMBERS[0]["telegram_id"] = primary_chat_id

    st.divider()
    st.caption("Pondicherry University Karaikal Campus · SIH 2026")

# ============================================================================
# PAGE 1: COMMAND CENTER
# ============================================================================
if page.startswith("🗺️"):
    header("Disaster Decision Support · Command Center", "KARAIKAL COASTAL ENCLAVE · PU KARAIKAL CAMPUS · SECTOR KKL-01")
    c1, c2, c3, c4 = st.columns(4)
    online = sum(d["status"] == "ONLINE" for d in DRONES)
    card(c1, "Current Inundation", "34.2%", "▲ 6.8% in last 6h · 41.3 km² flooded", "ro")
    card(c2, "⚠ Est. Max Depth", "1.4 m", "Arasalar river mouth · rising 8 cm/h", "am")
    card(c3, "Active Drone Feeds", f"{online} / {len(DRONES)}", "Drone-04 RTH · low battery", "em")
    card(c4, "Alerts Broadcasted", f"{ss.reached:,}", "residents reached via SMS / Telegram / CB", "sk")
    st.write("")

    m, side = st.columns([3, 1])
    with side:
        st.markdown("##### Map Layers")
        show_flood = st.toggle("Satellite Flood Vector", True)
        show_tracks = st.toggle("Drone Tracks", True)
        show_drones = st.toggle("Active Drone Assets", True)
        show_sos = st.toggle("Emergency Help Requests", True)
        map_engine = st.selectbox("GIS Engine", ["Pydeck 3D Vector", "Folium Leaflet Engine"])
        st.caption("Hover / click markers for tactical metadata.")

        st.markdown("##### Critical Requests")
        for r in HELP_REQUESTS:
            tone = "ro" if r["prio"] == "CRITICAL" else "am"
            st.markdown(
                f'<div class="panel" style="padding:.55rem;margin-bottom:.4rem"><span class="tag {tone}">{r["prio"]}</span>'
                f'<b class="mono" style="font-size:.8rem">{r["id"]}</b><div style="font-size:.78rem;color:#94a3b8">{r["info"]}</div></div>',
                unsafe_allow_html=True,
            )

    with m:
        if map_engine == "Pydeck 3D Vector":
            layers = []
            if show_flood:
                fp = pd.DataFrame(
                    [
                        {
                            **p,
                            "tip": f"<b>{p['name']}</b><br/>Est. depth: {p['depth']} m<br/>Source: Sentinel-1 SAR + Drone CV",
                        }
                        for p in FLOOD_POLYGONS
                    ]
                )
                layers.append(
                    pdk.Layer(
                        "PolygonLayer",
                        fp,
                        get_polygon="polygon",
                        get_fill_color=[56, 189, 248, 90],
                        get_line_color=[56, 189, 248, 220],
                        line_width_min_pixels=2,
                        pickable=True,
                    )
                )
            if show_tracks:
                tracks = []
                for d in DRONES:
                    rng = random.Random(d["id"])
                    path = [[d["lon"] + rng.uniform(-0.012, 0.012), d["lat"] + rng.uniform(-0.012, 0.012)] for _ in range(5)] + [
                        [d["lon"], d["lat"]]
                    ]
                    tracks.append({"path": path, "tip": f"<b>{d['id']} track</b><br/>Last 20 min"})
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        pd.DataFrame(tracks),
                        get_path="path",
                        get_color=[16, 185, 129, 180],
                        width_min_pixels=2,
                        get_width=3,
                        pickable=True,
                    )
                )
            if show_drones:
                dd = pd.DataFrame(
                    [
                        {
                            **d,
                            "color": [16, 185, 129] if d["status"] == "ONLINE" else [100, 116, 139],
                            "tip": f"<b>🚁 {d['id']}</b> · {d['status']}<br/>{d['info']}<br/>Battery: {d['bat']}%",
                        }
                        for d in DRONES
                    ]
                )
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        dd,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius=220,
                        radius_min_pixels=7,
                        stroked=True,
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=1,
                        pickable=True,
                    )
                )
            if show_sos:
                sd = pd.DataFrame(
                    [
                        {
                            **r,
                            "color": [244, 63, 94] if r["prio"] == "CRITICAL" else [245, 158, 11],
                            "tip": f"<b>🆘 {r['id']}</b> · {r['prio']}<br/>{r['info']}",
                        }
                        for r in HELP_REQUESTS
                    ]
                )
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        sd,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius=260,
                        radius_min_pixels=8,
                        pickable=True,
                    )
                )

            campus = pd.DataFrame(
                [
                    {
                        "lat": CAMPUS[0],
                        "lon": CAMPUS[1],
                        "label": "PU KARAIKAL CAMPUS",
                        "tip": "<b>🏫 Pondicherry University Karaikal Campus</b><br/>Shelter capacity: 1,200",
                    },
                    {"lat": KARAIKAL[0], "lon": KARAIKAL[1], "label": "KARAIKAL TOWN", "tip": "<b>Karaikal Town</b><br/>District EOC"},
                ]
            )
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    campus,
                    get_position=["lon", "lat"],
                    get_text="label",
                    get_size=13,
                    get_color=[226, 232, 240],
                    get_pixel_offset=[0, -18],
                    pickable=True,
                )
            )
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    campus,
                    get_position=["lon", "lat"],
                    get_fill_color=[226, 232, 240],
                    get_radius=120,
                    radius_min_pixels=5,
                    pickable=True,
                )
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(latitude=10.937, longitude=79.835, zoom=12.3, pitch=35),
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    tooltip={
                        "html": "{tip}",
                        "style": {
                            "backgroundColor": "#0f172a",
                            "color": "#e2e8f0",
                            "border": "1px solid #334155",
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                        },
                    },
                ),
                height=560,
            )
        else:
            # Folium Fallback Map
            fm = folium.Map(location=[10.937, 79.835], zoom_start=12, tiles="CartoDB dark_matter")
            if show_flood:
                for poly in FLOOD_POLYGONS:
                    folium.Polygon(
                        locations=[[p[1], p[0]] for p in poly["polygon"]],
                        color="#38bdf8",
                        fill=True,
                        fill_color="#38bdf8",
                        fill_opacity=0.35,
                        popup=f"{poly['name']} ({poly['depth']}m)",
                    ).add_to(fm)
            if show_drones:
                for d in DRONES:
                    folium.Marker(
                        [d["lat"], d["lon"]],
                        popup=f"{d['id']}: {d['info']}",
                        icon=folium.Icon(color="green" if d["status"] == "ONLINE" else "gray", icon="plane", prefix="fa"),
                    ).add_to(fm)
            if show_sos:
                for s_item in HELP_REQUESTS:
                    folium.Marker(
                        [s_item["lat"], s_item["lon"]],
                        popup=f"{s_item['id']}: {s_item['info']}",
                        icon=folium.Icon(color="red" if s_item["prio"] == "CRITICAL" else "orange", icon="warning-sign"),
                    ).add_to(fm)
            st_folium(fm, height=560, width="stretch")

    st.markdown("##### Inundation Trend (24h) · Karaikal")
    hrs = pd.date_range(end=datetime.now(), periods=24, freq="h")
    trend = pd.DataFrame(
        {
            "Inundation %": np.clip(np.cumsum(np.random.default_rng(1).normal(1.2, 0.8, 24)) + 6, 0, None)[-24:] * 34.2 / 34.2,
            "Max depth (dm)": np.linspace(3, 14, 24) + np.random.default_rng(2).normal(0, 0.4, 24),
        },
        index=hrs,
    )
    st.area_chart(trend, height=200, color=["#38bdf8", "#f59e0b"])

# ============================================================================
# PAGE 2: LIVE DRONE STREAMS & COMPUTER VISION ENGINE
# ============================================================================
elif page.startswith("🚁"):
    header("Live Drone Streams & Computer Vision Engine", "EDGE INFERENCE · HSV SEGMENTATION + MORPHOLOGY · 4 UAV FLEET")
    tab_sim, tab_real = st.tabs(["🚁 Fleet Simulation & Multi-UAV Grid", "📡 Real-Time Imagery & Video Upload Ingestion"])

    with tab_sim:
        grid, params = st.columns([3, 1])
        with params:
            st.markdown("##### CV Parameters")
            hue = st.slider("HSV Hue range (water)", 0, 180, (85, 140))
            thresh = st.slider("Threshold Value", 0, 255, 140)
            kernel = st.select_slider("Kernel Size", options=[3, 5, 7, 9, 11, 13], value=7)
            m_open = st.toggle("Morphological Opening", True)
            m_close = st.toggle("Morphological Closing", True)
            st.toggle("Gaussian Pre-blur", True)
            st.toggle("Depth Estimation (Stereo+DEM)", True)
            st.selectbox("Colour space", ["HSV", "LAB", "YCrCb"])
            st.caption("Adjustments re-render the CV overlay on all live feeds.")

        with grid:
            cols = st.columns(2)
            stats = []
            for i, d in enumerate(DRONES):
                with cols[i % 2]:
                    if d["status"] == "ONLINE":
                        gif, acc, depth, cover = drone_gif(i + 1, kernel, thresh, hue[0], hue[1], m_open, m_close, d["id"])
                        st.image(gif, width="stretch")
                        st.markdown(
                            f'<span class="tag em">ONLINE</span><span class="mono" style="font-size:.75rem;color:#94a3b8">{d["info"]}</span>',
                            unsafe_allow_html=True,
                        )
                        stats.append((d["id"], acc, depth, cover))
                    else:
                        st.image(offline_frame(d["id"]), width="stretch")
                        st.markdown(
                            f'<span class="tag ro">OFFLINE</span><span class="mono" style="font-size:.75rem;color:#94a3b8">{d["info"]}</span>',
                            unsafe_allow_html=True,
                        )
            st.write("")
            a, b, c, e = st.columns(4)
            a.metric("Mean Accuracy", f"{np.mean([s[1] for s in stats]):.1f}%")
            b.metric("Water Coverage", f"{np.mean([s[3] for s in stats]):.1f}%")
            c.metric("Inference", f"{18 + kernel * 1.6:.0f} ms/frame")
            e.metric("Edge GPU", "Jetson Orin · 71°C")

    with tab_real:
        st.subheader("Interactive OpenCV Live Water Ingestion Engine")
        source = st.radio(
            "Select Stream / Feed Mode:",
            ["🔴 Drone Preset Telemetry (RTSP / Aerial Link)", "📁 Upload Drone Snapshot / SD Image"],
            horizontal=True,
        )

        real_img = None
        if source == "🔴 Drone Preset Telemetry (RTSP / Aerial Link)":
            stream_preset = st.selectbox(
                "Select Active Aerial Drone Patrol Feed:",
                [
                    "Patrol 1: Arasalar River Inundation Zone",
                    "Patrol 2: Beach Road / Residential Sector",
                    "Patrol 3: Nagore Border Highway (Clear Sector)",
                ],
            )
            sample_urls = {
                "Patrol 1: Arasalar River Inundation Zone": "https://images.unsplash.com/photo-1547683905-f686c993aae5?w=800&q=80",
                "Patrol 2: Beach Road / Residential Sector": "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800&q=80",
                "Patrol 3: Nagore Border Highway (Clear Sector)": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&q=80",
            }
            try:
                resp = requests.get(sample_urls[stream_preset], timeout=6)
                real_img = Image.open(BytesIO(resp.content))
            except Exception:
                st.warning("Sample feed network timeout. Switching to local synth buffer.")
        else:
            up_file = st.file_uploader("Upload Snapshot from Drone Camera", type=["jpg", "png", "jpeg"])
            if up_file:
                real_img = Image.open(up_file)

        if real_img is not None:
            c_in, c_cv = st.columns(2)
            with c_in:
                st.markdown("##### Ingested Raw Frame")
                st.image(real_img, width="stretch")
            with c_cv:
                st.markdown("##### OpenCV Real-Time Segmentation")
                mask, overlay, inun_pct = analyze_flood_imagery(real_img)
                st.image(overlay, width="stretch")

            m1, m2, m3 = st.columns(3)
            m1.metric("Calculated Inundation %", f"{inun_pct:.2f}%")
            m2.metric("Estimated Max Water Column", f"{0.3 + (inun_pct / 45.0):.2f} m")
            status_txt = "CRITICAL INUNDATION" if inun_pct > 25.0 else ("MODERATE FLOODING" if inun_pct > 8.0 else "CLEAR / LOW HAZARD")
            m3.metric("Status Flag", status_txt)

# ============================================================================
# PAGE 3: SATELLITE ANALYTICS
# ============================================================================
elif page.startswith("📡"):
    header("Satellite Analytics · Sentinel-1 SAR", "C-BAND SAR PIERCES CLOUD COVER · ESA COPERNICUS · KARAIKAL AOI")
    mode = st.radio("SAR rendering", ["False-colour water extraction", "Grayscale backscatter (VV)"], horizontal=True)
    date = st.select_slider("Acquisition timeline", options=DATES, value="Oct 8, 2026")
    level = DATE_LEVEL[date]
    o, s = st.columns(2)
    with o:
        st.markdown("##### ☁️ Optical View · Sentinel-2")
        st.image(optical_gif(level), width="stretch")
        st.caption("Dense cyclonic cloud deck — optical sensors are completely blind during monsoon events.")
    with s:
        st.markdown("##### 📡 Sentinel-1 SAR Radar View")
        img, flooded = sar_image(level, mode)
        st.image(img, width="stretch")
        st.caption("Low radar backscatter = specular water reflection. Red boundary = extracted flood zone.")
    a, b, c, d = st.columns(4)
    a.metric("Flooded area (AOI)", f"{flooded:.1f}%", f"{(level - 0.62) * 40:+.1f}% vs Oct 6")
    b.metric("Orbit", "S1A · Desc. #156")
    c.metric("Polarisation", "VV + VH")
    d.metric("Pass time", f"{date.split(',')[0]} · 06:12 IST")
    st.markdown("##### Flood extent across timeline")
    st.bar_chart(pd.DataFrame({"Flooded %": [sar_image(DATE_LEVEL[x], mode)[1] for x in DATES]}, index=DATES), color="#38bdf8", height=220)

# ============================================================================
# PAGE 4: HYPER-LOCAL ALERT GATEWAY
# ============================================================================
elif page.startswith("📢"):
    header("Hyper-Local Alert Gateway", "LAST-MILE COMMUNICATION · GEOFENCED CELL BROADCAST · TELEGRAM BOT API")
    left, mid, right = st.columns([1.1, 1.6, 1])
    with left:
        st.markdown("##### Target Zones")
        sel = []
        for z, pop in ZONES.items():
            if st.checkbox(f"{z}", value=z.startswith(("Zone C", "Zone D")), key=z):
                sel.append(z)
            st.caption(f"   ↳ {pop:,} registered devices")
        total = sum(ZONES[z] for z in sel)
        st.metric("Recipients in geofence", f"{total:,}")
    with mid:
        st.markdown("##### Message Composer")
        tpl = st.selectbox("Quick emergency template", list(TEMPLATES), index=1)
        msg = st.text_area(
            "Alert message",
            value=TEMPLATES[tpl],
            height=170,
            max_chars=480,
            placeholder="Type alert in English / தமிழ் / Français…",
        )
        lang = st.multiselect("Languages", ["English", "தமிழ் (Tamil)", "Français", "తెలుగు"], default=["English", "தமிழ் (Tamil)"])
        sev = st.select_slider("Severity", ["Advisory", "Watch", "Warning", "Extreme"], value="Warning")
    with right:
        st.markdown("##### Channels")
        tg = st.toggle("Telegram API Gateway", True)
        cb = st.toggle("C-DOT Cell Broadcast System", True)
        sms = st.toggle("SMS Fallback (NIC)", False)
        siren = st.toggle("Campus PA / Sirens", False)
        st.markdown(
            f'<div class="panel mono" style="font-size:.75rem;margin-top:.5rem">'
            f"CB Towers in fence: {max(1, len(sel)*3)}<br>"
            f"Telegram subs: {int(total*0.42):,}<br>"
            f"Est. latency: &lt; 9 s</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="broadcast">', unsafe_allow_html=True)
    go = st.button("📢  BROADCAST HYPER-LOCAL ALERT", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    if go:
        chans = [n for n, on in [("C-DOT CB", cb), ("Telegram", tg), ("SMS", sms), ("PA/Siren", siren)] if on]
        if not sel:
            st.error("Select at least one target zone.")
        elif not msg.strip():
            st.error("Message cannot be empty.")
        elif not chans:
            st.error("Enable at least one distribution channel.")
        else:
            bar = st.progress(0, text="Sending… authenticating with C-DOT CBC")
            steps = [
                "Authenticating with C-DOT CBC",
                "Resolving geofence → cell IDs",
                "Pushing Telegram bot payloads",
                "Transmitting cell broadcast (CMAS ch. 4370)",
                "Awaiting delivery receipts",
            ]
            for k in range(100):
                time.sleep(0.015)
                bar.progress(k + 1, text=f"Sending… {steps[min(k // 20, 4)]}")
            bar.empty()

            # Optional live dispatch to Commander if configured
            if tg and telegram_bot_token:
                dispatch_team_telegram_alerts(telegram_bot_token, TEAM_MEMBERS, 34.2)

            ss.reached += total
            ss.history.insert(
                0,
                {
                    "Time (IST)": datetime.now().strftime("%H:%M:%S"),
                    "Zones": ", ".join(z.split("·")[0].strip() for z in sel),
                    "Channels": ", ".join(chans),
                    "Recipients": total,
                    "Message": (msg[:70] + "…") if len(msg) > 70 else msg,
                    "Status": "DELIVERED",
                },
            )
            st.success(f"Success: {total:,} Cell Broadcasts Triggered · {sev} · {', '.join(lang) or 'English'}")
            st.toast(f"✅ {total:,} devices alerted across {len(sel)} zone(s)", icon="📡")

    st.markdown("##### Recent Broadcast History")
    st.dataframe(
        pd.DataFrame(ss.history),
        width="stretch",
        hide_index=True,
        column_config={"Recipients": st.column_config.NumberColumn(format="%d")},
    )

# ============================================================================
# PAGE 5: FIRST RESPONDER DIRECTORY & LIVE MOBILE PUSH
# ============================================================================
elif page.startswith("👥"):
    header("First Responder Directory & Field Team Dispatch", "NDRF / SDMA / PU FIRST RESPONDER UNIT (KARAIKAL SECTOR)")

    fcol1, fcol2 = st.columns([1.2, 1])
    with fcol1:
        st.subheader("Field Officers Roster")
        for m_idx, member in enumerate(TEAM_MEMBERS, 1):
            st.markdown(
                f"""
            <div class="panel" style="margin-bottom:.8rem">
                <b>{m_idx}. {member['name']}</b> — <code>{member['role']}</code><br>
                📞 Phone: <code>{member['phone']}</code> | 📍 Station: <code>{member['sector']}</code><br>
                🌐 Coordinates: {member['lat']}° N, {member['lon']}° E
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.subheader("Direct Telegram Field Alert Trigger")
        test_inundation = st.slider("Simulated Inundation Level for Alert (%)", 5.0, 95.0, 42.0)

        if st.button("🚨 Broadcast to Officer Devices", type="primary"):
            res = dispatch_team_telegram_alerts(telegram_bot_token, TEAM_MEMBERS, test_inundation)
            if res.get("status") == "success" and res.get("count", 0) > 0:
                st.success(f"✅ Alert pushed via Telegram API to {res['count']} field officer device(s)!")
            else:
                st.info("ℹ️ Live Radio Simulation Mode active. Simulated radio packet cards:")

            for member in TEAM_MEMBERS:
                card_text = build_emergency_card(member["name"], test_inundation, member["role"], member["phone"])
                st.success(f"Dispatched ➡️ **{member['name']}** ({member['phone']})")
                st.code(card_text)

    with fcol2:
        st.subheader("Field Units Deployment Map")
        field_map = folium.Map(location=[10.936, 79.83], zoom_start=13, tiles="CartoDB dark_matter")
        for mem in TEAM_MEMBERS:
            folium.Marker(
                location=[mem["lat"], mem["lon"]],
                popup=f"{mem['name']} ({mem['role']}) - Ph: {mem['phone']}",
                icon=folium.Icon(color="orange", icon="user", prefix="fa"),
            ).add_to(field_map)
        folium.Marker(
            [10.9265, 79.8350],
            popup="Karaikal General Hospital (District Emergency HQ)",
            icon=folium.Icon(color="red", icon="plus"),
        ).add_to(field_map)
        st_folium(field_map, height=450, width="stretch")

# ============================================================================
# PAGE 6: SETTINGS
# ============================================================================
else:
    header("Campus & Regional Settings", "PONDICHERRY UNIVERSITY KARAIKAL CAMPUS · KARAIKAL DISTRICT EOC")
    a, b = st.columns(2)
    with a:
        st.markdown("##### Campus Profile")
        st.text_input("Institution", "Pondicherry University – Karaikal Campus")
        st.text_input("Campus coordinates", f"{CAMPUS[0]}° N, {CAMPUS[1]}° E")
        st.number_input("Designated shelter capacity", value=1200, step=50)
        st.text_input("Campus emergency coordinator", "Dr. — (Director, Karaikal Campus)")
        st.text_input("District helpline", "1077 · Karaikal Collectorate")
    with b:
        st.markdown("##### Regional Thresholds")
        st.slider("Inundation alert trigger (%)", 0, 100, 25)
        st.slider("Depth warning (m)", 0.0, 3.0, 0.8, 0.1)
        st.slider("Auto-broadcast surge level (m)", 0.0, 3.0, 1.5, 0.1)
        st.multiselect(
            "Monitored regions",
            list(ZONES) + ["Thirunallar", "Neravy", "T.R. Pattinam", "Nedungadu"],
            default=list(ZONES)[:4],
        )
        st.selectbox("SAR revisit source", ["Sentinel-1A/1C (ESA)", "EOS-04 RISAT-1A (ISRO)", "NISAR (NASA-ISRO)"])
    st.toggle("Fully autonomous mode (auto-dispatch drones & alerts above thresholds)", False)
    if st.button("💾 Save configuration"):
        st.success("Configuration saved for Karaikal sector KKL-01.")