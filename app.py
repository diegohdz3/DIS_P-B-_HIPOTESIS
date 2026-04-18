import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import google.generativeai as genai
import math

# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Z·STAT 296 · Race Analytics",
    layout="wide",
    page_icon="🏎",
    initial_sidebar_state="collapsed"
)

# ── MATPLOTLIB THEME ─────────────────────────────────────────
plt.rcParams.update({
    'axes.facecolor':    '#0A0A0A',
    'figure.facecolor':  '#080808',
    'text.color':        '#F0EDE8',
    'axes.labelcolor':   '#4A4040',
    'xtick.color':       '#4A4040',
    'ytick.color':       '#4A4040',
    'grid.color':        '#181414',
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'axes.edgecolor':    '#2A2020',
})

# ── FERRARI 296 GT3 PALETTE ──────────────────────────────────
C_ROSSO   = '#DC143C'   # Rosso Corsa — el rojo Ferrari
C_ROSSO2  = '#FF1744'   # Rojo vivo para highlights
C_ROSSO3  = '#FF6B6B'   # Rojo claro / accent suave
C_CARBON  = '#0D0D0D'   # Fibra de carbono negro
C_TITANIO = '#B8B4AE'   # Titanio / aluminio
C_CREAM   = '#F0EDE8'   # Crema de interiores
C_IVORY   = '#D4CFC8'   # Marfil
C_PITCH   = '#080808'   # Negro absoluto
C_EXHAUST = '#1A1614'   # Gris escape
C_KEVLAR  = '#2A2420'   # Kevlar oscuro
C_SCUDERIA= '#FF4B1F'   # Naranja Scuderia Ferrari
C_DIM     = '#2A2020'
C_MID     = '#3A3030'

def ax_ferrari(ax):
    ax.set_facecolor('#0A0A0A')
    ax.spines['bottom'].set_color('#2A1A1A')
    ax.spines['left'].set_color('#2A1A1A')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.tick_params(colors='#4A3030', length=3, pad=5)
    ax.grid(True, axis='y', color='#181010', lw=0.5, zorder=0)

# ── SESSION STATE ─────────────────────────────────────────────
defaults = dict(
    screen=0,
    datos=None, nombre_variable=None,
    z_stat=None, p_value=None, decision=None,
    mu0=0.0, sigma=1.0, alpha=0.05,
    tipo_cola="Bilateral (≠)",
    _z_c_inf=None, _z_c_sup=None,
    _n_z=None, _xbar=None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

SCREENS  = ['GRID', 'DATOS', 'TELEMETRÍA', 'PRUEBA·Z', 'RADIO·IA']
MAX_SCR  = 4

def go(n):
    st.session_state.screen = max(0, min(n, MAX_SCR))
    st.rerun()

# ── SVG TACHÓMETRO ───────────────────────────────────────────
def build_tacho(pct, color, label, val_str):
    """Semicircular tachometer SVG — Ferrari instrument cluster style"""
    cx, cy, r = 60, 62, 48
    def pt(deg):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)
    # Arc: 210° to -30° (240° sweep)
    bsx, bsy = pt(210); bex, bey = pt(-30)
    v_deg = 210 - (min(max(pct,0), 100) / 100) * 240
    vex, vey = pt(v_deg)
    # Segments: 16 ticks
    ticks_html = ''
    for i in range(16):
        a   = 210 - i * (240/15)
        rad = math.radians(a)
        is_active = (i / 15) <= (pct / 100)
        r_in  = r - 5
        r_out = r + 1
        x1 = cx + r_in  * math.cos(rad)
        y1 = cy - r_in  * math.sin(rad)
        x2 = cx + r_out * math.cos(rad)
        y2 = cy - r_out * math.sin(rad)
        seg_col = color if is_active else '#1E1818'
        ticks_html += '<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="3" stroke-linecap="round"/>' % (x1, y1, x2, y2, seg_col)
    # Background arc
    bg  = 'M %.2f %.2f A %d %d 0 1 0 %.2f %.2f' % (bsx, bsy, r, r, bex, bey)
    val = 'M %.2f %.2f A %d %d 0 1 0 %.2f %.2f' % (bsx, bsy, r, r, vex, vey)
    return (
        '<svg width="120" height="80" viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">'
        '<path d="%s" fill="none" stroke="#161010" stroke-width="4" stroke-linecap="round"/>'
        '<path d="%s" fill="none" stroke="%s" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/>'
        '<defs><filter id="glow"><feGaussianBlur stdDeviation="2" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        '%s'
        '<text x="60" y="56" text-anchor="middle" font-family="serif" font-size="20" font-weight="bold" fill="%s">%s</text>'
        '<text x="60" y="70" text-anchor="middle" font-family="monospace" font-size="7" fill="#FFFFFF" letter-spacing="2">%s</text>'
        '</svg>'
    ) % (bg, val, color, ticks_html, color, val_str, label.upper())

def build_rev_lights(scr, max_scr=4):
    """Ferrari-style F1 rev lights bar across top"""
    total  = 14
    filled = int((scr / max_scr) * total)
    lights = []
    for i in range(total):
        if i < filled:
            if filled >= 12:
                col, glow = '#DC143C', '#DC143C'
            elif filled >= 8:
                col, glow = '#FF8C00', '#FF8C00'
            else:
                col, glow = '#22DD44', '#22DD44'
            lights.append(
                '<rect x="%d" y="4" width="14" height="8" rx="2" fill="%s" '
                'filter="url(#lg)" opacity="0.95"/>' % (i * 17 + 2, col)
            )
        else:
            lights.append(
                '<rect x="%d" y="4" width="14" height="8" rx="2" fill="#141010" opacity="0.6"/>' % (i * 17 + 2)
            )
    inner = ''.join(lights)
    return (
        '<svg width="240" height="16" viewBox="0 0 240 16" xmlns="http://www.w3.org/2000/svg">'
        '<defs><filter id="lg"><feGaussianBlur stdDeviation="1.5" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        + inner + '</svg>'
    )


# ═══════════════════════════════════════════════════════════════
# CSS — Ferrari 296 GT3 · Full redesign
# ═══════════════════════════════════════════════════════════════
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

:root {
  --rosso:      #DC143C;
  --rosso2:     #FF1744;
  --rosso-dim:  #8B0A24;
  --rosso-dark: #3D0410;
  --rosso-glow: rgba(220,20,60,0.35);
  --carbon:     #0D0D0D;
  --pitch:      #080808;
  --exhaust:    #141212;
  --kevlar:     #1E1A18;
  --mid:        #2A2420;
  --titanio:    #B8B4AE;
  --cream:      #F0EDE8;
  --ivory:      #D4CFC8;
  --scuderia:   #FF4B1F;
  --yellow:     #FFD700;
}

html, body, [class*="css"], .stApp {
  font-family: 'Rajdhani', sans-serif !important;
  background: var(--pitch) !important;
  color: var(--cream) !important;
}
#MainMenu, footer, header { visibility: hidden; }
* { box-sizing: border-box; }

/* === BACKGROUND — Carbon weave + red heat === */
.stApp {
  background-color: var(--pitch) !important;
  background-image:
    radial-gradient(ellipse 80% 40% at 50% 0%, rgba(220,20,60,0.08) 0%, transparent 60%),
    repeating-linear-gradient(
      45deg,
      transparent 0px, transparent 3px,
      rgba(255,255,255,0.012) 3px, rgba(255,255,255,0.012) 4px
    ),
    repeating-linear-gradient(
      -45deg,
      transparent 0px, transparent 3px,
      rgba(255,255,255,0.008) 3px, rgba(255,255,255,0.008) 4px
    ) !important;
}
.stApp::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background: radial-gradient(ellipse 100% 100% at 50% 100%,
    rgba(0,0,0,0.7) 0%, transparent 60%);
}

/* === LAYOUT RESET === */
.main .block-container { padding: 0 !important; max-width: 100% !important; }
.block-container > div:first-child { gap: 0 !important; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stHorizontalBlock"] { padding: 0 !important; }
div[data-testid="column"] { padding: 0 10px !important; }
div[data-testid="column"]:first-child { padding-left: 0 !important; }
div[data-testid="column"]:last-child  { padding-right: 0 !important; }
.element-container { margin-bottom: 0 !important; }
.stButton { margin: 0 !important; padding: 0 !important; }
.main > div:first-child { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] > section > div { padding: 0 !important; }
.stMarkdown { margin: 0 !important; }
[data-testid="stMarkdownContainer"] { margin: 0 !important; }

/* === HEADER === */
.f296-header {
  position: relative;
  background: linear-gradient(180deg, #050303 0%, var(--exhaust) 100%);
  border-bottom: 1px solid rgba(220,20,60,0.2);
  overflow: hidden;
}
/* Top red racing stripe */
.f296-header::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg,
    transparent 0%,
    var(--rosso-dim) 8%,
    var(--rosso) 30%,
    var(--rosso2) 50%,
    var(--rosso) 70%,
    var(--rosso-dim) 92%,
    transparent 100%);
  box-shadow: 0 0 20px var(--rosso-glow), 0 0 40px rgba(220,20,60,0.15);
  animation: stripeFlare 3s ease-in-out infinite;
}
@keyframes stripeFlare {
  0%,100% { opacity: 1; }
  50%     { opacity: 0.7; box-shadow: 0 0 40px var(--rosso-glow), 0 0 80px rgba(220,20,60,0.2); }
}
.header-grid {
  display: grid;
  grid-template-columns: 280px 1fr 280px;
  align-items: center;
  padding: 16px 44px 12px;
  position: relative; z-index: 2;
}

/* Left: Instrument cluster */
.h-instruments {
  display: flex; align-items: center; gap: 10px;
}
.inst-stack { display: flex; flex-direction: column; gap: 2px; }
.inst-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 8px; letter-spacing: 3px;
  color: #FFFFFF; text-transform: uppercase;
}
.inst-value {
  font-family: 'Orbitron', sans-serif;
  font-size: 11px; font-weight: 700;
  letter-spacing: 1px;
}

/* Center: Scuderia branding */
.h-center { text-align: center; }
.prancing-horse {
  font-family: 'Orbitron', sans-serif;
  font-size: 9px; letter-spacing: 6px;
  color: var(--rosso-dim); text-transform: uppercase;
  margin-bottom: 4px;
}
.logo-z {
  font-family: 'Orbitron', sans-serif;
  font-size: 38px; font-weight: 900;
  letter-spacing: -2px; line-height: 1;
  background: linear-gradient(135deg, var(--cream) 0%, var(--titanio) 40%, var(--cream) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  filter: drop-shadow(0 0 20px rgba(220,20,60,0.3));
}
.logo-z .accent { 
  background: linear-gradient(135deg, var(--rosso2) 0%, var(--rosso) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.logo-subtitle {
  font-family: 'Share Tech Mono', monospace;
  font-size: 8px; letter-spacing: 4px;
  color: #FFFFFF; text-transform: uppercase; margin-top: 6px;
}
.rev-lights-wrap { margin-top: 10px; display: flex; justify-content: center; }

/* Right: Driver info */
.h-right { display: flex; align-items: center; justify-content: flex-end; gap: 16px; }
.driver-plate {
  text-align: right;
}
.dp-series {
  font-family: 'Share Tech Mono', monospace; font-size: 8px;
  letter-spacing: 3px; color: var(--rosso-dim); text-transform: uppercase;
}
.dp-name {
  font-family: 'Orbitron', sans-serif; font-size: 20px; font-weight: 700;
  letter-spacing: 1px; line-height: 1.1; color: var(--cream);
}
.dp-team {
  font-family: 'Share Tech Mono', monospace; font-size: 8px;
  letter-spacing: 2px; color: #FFFFFF; margin-top: 2px;
}

/* Bottom red line */
.header-underline {
  height: 2px;
  background: linear-gradient(90deg,
    transparent 0%, #3D0410 10%,
    var(--rosso) 35%, var(--rosso2) 50%,
    var(--rosso) 65%, #3D0410 90%, transparent 100%);
  animation: lineIn 1s cubic-bezier(.16,1,.3,1) both;
}
@keyframes lineIn { from{transform:scaleX(0)} to{transform:scaleX(1)} }

/* === NAV BAR === */
.nav-bar {
  background: var(--exhaust);
  border-bottom: 1px solid rgba(220,20,60,0.1);
  display: flex; align-items: stretch;
  position: relative;
}

/* === PADDLE BUTTONS === */
div[data-testid="stHorizontalBlock"].paddle-row {
  background: var(--exhaust);
  border-bottom: 1px solid rgba(220,20,60,0.10);
  padding: 0 !important; margin: 0 !important;
  align-items: stretch !important; min-height: 52px;
}
div[data-testid="stHorizontalBlock"].paddle-row
  > div[data-testid="column"]:first-child {
  padding: 6px 0 6px 28px !important;
  display: flex; align-items: center;
}
div[data-testid="stHorizontalBlock"].paddle-row
  > div[data-testid="column"]:last-child {
  padding: 6px 28px 6px 0 !important;
  display: flex; align-items: center; justify-content: flex-end;
}
div[data-testid="stHorizontalBlock"].paddle-row .stButton > button {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 16px !important; font-weight: 700 !important;
  letter-spacing: 2px !important; line-height: 1 !important;
  background: var(--kevlar) !important;
  color: var(--rosso) !important;
  border: 1px solid rgba(220,20,60,0.3) !important;
  border-radius: 2px !important;
  padding: 12px 32px 10px !important;
  min-width: 90px !important;
  transition: all 0.18s ease !important;
  clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%) !important;
}
div[data-testid="stHorizontalBlock"].paddle-row .stButton > button:hover {
  background: var(--rosso) !important;
  color: var(--pitch) !important;
  box-shadow: 0 0 30px var(--rosso-glow), inset 0 1px 0 rgba(255,255,255,0.1) !important;
  transform: translateY(-1px) !important;
}

/* === SECTION DOT INDICATORS === */
.seg-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #1E1818; border: 1px solid rgba(220,20,60,0.12);
  transition: all 0.3s cubic-bezier(.16,1,.3,1);
}
.seg-dot.done {
  background: rgba(220,20,60,0.25);
  border-color: rgba(220,20,60,0.4);
}
.seg-dot.active {
  background: var(--rosso);
  border-color: var(--rosso2);
  box-shadow: 0 0 10px var(--rosso-glow), 0 0 20px rgba(220,20,60,0.2);
  transform: scale(1.5);
}

/* === DISPLAY WRAP === */
.display-wrap {
  padding: 32px 48px 40px;
  animation: screenIn 0.4s cubic-bezier(.16,1,.3,1) both;
  position: relative; z-index: 1;
}
@keyframes screenIn {
  from { opacity:0; transform: translateY(16px); }
  to   { opacity:1; transform: translateY(0); }
}

/* === SECTION TITLE === */
.sect-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px; letter-spacing: 5px;
  color: #FFFFFF; text-transform: uppercase;
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 28px;
}
.sect-title::before {
  content: ''; display: block;
  width: 3px; height: 20px;
  background: linear-gradient(180deg, var(--rosso2), var(--rosso-dim));
  border-radius: 2px;
  box-shadow: 0 0 8px var(--rosso-glow);
}
.sect-title::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(220,20,60,0.25), transparent);
}

/* === WELCOME SCREEN === */
.welcome-arena {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center; padding: 20px 20px 30px;
}
.prancing-emblem {
  width: 160px; height: 160px;
  border-radius: 4px;
  border: 1px solid rgba(220,20,60,0.15);
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 32px;
  position: relative;
  background: linear-gradient(135deg, #0F0A0A 0%, #1A0A0A 100%);
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  animation: emblemPulse 3s ease-in-out infinite;
}
@keyframes emblemPulse {
  0%,100% { filter: drop-shadow(0 0 0px transparent); }
  50%     { filter: drop-shadow(0 0 25px var(--rosso-glow)); }
}
.emblem-inner {
  font-family: 'Orbitron', sans-serif;
  font-size: 44px; font-weight: 900;
  line-height: 0.85; text-align: center;
}
.emblem-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 7px; letter-spacing: 5px;
  color: var(--rosso-dim); text-transform: uppercase;
  margin-top: 8px;
}
.welcome-tag {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px; letter-spacing: 6px;
  color: var(--rosso-dim); text-transform: uppercase; margin-bottom: 14px;
}
.welcome-headline {
  font-family: 'Orbitron', sans-serif;
  font-size: 42px; font-weight: 900; letter-spacing: -1px;
  line-height: 1; margin-bottom: 16px;
  background: linear-gradient(135deg, var(--cream) 0%, var(--titanio) 60%, var(--cream) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.welcome-desc {
  font-family: 'Rajdhani', sans-serif; font-size: 15px;
  font-weight: 300; line-height: 1.9;
  color: #4A3838; max-width: 500px; margin: 0 auto 36px;
}
.stage-grid {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; max-width: 520px; margin: 0 auto 36px;
  background: rgba(220,20,60,0.08); border-radius: 3px; overflow: hidden;
}
.stage-cell {
  background: var(--kevlar); padding: 18px 14px;
  text-align: center; position: relative;
  transition: background 0.25s ease;
}
.stage-cell::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--rosso-dim); opacity: 0.5;
  transition: opacity 0.25s, background 0.25s;
}
.stage-cell:hover { background: #221818; }
.stage-cell:hover::before { background: var(--rosso); opacity: 1; }
.stage-num {
  font-family: 'Orbitron', sans-serif; font-size: 32px; font-weight: 900;
  color: var(--rosso); line-height: 1;
}
.stage-lbl {
  font-family: 'Share Tech Mono', monospace; font-size: 8px;
  letter-spacing: 2px; color: #FFFFFF; text-transform: uppercase; margin-top: 6px;
}
.ignition-hint {
  font-family: 'Orbitron', sans-serif; font-size: 9px;
  letter-spacing: 5px; color: #2A1818; text-transform: uppercase;
  animation: ignBlink 2s ease-in-out infinite;
}
@keyframes ignBlink { 0%,100%{opacity:1;color:var(--rosso-dim)} 50%{opacity:0.2;color:#2A1818} }

/* === CARDS === */
.f296-card {
  background: var(--kevlar);
  border: 1px solid rgba(255,255,255,0.04);
  border-top: 2px solid var(--rosso-dim);
  border-radius: 2px; padding: 20px 22px;
  position: relative; overflow: hidden;
  transition: all 0.22s cubic-bezier(.16,1,.3,1);
  clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
}
.f296-card::before {
  content: ''; position: absolute;
  top: 0; right: 0; width: 12px; height: 12px;
  background: linear-gradient(225deg, rgba(220,20,60,0.15) 0%, transparent 100%);
}
.f296-card:hover {
  border-top-color: var(--rosso);
  transform: translateY(-3px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(220,20,60,0.1);
}
.card-label {
  font-family: 'Share Tech Mono', monospace; font-size: 9px;
  letter-spacing: 3px; color: #FFFFFF; text-transform: uppercase; margin-bottom: 12px;
}
.card-value {
  font-family: 'Orbitron', sans-serif; font-size: 44px;
  font-weight: 900; letter-spacing: -2px; line-height: 0.9;
}
.card-desc {
  font-family: 'Rajdhani', sans-serif; font-size: 13px;
  color: #FFFFFF; margin-top: 10px; line-height: 1.5; font-weight: 400;
}
.card-bar { height: 2px; background: #141010; border-radius: 1px; margin-top: 16px; overflow: hidden; }
.card-fill { height: 100%; border-radius: 1px; }

/* === METRIC STRIP === */
.metric-strip {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: rgba(220,20,60,0.06);
  border-radius: 2px; overflow: hidden; margin-bottom: 16px;
}
.ms-cell {
  background: var(--kevlar); padding: 16px 18px; position: relative;
}
.ms-cell::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.ms-rosso::before  { background: var(--rosso); box-shadow: 0 0 8px var(--rosso-glow); }
.ms-white::before  { background: var(--titanio); }
.ms-orange::before { background: var(--scuderia); }
.ms-yellow::before { background: var(--yellow); }
.ms-lbl {
  font-family: 'Share Tech Mono', monospace; font-size: 8px;
  letter-spacing: 3px; color: #FFFFFF; text-transform: uppercase; margin-bottom: 8px;
}
.ms-val {
  font-family: 'Orbitron', sans-serif; font-size: 34px;
  font-weight: 900; letter-spacing: -1px; line-height: 1;
}
.ms-sub {
  font-family: 'Rajdhani', sans-serif; font-size: 11px;
  color: #FFFFFF; margin-top: 4px;
}

/* === INFO PANEL === */
.info-panel {
  background: var(--kevlar);
  border: 1px solid rgba(255,255,255,0.04);
  border-left: 3px solid var(--rosso);
  border-radius: 2px; padding: 16px 20px;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
}
.ip-label {
  font-family: 'Share Tech Mono', monospace; font-size: 9px;
  letter-spacing: 3px; color: var(--rosso); text-transform: uppercase; margin-bottom: 10px;
}
.ip-body {
  font-family: 'Rajdhani', sans-serif; font-size: 14px;
  color: #4A3838; line-height: 1.75; font-weight: 400;
}

/* === STATUS ROW === */
.status-row {
  display: flex; align-items: center; gap: 12px;
  padding: 11px 0; border-bottom: 1px solid rgba(255,255,255,0.03);
}
.s-indicator {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.s-label {
  font-family: 'Share Tech Mono', monospace; font-size: 9px;
  letter-spacing: 2px; text-transform: uppercase;
}

/* === Z HERO PANEL === */
.z-hero {
  background: linear-gradient(135deg, #120508 0%, #1A080C 100%);
  border: 1px solid rgba(220,20,60,0.25);
  border-top: 3px solid var(--rosso);
  border-radius: 2px; padding: 28px 24px; text-align: center;
  position: relative; overflow: hidden;
  clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 0 100%);
}
.z-hero::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 120% 60% at 50% 0%,
    rgba(220,20,60,0.12) 0%, transparent 65%);
}
.zh-tag {
  font-family: 'Share Tech Mono', monospace; font-size: 8px;
  letter-spacing: 5px; color: var(--rosso-dim);
  text-transform: uppercase; margin-bottom: 12px;
  position: relative;
}
.zh-number {
  font-family: 'Orbitron', sans-serif; font-size: 72px;
  font-weight: 900; letter-spacing: -4px; line-height: 0.9;
  position: relative;
}
.zh-decision {
  display: inline-block;
  font-family: 'Orbitron', sans-serif; font-size: 12px;
  font-weight: 700; letter-spacing: 3px;
  padding: 7px 20px; border-radius: 2px;
  margin-top: 12px; position: relative;
  clip-path: polygon(8px 0, 100% 0, calc(100% - 8px) 100%, 0 100%);
}
.zh-stats {
  font-family: 'Share Tech Mono', monospace; font-size: 9px;
  color: var(--mid); line-height: 2; margin-top: 16px;
  padding-top: 14px; border-top: 1px solid rgba(220,20,60,0.1);
  position: relative;
}

/* === DECISION BANNERS === */
.banner-rechaza {
  background: linear-gradient(135deg, rgba(220,20,60,0.12), rgba(220,20,60,0.05));
  border: 1px solid rgba(220,20,60,0.3);
  border-left: 4px solid var(--rosso);
  border-radius: 2px; padding: 18px 22px; margin-top: 16px;
  clip-path: polygon(0 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%);
}
.banner-ok {
  background: linear-gradient(135deg, rgba(34,200,68,0.10), rgba(34,200,68,0.04));
  border: 1px solid rgba(34,200,68,0.25);
  border-left: 4px solid #22C844;
  border-radius: 2px; padding: 18px 22px; margin-top: 16px;
  clip-path: polygon(0 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%);
}
.ban-tag {
  font-family: 'Share Tech Mono', monospace; font-size: 9px;
  letter-spacing: 3px; text-transform: uppercase; margin-bottom: 8px;
}
.ban-text {
  font-family: 'Rajdhani', sans-serif; font-size: 14px;
  font-weight: 500; line-height: 1.75;
}

/* === NATIVE WIDGET OVERRIDES === */
[data-testid="stFileUploader"] {
  background: var(--kevlar) !important;
  border: 1px dashed rgba(220,20,60,0.2) !important;
  border-radius: 2px !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] > div > div {
  background: var(--kevlar) !important;
  border: 1px solid rgba(220,20,60,0.15) !important;
  border-radius: 2px !important; color: var(--cream) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 15px !important; font-weight: 500 !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div:focus-within {
  border-color: rgba(220,20,60,0.5) !important;
  box-shadow: 0 0 12px rgba(220,20,60,0.1) !important;
}
label[data-testid="stWidgetLabel"] p {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 9px !important; letter-spacing: 3px !important;
  color: #FFFFFF !important; text-transform: uppercase !important;
}
.stButton > button {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 11px !important; font-weight: 700 !important;
  letter-spacing: 3px !important; text-transform: uppercase !important;
  background: var(--kevlar) !important; color: var(--rosso) !important;
  border: 1px solid rgba(220,20,60,0.3) !important;
  border-radius: 2px !important;
  padding: 14px 36px 12px !important;
  transition: all 0.18s ease !important;
  clip-path: polygon(8px 0, 100% 0, calc(100% - 8px) 100%, 0 100%) !important;
}
.stButton > button:hover {
  background: var(--rosso) !important; color: var(--pitch) !important;
  box-shadow: 0 0 24px var(--rosso-glow) !important;
  transform: translateY(-2px) !important;
}
[data-testid="stExpander"] {
  background: var(--kevlar) !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
  border-left: 2px solid var(--rosso-dim) !important;
  border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 9px !important; letter-spacing: 3px !important;
  text-transform: uppercase !important; color: #FFFFFF !important;
  padding: 14px 18px !important;
}
.stMarkdown p {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 14px !important; font-weight: 400 !important;
  color: #4A3838 !important;
}
[data-testid="stRadio"] label p {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 9px !important; letter-spacing: 2px !important;
  color: #FFFFFF !important; text-transform: uppercase !important;
}
[data-testid="column"]:nth-child(1) { animation: fadeSlide .35s ease .04s both; }
[data-testid="column"]:nth-child(2) { animation: fadeSlide .35s ease .10s both; }
[data-testid="column"]:nth-child(3) { animation: fadeSlide .35s ease .16s both; }
[data-testid="column"]:nth-child(4) { animation: fadeSlide .35s ease .22s both; }
@keyframes fadeSlide {
  from { opacity:0; transform: translateY(12px); }
  to   { opacity:1; transform: translateY(0); }
}
hr { border: none !important; border-top: 1px solid rgba(220,20,60,0.08) !important; margin: 20px 0 !important; }

/* Scrollbar Ferrari */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--pitch); }
::-webkit-scrollbar-thumb { background: var(--rosso-dim); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--rosso); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# BUILD HEADER
# ═══════════════════════════════════════════════════════════════
scr = st.session_state.screen
SCR_NAMES = SCREENS

z_s   = st.session_state.z_stat
z_pct = min(abs(z_s) / 4.0 * 100, 100) if z_s is not None else 0.0
z_col = C_ROSSO if (z_s is not None and st.session_state.p_value is not None
                    and st.session_state.p_value < st.session_state.alpha) else C_TITANIO

# Tachómetro: |Z|
tacho_html = build_tacho(
    z_pct, z_col,
    '|Z| stat',
    '%.2f' % abs(z_s) if z_s is not None else '--'
)

# Rev lights
rev_html = build_rev_lights(scr, MAX_SCR)

# Lap indicator: screen name
scr_label = SCR_NAMES[scr]

# Dot indicators
dot_parts  = []
for i, nm in enumerate(SCREENS):
    if   i == scr:  cls = 'seg-dot active'
    elif i < scr:   cls = 'seg-dot done'
    else:           cls = 'seg-dot'
    dot_parts.append('<div class="%s" title="%s"></div>' % (cls, nm))
dots_html = ''.join(dot_parts)

label_parts = []
for i, nm in enumerate(SCREENS):
    if i == scr:
        s = 'color:var(--rosso);font-weight:700;font-size:9px;letter-spacing:3px;'
    else:
        s = 'color:#FFFFFF;font-size:8px;letter-spacing:2px;'
    label_parts.append(
        '<span style="font-family:Share Tech Mono,monospace;text-transform:uppercase;%s">%s</span>' % (s, nm)
    )
labels_html = '&nbsp;<span style="color:#1E1818;">/</span>&nbsp;'.join(label_parts)

# Gear: screen number
gear_num = scr + 1
gear_col_map = {1:'#4A3838', 2:'#1A3A20', 3:'#1A2840', 4:'#5A2010', 5:'#DC143C'}
g_col = gear_col_map.get(gear_num, '#4A3838')

st.markdown("""
<div class="f296-header">
  <div class="header-grid">
    <div class="h-instruments">
      <div>{tacho}</div>
      <div class="inst-stack">
        <div class="inst-label">Sector</div>
        <div class="inst-value" style="color:{gcol};">{gnum}</div>
        <div class="inst-label" style="margin-top:4px;">Pantalla</div>
        <div class="inst-value" style="color:{gcol}88;font-size:9px;">{scrlbl}</div>
      </div>
    </div>
    <div class="h-center">
      <div class="prancing-horse">Ferrari 296 GT3 · Race Analytics</div>
      <div class="logo-z">Z<span class="accent">·</span>STAT</div>
      <div class="logo-subtitle">Prueba de Hip&oacute;tesis &middot; Cockpit Edition</div>
      <div class="rev-lights-wrap">{rev}</div>
    </div>
    <div class="h-right">
      <div class="driver-plate">
        <div class="dp-series">Estad&iacute;stica · UP Chiapas</div>
        <div class="dp-name">Z · TEST</div>
        <div class="dp-team">Distribuci&oacute;n Normal &middot; H₀ vs H₁</div>
      </div>
      <div style="width:2px;height:50px;background:linear-gradient(180deg,transparent,var(--rosso),transparent);border-radius:2px;"></div>
      <div class="inst-stack" style="text-align:right;">
        <div class="inst-label">α nivel</div>
        <div class="inst-value" style="color:var(--rosso);">%.2f</div>
        <div class="inst-label" style="margin-top:4px;">Decisión</div>
        <div class="inst-value" style="font-size:8px;color:{dcol};">{dtxt}</div>
      </div>
    </div>
  </div>
  <div class="header-underline"></div>
</div>
""".format(
    tacho=tacho_html, rev=rev_html,
    gnum=gear_num, gcol=g_col, scrlbl=scr_label,
    dcol=C_ROSSO if st.session_state.decision=='rechaza' else C_TITANIO,
    dtxt='RECHAZA H₀' if st.session_state.decision=='rechaza'
         else ('NO RECHAZA' if st.session_state.decision=='no_rechaza' else '---'),
) % st.session_state.alpha, unsafe_allow_html=True)

# ─── PADDLE NAV BAR ─────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"].paddle-row {
  background: var(--exhaust, #141212);
  border-bottom: 1px solid rgba(220,20,60,0.10);
  padding: 0 !important; margin: 0 !important;
  align-items: center !important;
}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="paddle-row" style="display:contents;">', unsafe_allow_html=True)
pb_l, pb_c, pb_r = st.columns([1, 3, 1], gap="small")

with pb_l:
    if scr > 0:
        if st.button("◀ PREV", key="dn"):
            go(scr - 1)

with pb_c:
    st.markdown(
        '<div style="display:flex;flex-direction:column;align-items:center;gap:8px;padding:12px 0 10px;">'
        '<div style="display:flex;gap:12px;align-items:center;">' + dots_html + '</div>'
        '<div style="display:flex;gap:8px;align-items:center;">' + labels_html + '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with pb_r:
    if scr < MAX_SCR:
        if st.button("NEXT ▶", key="up"):
            go(scr + 1)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div style="height:1px;background:rgba(220,20,60,0.08);"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SCREENS
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="display-wrap">', unsafe_allow_html=True)

# ─── SCREEN 0: GRID ──────────────────────────────────────────
if scr == 0:
    st.markdown("""
<div class="welcome-arena">
  <div class="prancing-emblem">
    <div>
      <div class="emblem-inner">
        Z<br><span style="background:linear-gradient(135deg,#DC143C,#FF1744);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;">296</span>
      </div>
      <div class="emblem-sub">GT3 &middot; Analytics</div>
    </div>
  </div>
  <div class="welcome-tag">Sistema de An&aacute;lisis Estad&iacute;stico</div>
  <div class="welcome-headline">PRUEBA Z</div>
  <p class="welcome-desc">
    Motor de an&aacute;lisis estad&iacute;stico inspirado en el Ferrari 296 GT3.<br>
    Visualiza, calcula y obtiene interpretaci&oacute;n IA de pruebas de hip&oacute;tesis.<br>
    Avanza con la paleta derecha para iniciar la sesi&oacute;n.
  </p>
  <div class="stage-grid">
    <div class="stage-cell"><div class="stage-num">1</div><div class="stage-lbl">Datos</div></div>
    <div class="stage-cell"><div class="stage-num">2</div><div class="stage-lbl">Telemetría</div></div>
    <div class="stage-cell"><div class="stage-num">3</div><div class="stage-lbl">Prueba Z</div></div>
    <div class="stage-cell"><div class="stage-num">4</div><div class="stage-lbl">Radio IA</div></div>
  </div>
  <div class="ignition-hint">&#9658; NEXT &middot; ENCENDER MOTOR</div>
</div>
""", unsafe_allow_html=True)


# ─── SCREEN 1: DATOS ─────────────────────────────────────────
elif scr == 1:
    st.markdown('<div class="sect-title">Ingesta de Datos &middot; Pit Lane · Configuraci&oacute;n</div>',
                unsafe_allow_html=True)

    c_up, c_gen, c_stat = st.columns([1, 1.6, 0.85], gap="large")

    with c_up:
        st.markdown('<div class="card-label">Cargar Archivo CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df_ = pd.read_csv(uploaded)
            num_cols = df_.select_dtypes(include=['int64','float64']).columns.tolist()
            if num_cols:
                col_sel = st.selectbox("VARIABLE A ANALIZAR", num_cols)
                datos   = df_[col_sel].dropna().values
                st.session_state.datos            = datos
                st.session_state.nombre_variable  = col_sel
                n_r, n_c = df_.shape
                st.markdown(
                    '<div class="info-panel" style="margin-top:14px;">'
                    '<div class="ip-label">Archivo Recibido</div>'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;">'
                    '<div><div class="card-label">Filas</div>'
                    '<div style="font-family:Orbitron,sans-serif;font-size:28px;font-weight:900;color:' + C_ROSSO + ';">' + str(n_r) + '</div></div>'
                    '<div><div class="card-label">Columnas</div>'
                    '<div style="font-family:Orbitron,sans-serif;font-size:28px;font-weight:900;color:' + C_TITANIO + ';">' + str(n_c) + '</div></div>'
                    '</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="info-panel" style="border-left-color:' + C_ROSSO + ';margin-top:14px;">'
                    '<div class="ip-label" style="color:' + C_ROSSO + ';">Sin columnas numéricas</div>'
                    '<div class="ip-body">El CSV no contiene variables numéricas.</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="height:160px;display:flex;align-items:center;justify-content:center;'
                'background:var(--kevlar,#1E1A18);border:1px dashed rgba(220,20,60,0.15);'
                'border-radius:2px;margin-top:14px;clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%);">'
                '<div style="text-align:center;">'
                '<div style="font-family:Orbitron,sans-serif;font-size:32px;font-weight:900;'
                'letter-spacing:4px;color:rgba(220,20,60,0.07);">NO DATA</div>'
                '<div style="font-family:Share Tech Mono,monospace;font-size:8px;'
                'letter-spacing:3px;color:#2A1818;margin-top:8px;">Carga un CSV para continuar</div>'
                '</div></div>',
                unsafe_allow_html=True
            )

    with c_gen:
        st.markdown('<div class="card-label">Generar Datos Sint&eacute;ticos</div>', unsafe_allow_html=True)
        tipo = st.selectbox("DISTRIBUCIÓN", ["Normal", "Sesgada a la derecha", "Sesgada a la izquierda"])
        n    = st.slider("NÚMERO DE OBSERVACIONES (n)", 30, 500, 100)
        if st.button("⚡ GENERAR DATOS"):
            if tipo == "Normal":
                datos = np.random.normal(loc=50, scale=10, size=n)
            elif tipo == "Sesgada a la derecha":
                datos = np.random.exponential(scale=10, size=n) + 40
            else:
                datos = 100 - np.random.exponential(scale=10, size=n)
            st.session_state.datos           = datos
            st.session_state.nombre_variable = tipo

        if st.session_state.datos is not None:
            d = st.session_state.datos
            sesgo_v = stats.skew(d)
            _, p_n  = stats.shapiro(d[:50] if len(d) > 50 else d)
            st.markdown(
                '<div class="info-panel" style="margin-top:14px;">'
                '<div class="ip-label">Datos en Memoria</div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px;">'
                '<div><div class="card-label">n</div>'
                '<div style="font-family:Orbitron,sans-serif;font-size:28px;font-weight:900;color:' + C_ROSSO + ';">' + str(len(d)) + '</div></div>'
                '<div><div class="card-label">x̄</div>'
                '<div style="font-family:Orbitron,sans-serif;font-size:28px;font-weight:900;color:' + C_TITANIO + ';">%.1f</div></div>' % np.mean(d) +
                '<div><div class="card-label">σ</div>'
                '<div style="font-family:Orbitron,sans-serif;font-size:28px;font-weight:900;color:' + C_SCUDERIA + ';">%.1f</div></div>' % np.std(d) +
                '</div>'
                '<div style="margin-top:12px;font-family:Share Tech Mono,monospace;font-size:9px;letter-spacing:2px;color:#FFFFFF;">'
                'VAR: <span style="color:' + C_ROSSO + ';">' + str(st.session_state.nombre_variable)[:28] + '</span>'
                '</div></div>',
                unsafe_allow_html=True
            )

    with c_stat:
        st.markdown('<div class="card-label">Estado del Sistema</div>', unsafe_allow_html=True)
        checks = [
            ("Datos cargados",   st.session_state.datos is not None),
            ("Variable definida",st.session_state.nombre_variable is not None),
            ("Prueba Z lista",   st.session_state.z_stat is not None),
        ]
        for lbl, ok in checks:
            dc_c  = C_ROSSO if ok else C_DIM
            txt_c = C_CREAM if ok else '#2A2020'
            glow  = 'box-shadow:0 0 8px ' + C_ROSSO + ';' if ok else ''
            st.markdown(
                '<div class="status-row">'
                '<div class="s-indicator" style="background:' + dc_c + ';' + glow + '"></div>'
                '<span class="s-label" style="color:' + txt_c + ';">' + lbl + '</span>'
                '<span style="margin-left:auto;font-family:Share Tech Mono,monospace;font-size:8px;'
                'letter-spacing:2px;color:' + dc_c + ';">' + ('OK' if ok else '—') + '</span>'
                '</div>',
                unsafe_allow_html=True
            )

        if st.session_state.datos is not None:
            d      = st.session_state.datos
            sesgo_ = stats.skew(d)
            _, p_n = stats.shapiro(d[:50] if len(d) > 50 else d)
            norm_c = '#22C844' if p_n > 0.05 else C_ROSSO
            norm_t = 'NORMAL' if p_n > 0.05 else 'NO NORMAL'
            st.markdown(
                '<div class="info-panel" style="margin-top:16px;">'
                '<div class="ip-label">Diagnóstico Rápido</div>'
                '<div class="ip-body">'
                'Shapiro-Wilk: <strong style="color:' + norm_c + ';">' + norm_t + '</strong><br>'
                'Sesgo: <strong style="color:' + C_SCUDERIA + ';">%.4f</strong><br>' % sesgo_ +
                'p (S-W): <strong style="color:' + C_TITANIO + ';">%.4f</strong>'   % p_n +
                '</div></div>',
                unsafe_allow_html=True
            )


# ─── SCREEN 2: TELEMETRÍA (Visualización) ───────────────────
elif scr == 2:
    st.markdown('<div class="sect-title">Telemetr&iacute;a &middot; Distribuci&oacute;n Visual &middot; An&aacute;lisis</div>',
                unsafe_allow_html=True)

    if st.session_state.datos is None:
        st.markdown(
            '<div class="info-panel"><div class="ip-label">Sin Señal</div>'
            '<div class="ip-body">Ve a DATOS e ingresa o genera datos primero.</div></div>',
            unsafe_allow_html=True
        )
    else:
        datos  = st.session_state.datos
        nombre = str(st.session_state.nombre_variable)

        media_v  = np.mean(datos)
        std_v    = np.std(datos)
        n_v      = len(datos)
        sesgo_v  = stats.skew(datos)
        _, p_norm = stats.shapiro(datos[:50] if len(datos) > 50 else datos)
        q1 = np.percentile(datos, 25); q3 = np.percentile(datos, 75); iqr = q3 - q1
        outs = int(np.sum((datos < q1 - 1.5*iqr) | (datos > q3 + 1.5*iqr)))

        # Metric strip
        metrics = [
            ('Media',    '%.2f' % media_v, C_ROSSO,   'ms-rosso'),
            ('Desv.Est.','%.2f' % std_v,   C_TITANIO, 'ms-white'),
            ('N',        str(n_v),          C_SCUDERIA,'ms-orange'),
            ('Sesgo',    '%.3f' % sesgo_v,  '#FFD700', 'ms-yellow'),
        ]
        strip = '<div class="metric-strip">'
        for lbl, val, col, cls in metrics:
            strip += (
                '<div class="ms-cell ' + cls + '">'
                '<div class="ms-lbl">' + lbl + '</div>'
                '<div class="ms-val" style="color:' + col + ';">' + val + '</div>'
                '</div>'
            )
        strip += '</div>'
        st.markdown(strip, unsafe_allow_html=True)

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        fig.patch.set_facecolor('#080808')

        # --- Histograma + KDE ---
        ax0 = axes[0]
        ax_ferrari(ax0)
        ax0.hist(datos, bins=22, color='#3D0410', edgecolor='none', density=True, alpha=1.0)
        xmin, xmax = ax0.get_xlim()
        x = np.linspace(xmin, xmax, 300)
        kde = stats.gaussian_kde(datos)
        ax0.plot(x, kde(x), color=C_ROSSO, lw=2, zorder=5)
        ax0.fill_between(x, kde(x), alpha=0.12, color=C_ROSSO)
        ax0.axvline(media_v, color=C_TITANIO, lw=1.2, linestyle='--', alpha=0.7)
        ax0.set_title('HISTOGRAMA + KDE', fontsize=8, fontweight='bold', color=C_ROSSO, pad=10, loc='left')
        ax0.set_xlabel(nombre[:22].upper(), fontsize=7, color='#FFFFFF')

        # --- Boxplot ---
        ax1 = axes[1]
        ax_ferrari(ax1)
        bp = ax1.boxplot(datos, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='#1A0508', color=C_ROSSO, linewidth=1.5),
                         whiskerprops=dict(color='#3A2020', lw=1),
                         capprops=dict(color=C_ROSSO, lw=2),
                         medianprops=dict(color=C_ROSSO2, lw=2.5),
                         flierprops=dict(marker='o', color=C_ROSSO, alpha=0.5, markersize=4,
                                         markeredgecolor='none'))
        ax1.set_title('BOXPLOT', fontsize=8, fontweight='bold', color=C_ROSSO, pad=10, loc='left')
        ax1.set_ylabel(nombre[:22].upper(), fontsize=7, color='#FFFFFF')

        # --- QQ Plot ---
        ax2 = axes[2]
        ax_ferrari(ax2)
        (osm, osr), (slope, intercept, r) = stats.probplot(datos, dist="norm")
        ax2.scatter(osm, osr, color=C_ROSSO, s=12, alpha=0.6, zorder=3, edgecolors='none')
        line_x = np.array([min(osm), max(osm)])
        ax2.plot(line_x, slope * line_x + intercept, color=C_TITANIO, lw=1.5, zorder=4, alpha=0.8)
        ax2.set_title('QQ PLOT · NORMALIDAD', fontsize=8, fontweight='bold', color=C_ROSSO, pad=10, loc='left')

        plt.tight_layout(pad=1.4)
        st.pyplot(fig); plt.close(fig)

        # Auto-analysis cards
        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3, gap="small")
        for col_w, lbl, val, ok, ok_txt, no_txt in [
            (a1, 'Normalidad (S-W)', 'p = %.4f' % p_norm, p_norm > 0.05,
             'Distribución normal ✓', 'Distribución no normal ✗'),
            (a2, 'Sesgo', '%.4f' % sesgo_v, abs(sesgo_v) < 0.5,
             'Sin sesgo significativo', 'Sesgo detectado'),
            (a3, 'Outliers', '%d detectados' % outs, outs == 0,
             'Sin outliers ✓', '%d outliers detectados' % outs),
        ]:
            c   = '#22C844' if ok else C_ROSSO
            with col_w:
                st.markdown(
                    '<div class="f296-card" style="border-top-color:' + c + ';padding:16px 18px;">'
                    '<div class="card-label">' + lbl + '</div>'
                    '<div style="font-family:Orbitron,sans-serif;font-size:22px;font-weight:900;color:' + c + ';">'
                    + val + '</div>'
                    '<div class="card-desc" style="color:' + c + '88;margin-top:8px;">' + (ok_txt if ok else no_txt) + '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )


# ─── SCREEN 3: PRUEBA Z ──────────────────────────────────────
elif scr == 3:
    st.markdown('<div class="sect-title">Prueba Z &middot; Qualifying &middot; C&aacute;lculo &amp; Decisi&oacute;n</div>',
                unsafe_allow_html=True)

    c_cfg, c_dat, c_res = st.columns([1, 1, 1.1], gap="large")

    with c_cfg:
        st.markdown('<div class="card-label">Par&aacute;metros de Hip&oacute;tesis</div>', unsafe_allow_html=True)
        mu0      = st.number_input("VALOR BAJO H₀ (μ₀)", value=st.session_state.mu0, step=0.1)
        sigma    = st.number_input("DESV. EST. POBLACIONAL (σ)", value=st.session_state.sigma,
                                   min_value=0.0001, step=0.1)
        alpha_v  = st.selectbox("NIVEL DE SIGNIFICANCIA (α)", [0.01, 0.05, 0.10],
                                index=[0.01, 0.05, 0.10].index(st.session_state.alpha)
                                if st.session_state.alpha in [0.01, 0.05, 0.10] else 1)
        tipo_    = st.selectbox("TIPO DE PRUEBA (H₁)", [
            "Bilateral (≠)", "Unilateral Derecha (>)", "Unilateral Izquierda (<)"
        ], index=["Bilateral (≠)", "Unilateral Derecha (>)", "Unilateral Izquierda (<)"].index(
            st.session_state.tipo_cola) if st.session_state.tipo_cola in
            ["Bilateral (≠)", "Unilateral Derecha (>)", "Unilateral Izquierda (<)"] else 0)

        st.session_state.mu0       = mu0
        st.session_state.sigma     = sigma
        st.session_state.alpha     = alpha_v
        st.session_state.tipo_cola = tipo_

    with c_dat:
        st.markdown('<div class="card-label">Fuente de Datos</div>', unsafe_allow_html=True)
        fuente = st.radio("FUENTE", ["Sesión actual", "Ingreso manual"], label_visibility="collapsed")

        if fuente == "Sesión actual":
            if st.session_state.datos is not None:
                d_z   = st.session_state.datos
                n_z   = len(d_z)
                xbar  = float(np.mean(d_z))
                st.markdown(
                    '<div class="info-panel" style="margin-top:12px;">'
                    '<div class="ip-label">Datos Detectados</div>'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;">'
                    '<div><div class="card-label">n</div>'
                    '<div style="font-family:Orbitron,sans-serif;font-size:32px;font-weight:900;color:' + C_ROSSO + ';">' + str(n_z) + '</div></div>'
                    '<div><div class="card-label">x̄</div>'
                    '<div style="font-family:Orbitron,sans-serif;font-size:32px;font-weight:900;color:' + C_TITANIO + ';">%.3f</div></div>' % xbar +
                    '</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="info-panel" style="border-left-color:' + C_ROSSO + ';margin-top:12px;">'
                    '<div class="ip-label" style="color:' + C_ROSSO + ';">Sin datos en sesión</div>'
                    '<div class="ip-body">Ve a DATOS primero.</div></div>',
                    unsafe_allow_html=True
                )
                n_z = None; xbar = None
        else:
            n_z  = st.number_input("TAMAÑO DE MUESTRA (n)", min_value=1, value=30, step=1)
            xbar = st.number_input("MEDIA MUESTRAL (x̄)", value=0.0, step=0.1)

        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        if st.button("⚡ CALCULAR PRUEBA Z"):
            if n_z is None or xbar is None:
                st.error("Ingresa datos o carga desde sesión.")
            else:
                se     = sigma / np.sqrt(n_z)
                z_calc = (xbar - mu0) / se
                if tipo_ == "Bilateral (≠)":
                    p_val   = 2 * (1 - stats.norm.cdf(abs(z_calc)))
                    z_c_i   = stats.norm.ppf(alpha_v / 2)
                    z_c_s   = stats.norm.ppf(1 - alpha_v / 2)
                elif tipo_ == "Unilateral Derecha (>)":
                    p_val   = 1 - stats.norm.cdf(z_calc)
                    z_c_i   = None
                    z_c_s   = stats.norm.ppf(1 - alpha_v)
                else:
                    p_val   = stats.norm.cdf(z_calc)
                    z_c_i   = stats.norm.ppf(alpha_v)
                    z_c_s   = None

                st.session_state.z_stat   = z_calc
                st.session_state.p_value  = p_val
                st.session_state.decision = 'rechaza' if p_val < alpha_v else 'no_rechaza'
                st.session_state._z_c_inf = z_c_i
                st.session_state._z_c_sup = z_c_s
                st.session_state._n_z     = n_z
                st.session_state._xbar    = xbar
                st.rerun()

    with c_res:
        st.markdown('<div class="card-label">Resultado</div>', unsafe_allow_html=True)
        z_s  = st.session_state.z_stat
        p_s  = st.session_state.p_value
        dec  = st.session_state.decision

        if z_s is not None:
            z_hero_col = C_ROSSO if dec == 'rechaza' else '#22C844'
            dec_lbl    = 'RECHAZA H₀' if dec == 'rechaza' else 'NO RECHAZA H₀'
            st.markdown(
                '<div class="z-hero">'
                '<div class="zh-tag">Estad&iacute;stico Z Calculado</div>'
                '<div class="zh-number" style="color:' + z_hero_col + ';">'
                + ('%.4f' % z_s) + '</div>'
                '<div class="zh-decision" style="background:' + z_hero_col + '22;color:' + z_hero_col + ';">'
                + dec_lbl + '</div>'
                '<div class="zh-stats">'
                'p-valor &nbsp;=&nbsp; ' + ('%.2e' % p_s if p_s < 0.001 else '%.4f' % p_s) + '<br>'
                '&alpha; &nbsp;&nbsp;&nbsp;&nbsp;=&nbsp; %.2f' % alpha_v + '<br>'
                'SE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp; σ / √n = %.4f' % (sigma / np.sqrt(n_z if isinstance(st.session_state._n_z, (int,float)) and st.session_state._n_z else 1))
                + '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            if dec == 'rechaza':
                st.markdown(
                    '<div class="banner-rechaza">'
                    '<div class="ban-tag" style="color:' + C_ROSSO + ';">Decisión &middot; Race Control</div>'
                    '<div class="ban-text">Se <strong style="color:' + C_ROSSO + ';">RECHAZA H₀</strong>. '
                    'Existe evidencia estadística suficiente para respaldar H₁.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="banner-ok">'
                    '<div class="ban-tag" style="color:#22C844;">Decisión &middot; Race Control</div>'
                    '<div class="ban-text"><strong style="color:#22C844;">NO SE RECHAZA H₀</strong>. '
                    'No hay evidencia estadística suficiente para rechazar H₀.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="info-panel"><div class="ip-label">En espera</div>'
                '<div class="ip-body">Configura parámetros y presiona CALCULAR PRUEBA Z.</div></div>',
                unsafe_allow_html=True
            )

    # Gráfica zona de rechazo
    if st.session_state.z_stat is not None:
        st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-label" style="margin-bottom:10px;">'
            'Zona de Rechazo &middot; Distribuci&oacute;n Normal Est&aacute;ndar &middot; '
            'α = %.2f' % st.session_state.alpha + '</div>',
            unsafe_allow_html=True
        )

        z_s   = st.session_state.z_stat
        z_c_i = getattr(st.session_state, '_z_c_inf', None)
        z_c_s = getattr(st.session_state, '_z_c_sup', None)
        alp   = st.session_state.alpha
        dec   = st.session_state.decision
        tipo_ = st.session_state.tipo_cola

        fig2, ax2 = plt.subplots(figsize=(14, 4.5))
        fig2.patch.set_facecolor('#080808')
        ax_ferrari(ax2)

        x = np.linspace(-4.5, 4.5, 1200)
        y = stats.norm.pdf(x)
        ax2.plot(x, y, color='#4A3030', lw=1.5)
        ax2.fill_between(x, y, alpha=0.03, color=C_CREAM)

        if tipo_ == "Bilateral (≠)":
            for fx in [np.linspace(-4.5, z_c_i, 200), np.linspace(z_c_s, 4.5, 200)]:
                ax2.fill_between(fx, stats.norm.pdf(fx), color=C_ROSSO, alpha=0.3)
                ax2.fill_between(fx, stats.norm.pdf(fx), alpha=0.05, color=C_ROSSO2)
            ax2.axvline(z_c_i, color=C_ROSSO, ls='--', lw=1.2, alpha=0.7,
                        label='Z crit. %.3f / %.3f' % (z_c_i, z_c_s))
            ax2.axvline(z_c_s, color=C_ROSSO, ls='--', lw=1.2, alpha=0.7)
        elif tipo_ == "Unilateral Derecha (>)":
            fx = np.linspace(z_c_s, 4.5, 200)
            ax2.fill_between(fx, stats.norm.pdf(fx), color=C_ROSSO, alpha=0.3)
            ax2.axvline(z_c_s, color=C_ROSSO, ls='--', lw=1.2, alpha=0.7,
                        label='Z crit. %.3f' % z_c_s)
        else:
            fx = np.linspace(-4.5, z_c_i, 200)
            ax2.fill_between(fx, stats.norm.pdf(fx), color=C_ROSSO, alpha=0.3)
            ax2.axvline(z_c_i, color=C_ROSSO, ls='--', lw=1.2, alpha=0.7,
                        label='Z crit. %.3f' % z_c_i)

        z_line_col = C_ROSSO if dec == 'rechaza' else '#22C844'
        ax2.axvline(z_s, color=z_line_col, lw=2.5, zorder=5,
                    label='Z calc. %.4f' % z_s)
        ax2.scatter([z_s], [stats.norm.pdf(z_s)],
                    color=z_line_col, s=70, zorder=6, edgecolors='none')
        # Glow effect point
        ax2.scatter([z_s], [stats.norm.pdf(z_s)],
                    color=z_line_col, s=200, zorder=4, alpha=0.15, edgecolors='none')

        ax2.set_title('ZONA DE RECHAZO · PRUEBA Z',
                      fontsize=8, fontweight='bold', color=C_ROSSO, pad=10, loc='left')
        ax2.set_xlabel('Z', fontsize=8, color='#FFFFFF')
        ax2.set_ylabel('Densidad', fontsize=8, color='#FFFFFF')
        leg = ax2.legend(fontsize=8, facecolor='#0D0808', edgecolor='#2A1818',
                         labelcolor=C_CREAM, framealpha=0.95)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig2); plt.close(fig2)


# ─── SCREEN 4: RADIO IA ──────────────────────────────────────
elif scr == 4:
    st.markdown('<div class="sect-title">Radio IA &middot; Gemini 2.5 Flash &middot; S&iacute;ntesis Ejecutiva</div>',
                unsafe_allow_html=True)

    ai_l, ai_r = st.columns([1.3, 1], gap="large")

    with ai_l:
        st.markdown('<div class="card-label" style="margin-bottom:10px;">Acceso Gemini &middot; API Key</div>',
                    unsafe_allow_html=True)
        api_key = st.text_input("GEMINI API KEY", type="password", placeholder="AIzaSy...")
        run_btn = st.button("⚡ TRANSMITIR AL IA")

        if run_btn:
            if not api_key:
                st.error("Ingresa una API Key de Google Gemini.")
            elif st.session_state.datos is None:
                st.error("Ve a DATOS primero y carga o genera datos.")
            elif st.session_state.z_stat is None:
                st.error("Ejecuta la Prueba Z antes de solicitar interpretación.")
            else:
                with st.spinner("Transmitiendo al motor IA..."):
                    try:
                        genai.configure(api_key=api_key)
                        gm = genai.GenerativeModel('gemini-2.5-flash')

                        datos_  = st.session_state.datos
                        nombre_ = st.session_state.nombre_variable
                        media_  = np.mean(datos_)
                        std_    = np.std(datos_)
                        n_      = len(datos_)
                        sesgo_  = stats.skew(datos_)
                        _, p_n  = stats.shapiro(datos_[:50] if len(datos_) > 50 else datos_)
                        z_s_    = st.session_state.z_stat
                        p_s_    = st.session_state.p_value
                        dec_    = st.session_state.decision
                        alp_    = st.session_state.alpha
                        mu0_    = st.session_state.mu0
                        sigma_  = st.session_state.sigma
                        cola_   = st.session_state.tipo_cola
                        dec_txt = 'Se RECHAZA H0' if dec_ == 'rechaza' else 'NO se rechaza H0'

                        prompt = (
                            "Eres un profesor de estadística experto pero accesible.\n"
                            "Analiza los resultados de esta Prueba Z y proporciona interpretación educativa en máximo 160 palabras.\n\n"
                            "Variable: '%s' | n=%d | Media=%.4f | σ=%.4f | Sesgo=%.3f\n"
                            "Shapiro-Wilk p=%.4f (%s)\n\n"
                            "Prueba Z: H0: μ=%.3f | Tipo: %s | α=%.2f\n"
                            "Z calculado=%.4f | p-valor=%.4f | Decisión: %s\n\n"
                            "1. Interpreta el resultado en lenguaje claro.\n"
                            "2. ¿Qué implica estadísticamente esta decisión?\n"
                            "3. ¿Alguna advertencia sobre la validez de la prueba?\n"
                            "Sé directo, técnico y conciso."
                        ) % (
                            nombre_, n_, media_, sigma_, sesgo_,
                            p_n, 'normal' if p_n > 0.05 else 'no normal',
                            mu0_, cola_, alp_,
                            z_s_, p_s_, dec_txt
                        )

                        resp = gm.generate_content(prompt)
                        st.balloons()
                        st.markdown(
                            '<div class="info-panel" style="margin-top:16px;border-left-color:#22C844;">'
                            '<div class="ip-label" style="color:#22C844;">IA &middot; Interpretaci&oacute;n Recibida</div>'
                            '<div class="ip-body" style="color:#A0C8A0;font-size:14px;line-height:1.9;">'
                            + resp.text.replace('\n', '<br>') +
                            '</div></div>',
                            unsafe_allow_html=True
                        )

                        st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
                        st.markdown('<div class="card-label">Reflexi&oacute;n del Estudiante</div>',
                                    unsafe_allow_html=True)
                        conclusion = st.text_area(
                            "CONCLUSIÓN",
                            label_visibility="collapsed",
                            placeholder="Mi conclusión basada en los resultados y el análisis de la IA...",
                            height=120
                        )
                        if st.button("GUARDAR REFLEXIÓN"):
                            if conclusion:
                                st.success("✓ Reflexión guardada. Excelente análisis.")
                                st.balloons()
                            else:
                                st.warning("Escribe tu conclusión antes de guardar.")

                    except Exception as e:
                        st.error("Error de conexión con Gemini: " + str(e))

    with ai_r:
        st.markdown('<div class="card-label" style="margin-bottom:10px;">Resumen de Carrera</div>',
                    unsafe_allow_html=True)

        z_s = st.session_state.z_stat
        p_s = st.session_state.p_value
        dec = st.session_state.decision

        if z_s is not None and st.session_state.datos is not None:
            datos_ = st.session_state.datos
            dec_c  = C_ROSSO if dec == 'rechaza' else '#22C844'
            dec_l  = 'RECHAZA H₀' if dec == 'rechaza' else 'NO RECHAZA H₀'

            rows = [
                ("Variable",      str(st.session_state.nombre_variable or '—')[:28], C_ROSSO),
                ("n",             str(len(datos_)),                                    C_SCUDERIA),
                ("x̄ (media)",    "%.4f" % np.mean(datos_),                           C_TITANIO),
                ("H₀: μ =",       "%.3f" % st.session_state.mu0,                     '#4488CC'),
                ("σ pob.",        "%.4f" % st.session_state.sigma,                    C_IVORY),
                ("Cola",          str(st.session_state.tipo_cola),                    '#B8B4AE'),
                ("α",             "%.2f" % st.session_state.alpha,                    C_ROSSO),
                ("Z calculado",   "%.4f" % z_s,                                       dec_c),
                ("p-valor",       ('%.2e' % p_s if p_s < 0.001 else '%.4f' % p_s),   dec_c),
                ("Decisión",      dec_l,                                               dec_c),
            ]
            C_IVORY = '#C0B898'
            for lbl, val, col in rows:
                st.markdown(
                    '<div style="display:flex;justify-content:space-between;align-items:flex-start;'
                    'padding:10px 0;border-bottom:1px solid rgba(220,20,60,0.06);">'
                    '<span style="font-family:Share Tech Mono,monospace;font-size:8px;'
                    'letter-spacing:2px;color:#FFFFFF;text-transform:uppercase;'
                    'flex-shrink:0;padding-right:12px;">' + lbl + '</span>'
                    '<span style="font-family:Orbitron,sans-serif;font-size:12px;'
                    'font-weight:700;color:' + col + ';text-align:right;word-break:break-all;">'
                    + val + '</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="info-panel"><div class="ip-label">Sin transmisión</div>'
                '<div class="ip-body">Ejecuta la Prueba Z en la marcha 4 primero.</div></div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)