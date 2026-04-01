import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np
from src.detector import DrowsinessDetector

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="Driver AI Dashboard", layout="wide", initial_sidebar_state="expanded")

for k in ["ear_list", "mar_list", "gaze_list", "head_list", "drive_history"]:
    if k not in st.session_state: st.session_state[k] = []
if "fatigue_score" not in st.session_state: st.session_state.fatigue_score = 0
if "cap" not in st.session_state: st.session_state.cap = None

# ==========================================
# 2. THEME & DARK BUTTON CSS
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #020617; color: #f8fafc; }
    .dashboard-header { background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 25px; margin-bottom: 25px; text-align: center; backdrop-filter: blur(12px); }
    .dashboard-header h1 { color: #38bdf8; margin: 0; font-size: 2rem; font-weight: 800; }
    .dashboard-header p { color: #94a3b8; font-size: 0.9rem; margin-top: 5px; text-transform: uppercase; letter-spacing: 2px; }

    .glass-card { background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 20px; margin-bottom: 15px; backdrop-filter: blur(12px); text-align: center; }
    .card-label { color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
    .card-value { color: #f8fafc; font-size: 2.2rem; font-weight: 700; margin: 0; line-height: 1; }
    
    .stButton>button, .stDownloadButton>button {
        background-color: #0f172a !important; color: #f8fafc !important;
        border: 1px solid #334155 !important; border-radius: 12px !important;
        padding: 10px 24px !important; width: 100% !important; transition: 0.3s !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #38bdf8 !important; color: #020617 !important; border: 1px solid #38bdf8 !important;
    }
    .status-text { font-size: clamp(1.4rem, 1.8vw, 2.2rem); font-weight: 900; margin: 0; }
    .color-alert { color: #10b981; } .color-warn { color: #facc15; } .color-danger { color: #ef4444; } .color-calib { color: #38bdf8; }
    .blink { animation: pulse 1s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .section-label { font-size: 0.85rem; color: #cbd5e1; font-weight: 600; margin-bottom: 10px; display: block; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-header"><h1>🚗 Driver AI Monitoring System</h1><p>Drowsiness, Fatigue, and Multi-Axis Distraction Detection</p></div>', unsafe_allow_html=True)

# ==========================================
# 3. LAYOUT
# ==========================================
run = st.sidebar.checkbox("🟢 Start Monitoring", value=False)
if st.sidebar.button("🗑️ Reset Session"):
    for k in ["ear_list", "mar_list", "gaze_list", "head_list", "drive_history"]: st.session_state[k] = []
    st.session_state.fatigue_score = 0; st.rerun()

col_main, col_dash = st.columns([2.8, 1.2], gap="large")

with col_main:
    main_placeholder = st.empty()
    r1c1, r1c2 = st.columns(2); r2c1, r2c2 = st.columns(2)
    with r1c1: st.markdown('<div class="section-label">📈 EAR (Eyes)</div>', unsafe_allow_html=True); ear_g = st.empty()
    with r1c2: st.markdown('<div class="section-label">👄 MAR (Mouth)</div>', unsafe_allow_html=True); mar_g = st.empty()
    with r2c1: st.markdown('<div class="section-label">👀 GAZE (Pupils)</div>', unsafe_allow_html=True); gaze_g = st.empty()
    with r2c2: st.markdown('<div class="section-label">📐 HEAD (Yaw)</div>', unsafe_allow_html=True); head_g = st.empty()

with col_dash:
    status_box = st.empty(); ear_box = st.empty(); mar_box = st.empty(); fatigue_box = st.empty()

def update_dashboard(status, ear, mar, fatigue):
    """
    Updates the UI dashboard cards with the latest status, metrics, and fatigue score.
    """
    # 1. Determine Color, Icon, and Animation based on Status
    if "CALIBRATING" in status:
        color, icon, blink = "color-calib", "🔄", "blink"
    elif "SUNGLASSES" in status or "GLARE" in status:
        color, icon, blink = "color-calib", "🕶️", ""
    elif status == "ALERT":
        color, icon, blink = "color-alert", "🟢", ""
    elif status == "CRITICAL DISTRACTION":
        color, icon, blink = "color-danger", "🚨", "blink"
    elif status in ["DROWSY", "YAWNING", "HEAD DROP", "DISTRACTED"]:
        color, icon, blink = "color-danger", "🔴", "blink"
    elif "SEMI" in status: # Handles SEMI-DROWSY or SEMI-YAWNING
        color, icon, blink = "color-warn", "🟡", ""
    else:
        color, icon, blink = "color-warn", "⚪", ""

    # 2. Render Status Card
    status_box.markdown(f"""
        <div class="glass-card {blink}">
            <div class="card-label">System Status</div>
            <div class="status-text {color}">{icon} {status}</div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Render EAR Card
    ear_box.markdown(f"""
        <div class="glass-card">
            <div class="card-label">Eye Aspect Ratio</div>
            <div class="card-value">{ear:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    # 4. Render MAR Card
    mar_box.markdown(f"""
        <div class="glass-card">
            <div class="card-label">Mouth Aspect Ratio</div>
            <div class="card-value">{mar:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    # 5. Render Fatigue Progress Bar
    # Color logic: Green < 30, Yellow < 70, Red >= 70
    bar_color = "#10b981" if fatigue < 30 else "#facc15" if fatigue < 70 else "#ef4444"
    fatigue_box.markdown(f"""
        <div class="glass-card">
            <div class="card-label">Fatigue Level ({fatigue}%)</div>
            <div style="width:100%; height:12px; background:rgba(0,0,0,0.3); border-radius:6px; margin-top:10px; overflow:hidden;">
                <div style="width:{fatigue}%; height:100%; background:{bar_color}; transition:0.3s; box-shadow:0 0 10px {bar_color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. MONITORING
# ==========================================
if run:
    if "detector" not in st.session_state: st.session_state.detector = DrowsinessDetector()
    if st.session_state.cap is None: st.session_state.cap = cv2.VideoCapture(0)
    det, cap = st.session_state.detector, st.session_state.cap
    
    while run:
        ret, frame = cap.read()
        if not ret: break
        frame = det.process_frame(cv2.flip(frame, 1))
        ear, mar, gaze, head, status = det.last_ear, det.last_mar, det.last_gaze, det.last_head_h, det.last_status

        for l, v in zip([st.session_state.ear_list, st.session_state.mar_list, st.session_state.gaze_list, st.session_state.head_list], [ear, mar, gaze, head]):
            l.append(v)
            if len(l) > 50: l.pop(0)

        if status in ["DROWSY", "YAWNING", "HEAD DROP", "DISTRACTED", "CRITICAL DISTRACTION"]: 
            st.session_state.fatigue_score = min(100, st.session_state.fatigue_score + 2)
        else: st.session_state.fatigue_score = max(0, st.session_state.fatigue_score - 1)

        main_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        ear_g.line_chart(st.session_state.ear_list, height=150); mar_g.line_chart(st.session_state.mar_list, height=150, color="#facc15"); gaze_g.line_chart(st.session_state.gaze_list, height=150, color="#60a5fa"); head_g.line_chart(st.session_state.head_list, height=150, color="#c084fc")
        
        d_status = f"CALIBRATING ({det.calibration_frame_count}/60)" if status == "CALIBRATING" else status
        update_dashboard(d_status, ear, mar, st.session_state.fatigue_score)
        st.session_state.drive_history.append({"Time": time.strftime("%H:%M:%S"), "EAR": ear, "MAR": mar, "Gaze": gaze, "Head": head, "Status": status})
        time.sleep(0.01)

# ==========================================
# 5. REPORT
# ==========================================
else:
    if st.session_state.cap: st.session_state.cap.release(); st.session_state.cap = None
    if st.session_state.drive_history:
        if "detector" in st.session_state: del st.session_state.detector
        df = pd.DataFrame(st.session_state.drive_history)
        with main_placeholder.container():
            st.markdown("### 📊 Session Analytics Report")
            m1, m2, m3, m4 = st.columns(4)
            stats = [("Avg Alertness", f"{df['EAR'].mean():.2f}", "#38bdf8"), ("Drowsiness", len(df[df['Status'].str.contains('DROWSY', na=False)]), "#ef4444"), ("Distraction", len(df[df['Status'].str.contains('DISTRACT', na=False)]), "#60a5fa"), ("Yawns", len(df[df['Status'].str.contains('YAWN', na=False)]), "#facc15")]
            for col, (l, v, c) in zip([m1, m2, m3, m4], stats):
                col.markdown(f'<div class="glass-card" style="border-top:3px solid {c}; background:rgba(15,23,42,0.8);"><div class="card-label" style="font-size:0.7rem;">{l}</div><div class="card-value" style="color:{c}; font-size:1.8rem;">{v}</div></div>', unsafe_allow_html=True)
            st.line_chart(df.set_index('Time')[['EAR', 'MAR', 'Gaze', 'Head']], color=["#38bdf8", "#facc15", "#60a5fa", "#c084fc"], height=300)
            st.download_button("📥 Download Report", df.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")
            if st.button("🔄 New Session"): st.session_state.drive_history = []; st.session_state.fatigue_score = 0; st.rerun()
    else:
        update_dashboard("STANDBY", 0, 0, 0)
        main_placeholder.markdown('<div style="height:450px; display:flex; justify-content:center; align-items:center; color:#94a3b8; border:1px dashed #334155; border-radius:16px;">SYSTEM IDLE. ACTIVATE IN SIDEBAR.</div>', unsafe_allow_html=True)