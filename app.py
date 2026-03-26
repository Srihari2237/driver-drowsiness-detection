import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np

# Import the updated detector
from src.detector import DrowsinessDetector

# ==========================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="Driver AI Monitoring System", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "ear_list" not in st.session_state:
    st.session_state.ear_list = []
if "mar_list" not in st.session_state:
    st.session_state.mar_list = []
if "fatigue_score" not in st.session_state:
    st.session_state.fatigue_score = 0
if "drive_history" not in st.session_state:
    st.session_state.drive_history = []
if "cap" not in st.session_state:
    st.session_state.cap = None

# ==========================================
# 2. ADVANCED CUSTOM CSS (THEME & UI)
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #020617; color: #f8fafc; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    
    /* Dashboard Glassmorphism Title */
    .dashboard-title {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 25px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .dashboard-title h1 { color: #38bdf8; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: 1px; }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 22px 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(12px);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .card-label { color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
    .card-value { color: #f8fafc; font-size: 2.4rem; font-weight: 700; margin: 0; line-height: 1; }
    
    /* Status Text Constraints */
    .status-container { width: 100%; white-space: nowrap; overflow: hidden; }
    .status-text { font-size: clamp(1.4rem, 1.8vw, 2rem); font-weight: 900; margin: 0; }
    
    /* Status Colors */
    .color-alert { color: #10b981; text-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }
    .color-warn { color: #facc15; text-shadow: 0 0 15px rgba(250, 204, 21, 0.3); }
    .color-danger { color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
    .color-calib { color: #38bdf8; text-shadow: 0 0 15px rgba(56, 189, 248, 0.4); }

    /* Animations */
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    .blink { animation: pulse 1s infinite ease-in-out; }
    
    .section-label { font-size: 0.9rem; color: #cbd5e1; font-weight: 600; margin-bottom: 10px; display: block; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("### ⚙️ System Controls")
run = st.sidebar.checkbox("🟢 Start Monitoring", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reset All Session Data"):
    st.session_state.drive_history = []
    st.session_state.fatigue_score = 0
    st.session_state.ear_list = []
    st.session_state.mar_list = []
    st.rerun()

# ==========================================
# 4. DASHBOARD LAYOUT
# ==========================================
col_main, col_dash = st.columns([2.8, 1.2], gap="large")

with col_main:
    main_placeholder = st.empty() # Video Feed OR Analytics Report
    
    # Dual Graph Layout
    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        st.markdown('<div class="section-label">📈 Eye Alertness (EAR)</div>', unsafe_allow_html=True)
        ear_graph_placeholder = st.empty()
    with graph_col2:
        st.markdown('<div class="section-label">👄 Mouth Activity (MAR)</div>', unsafe_allow_html=True)
        mar_graph_placeholder = st.empty()

with col_dash:
    status_box = st.empty()
    ear_box = st.empty()
    mar_box = st.empty()
    fatigue_box = st.empty()

def update_dashboard(status, ear, mar, fatigue):
    # Mapping logic for UI Styles
    if "CALIBRATING" in status:
        color_class, icon, blink = "color-calib", "🔄", "blink"
    elif status == "ALERT":
        color_class, icon, blink = "color-alert", "🟢", ""
    elif status in ["SEMI-DROWSY", "SEMI-YAWNING"]:
        color_class, icon, blink = "color-warn", "🟡", ""
    elif status in ["DROWSY", "YAWNING", "HEAD DROP", "DISTRACTED"]:
        color_class, icon, blink = "color-danger", "🔴", "blink"
    else:
        color_class, icon, blink = "color-warn", "⚪", ""

    status_box.markdown(f'<div class="glass-card status-container {blink}"><div class="card-label">System Status</div><div class="status-text {color_class}">{icon} {status}</div></div>', unsafe_allow_html=True)
    ear_box.markdown(f'<div class="glass-card"><div class="card-label">Eye Aspect Ratio</div><div class="card-value">{ear:.2f}</div></div>', unsafe_allow_html=True)
    mar_box.markdown(f'<div class="glass-card"><div class="card-label">Mouth Aspect Ratio</div><div class="card-value">{mar:.2f}</div></div>', unsafe_allow_html=True)
    
    # Dynamic Fatigue Meter
    bar_color = "#10b981" if fatigue < 30 else "#facc15" if fatigue < 70 else "#ef4444"
    fatigue_box.markdown(f'<div class="glass-card"><div class="card-label">Fatigue Level ({fatigue}%)</div><div style="width:100%; height:12px; background:rgba(0,0,0,0.3); border-radius:6px; margin-top:10px; overflow:hidden;"><div style="width:{fatigue}%; height:100%; background:{bar_color}; transition:0.3s; box-shadow: 0 0 10px {bar_color};"></div></div></div>', unsafe_allow_html=True)

# ==========================================
# 5. CORE MONITORING LOGIC
# ==========================================

if run:
    # Initialize detector and camera
    if "detector" not in st.session_state:
        st.session_state.detector = DrowsinessDetector()
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    detector = st.session_state.detector
    cap = st.session_state.cap

    while run:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame = detector.process_frame(frame)

        # Get values from detector
        ear = getattr(detector, "last_ear", 0.0)
        mar = getattr(detector, "last_mar", 0.0)
        status = getattr(detector, "last_status", "UNKNOWN")

        # Telemetry Logging
        st.session_state.ear_list.append(ear)
        st.session_state.mar_list.append(mar)
        if len(st.session_state.ear_list) > 50: st.session_state.ear_list.pop(0)
        if len(st.session_state.mar_list) > 50: st.session_state.mar_list.pop(0)
        
        # Fatigue Accumulation
        if status in ["DROWSY", "YAWNING", "HEAD DROP", "DISTRACTED"]: 
            st.session_state.fatigue_score += 2
        else: 
            st.session_state.fatigue_score = max(0, st.session_state.fatigue_score - 1)
        st.session_state.fatigue_score = min(100, st.session_state.fatigue_score)

        # Log Data for Analytics Report
        curr_time = time.strftime("%H:%M:%S")
        st.session_state.drive_history.append({"Time": curr_time, "EAR": ear, "MAR": mar, "Status": status})

        # --- UI REFRESH ---
        main_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        ear_graph_placeholder.line_chart(st.session_state.ear_list, height=180)
        mar_graph_placeholder.line_chart(st.session_state.mar_list, height=180, color="#facc15")
        
        # Display special calibration status if active
        display_status = status
        if status == "CALIBRATING":
            count = getattr(detector, 'calibration_frame_count', 0)
            display_status = f"CALIBRATING ({count}/60)"
            
        update_dashboard(display_status, ear, mar, st.session_state.fatigue_score)
        time.sleep(0.01)

# ==========================================
# 6. POST-DRIVE ANALYTICS & STANDBY
# ==========================================
else:
    # Safely release camera
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    
    # Case 1: No data yet (Initial Standby)
    if not st.session_state.drive_history:
        update_dashboard("STANDBY", 0, 0, 0)
        main_placeholder.markdown('<div style="display:flex; justify-content:center; align-items:center; height:450px; background:rgba(30,41,59,0.2); border-radius:16px; border:1px dashed rgba(255,255,255,0.1); color:#94a3b8; font-weight:600;">AI MONITORING INACTIVE. TOGGLE "START MONITORING" TO BEGIN.</div>', unsafe_allow_html=True)
    
    # Case 2: Session just ended (Show Report)
    else:
        # Reset detector for next run
        if "detector" in st.session_state: del st.session_state.detector
        
        df = pd.DataFrame(st.session_state.drive_history)
        
        avg_ear = df['EAR'].mean()
        drowsy_count = len(df[df['Status'] == 'DROWSY'])
        distracted_count = len(df[df['Status'] == 'DISTRACTED'])
        yawn_count = len(df[df['Status'] == 'YAWNING'])

        with main_placeholder.container():
            st.markdown("### 📊 Comprehensive Session Report")
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg. Alertness", f"{avg_ear:.2f}")
            m2.metric("Microsleeps", drowsy_count, delta_color="inverse")
            m3.metric("Distractions", distracted_count, delta_color="inverse")
            m4.metric("Total Yawns", yawn_count)
            
            # Full Timeline Chart
            st.markdown('<div class="section-label" style="margin-top:20px;">Alertness & Fatigue Timeline</div>', unsafe_allow_html=True)
            st.line_chart(df.set_index('Time')[['EAR', 'MAR']])
            
            # Export Utility
            st.markdown("---")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Driving Analytics (.CSV)",
                data=csv,
                file_name=f"driver_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            if st.button("🔄 Start New Session"):
                st.session_state.drive_history = []
                st.rerun()