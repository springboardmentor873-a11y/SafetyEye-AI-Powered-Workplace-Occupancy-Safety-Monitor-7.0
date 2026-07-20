import os
import time
import tempfile
from datetime import datetime

import cv2
import pandas as pd
from prometheus_client import Counter, Gauge, start_http_server, REGISTRY
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. PROMETHEUS METRICS SETUP
# -----------------------------------------------------------------------------
def create_or_get_metric(metric_type, name, documentation):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_type(name, documentation)

# FIXED: Swapped to 8001 and added addr='0.0.0.0' to allow Docker traffic through
try:
    start_http_server(8001, addr='0.0.0.0')
except Exception:
    pass

GAUGE_OCCUPANCY = create_or_get_metric(Gauge, "safetyeye_occupancy_total", "Current total people detected")
GAUGE_VIOLATIONS = create_or_get_metric(Gauge, "safetyeye_violations_total", "Current total safety violations")
COUNTER_NO_HARDHAT = create_or_get_metric(Counter, "safetyeye_no_hardhat_total", "Cumulative missing hardhats detected")
COUNTER_NO_MASK = create_or_get_metric(Counter, "safetyeye_no_mask_total", "Cumulative missing masks detected")
COUNTER_NO_VEST = create_or_get_metric(Counter, "safetyeye_no_safety_vest_total", "Cumulative missing safety vests detected")


# -----------------------------------------------------------------------------
# 2. STREAMLIT PAGE CONFIG & SESSION STATE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafetyEye",
    page_icon="⛑️",
    layout="wide"
)

if "logs" not in st.session_state:
    st.session_state.logs = []


# -----------------------------------------------------------------------------
# 3. MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    possible_paths = [
        "best.pt",
        r"G:\My Drive\YOLO_Training\safety_project-6\weights\best.pt",
        "weights/best.pt"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return YOLO(path)
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading model: {e}")


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("System Configuration")
weights_input = st.sidebar.text_input("Path to trained YOLOv8 weights:", value="best.pt")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.40, 0.05)
uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4", "avi", "mov"])


# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
st.title("⛑️ SafetyEye: AI-Powered Workplace Safety Monitor")
st.caption("Real-time PPE compliance tracking and occupancy analytics.")

col_video, col_metrics = st.columns([2, 1])

with col_video:
    st.subheader("📺 Live Surveillance Feed")
    video_placeholder = st.empty()

with col_metrics:
    st.subheader("📊 Real-Time Analytics & Logs")
    
    st.write("Current Occupancy")
    metric_occ = st.empty()
    metric_occ.markdown("### 0")
    
    st.write("Active Violations")
    metric_viol = st.empty()
    metric_viol.markdown("### 0")
    
    alert_placeholder = st.empty()
    
    st.subheader("📋 Recent Compliance Logs")
    table_placeholder = st.empty()


# -----------------------------------------------------------------------------
# 6. VIDEO PROCESSING & INFERENCE LOOP
# -----------------------------------------------------------------------------
if uploaded_file and st.sidebar.button("🚀 Start Monitor Stream"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    last_log_time = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run Inference
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        annotated_frame = results.plot()

        # Extract Detections
        total_people = 0
        total_violations = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            if "person" in label:
                total_people += 1
            if "no-" in label or "missing" in label or "without" in label:
                total_violations += 1
                if "hardhat" in label or "helmet" in label:
                    COUNTER_NO_HARDHAT.inc()
                elif "mask" in label:
                    COUNTER_NO_MASK.inc()
                elif "vest" in label:
                    COUNTER_NO_VEST.inc()

        # Update Prometheus Gauges
        GAUGE_OCCUPANCY.set(total_people)
        GAUGE_VIOLATIONS.set(total_violations)

        # Update Live Dashboard Metrics
        metric_occ.markdown(f"### {total_people}")
        metric_viol.markdown(f"### {total_violations}")

        if total_violations > 0:
            alert_placeholder.warning(f"🚨 WARNING: {total_violations} Safety Equipment Violations Flagged!")
        else:
            alert_placeholder.success("✅ Full Compliance Maintained")

        # Rate-Limited Logging (Logs ONCE per second max)
        current_time = datetime.now().strftime("%H:%M:%S")
        if current_time != last_log_time:
            st.session_state.logs.insert(0, {
                "Timestamp": current_time,
                "Occupancy": total_people,
                "Violations": total_violations,
                "Status": "Violation" if total_violations > 0 else "Safe"
            })
            if len(st.session_state.logs) > 50:
                st.session_state.logs.pop()
            last_log_time = current_time

        # Increased to 300px for a larger, clearer view of log rows
        df_logs = pd.DataFrame(st.session_state.logs)
        table_placeholder.dataframe(
            df_logs,
            use_container_width=True,
            hide_index=True,
            height=300,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="medium"),
            }
        )

        # Display Frame
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()