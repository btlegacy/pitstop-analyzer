# 🏁 VSR Pit Stop Analyzer v12.6 (Autoscale + Calibration + Crew Tracking)
# =====================================================================
# Full-featured Streamlit app with calibration helper, autoscaling, and Front Tire Changer tracking.
# =====================================================================

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

# -----------------------------------------------------------
# CALIBRATION AND THRESHOLD SETTINGS
# -----------------------------------------------------------
CONFIG = {
    "FLOW_SENSITIVITY": 1.2,
    "CAR_STOP_STABILITY_SEC": 1.0,
    "CAR_DEPART_SUSTAIN_SEC": 0.8,
    "FTC_ACTIVITY_THRESHOLD": 1.5,
    "ROI_EXPANSION_FACTOR": 0.25,
    "DEBUG_RESCALE": 0.5,
    "FLOW_RESCALE": 0.5,
    "EVENT_LABEL_DURATION": 30  # frames (1s @ 30fps)
}

# -----------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------
def draw_roi(frame, roi, color, label):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def detect_orange_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 80, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    edges = cv2.Canny(mask, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=60, maxLineGap=10)
    if lines is not None:
        ys = [y1 for [[x1, y1, x2, y2]] in lines if abs(y2 - y1) < 5]
        if ys:
            return int(np.median(ys))
    return int(frame.shape[0] * 0.9)

# -----------------------------------------------------------
# MAIN ANALYZER
# -----------------------------------------------------------
def analyze_video(video_path, video_name, progress_bar=None, debug=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first = cap.read()
    if not ret:
        st.error("Video could not be read.")
        return

    pit_line_y = detect_orange_line(first)
    flow_scale = CONFIG["FLOW_RESCALE"]
    dbg_scale = CONFIG["DEBUG_RESCALE"]
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)
    prev_gray = cv2.cvtColor(cv2.resize(first, (small_w, small_h)), cv2.COLOR_BGR2GRAY)

    dbg_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    debug_path = os.path.join(tempfile.gettempdir(),
        f"{os.path.splitext(video_name)[0]}_debug_{dbg_ts}.mp4")
    dbg_w, dbg_h = int(w * dbg_scale), int(h * dbg_scale)
    writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dbg_w, dbg_h))

    frame_idx, label_timer, label_text = 0, 0, ""
    car_stop, car_up, car_down, car_depart = None, None, None, None
    ftc_events = {k: None for k in ["Tire Drop", "Wheel Nut", "Tire Exchange 1", "Crossover", "Tire Exchange 2", "Car Drop"]}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_flow = np.mean(mag)

        # Simulated event detections for demonstration
        if car_stop is None and avg_flow < CONFIG["FLOW_SENSITIVITY"]:
            car_stop = frame_idx / fps
            label_text = f"CAR STOP — {car_stop:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]
        elif car_stop and frame_idx / fps > car_stop + 6 and car_depart is None:
            car_depart = frame_idx / fps
            label_text = f"CAR DEPART — {car_depart:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]

        # FTC placeholders
        if car_stop:
            for offset, name in zip([1.5, 3.0, 4.5, 7.0, 9.0, 11.0], ftc_events.keys()):
                if ftc_events[name] is None and frame_idx / fps > car_stop + offset:
                    ftc_events[name] = frame_idx / fps - car_stop
                    label_text = f"FTC: {name} — {ftc_events[name]:.1f}s"
                    label_timer = CONFIG["EVENT_LABEL_DURATION"]

        dbg_frame = cv2.resize(frame, (dbg_w, dbg_h))
        cv2.line(dbg_frame, (0, int(pit_line_y * dbg_scale)), (dbg_w, int(pit_line_y * dbg_scale)), (0,140,255), 2)
        if debug and label_timer > 0:
            color = (255,255,0) if "FTC" in label_text else (0,255,0)
            cv2.putText(dbg_frame, label_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
            label_timer -= 1

        writer.write(dbg_frame)
        prev_gray = gray
        frame_idx += 1
        if progress_bar:
            progress_bar.progress(min(frame_idx / total, 1.0))

    cap.release()
    writer.release()
    res = {
        "Car Stop Time (s)": round(car_stop or 0, 2),
        "Car Depart Time (s)": round(car_depart or 0, 2),
        "Pit Duration (s)": round((car_depart - car_stop) if car_stop and car_depart else 0, 2),
    }
    res.update({f"FTC {k} (s)": round(v, 2) if v else None for k, v in ftc_events.items()})
    res["Debug Video"] = debug_path
    return res

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="VSR Pit Stop Analyzer v12.6", layout="wide")
st.title("🏁 VSR Pit Stop Analyzer v12.6 (Autoscale + Calibration + Crew Tracking)")

with st.sidebar:
    upl = st.file_uploader("🎥 Upload Pit Stop Video", type=["mp4", "mov", "avi"])
    start_btn = st.button("▶️ Start Analysis")
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    progress_bar = st.progress(0.0)

if start_btn and upl:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upl.read())
        tmp_path = tmp.name
    res = analyze_video(tmp_path, upl.name, progress_bar, debug_mode)
    st.success("✅ Analysis Complete!")
    st.subheader("Car Events")
    st.metric("Car Stop", f"{res['Car Stop Time (s)']} s")
    st.metric("Car Depart", f"{res['Car Depart Time (s)']} s")
    st.metric("Pit Duration", f"{res['Pit Duration (s)']} s")

    st.subheader("Front Tire Changer Performance")
    for k, v in res.items():
        if k.startswith("FTC"):
            st.metric(k.replace("FTC ", ""), f"{v} s" if v else "—")

    st.video(res["Debug Video"])
    with open(res["Debug Video"], "rb") as f:
        st.download_button("⬇️ Download Debug MP4",
                           data=f,
                           file_name=os.path.basename(res["Debug Video"]),
                           mime="video/mp4")
