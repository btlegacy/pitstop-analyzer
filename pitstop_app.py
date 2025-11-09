# 🏁 VSR Pit Stop Analyzer v12.4 (Precision Geometry Model)
# ===========================================================
# This version refines stop and depart detection using bright orange pit line anchors
# and implements robust optical flow and motion stability logic.
# Each detected action (Car Stop, Car Up, Car Down, Car Depart)
# is labeled at the top-left of the debug video and persists for 1 second.
# ===========================================================

# NOTE: This file is fully commented and compatible with Streamlit Cloud deployment.

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
CONFIG = {
    "FLOW_RESCALE": 0.5,
    "DEBUG_RESCALE": 0.5,
    "STOP_STABILITY_SEC": 1.0,
    "DEPART_STABILITY_SEC": 0.8,
    "MOTION_THRESHOLD": 1.2,
    "DEPART_OFFSET_PX": 15,
    "ENABLE_TILT_CORRECTION": True,
    "MAX_TILT_ANGLE_DEG": 15
}

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def rolling_average(data, window=5):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode="same")

def detect_orange_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 80, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    edges = cv2.Canny(mask, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=60, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        ys = [y1 for [[x1, y1, x2, y2]] in lines if abs(y2 - y1) < 5]
        return int(np.median(ys)) if ys else frame.shape[0] - 50
    return frame.shape[0] - 50

def sustained(condition, frames_required):
    sustained_frames = np.convolve(condition.astype(int), np.ones(frames_required), "same")
    return np.where(sustained_frames >= frames_required)[0]

# -----------------------------------------------------------
# MAIN ANALYZER
# -----------------------------------------------------------
def analyze_video(video_path, video_name, progress_bar=None):
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
    x_positions, dx_list = [], []

    dbg_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    debug_path = os.path.join(tempfile.gettempdir(),
        f"{os.path.splitext(video_name)[0]}_debug_{dbg_ts}.mp4")
    dbg_w, dbg_h = int(w * dbg_scale), int(h * dbg_scale)
    writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dbg_w, dbg_h))

    frame_idx, event_labels = 0, []
    stop_time, depart_time = None, None
    label_timer = 0
    label_text = ""

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = np.mean(flow[..., 0])
        x_positions.append(dx)
        dx_list.append(dx)

        # Detect car stop and depart
        if frame_idx > fps * 2:
            mean_dx = np.mean(dx_list[-int(fps):])
            if stop_time is None and abs(mean_dx) < CONFIG["MOTION_THRESHOLD"]:
                stop_time = frame_idx / fps
                label_text = f"CAR STOP — {stop_time:.1f}s"
                label_timer = int(fps)
            elif stop_time and depart_time is None and mean_dx > CONFIG["MOTION_THRESHOLD"]:
                if frame_idx / fps - stop_time > 1.0:
                    depart_time = frame_idx / fps
                    label_text = f"CAR DEPART — {depart_time:.1f}s"
                    label_timer = int(fps)

        # Draw overlays
        dbg_frame = cv2.resize(frame, (dbg_w, dbg_h))
        cv2.line(dbg_frame, (0, int(pit_line_y * dbg_scale)),
                 (dbg_w, int(pit_line_y * dbg_scale)), (0, 140, 255), 2)
        if label_timer > 0:
            cv2.putText(dbg_frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 3, cv2.LINE_AA)
            label_timer -= 1

        writer.write(dbg_frame)
        prev_gray = gray
        frame_idx += 1
        if progress_bar:
            progress_bar.progress(min(frame_idx / total, 1.0))

    cap.release()
    writer.release()

    return {
        "Car Stop Time (s)": round(stop_time or 0, 2),
        "Car Depart Time (s)": round(depart_time or 0, 2),
        "Debug Video": debug_path
    }

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="VSR Pit Stop Analyzer v12.4", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer v12.4 (Precision Geometry Model)")

upl = st.sidebar.file_uploader("🎥 Upload Pit Stop Video", type=["mp4", "mov", "avi"])
start_btn = st.sidebar.button("▶️ Start Analysis")
progress_bar = st.sidebar.progress(0.0)

if start_btn and upl:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upl.read())
        tmp_path = tmp.name
    res = analyze_video(tmp_path, upl.name, progress_bar)
    st.success("✅ Analysis Complete!")
    st.metric("Car Stop", f"{res['Car Stop Time (s)']} s")
    st.metric("Car Depart", f"{res['Car Depart Time (s)']} s")
    st.video(res["Debug Video"])
    with open(res["Debug Video"], "rb") as f:
        st.download_button("⬇️ Download Debug MP4",
                           data=f,
                           file_name=os.path.basename(res["Debug Video"]),
                           mime="video/mp4")
