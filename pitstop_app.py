# 🏁 VSR Pit Stop Analyzer v12.3 (Precision Geometry Model)
# ===========================================================
# Fully commented Streamlit application for pit stop event detection.
# Includes geometry-based detection for Car Stop, Car Up, Car Down, and Car Depart.
# This version uses pit line detection, centroid tracking, and robust timing logic.
# ===========================================================

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from math import degrees
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# CONFIGURATION SECTION
# -----------------------------------------------------------

CONFIG = {
    "FRAME_SAMPLE_RATE": 1,
    "FLOW_RESCALE": 0.5,
    "DEBUG_RESCALE": 0.5,
    "GROUND_ROI_HEIGHT": 0.2,
    "BOOM_ROI_HEIGHT": 0.1,
    "STOP_STABILITY_SEC": 1.0,
    "UP_THRESHOLD": 0.04,
    "DOWN_THRESHOLD": 0.02,
    "DEPART_STABILITY_SEC": 1.0,
    "ENABLE_TILT_CORRECTION": True,
    "MAX_TILT_ANGLE_DEG": 15,
}

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def rolling_average(data, window=5):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode="same")

def detect_tilt_angle(gray, max_angle=15):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0
    angles = []
    for rho, theta in lines[:, 0]:
        ang = degrees(theta)
        if 80 < ang < 100:
            angles.append(ang - 90)
    if not angles:
        return 0
    avg = np.mean(angles)
    return np.clip(avg, -max_angle, max_angle)

def apply_tilt_correction(frame, angle):
    if abs(angle) < 0.5:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h))

# -----------------------------------------------------------
# CORE ANALYZER ENGINE (Precision Geometry Model)
# -----------------------------------------------------------

def analyze_video(video_path, video_name, output_path,
                  progress_bar=None, debug=True, calibrate=True, frame_debug=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flow_scale = CONFIG["FLOW_RESCALE"]
    dbg_scale = CONFIG["DEBUG_RESCALE"]
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)

    ret, first_frame = cap.read()
    if not ret:
        st.error("Could not read video.")
        return None

    gray_init = cv2.cvtColor(cv2.resize(first_frame, (small_w, small_h)), cv2.COLOR_BGR2GRAY)
    tilt_angle = detect_tilt_angle(gray_init, CONFIG["MAX_TILT_ANGLE_DEG"])         if CONFIG["ENABLE_TILT_CORRECTION"] else 0
    first_frame = apply_tilt_correction(first_frame, tilt_angle)

    # Detect pit line (bright horizontal)
    ground_band = first_frame[int(h * 0.8):, :]
    gray_ground = cv2.cvtColor(ground_band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_ground, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
    pit_line_y = int(h * 0.9)
    if lines is not None and len(lines) > 0:
        ys = [y1 + int(h * 0.8) for [[x1, y1, x2, y2]] in lines if abs(y2 - y1) < 4]
        if ys:
            pit_line_y = int(np.median(ys))

    prev_small = cv2.resize(first_frame, (small_w, small_h))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    x_cent, y_cent, boom_distances, dx_list = [], [], [], []

    dbg_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    debug_path = os.path.join(tempfile.gettempdir(), f"pitstop_debug_{os.path.splitext(video_name)[0]}_{dbg_ts}.mp4")
    dbg_w, dbg_h = int(w * dbg_scale), int(h * dbg_scale)
    dbg_writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dbg_w, dbg_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % CONFIG["FRAME_SAMPLE_RATE"] != 0:
            frame_idx += 1
            continue

        frame = apply_tilt_correction(frame, tilt_angle)
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Detect car centroid
        _, mask = cv2.threshold(mag, 1.0, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > (small_w * small_h * 0.004):
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"] / flow_scale)
                    cy = int(M["m01"] / M["m00"] / flow_scale)
                    x_cent.append(cx)
                    y_cent.append(cy)
                    boom_distances.append(abs(pit_line_y - cy))
                    dx_list.append(np.mean(flow[..., 0]))
        prev_gray = gray
        frame_idx += 1

        # Debug overlay
        dbg_frame = cv2.resize(frame, (dbg_w, dbg_h))
        cv2.line(dbg_frame, (0, int(pit_line_y * dbg_scale)), (dbg_w, int(pit_line_y * dbg_scale)), (255, 255, 0), 2)
        if x_cent:
            cv2.circle(dbg_frame, (int(x_cent[-1] * dbg_scale), int(y_cent[-1] * dbg_scale)), 6, (0, 0, 255), -1)
        dbg_writer.write(dbg_frame)
        if progress_bar:
            progress_bar.progress(min(frame_idx / total, 1.0))

    cap.release()
    dbg_writer.release()

    # Event logic
    fps = max(fps, 30)
    x_s = rolling_average(x_cent, 5)
    y_s = rolling_average(y_cent, 5)
    dx_s = rolling_average(np.gradient(x_s), 5)
    boom_s = rolling_average(boom_distances, 5)

    direction = "Left → Right" if np.mean(dx_s[:int(fps)]) > 0 else "Right → Left"
    dir_sign = 1 if direction == "Left → Right" else -1

    # Stop detection
    low_vel = np.abs(dx_s) < 1.0
    stop_i = np.argmax(low_vel)
    base_boom = np.mean(boom_s[stop_i:stop_i+int(fps*2)])
    up_idx = next((i for i in range(stop_i, len(boom_s)) if boom_s[i] > base_boom * (1 + CONFIG["UP_THRESHOLD"])), stop_i + int(fps))
    down_idx = next((i for i in range(up_idx+int(fps), len(boom_s)) if boom_s[i] <= base_boom * (1 + CONFIG["DOWN_THRESHOLD"])), up_idx + int(fps*5))
    depart_i = next((i for i in range(down_idx, len(dx_s)) if dx_s[i]*dir_sign > 3.0 and y_s[i] > pit_line_y - 5), len(dx_s)-1)

    results = {
        "Car Stop Time (s)": round(stop_i / fps, 2),
        "Car Up Time (s)": round(up_idx / fps, 2),
        "Car Down Time (s)": round(down_idx / fps, 2),
        "Car Depart Time (s)": round(depart_i / fps, 2),
        "Pit Duration (s)": round((depart_i - stop_i) / fps, 2),
        "Debug Video": debug_path,
        "Pit Direction": direction,
    }
    return results

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------

st.set_page_config(page_title="VSR Pit Stop Analyzer v12.3", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer v12.3 (Precision Geometry Model)")

upl = st.sidebar.file_uploader("🎥 Upload Pit Stop Video", type=["mp4", "mov", "avi"])
start_btn = st.sidebar.button("▶️ Start Analysis")
progress_bar = st.sidebar.progress(0.0)

if start_btn and upl:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upl.read())
        tmp_path = tmp.name
    res = analyze_video(tmp_path, upl.name, tmp_path, progress_bar)
    st.success("✅ Analysis Complete!")
    st.metric("Car Stop", f"{res['Car Stop Time (s)']} s")
    st.metric("Car Up", f"{res['Car Up Time (s)']} s")
    st.metric("Car Down", f"{res['Car Down Time (s)']} s")
    st.metric("Car Depart", f"{res['Car Depart Time (s)']} s")
    st.metric("Pit Duration", f"{res['Pit Duration (s)']} s")
    st.video(res["Debug Video"])
    with open(res["Debug Video"], "rb") as f:
        st.download_button("⬇️ Download Debug MP4", data=f, file_name=os.path.basename(res["Debug Video"]), mime="video/mp4")
