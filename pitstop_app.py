# ============================================================
# 🏁 VSR Pit Stop Analyzer v12.9 (Precision + Crew Edition)
# ------------------------------------------------------------
# Complete, production-ready Streamlit application for analyzing
# racing pit stops using optical flow and object motion tracking.
# Includes Car Events + Front Tire Changer performance stats.
# ============================================================

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="VSR Pit Stop Analyzer v12.9",
                   layout="wide",
                   page_icon="🏁")

# ============================================================
# DRAW HELPER (ROI BOXES + LABELS)
# ============================================================
def draw_roi(frame, roi, color=(0, 255, 0), label="ROI"):
    """Draws a labeled bounding box for calibration/debug overlays."""
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ============================================================
# FLOW FIELD VISUALIZATION (DEBUG OVERLAY)
# ============================================================
def draw_flow_field(frame, flow, step=16, scale=5):
    """Draws dense optical flow vectors with automatic shape handling."""
    try:
        h, w = flow.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].astype(np.int32)

        fx, fy = flow[y, x].T
        if fx.shape != fy.shape:
            min_h, min_w = min(fx.shape[0], fy.shape[0]), min(fx.shape[1], fy.shape[1])
            fx, fy = fx[:min_h, :min_w], fy[:min_h, :min_w]
            x, y = x[:min_h, :min_w], y[:min_h, :min_w]

        lines = np.vstack([
            x.flatten(),
            y.flatten(),
            (x + fx * scale).flatten(),
            (y + fy * scale).flatten()
        ]).T.reshape(-1, 2, 2)

        vis = frame.copy()
        for (x1, y1), (x2, y2) in lines.astype(np.int32):
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.3)
        return vis
    except Exception as e:
        print(f"[Flow Visualization Error] {e}")
        return frame
# ============================================================
# CALIBRATION PREVIEW FUNCTION
# ============================================================
def calibration_preview(frame):
    """Draws calibration overlays to verify ROIs."""
    try:
        h, w, _ = frame.shape
        car_roi = (int(w * 0.2), int(h * 0.4), int(w * 0.6), int(h * 0.25))
        ftc_outside_roi = (int(w * 0.65), int(h * 0.5), int(w * 0.3), int(h * 0.4))
        ftc_inside_roi = (int(w * 0.05), int(h * 0.5), int(w * 0.3), int(h * 0.4))

        overlay = frame.copy()
        draw_roi(overlay, car_roi, (0, 255, 0), "Car ROI")
        draw_roi(overlay, ftc_outside_roi, (255, 255, 0), "FTC Outside")
        draw_roi(overlay, ftc_inside_roi, (255, 255, 0), "FTC Inside")

        pit_line_x = int(w * 0.5)
        cv2.line(overlay, (pit_line_x, 0), (pit_line_x, h), (0, 165, 255), 2)
        cv2.putText(overlay, "Reference Pit Line", (pit_line_x + 10, int(h * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        return overlay
    except Exception:
        return np.zeros((480, 640, 3), dtype=np.uint8)

# ============================================================
# CAR EVENT DETECTION
# ============================================================
def analyze_car_events(video_path, debug=False):
    """Analyzes car stop, up, down, and depart based on optical flow in the car ROI."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev_frame = cap.read()
    if not ret:
        return {}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    roi = (int(w * 0.25), int(h * 0.35), int(w * 0.5), int(h * 0.3))

    motion_mags = []
    frame_idx = 0
    events = {"Car Stop Time (s)": None, "Car Up Time (s)": None,
              "Car Down Time (s)": None, "Car Depart Time (s)": None,
              "Pit Duration (s)": None}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        x, y, w_roi, h_roi = roi
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        roi_mag = np.mean(mag[y:y + h_roi, x:x + w_roi])
        motion_mags.append(roi_mag)
        prev_gray = gray
        frame_idx += 1

    motion_mags = np.array(motion_mags)
    t = np.arange(len(motion_mags)) / fps

    # Detect events using thresholds
    stop_idx = np.argmax(motion_mags < 0.5)
    up_idx = np.argmax(motion_mags > 2.0)
    down_idx = np.argmax((t > t[up_idx]) & (motion_mags < 1.0))
    depart_idx = np.argmax((t > t[down_idx]) & (motion_mags > 2.0))

    if stop_idx: events["Car Stop Time (s)"] = round(t[stop_idx], 2)
    if up_idx: events["Car Up Time (s)"] = round(t[up_idx], 2)
    if down_idx: events["Car Down Time (s)"] = round(t[down_idx], 2)
    if depart_idx: events["Car Depart Time (s)"] = round(t[depart_idx], 2)
    if stop_idx and depart_idx:
        events["Pit Duration (s)"] = round(t[depart_idx] - t[stop_idx], 2)

    cap.release()
    return events
# ============================================================
# FRONT TIRE CHANGER (FTC) PERFORMANCE ANALYSIS
# ============================================================
def analyze_ftc_performance(video_path, car_events, debug=False):
    """
    Estimates timing of key Front Tire Changer actions.
    Uses optical flow magnitude changes within defined ROIs.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev = cap.read()
    if not ret:
        return {}

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    # Define FTC work areas
    outside_roi = (int(w * 0.65), int(h * 0.55), int(w * 0.3), int(h * 0.35))
    inside_roi  = (int(w * 0.05), int(h * 0.55), int(w * 0.3), int(h * 0.35))

    mags_out, mags_in, t = [], [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        x, y, w1, h1 = outside_roi
        mags_out.append(np.mean(mag[y:y + h1, x:x + w1]))
        x, y, w2, h2 = inside_roi
        mags_in.append(np.mean(mag[y:y + h2, x:x + w2]))
        prev_gray = gray
        frame_idx += 1
        t.append(frame_idx / fps)
    cap.release()

    mags_out, mags_in, t = np.array(mags_out), np.array(mags_in), np.array(t)
    base_time = car_events.get("Car Stop Time (s)", 0)

    # Action inference
    stats = {
        "Time To Tire Drop (s)": None,
        "Time To Wheel Nut (s)": None,
        "First Tire Exchange (s)": None,
        "Crossover Time (s)": None,
        "Second Tire Exchange (s)": None,
        "Tire to Car Drop (s)": None
    }

    drop_idx = np.argmax(mags_out > 1.5)
    if drop_idx:
        stats["Time To Tire Drop (s)"] = round(t[drop_idx] - base_time, 2)
    nut_idx = np.argmax(mags_out > 2.5)
    if nut_idx:
        stats["Time To Wheel Nut (s)"] = round(t[nut_idx] - base_time, 2)
    exch1_idx = np.argmax((t > t[nut_idx]) & (mags_out < 1.0))
    cross_idx = np.argmax(mags_in > 1.5)
    exch2_idx = np.argmax((t > t[cross_idx]) & (mags_in < 1.0))
    cardrop_idx = np.argmax(t > car_events.get("Car Down Time (s)", 0))

    if exch1_idx: stats["First Tire Exchange (s)"] = round(t[exch1_idx] - base_time, 2)
    if cross_idx: stats["Crossover Time (s)"] = round(t[cross_idx] - base_time, 2)
    if exch2_idx: stats["Second Tire Exchange (s)"] = round(t[exch2_idx] - base_time, 2)
    if cardrop_idx: stats["Tire to Car Drop (s)"] = round(t[cardrop_idx] - base_time, 2)
    return stats


# ============================================================
# MAIN VIDEO ANALYSIS PIPELINE
# ============================================================
def analyze_video(video_path, progress_bar, debug=False, calibration=False):
    """
    Runs full analysis (Car events + FTC metrics) and produces annotated MP4.
    """
    car_events = analyze_car_events(video_path, debug)
    ftc_stats = analyze_ftc_performance(video_path, car_events, debug)

    # Prepare annotated video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/tmp/annotated_{timestamp}.mp4"
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w, h))
    frame_idx, total = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, prev_frame = cap.read()
    if not ret:
        return {}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        if debug:
            frame = draw_flow_field(frame, flow, step=16, scale=6)

        # Annotate car event markers
        now_t = frame_idx / fps
        for k, v in car_events.items():
            if v is not None and abs(now_t - v) < 0.3:
                cv2.putText(frame, k.replace("(s)", ""), (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        prev_gray = gray
        out.write(frame)
        frame_idx += 1
        if frame_idx % 10 == 0:
            progress_bar.progress(frame_idx / total)
    cap.release()
    out.release()
    progress_bar.progress(1.0)
    return {"Car Events": car_events, "FTC Stats": ftc_stats, "Annotated Video": output_path}


# ============================================================
# STREAMLIT INTERFACE
# ============================================================
st.title("🏁 VSR Pit Stop Analyzer v12.9 (Precision + Crew Edition)")
st.markdown("---")

debug_mode = st.sidebar.checkbox("🔍 Enable Debug Mode", value=False)
calib_mode = st.sidebar.checkbox("🧭 Calibration Preview", value=False)

uploaded_file = st.sidebar.file_uploader("🎥 Upload Pit Stop Video",
                                         type=["mp4", "mov", "avi", "mpeg4"])

if uploaded_file:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(uploaded_file.read())
    tmp_file.close()

    st.video(tmp_file.name)
    if st.button("▶️ Start Analysis"):
        progress_bar = st.progress(0)
        if calib_mode:
            cap = cv2.VideoCapture(tmp_file.name)
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.image(cv2.cvtColor(calibration_preview(frame), cv2.COLOR_BGR2RGB),
                         caption="Calibration Overlay", use_container_width=True)

        st.info("⏱️ Analyzing video — please wait while optical flow is processed...")
        result = analyze_video(tmp_file.name, progress_bar, debug_mode, calib_mode)

        st.success("✅ Analysis Complete!")
        st.markdown("### Car Event Summary")
        st.json(result["Car Events"])

        st.markdown("### Front Tire Changer Performance")
        st.json(result["FTC Stats"])

        st.markdown("### 🎬 Annotated Debug Video")
        st.video(result["Annotated Video"])
