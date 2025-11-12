# 🏁 VSR Pit Stop Analyzer v12.2 (Precision, Half-Res Debug)
# -----------------------------------------------------------
# Default Debug Mode: ON
# Default Calibration Mode: ON
# -----------------------------------------------------------

# =============================
# VSR Pit Stop Analyzer Configuration
# =============================
CONFIG = {
    "FRAME_SAMPLE_RATE": 1,         # Analyze every Nth frame (1 = all frames)
    "FLOW_RESCALE": 0.5,            # Optical flow downscale ratio
    "DEBUG_RESCALE": 0.5,           # Debug MP4 output resolution scale
    "GROUND_ROI_HEIGHT": 0.2,       # % of frame height for ground reference
    "BOOM_ROI_HEIGHT": 0.1,         # % of frame height for boom region
    "STOP_STABILITY_SEC": 1.0,      # Sustained stillness for Stop event
    "UP_THRESHOLD": 0.04,           # Δ boom distance (fraction of frame height)
    "DOWN_THRESHOLD": 0.02,         # Δ return threshold for Car Down
    "DEPART_STABILITY_SEC": 1.0,    # Sustained forward motion for Depart
    "CONFIDENCE_BASE": 0.9,         # Default detection confidence
    "ENABLE_TILT_CORRECTION": True, # Enable Hough tilt normalization
    "MAX_TILT_ANGLE_DEG": 15,       # Max tilt correction angle
}

# =============================
# Imports
# =============================
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime

# =============================
# Utility Functions
# =============================

def detect_tilt_angle(gray, max_deg=15):
    """Estimate scene tilt using Hough lines."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=gray.shape[1]//5, maxLineGap=15)
    if lines is None:
        return 0
    angles = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        if abs(angle) < max_deg:
            angles.append(angle)
    if len(angles) == 0:
        return 0
    return float(np.median(angles))


def apply_tilt_correction(frame, angle):
    if abs(angle) < 0.5:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h))
# =============================
# Core Analyzer Function
# =============================

def analyze_video(video_path, video_name, output_path,
                  progress_bar=None, debug=True,
                  calibrate=True, frame_debug=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total <= 0:
        st.error("Invalid video stream/fps")
        return

    # Rescale for flow & debug
    flow_scale = CONFIG["FLOW_RESCALE"]
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)
    dbg_w, dbg_h = int(w * CONFIG["DEBUG_RESCALE"]), int(h * CONFIG["DEBUG_RESCALE"]) 

    # Paths
    videostem = os.path.splitext(os.path.basename(video_name))[0]
    safe_stem = "".join([c for c in videostem if c.isalnum() or c in ("-","_")])
    debug_path = os.path.join(output_path, f"{safe_stem}_debug.mp4")

    # Initial read
    ret, prev = cap.read()
    if not ret:
        st.error("Video could not be read.")
        return
    prev_small = cv2.resize(prev, (small_w, small_h))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    # Tilt correction (on first frame)
    tilt_angle = 0
    if CONFIG["ENABLE_TILT_CORRECTION"]:
        tilt_angle = detect_tilt_angle(prev_gray, CONFIG["MAX_TILT_ANGLE_DEG"])

    x_cent, y_cent, dx_list, dy_list = [], [], [], []
    boom_distances = []

    frame_idx = 0

    # ---------- First pass: collect motion & geometry (no writing) ----------
    while True:
        if frame_idx > 0:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = prev

        frame = apply_tilt_correction(frame, tilt_angle)
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Split ground ROI vs car ROI
        ground_y1 = int(small_h * (1 - CONFIG["GROUND_ROI_HEIGHT"]))
        ground_roi = flow[ground_y1:, :, :]
        car_roi = flow[:ground_y1, :, :]

        ground_motion = np.mean(np.abs(ground_roi[..., 0]))
        car_motion_x = np.mean(np.abs(car_roi[..., 0]))
        car_motion_y = np.mean(np.abs(car_roi[..., 1]))
        rel_motion_x = car_motion_x - ground_motion

        # Track car centroid (largest moving contour)
        _, mask = cv2.threshold(mag, 1.0, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > (small_w * small_h * 0.005):
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x_cent.append(int(cx / flow_scale))
                    y_cent.append(int(cy / flow_scale))
                    dx_list.append(rel_motion_x)
                    dy_list.append(car_motion_y)

        # Detect boom region automatically
        top_motion = np.mean(np.abs(flow[:int(small_h * CONFIG["BOOM_ROI_HEIGHT"]), :, 1]))
        bottom_motion = np.mean(np.abs(flow[int(small_h * (1 - CONFIG["BOOM_ROI_HEIGHT"])):, :, 1]))
        boom_region_top = top_motion < bottom_motion  # True if boom likely on top

        # Approximate boom distance proxy (vertical flow in boom region)
        if boom_region_top:
            boom_flow = flow[:int(small_h * CONFIG["BOOM_ROI_HEIGHT"]), :, 1]
        else:
            boom_flow = flow[int(small_h * (1 - CONFIG["BOOM_ROI_HEIGHT"])):, :, 1]
        boom_distances.append(float(np.mean(np.abs(boom_flow))))

        prev_gray = gray
        frame_idx += 1
        if progress_bar and frame_idx % 10 == 0:
            progress_bar.progress(min(frame_idx / total, 1.0))

    # ---------- Event detection ----------
    def rolling_average(data, window=5):
        """Smooth numerical data using a moving average."""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode="same")

    def sustained(condition, frames_required):
        """Find sustained True intervals lasting at least given frame count."""
        sustained_frames = np.convolve(condition.astype(int), np.ones(frames_required), "same")
        return np.where(sustained_frames >= frames_required)[0]

    def confidence_label(level):
        """Color-coded Streamlit confidence label."""
        if level == "High":
            return ":green[✅ High]"
        elif level == "Medium":
            return ":orange[🟧 Medium]"
        return ":red[⚠ Low]"

    # Smooth motion data
    x_s = rolling_average(x_cent, 5)
    y_s = rolling_average(y_cent, 5)
    dx = rolling_average(np.gradient(x_s), 5)
    dy = rolling_average(np.gradient(y_s), 5)
    boom_s = rolling_average(boom_distances, 5)

    # Determine pit direction
    direction = "Left → Right" if np.mean(dx[:int(fps * 2)]) > 0 else "Right → Left"
    dir_sign = 1 if direction == "Left → Right" else -1

    # Car Stop Detection (relative motion)
    low_vel = np.abs(dx) < 1.0
    sustained_stop = sustained(low_vel, int(fps * CONFIG["STOP_STABILITY_SEC"]))
    stop_i = sustained_stop[0] if len(sustained_stop) > 0 else int(fps * 2)

    # Car Up/Down Detection (based on boom distance)
    base_boom = np.mean(boom_s[stop_i:stop_i + int(fps * 2)]) if len(boom_s) > stop_i else np.mean(boom_s)
    up_thresh = base_boom * (1 + CONFIG["UP_THRESHOLD"])
    down_thresh = base_boom * (1 + CONFIG["DOWN_THRESHOLD"])
    up_idx = next((i for i in range(stop_i, len(boom_s)) if boom_s[i] > up_thresh), stop_i + int(fps))
    down_idx = next((i for i in range(up_idx + int(fps * 5), len(boom_s)) if boom_s[i] <= down_thresh),
                    up_idx + int(fps * 10))

    # Car Depart Detection (forward motion)
    depart_i = next((i for i in range(down_idx, len(dx)) if dx[i] * dir_sign > 4.0), len(dx) - 1)

    # Logical sequence enforcement
    if up_idx <= stop_i:
        up_idx = stop_i + int(fps)
    if down_idx <= up_idx:
        down_idx = up_idx + int(fps * 5)
    if depart_i <= down_idx:
        depart_i = down_idx + int(fps * 3)

    # --- Second pass: write debug video with prominent STOP/DEPART banners ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    dbg_writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dbg_w, dbg_h))
    frame_idx = 0
    # show banners for 2 seconds
    show_len = int(2 * fps)
    stop_start, stop_end = max(0, int(stop_i)), min(total-1, int(stop_i)+show_len)
    depart_start, depart_end = max(0, int(depart_i)), min(total-1, int(depart_i)+show_len)

    def draw_banner(img, text):
        h2, w2 = img.shape[:2]
        overlay = img.copy()
        # semi-transparent bar
        cv2.rectangle(overlay, (0, int(0.1*h2)), (w2, int(0.35*h2)), (0,0,0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        # bold, big text with shadow
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0
        thickness = 5
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w2 - text_size[0]) // 2
        y = int(0.27*h2)
        cv2.putText(img, text, (x+3, y+3), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, (255,255,255), thickness, cv2.LINE_AA)

    # Recompute flow for tilt-corrected frames (for consistency of other small overlays)
    cap_ok = True
    prev = None
    prev_small = None
    prev_gray = None
    cap_ok, prev = cap.read()
    if not cap_ok:
        st.error("Video could not be re-read for annotation.")
    else:
        prev = apply_tilt_correction(prev, tilt_angle)
        prev_small = cv2.resize(prev, (small_w, small_h))
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        # write first frame
        while True:
            if frame_idx>0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = apply_tilt_correction(frame, tilt_angle)
                frame_small = cv2.resize(frame, (small_w, small_h))
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                prev_gray = gray
            else:
                frame = prev.copy()
                mag = None

            # build base debug frame (same as first pass but minimal)
            dbg_frame = cv2.resize(frame, (dbg_w, dbg_h))
            cv2.putText(dbg_frame, f"Frame: {frame_idx}/{total}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(dbg_frame, f"Tilt: {tilt_angle:.1f} deg", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            # event banners
            if stop_start <= frame_idx < stop_end:
                draw_banner(dbg_frame, "🚨 CAR STOP")
            if depart_start <= frame_idx < depart_end:
                draw_banner(dbg_frame, "🏁 CAR DEPART")

            dbg_writer.write(dbg_frame)
            frame_idx += 1
            if progress_bar:
                progress_bar.progress(min(frame_idx / total, 1.0))

    dbg_writer.release()
    cap.release()

    # Package results
    results = {
        "Car Stop Time (s)": round(stop_i / fps, 2),
        "Car Up Time (s)": round(up_idx / fps, 2),
        "Car Down Time (s)": round(down_idx / fps, 2),
        "Car Depart Time (s)": round(depart_i / fps, 2),
        "Pit Duration (s)": round((depart_i - stop_i) / fps, 2),
        "Annotated Video": debug_path,
        "Debug Video": debug_path,
        "Pit Direction": direction,
    }

    return results
# =============================
# Streamlit User Interface
# =============================
st.set_page_config(page_title="VSR Pit Stop Analyzer", page_icon="🏁", layout="wide")

st.title("🏁 VSR Pit Stop Analyzer — Precision Geometry Model")
st.caption("v12.2 — Optical flow + geometry without heavy detectors.")

uploaded_file = st.file_uploader("Upload overhead pit-stop video", type=["mp4", "mov", "m4v", "avi"]) 

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = os.path.join(tmpdir, uploaded_file.name)
        with open(tmp_video, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Analyzing… (two-pass: detect events, then render debug overlays)")
        progress = st.progress(0.0)
        results = analyze_video(tmp_video, uploaded_file.name, tmpdir, progress_bar=progress, debug=True)

        st.subheader("Results")
        col1, col2 = st.columns([1,1])
        with col1:
            st.metric("Car Stop (s)", results["Car Stop Time (s)"])
            st.metric("Car Up (s)", results["Car Up Time (s)"])
            st.metric("Car Down (s)", results["Car Down Time (s)"])
            st.metric("Car Depart (s)", results["Car Depart Time (s)"])
            st.metric("Pit Duration (s)", results["Pit Duration (s)"])
        with col2:
            st.video(results["Debug Video"])

        st.markdown("---")
        st.markdown("#### Notes")
        st.markdown("""
- The debug video now shows **large on-screen banners** for **STOP** and **DEPART** lasting ~2 seconds at the detected frames.
- The analyzer runs in **two passes**: first to detect timing, second to render the overlays.
- Tilt correction is applied once and reused for the overlay pass.
- This keeps your existing analysis logic intact.
- Debug video is written to `<video_stem>_debug.mp4`.
        """)

else:
    st.info("Upload a video to begin.")

st.markdown("---")
st.markdown("#### ℹ️ How it works")
st.markdown("""
1. **Optical Flow & ROIs** — Splits ground vs. car ROIs to estimate relative motion.
2. **Stop/Depart** — Finds sustained low velocity for **Stop** and strong directional motion for **Depart**.
3. **Two-pass output** — Overlays big **STOP/DEPART** banners for ~2 seconds when they occur.
4. **Confidence** — Basic heuristics kept lightweight for Streamlit Cloud.

- Debug MP4 provides visual diagnostics for engineering validation.
""")

st.markdown("#### 💾 Deployment")
st.text("Ready for Streamlit Cloud. Place pitstop_app.py and requirements.txt in your repo root.")

st.markdown("✅ *VSR Pit Stop Analyzer v12.2 – Precision Geometry Model*")
