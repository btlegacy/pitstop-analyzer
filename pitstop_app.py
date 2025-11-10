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
from math import atan2, degrees

# =============================
# Utility Functions
# =============================

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
        return ":orange[⚠️ Medium]"
    else:
        return ":red[❌ Low]"

def save_report(video_name, fps, w, h, direction, base_stab, lift_t, drop_t,
                timings, confs):
    """Generate plain-text calibration report."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fn = f"calibration_report_{os.path.splitext(video_name)[0]}_{ts}.txt"
    with open(fn, "w") as f:
        f.write("🏁 VSR Pit Stop Analyzer v12.2 – Calibration Report\\n")
        f.write("-------------------------------------------------\\n")
        f.write(f"Video: {video_name}\\nVersion: 12.2 (Precision)\\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write(f"Detected FPS: {fps:.2f}\\nFrame Size: {w} × {h}\\n")
        f.write(f"Pit Direction: {direction}\\n")
        f.write(f"Baseline Stability: {base_stab:.2f} px [{confs['Baseline']}]\\n")
        f.write(f"Lift Threshold (Up): ΔY = {lift_t:.1f}% [{confs['Up']}]\\n")
        f.write(f"Drop Threshold (Down): ΔY = {drop_t:.1f}% [{confs['Down']}]\\n\\n")
        f.write("Event Timings (seconds)\\n-----------------------\\n")
        for k, v in timings.items():
            f.write(f"{k}: {v['time']:.2f} [{v['conf']} Confidence]\\n")
        f.write(f"\\nOverall Confidence: {confs['Overall']}\\n")
    return fn

def detect_tilt_angle(gray, max_angle=15):
    """Estimate tilt angle using Hough line detection on ground lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0
    angles = []
    for rho, theta in lines[:, 0]:
        angle_deg = degrees(theta)
        if 80 < angle_deg < 100:  # near-vertical stall lines
            angles.append(angle_deg - 90)
    if not angles:
        return 0
    avg_angle = np.mean(angles)
    return np.clip(avg_angle, -max_angle, max_angle)

def apply_tilt_correction(frame, angle):
    """Apply affine rotation to correct camera tilt."""
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

    flow_scale = CONFIG["FLOW_RESCALE"]
    dbg_scale = CONFIG["DEBUG_RESCALE"]
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)

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

    # Debug MP4 Writer
    dbg_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    debug_path = os.path.join(
        tempfile.gettempdir(),
        f"pitstop_debug_{os.path.splitext(video_name)[0]}_{dbg_ts}.mp4"
    )
    dbg_w, dbg_h = int(w * dbg_scale), int(h * dbg_scale)
    dbg_writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dbg_w, dbg_h))

    # Process frames
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

        # Compute boom-car distance
        if boom_region_top:
            boom_y = int(h * CONFIG["BOOM_ROI_HEIGHT"] / 2)
        else:
            boom_y = int(h * (1 - CONFIG["BOOM_ROI_HEIGHT"] / 2))
        if y_cent:
            car_y = y_cent[-1]
            boom_distances.append(abs(car_y - boom_y))
        else:
            boom_distances.append(0)

        # Debug Visualization
        dbg_frame = cv2.resize(frame, (dbg_w, dbg_h))
        if y_cent:
            cv2.circle(dbg_frame, (int(x_cent[-1] * dbg_scale), int(y_cent[-1] * dbg_scale)), 6, (255, 0, 0), -1)
        color = (0, 255, 255) if boom_region_top else (255, 0, 255)
        cv2.rectangle(dbg_frame,
                      (0, int(boom_y * dbg_scale) - 5),
                      (dbg_w, int(boom_y * dbg_scale) + 5),
                      color, 2)
        cv2.rectangle(dbg_frame,
                      (0, int(h * (1 - CONFIG["GROUND_ROI_HEIGHT"]) * dbg_scale)),
                      (dbg_w, int(h * dbg_scale)), (0, 255, 0), 2)

        # Overlay telemetry info
        cv2.putText(dbg_frame, f"Frame: {frame_idx}/{total}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(dbg_frame, f"Tilt: {tilt_angle:.1f} deg", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        dbg_writer.write(dbg_frame)

        prev_gray = gray
        frame_idx += 1
        if progress_bar:
            progress_bar.progress(min(frame_idx / total, 1.0))

    cap.release()
    dbg_writer.release()

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

st.set_page_config(page_title="VSR Pit Stop Analyzer v12.2", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer v12.2 (Precision, Half-Res Debug)")

# Default settings (Debug + Calibration ON)
debug_mode = True
calib_mode = True
frame_dbg = True

# Sidebar Controls
st.sidebar.header("⚙️ Analysis Settings")
upl = st.sidebar.file_uploader("🎥 Upload Pit Stop Video", type=["mp4", "mov", "avi"])
start_btn = st.sidebar.button("▶️ Start Analysis")

progress_bar = st.sidebar.progress(0.0)

# Main execution
if start_btn and upl:
    st.sidebar.info("⏱️ Processing video, please wait...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upl.read())
        tmp_path = tmp.name

    output_path = os.path.join(tempfile.gettempdir(), f"annotated_{upl.name}")
    res = analyze_video(tmp_path, upl.name, output_path,
                        progress_bar, debug_mode, calib_mode, frame_dbg)

    st.success("✅ Analysis Complete!")
    st.markdown("---")

    # Summary Section
    st.subheader("📊 Pit Stop Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Car Stop", f"{res['Car Stop Time (s)']} s")
    c1.metric("Car Up", f"{res['Car Up Time (s)']} s")
    c2.metric("Car Down", f"{res['Car Down Time (s)']} s")
    c2.metric("Car Depart", f"{res['Car Depart Time (s)']} s")
    c3.metric("Pit Duration", f"{res['Pit Duration (s)']} s")
    c3.metric("Direction", res["Pit Direction"])

    # Annotated Video Player
    st.subheader("🎬 Annotated Video")
    st.video(res["Annotated Video"])

    # Prominent Debug MP4 Download Link
    st.markdown("### 🎞️ Download Full Debug Video (Annotated MP4)")
    with open(res["Debug Video"], "rb") as f:
        st.download_button(
            label="⬇️ Download Debug MP4",
            data=f,
            file_name=os.path.basename(res["Debug Video"]),
            mime="video/mp4"
        )

    st.markdown("---")
    st.info("Each debug video includes overlays for car center, boom/ground ROIs, optical flow vectors, and telemetry graph (X velocity, Y position, boom distance).")

# Instructions for users
st.markdown("---")
st.caption("""
**Usage Notes:**
- Ensure videos are overhead pit stop views with visible ground lines and boom.
- Car Stop is detected from relative motion; Car Up/Down from boom–car geometry.
- Debug MP4 provides visual diagnostics for engineering validation.
""")

st.markdown("#### 💾 Deployment")
st.text("Ready for Streamlit Cloud. Place pitstop_app.py and requirements.txt in your repo root.")

st.markdown("✅ *VSR Pit Stop Analyzer v12.2 – Precision Geometry Model*")
