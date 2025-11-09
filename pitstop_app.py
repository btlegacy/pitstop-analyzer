# 🏁 VSR Pit Stop Analyzer v10 – Fixed-Camera, Multi-Run, Calibration Mode
# Streamlit-based app for analyzing pit stop events (Stop, Up, Down, Depart)
# using motion-only sustained detection logic for overhead fixed-camera videos.

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import matplotlib.pyplot as plt
from datetime import datetime

# =============================
# Utility Functions
# =============================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

def rolling_average(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='same')

def sustained(condition, frames_required):
    sustained_frames = np.convolve(condition.astype(int), np.ones(frames_required), 'same')
    return np.where(sustained_frames >= frames_required)[0]

def confidence_label(conf):
    if conf == "High":
        return f":green[✅ {conf}]"
    elif conf == "Medium":
        return f":orange[⚠️ {conf}]"
    else:
        return f":red[❌ {conf}]"

def save_calibration_report(video_name, fps, width, height, baseline_stability,
                            lift_thresh, drop_thresh, smoothing, stop_window,
                            timings, confs):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"calibration_report_{os.path.splitext(video_name)[0]}_{ts}.txt"
    with open(filename, "w") as f:
        f.write("🏁 VSR Pit Stop Analyzer v10 – Calibration Report\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Video: {video_name}\n")
        f.write("Version: 10.0\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Detected FPS: {fps:.2f}\n")
        f.write(f"Frame Size: {width} × {height}\n")
        f.write(f"Baseline Stability: {baseline_stability:.2f} px [{confs['Baseline']}]\n")
        f.write(f"Lift Threshold (Up): ΔY = {lift_thresh:.1f}% [{confs['Up']}]\n")
        f.write(f"Drop Threshold (Down): ΔY = {drop_thresh:.1f}% [{confs['Down']}]\n")
        f.write(f"Smoothing Window: {smoothing} frames\n")
        f.write(f"Stop Stability Window: {stop_window:.1f} s\n\n")

        f.write("Event Timings (seconds)\n-----------------------\n")
        for k, v in timings.items():
            f.write(f"{k}: {v['time']:.2f} [{v['conf']} Confidence]\n")
        f.write(f"\nOverall Confidence: {confs['Overall']}\n")
        f.write(f"Report saved as: {filename}\n")
    return filename

# =============================
# Core Analyzer
# =============================

def analyze_and_visualize_pitstop(video_path, video_name, output_path,
                                  progress_bar=None, debug=False, calibrate=False):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    x_centers, y_centers = [], []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            cars = [b for b in boxes if int(b.cls) in [2, 7]]
            if cars:
                b = cars[0]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                x_centers.append(cx)
                y_centers.append(cy)
            else:
                x_centers.append(np.nan)
                y_centers.append(np.nan)
        else:
            x_centers.append(np.nan)
            y_centers.append(np.nan)
        frame_idx += 1
        if progress_bar:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
    cap.release()

    # Replace NaNs
    x_centers = np.nan_to_num(x_centers, nan=np.nanmean(x_centers))
    y_centers = np.nan_to_num(y_centers, nan=np.nanmean(y_centers))

    # Smooth signals
    x_smooth = rolling_average(x_centers, 5)
    y_smooth = rolling_average(y_centers, 5)
    dx = rolling_average(np.gradient(x_smooth), 5)
    dy = rolling_average(np.gradient(y_smooth), 5)
    dxn = dx / (np.max(np.abs(dx)) + 1e-6)
    dyn = dy / (np.max(np.abs(dy)) + 1e-6)

    # Threshold logic
    near_center = np.abs(x_smooth - width/2) < width * 0.08
    stationary = np.abs(dxn) < 0.02

    stop_candidates = sustained(near_center & stationary, int(fps * 1.0))
    stop_idx = stop_candidates[0] if len(stop_candidates) > 0 else 0

    baseline_y = np.mean(y_smooth[stop_idx:stop_idx + int(fps * 2)])
    lift_thresh = 0.02 * height
    drop_thresh = 0.02 * height

    up_candidates = np.where(y_smooth < baseline_y - lift_thresh)[0]
    car_up_idx = up_candidates[0] if len(up_candidates) > 0 else stop_idx + int(fps * 2)

    down_candidates = np.where(y_smooth > baseline_y + drop_thresh)[0]
    down_candidates = down_candidates[down_candidates > car_up_idx + int(fps * 10)]
    car_down_idx = down_candidates[0] if len(down_candidates) > 0 else car_up_idx + int(fps * 35)

    depart_candidates = np.where((dxn > 0.05) & (x_smooth > width * 0.8))[0]
    depart_candidates = depart_candidates[depart_candidates > car_down_idx]
    depart_idx = depart_candidates[0] if len(depart_candidates) > 0 else len(x_smooth) - 1

    stop_time = round(stop_idx / fps, 2)
    car_up_time = round(car_up_idx / fps, 2)
    car_down_time = round(car_down_idx / fps, 2)
    depart_time = round(depart_idx / fps, 2)

    # Confidence approximation
    confs = {
        "Stop": "Medium",
        "Up": "High" if len(up_candidates) else "Medium",
        "Down": "High" if len(down_candidates) else "Medium",
        "Depart": "High" if len(depart_candidates) else "Medium",
        "Baseline": "High",
        "Overall": "High"
    }

    # Annotate video
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    events = {
        "Car Stop": stop_idx,
        "Car Up": car_up_idx,
        "Car Down": car_down_idx,
        "Car Depart": depart_idx
    }
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for event, idx in events.items():
            if abs(frame_num - idx) < fps * 0.5:
                cv2.putText(frame, f"{event} ({frame_num / fps:.2f}s)", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        out.write(frame)
        frame_num += 1
    cap.release()
    out.release()

    results = {
        "Car Stop Time (s)": stop_time,
        "Car Up Time (s)": car_up_time,
        "Car Down Time (s)": car_down_time,
        "Car Depart Time (s)": depart_time,
        "Pit Duration (s)": round(depart_time - stop_time, 2),
        "Annotated Video": output_path,
    }

    if debug:
        times = np.arange(len(x_smooth)) / fps
        # X motion
        plt.figure(figsize=(8, 3))
        plt.plot(times, x_smooth, label='X Position')
        plt.axvspan(stop_time, depart_time, color='orange', alpha=0.2)
        plt.xlabel("Time (s)")
        plt.ylabel("X Center")
        plt.legend()
        plt.tight_layout()
        plt.savefig("x_motion.png")
        plt.close()

        # Y motion
        plt.figure(figsize=(8, 3))
        plt.plot(times, y_smooth, label='Y Position')
        plt.axvspan(car_up_time, car_down_time, color='green', alpha=0.2)
        plt.xlabel("Time (s)")
        plt.ylabel("Y Center")
        plt.legend()
        plt.tight_layout()
        plt.savefig("y_motion.png")
        plt.close()
        results["X Motion Plot"] = "x_motion.png"
        results["Y Motion Plot"] = "y_motion.png"

    if calibrate:
        timings = {
            "Car Stop": {"time": stop_time, "conf": confs["Stop"]},
            "Car Up": {"time": car_up_time, "conf": confs["Up"]},
            "Car Down": {"time": car_down_time, "conf": confs["Down"]},
            "Car Depart": {"time": depart_time, "conf": confs["Depart"]}
        }
        report = save_calibration_report(video_name, fps, width, height,
                                         np.std(y_smooth[stop_idx:stop_idx + int(fps * 2)]),
                                         (lift_thresh / height) * 100,
                                         (drop_thresh / height) * 100,
                                         5, 1.0, timings, confs)
        results["Calibration Report"] = report

    return results

# =============================
# Streamlit App
# =============================

st.set_page_config(page_title="VSR Pit Stop Analyzer v10", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer v10")

if "run_count" not in st.session_state:
    st.session_state["run_count"] = 0

debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
calibration_mode = st.sidebar.checkbox("Enable Calibration Mode", value=False)
uploaded = st.sidebar.file_uploader("🎥 Upload pit stop video", type=["mp4", "mov", "avi"])
analyze_btn = st.sidebar.button("Start Analysis")
progress_bar = st.sidebar.progress(0.0)

if analyze_btn and uploaded:
    st.session_state["run_count"] += 1
    run_id = st.session_state["run_count"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    output_path = os.path.join(tempfile.gettempdir(), f"annotated_run{run_id}.mp4")
    st.sidebar.info("⏱️ Analyzing... please wait")

    results = analyze_and_visualize_pitstop(tmp_path, uploaded.name, output_path,
                                            progress_bar, debug_mode, calibration_mode)

    st.sidebar.success("✅ Analysis Complete!")

    st.markdown(f"---\n## 🏁 Run #{run_id}: {uploaded.name}\n")
    st.subheader("📊 Pit Stop Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Car Stop", f"{results['Car Stop Time (s)']} s")
    col1.metric("Car Up", f"{results['Car Up Time (s)']} s")
    col2.metric("Car Down", f"{results['Car Down Time (s)']} s")
    col2.metric("Car Depart", f"{results['Car Depart Time (s)']} s")
    col3.metric("Pit Duration", f"{results['Pit Duration (s)']} s")

    st.subheader("🎬 Annotated Video")
    st.video(results["Annotated Video"])

    if debug_mode:
        st.subheader("📈 Motion Analysis (Debug Mode)")
        if "X Motion Plot" in results:
            st.image(results["X Motion Plot"], caption="X Motion (Left-Right)")
        if "Y Motion Plot" in results:
            st.image(results["Y Motion Plot"], caption="Y Motion (Lift-Drop)")

    if calibration_mode and "Calibration Report" in results:
        st.subheader("🧮 Calibration Report")
        with open(results["Calibration Report"], "r") as f:
            st.text(f.read())
        with open(results["Calibration Report"], "rb") as f:
            st.download_button("💾 Save Calibration Report", data=f, file_name=os.path.basename(results["Calibration Report"]))

st.markdown("---\n### 📤 Analyze Another Video")
st.info("Upload a new video above to start a new run. Each analysis will appear below the previous one.")
