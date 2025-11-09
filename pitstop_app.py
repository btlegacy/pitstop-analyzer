import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

def rolling_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='same')

def analyze_and_visualize_pitstop(video_path, output_path="pitstop_annotated.mp4", progress_bar=None, debug=False):
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

    x_centers = np.nan_to_num(x_centers, nan=np.nanmean(x_centers))
    y_centers = np.nan_to_num(y_centers, nan=np.nanmean(y_centers))
    x_smooth = rolling_average(x_centers, 5)
    y_smooth = rolling_average(y_centers, 5)

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    dx = rolling_average(dx, 5)
    dy = rolling_average(dy, 5)

    dxn = dx / (np.max(np.abs(dx)) + 1e-6)
    dyn = dy / (np.max(np.abs(dy)) + 1e-6)

    near_center = np.abs(x_smooth - width/2) < width * 0.08
    stationary = np.abs(dxn) < 0.02

    # --- Sustained motion logic ---
    def sustained(condition, frames_required):
        sustained_frames = np.convolve(condition.astype(int), np.ones(frames_required), 'same')
        return np.where(sustained_frames >= frames_required)[0]

    stop_candidates = sustained(near_center & stationary, int(fps * 0.7))
    stop_idx = stop_candidates[0] if len(stop_candidates) > 0 else 0

    up_candidates = sustained(dyn < -0.05, int(fps * 0.3))
    car_up_idx = up_candidates[0] if len(up_candidates) > 0 else stop_idx + int(fps * 2)

    down_candidates = sustained(dyn > 0.05, int(fps * 0.3))
    down_candidates = down_candidates[down_candidates > car_up_idx]
    car_down_idx = down_candidates[0] if len(down_candidates) > 0 else car_up_idx + int(fps * 36)

    depart_candidates = sustained(dxn > 0.05, int(fps * 1.0))
    depart_candidates = depart_candidates[depart_candidates > car_down_idx]
    depart_idx = depart_candidates[0] if len(depart_candidates) > 0 else len(x_smooth) - 1

    stop_time = round(stop_idx / fps, 2)
    car_up_time = round(car_up_idx / fps, 2)
    car_down_time = round(car_down_idx / fps, 2)
    depart_time = round(depart_idx / fps, 2)

    # --- Annotate video ---
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

    if debug:
        times = np.arange(len(x_smooth)) / fps
        # X Motion Plot
        plt.figure(figsize=(8, 3))
        plt.plot(times, x_smooth, label='X Position (Left-Right)', color='blue')
        plt.axvspan(stop_time, depart_time, color='orange', alpha=0.2, label='Pit Window')
        plt.axvline(stop_time, color='orange', linestyle='--', label='Stop')
        plt.axvline(depart_time, color='blue', linestyle='--', label='Depart')
        plt.xlabel('Time (s)')
        plt.ylabel('X Center')
        plt.legend()
        plt.tight_layout()
        plt.savefig("x_motion.png")
        plt.close()

        # Y Motion Plot
        plt.figure(figsize=(8, 3))
        plt.plot(times, y_smooth, label='Y Position (Lift-Drop)', color='black')
        plt.axvspan(car_up_time, car_down_time, color='green', alpha=0.2, label='Lift/Drop Window')
        plt.axvline(car_up_time, color='green', linestyle='--', label='Up')
        plt.axvline(car_down_time, color='red', linestyle='--', label='Down')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Center')
        plt.legend()
        plt.tight_layout()
        plt.savefig("y_motion.png")
        plt.close()

    return {
        "Car Stop Time (s)": stop_time,
        "Car Up Time (s)": car_up_time,
        "Car Down Time (s)": car_down_time,
        "Car Depart Time (s)": depart_time,
        "Pit Duration (s)": round(depart_time - stop_time, 2),
        "Annotated Video": output_path,
        "X Motion Plot": "x_motion.png" if debug else None,
        "Y Motion Plot": "y_motion.png" if debug else None
    }

# --- Streamlit UI ---
st.set_page_config(page_title="VSR Pit Stop Analyzer v9", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer")

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("🎥 Upload pit stop video", type=["mp4", "mov", "avi"])
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
analyze_btn = st.sidebar.button("Start Analysis")
progress_bar = st.sidebar.progress(0.0)

if analyze_btn and uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    output_path = os.path.join(tempfile.gettempdir(), "annotated_pitstop.mp4")
    st.sidebar.info("⏱️ Analyzing... please wait")

    results = analyze_and_visualize_pitstop(tmp_path, output_path, progress_bar, debug_mode)
    st.sidebar.success("✅ Analysis Complete!")

    st.subheader("📊 Pit Stop Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Car Stop", f"{results['Car Stop Time (s)']} s")
    col1.metric("Car Up", f"{results['Car Up Time (s)']} s")
    col2.metric("Car Down", f"{results['Car Down Time (s)']} s")
    col2.metric("Car Depart", f"{results['Car Depart Time (s)']} s")
    col3.metric("Pit Duration", f"{results['Pit Duration (s)']} s")

    st.divider()
    st.subheader("🎬 Annotated Video")
    st.video(results["Annotated Video"])

    if debug_mode:
        st.divider()
        st.subheader("📈 Motion Analysis (Debug Mode)")
        if results["X Motion Plot"]:
            st.image(results["X Motion Plot"], caption="X Motion (Left-Right)")
        if results["Y Motion Plot"]:
            st.image(results["Y Motion Plot"], caption="Y Motion (Lift-Drop)")
