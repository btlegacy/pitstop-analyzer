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

def analyze_and_visualize_pitstop(video_path, output_path="pitstop_annotated.mp4", progress_placeholder=None):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_gray, motion_scores, car_center_y, boxes_list = None, [], [], []

    progress = 0
    progress_bar = None
    if progress_placeholder:
        progress_bar = progress_placeholder.progress(0)

    # --- Pass 1: analyze video frame-by-frame ---
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_scores.append(np.sum(diff > 25))
        prev_gray = gray

        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            cars = [b for b in boxes if int(b.cls) in [2,7]]
            if cars:
                b = cars[0]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                y_center = int((y1 + y2) / 2)
                car_center_y.append(y_center)
                boxes_list.append((x1, y1, x2, y2))
            else:
                car_center_y.append(np.nan)
                boxes_list.append(None)
        else:
            car_center_y.append(np.nan)
            boxes_list.append(None)

        # Update progress bar
        frame_count += 1
        if progress_bar and total_frames > 0:
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))

    cap.release()

    # --- Signal processing ---
    motion_scores = np.array(motion_scores)
    car_center_y = np.array(car_center_y)
    fps = max(fps, 1)
    total_frames = len(motion_scores)
    video_duration = total_frames / fps
    times = np.arange(len(motion_scores)) / fps

    motion_smooth = np.convolve(motion_scores, np.ones(15)/15, mode='same')
    y_smooth = np.convolve(np.nan_to_num(car_center_y, nan=np.nanmean(car_center_y)), np.ones(5)/5, mode='same')

    stop_threshold = np.percentile(motion_smooth, 20)
    depart_threshold = np.percentile(motion_smooth, 80)
    window = int(fps * 0.5)
    stop_idx, depart_idx = 0, total_frames - 1

    for i in range(window, total_frames - window):
        if np.all(motion_smooth[i-window:i] < stop_threshold):
            stop_idx = i
            break
    for i in range(stop_idx + window, total_frames - window):
        if np.all(motion_smooth[i-window:i] > depart_threshold):
            depart_idx = i
            break

    dy = np.diff(y_smooth)
    dy_smooth = np.convolve(dy, np.ones(5)/5, mode='same')
    up_idx = np.argmin(dy_smooth)
    down_idx = np.argmax(dy_smooth)

    stop_time = min(stop_idx / fps, video_duration)
    car_up_time = min(up_idx / fps, video_duration)
    car_down_time = min(down_idx / fps, video_duration)
    depart_time = min(depart_idx / fps, video_duration)

    # Sanity adjustments
    if car_up_time < stop_time or (car_up_time - stop_time) > 10:
        car_up_time = stop_time + 2
    if car_down_time < car_up_time or (car_down_time - car_up_time) > 60:
        car_down_time = car_up_time + 36
    if depart_time < car_down_time or (depart_time - car_down_time) > 10:
        depart_time = car_down_time + 5

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    events = {"Car Stop": int(stop_idx), "Car Up": int(up_idx), "Car Down": int(down_idx), "Car Depart": int(depart_idx)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(boxes_list) and boxes_list[frame_idx] is not None:
            x1, y1, x2, y2 = boxes_list[frame_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        for event, idx in events.items():
            if abs(frame_idx - idx) < fps * 0.5:
                cv2.putText(frame, f"{event} ({frame_idx / fps:.2f}s)", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Finalize progress bar
    if progress_bar:
        progress_bar.progress(100)

    plt.figure(figsize=(8,3))
    plt.plot(times, motion_smooth, label="Motion Intensity")
    plt.plot(times[:len(dy_smooth)], dy_smooth*10, label="Lift/Drop Derivative (x10)")
    plt.axvline(stop_time, color="orange", linestyle="--", label="Stop")
    plt.axvline(car_up_time, color="green", linestyle="--", label="Up")
    plt.axvline(car_down_time, color="red", linestyle="--", label="Down")
    plt.axvline(depart_time, color="blue", linestyle="--", label="Depart")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pitstop_signals.png")
    plt.close()

    return {
        "Car Stop Time (s)": round(stop_time, 2),
        "Car Up Time (s)": round(car_up_time, 2),
        "Car Down Time (s)": round(car_down_time, 2),
        "Car Depart Time (s)": round(depart_time, 2),
        "Pit Duration (s)": round(depart_time - stop_time, 2),
        "Annotated Video": output_path,
        "Signal Plot": "pitstop_signals.png"
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Pit Stop Analyzer v7", layout="centered")
st.title("🏎️ IMSA Pit Stop Analyzer v7")
st.markdown("""
Upload a pit-stop video to automatically detect:
- 🟠 **Car Stop**
- 🟢 **Car Up (on air jacks)**
- 🔴 **Car Down**
- 🔵 **Car Depart**
""")

uploaded = st.file_uploader("🎥 Upload your pit-stop video", type=["mp4", "mov", "avi"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    status_placeholder = st.empty()
    video_placeholder = st.empty()
    progress_placeholder = st.empty()

    status_placeholder.info("⏱️ **Analyzing… please wait ⏳**")
    video_placeholder.video(tmp_path)

    output_path = os.path.join(tempfile.gettempdir(), "annotated_pitstop.mp4")
    results = analyze_and_visualize_pitstop(tmp_path, output_path=output_path, progress_placeholder=progress_placeholder)

    status_placeholder.success("✅ **Analysis Complete!**")

    st.subheader("📊 Pit Stop Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Car Stop", f"{results['Car Stop Time (s)']} s")
    col1.metric("Car Up", f"{results['Car Up Time (s)']} s")
    col2.metric("Car Down", f"{results['Car Down Time (s)']} s")
    col2.metric("Car Depart", f"{results['Car Depart Time (s)']} s")
    col3.metric("Pit Duration", f"{results['Pit Duration (s)']} s")

    st.divider()
    st.subheader("🎬 Annotated Video")
    st.video(output_path)

    if os.path.exists(results["Signal Plot"]):
        st.divider()
        st.subheader("📈 Motion & Lift Signal Plot")
        st.image(results["Signal Plot"], caption="Motion and lift signal analysis", use_column_width=True)
