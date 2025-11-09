import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

def analyze_and_visualize_pitstop(video_path, output_path="pitstop_annotated.mp4"):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    prev_gray, motion_scores, car_center_y, boxes_list = None, [], [], []

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
            cars = [b for b in boxes if int(b.cls) in [2, 7]]
            if cars:
                b = cars[0]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                y_center = int((y1 + y2)/2)
                car_center_y.append(y_center)
                boxes_list.append((x1,y1,x2,y2))
            else:
                car_center_y.append(np.nan)
                boxes_list.append(None)
        else:
            car_center_y.append(np.nan)
            boxes_list.append(None)

    cap.release()
    motion_scores = np.array(motion_scores)
    car_center_y = np.array(car_center_y)
    fps = max(fps, 1)
    times = np.arange(len(motion_scores))/fps
    motion_smooth = np.convolve(motion_scores, np.ones(15)/15, mode='same')
    y_smooth = np.convolve(np.nan_to_num(car_center_y, nan=np.nanmean(car_center_y)), np.ones(5)/5, mode='same')

    # --- Adaptive detection logic ---
    stop_threshold = np.percentile(motion_smooth, 20)
    depart_threshold = np.percentile(motion_smooth, 80)

    window = int(fps * 0.5)
    stop_idx, depart_idx = 0, len(motion_smooth) - 1
    for i in range(window, len(motion_smooth) - window):
        if np.all(motion_smooth[i-window:i] < stop_threshold):
            stop_idx = i
            break
    for i in range(stop_idx + window, len(motion_smooth) - window):
        if np.all(motion_smooth[i-window:i] > depart_threshold):
            depart_idx = i
            break

    dy = np.diff(y_smooth)
    dy_smooth = np.convolve(dy, np.ones(5)/5, mode='same')
    up_idx = np.argmin(dy_smooth)
    down_idx = np.argmax(dy_smooth)

    stop_time, car_up_time, car_down_time, depart_time = (
        stop_idx / fps,
        up_idx / fps,
        down_idx / fps,
        depart_idx / fps,
    )

    # --- Sanity corrections ---
    if car_up_time < stop_time:
        car_up_time = stop_time + 2
    if car_down_time < car_up_time:
        car_down_time = car_up_time + 35
    if depart_time < car_down_time:
        depart_time = car_down_time + 5

    events = {"Car Stop": int(stop_idx), "Car Up": int(up_idx), "Car Down": int(down_idx), "Car Depart": int(depart_idx)}

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(boxes_list) and boxes_list[frame_idx] is not None:
            x1, y1, x2, y2 = boxes_list[frame_idx]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        for event, idx in events.items():
            if abs(frame_idx-idx) < fps*0.5:
                cv2.putText(frame,f"{event} ({frame_idx/fps:.2f}s)",(50,80),
                            cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

    return {
        "Car Stop Time (s)": round(stop_time,2),
        "Car Up Time (s)": round(car_up_time,2),
        "Car Down Time (s)": round(car_down_time,2),
        "Car Depart Time (s)": round(depart_time,2),
        "Pit Duration (s)": round(depart_time-stop_time,2),
        "Annotated Video": output_path
    }

# ---- Streamlit UI ----
st.set_page_config(page_title="Pit Stop Analyzer", layout="wide")
st.title("🏎️ IMSA Pit Stop Analyzer v2")
st.markdown("Upload a pit-stop video to detect **Car Stop**, **Car Up**, **Car Down**, and **Car Depart** with adaptive timing logic.")

uploaded = st.file_uploader("Upload your pit-stop video", type=["mp4","mov","avi"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(tmp_path)
    st.write("⏱️ Analyzing… please wait.")
    output_path = os.path.join(tempfile.gettempdir(), "annotated_pitstop.mp4")
    results = analyze_and_visualize_pitstop(tmp_path, output_path=output_path)

    st.success("✅ Analysis complete!")
    st.json(results)
    st.video(output_path)
