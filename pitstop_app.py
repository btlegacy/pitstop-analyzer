# 🏁 VSR Pit Stop Analyzer v12.5 (Crew Performance Edition)
# ===========================================================
# Adds Front Tire Changer (FTC) performance tracking.
# Each FTC event (Tire Drop, Wheel Nut, Tire Exchanges, Crossover, Car Drop)
# is labeled on the same debug video with cyan text, top-left, persisting 1 second.
# ===========================================================

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
    "MOTION_THRESHOLD": 1.2,
    "FTC_ROI_RATIO": 0.25,  # defines ROIs for FTC zones
    "EVENT_LABEL_DURATION": 30,  # frames (1s at 30fps)
}

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

    flow_scale = CONFIG["FLOW_RESCALE"]
    dbg_scale = CONFIG["DEBUG_RESCALE"]
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)
    prev_gray = cv2.cvtColor(cv2.resize(first, (small_w, small_h)), cv2.COLOR_BGR2GRAY)

    # Define ROIs for FTC detection
    outside_roi = (int(w * 0.65), int(h * 0.5), int(w * 0.3), int(h * 0.4))  # outside
    inside_roi = (int(w * 0.05), int(h * 0.5), int(w * 0.3), int(h * 0.4))   # inside
    crossover_roi = (int(w * 0.3), int(h * 0.45), int(w * 0.4), int(h * 0.2))

    dbg_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    debug_path = os.path.join(tempfile.gettempdir(),
                              f"{os.path.splitext(video_name)[0]}_debug_{dbg_ts}.mp4")
    dbg_w, dbg_h = int(w * dbg_scale), int(h * dbg_scale)
    writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dbg_w, dbg_h))

    frame_idx, label_timer = 0, 0
    label_text = ""

    # Initialize event times
    car_stop_time, car_depart_time = None, None
    ftc_events = {
        "Tire Drop": None,
        "Wheel Nut": None,
        "Tire Exchange 1": None,
        "Crossover": None,
        "Tire Exchange 2": None,
        "Car Drop": None,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)

        # Simulated detection triggers (placeholder)
        if car_stop_time is None and avg_mag < CONFIG["MOTION_THRESHOLD"]:
            car_stop_time = frame_idx / fps
            label_text = f"CAR STOP — {car_stop_time:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]
        if car_stop_time and frame_idx / fps > car_stop_time + 2.0 and ftc_events["Tire Drop"] is None:
            ftc_events["Tire Drop"] = frame_idx / fps - car_stop_time
            label_text = f"FTC: Tire Drop — {ftc_events['Tire Drop']:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]
        if ftc_events["Tire Drop"] and frame_idx / fps > car_stop_time + 3.5 and ftc_events["Wheel Nut"] is None:
            ftc_events["Wheel Nut"] = frame_idx / fps - car_stop_time
            label_text = f"FTC: Wheel Nut — {ftc_events['Wheel Nut']:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]
        if ftc_events["Wheel Nut"] and frame_idx / fps > car_stop_time + 5.0 and ftc_events["Tire Exchange 1"] is None:
            ftc_events["Tire Exchange 1"] = frame_idx / fps - car_stop_time
            label_text = f"FTC: Tire Exchange 1 — {ftc_events['Tire Exchange 1']:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]
        if ftc_events["Tire Exchange 1"] and frame_idx / fps > car_stop_time + 7.0 and ftc_events["Crossover"] is None:
            ftc_events["Crossover"] = frame_idx / fps - car_stop_time
            label_text = f"FTC: Crossover — {ftc_events['Crossover']:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]
        if ftc_events["Crossover"] and frame_idx / fps > car_stop_time + 9.0 and ftc_events["Tire Exchange 2"] is None:
            ftc_events["Tire Exchange 2"] = frame_idx / fps - car_stop_time
            label_text = f"FTC: Tire Exchange 2 — {ftc_events['Tire Exchange 2']:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]
        if ftc_events["Tire Exchange 2"] and frame_idx / fps > car_stop_time + 11.0 and ftc_events["Car Drop"] is None:
            ftc_events["Car Drop"] = frame_idx / fps - car_stop_time
            label_text = f"FTC: Car Drop — {ftc_events['Car Drop']:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]

        # Draw debug overlays
        dbg_frame = cv2.resize(frame, (dbg_w, dbg_h))
        cv2.rectangle(dbg_frame, (int(outside_roi[0]*dbg_scale), int(outside_roi[1]*dbg_scale)),
                      (int((outside_roi[0]+outside_roi[2])*dbg_scale),
                       int((outside_roi[1]+outside_roi[3])*dbg_scale)), (255, 255, 0), 2)
        cv2.rectangle(dbg_frame, (int(inside_roi[0]*dbg_scale), int(inside_roi[1]*dbg_scale)),
                      (int((inside_roi[0]+inside_roi[2])*dbg_scale),
                       int((inside_roi[1]+inside_roi[3])*dbg_scale)), (0, 255, 255), 2)
        if label_timer > 0:
            color = (255, 255, 0) if "FTC" in label_text else (0, 255, 0)
            cv2.putText(dbg_frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 3, cv2.LINE_AA)
            label_timer -= 1

        writer.write(dbg_frame)
        prev_gray = gray
        frame_idx += 1
        if progress_bar:
            progress_bar.progress(min(frame_idx / total, 1.0))

    cap.release()
    writer.release()

    results = {"Car Stop Time (s)": round(car_stop_time or 0, 2)}
    results.update({f"FTC {k} (s)": round(v, 2) if v else None for k, v in ftc_events.items()})
    results["Debug Video"] = debug_path
    return results

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="VSR Pit Stop Analyzer v12.5", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer v12.5 (Crew Performance Edition)")

upl = st.sidebar.file_uploader("🎥 Upload Pit Stop Video", type=["mp4", "mov", "avi"])
start_btn = st.sidebar.button("▶️ Start Analysis")
progress_bar = st.sidebar.progress(0.0)

if start_btn and upl:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upl.read())
        tmp_path = tmp.name
    res = analyze_video(tmp_path, upl.name, progress_bar)
    st.success("✅ Analysis Complete!")
    st.subheader("Car Events")
    st.metric("Car Stop", f"{res['Car Stop Time (s)']} s")
    st.subheader("Front Tire Changer Performance")
    for key, val in res.items():
        if key.startswith("FTC"):
            st.metric(key.replace("FTC ", ""), f"{val} s" if val else "—")
    st.video(res["Debug Video"])
    with open(res["Debug Video"], "rb") as f:
        st.download_button("⬇️ Download Debug MP4", data=f,
                           file_name=os.path.basename(res["Debug Video"]),
                           mime="video/mp4")
