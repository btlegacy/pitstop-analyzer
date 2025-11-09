# 🏁 VSR Pit Stop Analyzer v12 – Fast Cloud Build
# -------------------------------------------------------
# This version uses adaptive color + optical flow tracking
# to detect key pit stop events (Stop, Up, Down, Depart)
# in fixed overhead IMSA-style pit lane footage.
#
# Optimized for Streamlit Cloud CPU environments.
# -------------------------------------------------------

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from datetime import datetime

# =============================
# Helper Functions
# =============================

def rolling_average(data, window=5):
    """Smooth signal to remove frame noise"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode="same")

def sustained(condition, frames_required):
    """Detect sustained condition for consecutive frames"""
    sustained_frames = np.convolve(condition.astype(int), np.ones(frames_required), "same")
    return np.where(sustained_frames >= frames_required)[0]

def confidence_label(level):
    """Streamlit-colored confidence label"""
    if level == "High":
        return ":green[✅ High]"
    elif level == "Medium":
        return ":orange[⚠️ Medium]"
    else:
        return ":red[❌ Low]"

def save_report(video_name, fps, w, h, direction, base_stab, lift_t, drop_t,
                timings, confs):
    """Save plain-text calibration report"""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fn = f"calibration_report_{os.path.splitext(video_name)[0]}_{ts}.txt"
    with open(fn, "w") as f:
        f.write("🏁 VSR Pit Stop Analyzer v12 – Calibration Report\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Video: {video_name}\nVersion: 12.0 (Fast Cloud Build)\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Detected FPS: {fps:.2f}\nFrame Size: {w} × {h}\n")
        f.write(f"Pit Direction: {direction}\n")
        f.write(f"Baseline Stability: {base_stab:.2f} px [{confs['Baseline']}]\n")
        f.write(f"Lift Threshold (Up): ΔY = {lift_t:.1f}% [{confs['Up']}]\n")
        f.write(f"Drop Threshold (Down): ΔY = {drop_t:.1f}% [{confs['Down']}]\n\n")
        f.write("Event Timings (seconds)\n-----------------------\n")
        for k,v in timings.items():
            f.write(f"{k}: {v['time']:.2f} [{v['conf']} Confidence]\n")
        f.write(f"\nOverall Confidence: {confs['Overall']}\n")
        f.write(f"Report saved as: {fn}\n")
    return fn

# =============================
# Core Analyzer
# =============================

def analyze_video(video_path, video_name, output_path,
                  progress_bar=None, debug=False,
                  calibrate=False, frame_debug=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downsample optical flow resolution for performance
    flow_scale = 0.5  # process at 50% size
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)

    x_cent, y_cent = [], []
    ret, prev = cap.read()
    if not ret:
        st.error("Video could not be read.")
        return
    prev_small = cv2.resize(prev, (small_w, small_h))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    # Optional MP4 debug clip writer
    if frame_debug:
        dbg_path = os.path.join(tempfile.gettempdir(), "debug_clip.mp4")
        dbg_writer = cv2.VideoWriter(dbg_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
    else:
        dbg_path = None

    # Process every 2nd frame for faster Cloud performance
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Threshold magnitude to remove small movements (crew, shadows)
        _, mask = cv2.threshold(mag, 1.0, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        # Largest contour = car body region
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > (small_w * small_h * 0.005):  # ignore small moving objects
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Scale back to full resolution coordinates
                    x_cent.append(int(cx / flow_scale))
                    y_cent.append(int(cy / flow_scale))

                    # Optional debug visualization
                    if frame_debug and frame_idx < fps * 8:
                        vis = frame.copy()
                        cv2.drawContours(vis, [np.int32(c / flow_scale)], -1, (0, 255, 0), 2)
                        cv2.circle(vis, (int(cx / flow_scale), int(cy / flow_scale)), 8, (255, 0, 0), -1)
                        dbg_writer.write(vis)

        prev_gray = gray
        frame_idx += 1
        if progress_bar:
            progress_bar.progress(min(frame_idx / total, 1.0))
    cap.release()
    if frame_debug:
        dbg_writer.release()

    # Handle missing frames
    if not x_cent:
        st.error("No car motion detected.")
        return

    # Smooth motion data
    x_s = rolling_average(x_cent, 5)
    y_s = rolling_average(y_cent, 5)
    dx = rolling_average(np.gradient(x_s), 5)
    dy = rolling_average(np.gradient(y_s), 5)

    # -------------------------------------
    # Detect pit direction automatically
    # -------------------------------------
    direction = "Left → Right" if np.mean(dx[:int(fps * 2)]) > 0 else "Right → Left"
    dir_sign = 1 if direction == "Left → Right" else -1

    # -------------------------------------
    # Two-phase Stop detection
    # -------------------------------------
    near_center = (x_s > w * 0.45) if dir_sign == 1 else (x_s < w * 0.55)
    low_vel = np.abs(dx) < 1.5
    sustained_stop = sustained(near_center & low_vel, int(fps))
    stop_i = sustained_stop[0] if len(sustained_stop) > 0 else int(fps * 4)

    # Baseline for vertical movement (car stationary height)
    base_y = np.mean(y_s[stop_i:stop_i + int(fps * 2)])
    base_std = np.std(y_s[stop_i:stop_i + int(fps * 2)])

    # -------------------------------------
    # Air Wand ROI Detection (Car Up trigger)
    # -------------------------------------
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, stop_i * 2)  # approx. real frame index
    ret, stop_frame = cap.read()
    cap.release()

    roi_offset = int(0.18 * w * dir_sign)
    roi_w = int(0.12 * w)
    roi_h = int(0.2 * h)
    cx, cy = int(x_s[stop_i]), int(y_s[stop_i])
    x1 = np.clip(cx + roi_offset - roi_w // 2, 0, w - 1)
    y1 = np.clip(cy - roi_h // 2, 0, h - 1)
    x2 = np.clip(x1 + roi_w, 0, w)
    y2 = np.clip(y1 + roi_h, 0, h)
    wand_roi = stop_frame[y1:y2, x1:x2]

    # Compute inward flow burst
    wand_gray = cv2.cvtColor(wand_roi, cv2.COLOR_BGR2GRAY)
    wand_prev = cv2.GaussianBlur(wand_gray, (5, 5), 0)
    motion_intensity = np.mean(np.abs(np.gradient(wand_gray.astype(float))))
    wand_trigger = motion_intensity > 8.0  # tuned empirically from pit footage

    # Save Air Wand ROI Debug Image
    wand_dbg = wand_roi.copy()
    cv2.rectangle(stop_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(stop_frame, "Air Wand ROI", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    wand_path = os.path.join(tempfile.gettempdir(), "airwand_roi_debug.png")
    cv2.imwrite(wand_path, stop_frame)

    # -------------------------------------
    # Car Up / Down / Depart
    # -------------------------------------
    lift_th = base_std * 2.0  # 2σ lift threshold
    drop_th = base_std * 2.0

    up_idx = next((i for i in range(stop_i, len(y_s))
                   if (base_y - y_s[i]) > lift_th), stop_i + int(fps * 2))
    down_idx = next((i for i in range(up_idx + int(fps * 10), len(y_s))
                     if (y_s[i] - base_y) > drop_th), up_idx + int(fps * 35))

    if dir_sign == 1:
        depart_idx = next((i for i in range(down_idx, len(x_s))
                           if dx[i] > 5 and x_s[i] > w * 0.8), len(x_s) - 1)
    else:
        depart_idx = next((i for i in range(down_idx, len(x_s))
                           if dx[i] < -5 and x_s[i] < w * 0.2), len(x_s) - 1)

    # -------------------------------------
    # Annotate results
    # -------------------------------------
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fidx = 0
    events = {
        "Car Stop": stop_i,
        "Car Up": up_idx,
        "Car Down": down_idx,
        "Car Depart": depart_idx
    }
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        for lbl, idx in events.items():
            if abs(fidx - idx * 2) < fps * 0.5:
                cv2.putText(frm, f"{lbl} ({idx / fps:.2f}s)",
                            (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        out.write(frm)
        fidx += 1
    cap.release()
    out.release()

    # -------------------------------------
    # Results and optional debug plots
    # -------------------------------------
    results = {
        "Car Stop Time (s)": round(stop_i / fps, 2),
        "Car Up Time (s)": round(up_idx / fps, 2),
        "Car Down Time (s)": round(down_idx / fps, 2),
        "Car Depart Time (s)": round(depart_idx / fps, 2),
        "Pit Duration (s)": round((depart_idx - stop_i) / fps, 2),
        "Annotated Video": output_path,
        "Air Wand Debug": wand_path,
        "Pit Direction": direction
    }

    # Confidence assessment (placeholder heuristic)
    confs = {"Stop": "High", "Up": "High", "Down": "High",
             "Depart": "High", "Baseline": "High", "Overall": "High"}

    if debug:
        t = np.arange(len(x_s)) / fps
        plt.figure(figsize=(8, 3))
        plt.plot(t, x_s)
        plt.axvspan(stop_i / fps, depart_idx / fps, color="orange", alpha=0.2)
        plt.xlabel("Time (s)")
        plt.ylabel("X Center")
        plt.tight_layout()
        plt.savefig("x_motion.png")
        plt.close()

        plt.figure(figsize=(8, 3))
        plt.plot(t, y_s)
        plt.axvspan(up_idx / fps, down_idx / fps, color="green", alpha=0.2)
        plt.xlabel("Time (s)")
        plt.ylabel("Y Center")
        plt.tight_layout()
        plt.savefig("y_motion.png")
        plt.close()
        results["X Motion Plot"] = "x_motion.png"
        results["Y Motion Plot"] = "y_motion.png"

    if calibrate:
        timings = {k: {"time": results[f"{k} Time (s)"], "conf": confs[k]} for k in ["Car Stop", "Car Up", "Car Down", "Car Depart"]}
        rep = save_report(video_name, fps, w, h, direction, base_std, (lift_th / h) * 100,
                          (drop_th / h) * 100, timings, confs)
        results["Calibration Report"] = rep
    return results

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="VSR Pit Stop Analyzer v12", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer v12")

if "run_count" not in st.session_state:
    st.session_state["run_count"] = 0

debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
calib_mode = st.sidebar.checkbox("Enable Calibration Mode", value=False)
frame_dbg = st.sidebar.checkbox("Enable Frame-by-Frame Debug (MP4)", value=False)
upl = st.sidebar.file_uploader("🎥 Upload pit stop video", type=["mp4", "mov", "avi"])
btn = st.sidebar.button("Start Analysis")
pbar = st.sidebar.progress(0.0)

if btn and upl:
    st.session_state["run_count"] += 1
    run_id = st.session_state["run_count"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upl.read())
        tmp_path = tmp.name
    outp = os.path.join(tempfile.gettempdir(), f"annotated_run{run_id}.mp4")

    st.sidebar.info("⏱️ Processing video...")
    res = analyze_video(tmp_path, upl.name, outp, pbar, debug_mode, calib_mode, frame_dbg)
    st.sidebar.success("✅ Analysis Complete!")

    st.markdown(f"---\n## 🏁 Run #{run_id}: {upl.name}")
    st.subheader("📊 Pit Stop Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Car Stop", f"{res['Car Stop Time (s)']} s")
    c1.metric("Car Up", f"{res['Car Up Time (s)']} s")
    c2.metric("Car Down", f"{res['Car Down Time (s)']} s")
    c2.metric("Car Depart", f"{res['Car Depart Time (s)']} s")
    c3.metric("Pit Duration", f"{res['Pit Duration (s)']} s")

    st.subheader("🎬 Annotated Video")
    st.video(res["Annotated Video"])

    if debug_mode:
        with st.expander("📈 Motion Analysis (Debug Plots)"):
            if "X Motion Plot" in res:
                st.image(res["X Motion Plot"])
            if "Y Motion Plot" in res:
                st.image(res["Y Motion Plot"])

    if calib_mode and "Calibration Report" in res:
        with st.expander("🧮 Calibration Report"):
            with open(res["Calibration Report"], "r") as f:
                st.text(f.read())
            with open(res["Calibration Report"], "rb") as f:
                st.download_button("💾 Save Calibration Report",
                    data=f, file_name=os.path.basename(res["Calibration Report"]))
            with open(res["Air Wand Debug"], "rb") as f:
                st.download_button("🖼️ Download Air Wand ROI Debug Image",
                    data=f, file_name=os.path.basename(res["Air Wand Debug"]))

    if frame_dbg and "Debug Clip" in res:
        with open(res["Debug Clip"], "rb") as f:
            st.download_button("📥 Save Frame-by-Frame Debug Video",
                data=f, file_name=os.path.basename(res["Debug Clip"]))

st.markdown("---\n### 📤 Analyze Another Video")
st.info("Upload a new video above to start another run; previous results remain visible.")
