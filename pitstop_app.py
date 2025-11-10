# 🏁 VSR Pit Stop Analyzer v12.8 (Precision + Crew Edition)
# ==========================================================
# Full production version combining:
# - Car Stop / Up / Down / Depart logic (optical flow + ROI-based detection)
# - Front Tire Changer (FTC) performance tracking
# - Autoscaling (720p–4K)
# - Calibration & Debug Modes
# - Max-detail optical flow visualization (every 5–10 px)
# - Fully Streamlit-ready layout
# ==========================================================

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

# -----------------------------------------------------------
# CONFIGURATION & THRESHOLD SETTINGS
# -----------------------------------------------------------
CONFIG = {
    # Optical flow tuning
    "FLOW_SENSITIVITY": 1.2,          # Base motion threshold for horizontal flow
    "VERTICAL_FLOW_SENSITIVITY": 0.9, # For detecting air jack lift/drop
    "FLOW_RESCALE": 0.5,              # Processing resolution scale
    "DEBUG_RESCALE": 0.5,             # Output resolution scale

    # Event timing filters
    "CAR_STOP_STABILITY_SEC": 1.0,    # Required stability before confirming stop
    "CAR_DEPART_SUSTAIN_SEC": 0.8,    # Sustained motion duration for depart
    "FTC_ACTIVITY_THRESHOLD": 1.4,    # Crew motion trigger level
    "ROI_EXPANSION_FACTOR": 0.25,     # ROI padding to handle tilt
    "EVENT_LABEL_DURATION": 30,       # Frames to display overlay label (~1 sec @ 30fps)
    "FLOW_VECTOR_SPACING": 8          # Pixel spacing for debug flow arrows
}
# -----------------------------------------------------------
# CALIBRATION PREVIEW FUNCTION
# -----------------------------------------------------------
def calibration_preview(frame):
    """Draws static calibration overlays to verify ROI alignment."""
    try:
        h, w, _ = frame.shape

        # Example ROIs (same as in car + FTC tracking)
        car_roi = (int(w * 0.2), int(h * 0.4), int(w * 0.6), int(h * 0.25))
        ftc_outside_roi = (int(w * 0.65), int(h * 0.5), int(w * 0.3), int(h * 0.4))
        ftc_inside_roi = (int(w * 0.05), int(h * 0.5), int(w * 0.3), int(h * 0.4))

        overlay = frame.copy()

        # Draw ROIs
        draw_roi(overlay, car_roi, (0, 255, 0), "Car ROI")
        draw_roi(overlay, ftc_outside_roi, (255, 255, 0), "FTC Outside")
        draw_roi(overlay, ftc_inside_roi, (255, 255, 0), "FTC Inside")

        # Draw reference pit lines
        pit_line_x = int(w * 0.5)
        cv2.line(overlay, (pit_line_x, 0), (pit_line_x, h), (0, 165, 255), 2)
        cv2.putText(overlay, "Reference Pit Line", (pit_line_x + 10, int(h * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

        return overlay

    except Exception as e:
        print(f"[Calibration Error] {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)

# -----------------------------------------------------------
# DRAWING & VISUAL UTILITIES
# -----------------------------------------------------------
def draw_roi(frame, roi, color, label=None):
    """Draws a rectangular ROI on a frame with an optional label."""
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def draw_flow_field(frame, flow, step=8, scale=1.0, color=(255, 0, 0)):
    """Draw optical flow vectors on frame for debug visualization."""
    h, w = flow.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx * scale, y + fy * scale]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = frame.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, tipLength=0.3)
    return vis

def detect_orange_line(frame):
    """Detects the bright orange pit line to anchor horizontal car motion."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 80, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    edges = cv2.Canny(mask, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=60, maxLineGap=10)
    if lines is not None:
        ys = [y1 for [[x1, y1, x2, y2]] in lines if abs(y2 - y1) < 5]
        if ys:
            return int(np.median(ys))
    return int(frame.shape[0] * 0.85)

# -----------------------------------------------------------
# CALIBRATION MODE VISUALIZATION
# -----------------------------------------------------------
def calibration_preview(frame):
    """Displays automatically scaled ROIs to verify correct alignment."""
    h, w = frame.shape[:2]
    car_roi = (int(w * 0.25), int(h * 0.35), int(w * 0.5), int(h * 0.4))
    outside_roi = (int(w * 0.65), int(h * 0.5), int(w * 0.3), int(h * 0.4))
    inside_roi = (int(w * 0.05), int(h * 0.5), int(w * 0.3), int(h * 0.4))
    crossover_roi = (int(w * 0.35), int(h * 0.45), int(w * 0.3), int(h * 0.25))
    pit_line_y = detect_orange_line(frame)

    calib_frame = frame.copy()
    draw_roi(calib_frame, car_roi, (0, 255, 0), "Car ROI")
    draw_roi(calib_frame, outside_roi, (255, 255, 0), "Outside ROI")
    draw_roi(calib_frame, inside_roi, (255, 255, 0), "Inside ROI")
    draw_roi(calib_frame, crossover_roi, (0, 165, 255), "Crossover ROI")
    cv2.line(calib_frame, (0, pit_line_y), (w, pit_line_y), (0, 140, 255), 2)
    cv2.putText(calib_frame, "Calibration Mode - Verify ROI Alignment",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    return calib_frame
# -----------------------------------------------------------
# MAIN ANALYSIS ENGINE — CAR EVENT DETECTION
# -----------------------------------------------------------
def analyze_video(video_path, video_name, progress_bar=None, debug=False):
    """Processes video frame-by-frame to detect car and crew events."""

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read first frame
    ret, first = cap.read()
    if not ret:
        st.error("Unable to read video.")
        return None

    # Autoscaled ROIs (relative coordinates)
    car_roi = (int(w * 0.25), int(h * 0.35), int(w * 0.5), int(h * 0.4))
    pit_line_y = detect_orange_line(first)

    # Flow pre-processing
    flow_scale = CONFIG["FLOW_RESCALE"]
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)
    prev_gray = cv2.cvtColor(cv2.resize(first, (small_w, small_h)), cv2.COLOR_BGR2GRAY)

    # Debug writer setup
    dbg_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dbg_name = f"{os.path.splitext(video_name)[0]}_debug_{dbg_ts}.mp4"
    debug_path = os.path.join(tempfile.gettempdir(), dbg_name)
    dbg_w, dbg_h = int(w * CONFIG["DEBUG_RESCALE"]), int(h * CONFIG["DEBUG_RESCALE"])
    writer = cv2.VideoWriter(debug_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps,
                             (dbg_w, dbg_h))

    # Event state tracking
    frame_idx = 0
    label_timer = 0
    label_text = ""

    car_stop = None
    car_up = None
    car_down = None
    car_depart = None
    motion_buffer = []   # stores rolling horizontal flow averages
    vertical_buffer = [] # stores rolling vertical flow averages

    # -------------------------------------------------------
    # Helper: sustained-motion test
    # -------------------------------------------------------
    def sustained_motion(buffer, threshold, frames_required):
        """Returns True if avg motion exceeds threshold for N consecutive frames."""
        if len(buffer) < frames_required:
            return False
        recent = buffer[-frames_required:]
        return np.mean(np.abs(recent)) > threshold

    # -------------------------------------------------------
    # Frame loop
    # -------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize + gray for optical flow
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        horiz_mean = np.mean(np.abs(fx))
        vert_mean = np.mean(fy)
        motion_buffer.append(horiz_mean)
        vertical_buffer.append(vert_mean)

        # Limit buffer length (approx 2 seconds)
        max_len = int(fps * 2)
        if len(motion_buffer) > max_len:
            motion_buffer.pop(0)
            vertical_buffer.pop(0)

        # ---------------------------------------------------
        # CAR EVENT DETECTION
        # ---------------------------------------------------
        # 1. Car Stop — must be still for configured seconds
        if car_stop is None and not sustained_motion(motion_buffer,
                                                     CONFIG["FLOW_SENSITIVITY"],
                                                     int(fps * CONFIG["CAR_STOP_STABILITY_SEC"])):
            car_stop = frame_idx / fps
            label_text = f"CAR STOP — {car_stop:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]

        # 2. Car Up — vertical motion toward top (negative fy)
        if car_stop and car_up is None and np.mean(vertical_buffer) < -CONFIG["VERTICAL_FLOW_SENSITIVITY"]:
            car_up = frame_idx / fps
            label_text = f"CAR UP — {car_up:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]

        # 3. Car Down — positive vertical flow after Up
        if car_up and car_down is None and np.mean(vertical_buffer) > CONFIG["VERTICAL_FLOW_SENSITIVITY"]:
            car_down = frame_idx / fps
            label_text = f"CAR DOWN — {car_down:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]

        # 4. Car Depart — sustained rightward motion (fx > threshold)
        if car_down and car_depart is None and sustained_motion(motion_buffer,
                                                                CONFIG["FLOW_SENSITIVITY"],
                                                                int(fps * CONFIG["CAR_DEPART_SUSTAIN_SEC"])):
            car_depart = frame_idx / fps
            label_text = f"CAR DEPART — {car_depart:.1f}s"
            label_timer = CONFIG["EVENT_LABEL_DURATION"]

        # ---------------------------------------------------
        # DEBUG FRAME GENERATION
        # ---------------------------------------------------
        dbg_frame = cv2.resize(frame, (dbg_w, dbg_h))
        cv2.line(dbg_frame,
                 (0, int(pit_line_y * CONFIG["DEBUG_RESCALE"])),
                 (dbg_w, int(pit_line_y * CONFIG["DEBUG_RESCALE"])),
                 (0, 140, 255), 2)

        if debug:
            # Draw ROI and flow vectors
            scaled_flow = cv2.resize(flow, (dbg_w, dbg_h))
            dbg_frame = draw_flow_field(dbg_frame,
                                        scaled_flow,
                                        step=CONFIG["FLOW_VECTOR_SPACING"],
                                        scale=1.0,
                                        color=(255, 0, 0))
            draw_roi(dbg_frame, (int(car_roi[0] * CONFIG["DEBUG_RESCALE"]),
                                 int(car_roi[1] * CONFIG["DEBUG_RESCALE"]),
                                 int(car_roi[2] * CONFIG["DEBUG_RESCALE"]),
                                 int(car_roi[3] * CONFIG["DEBUG_RESCALE"])),
                     (0, 255, 0), "Car ROI")

        # Display temporary event labels
        if label_timer > 0:
            cv2.putText(dbg_frame, label_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
            label_timer -= 1

        writer.write(dbg_frame)

        # Prepare for next frame
        prev_gray = gray
        frame_idx += 1

        if progress_bar:
            progress_bar.progress(min(frame_idx / total, 1.0))

    cap.release()
    writer.release()

    # Collect results
    results = {
        "Car Stop Time (s)": round(car_stop or 0, 2),
        "Car Up Time (s)": round(car_up or 0, 2),
        "Car Down Time (s)": round(car_down or 0, 2),
        "Car Depart Time (s)": round(car_depart or 0, 2),
        "Pit Duration (s)": round((car_depart - car_stop)
                                  if car_stop and car_depart else 0, 2),
        "Debug Video": debug_path
    }

    return results
# -----------------------------------------------------------
# FRONT TIRE CHANGER (FTC) TRACKING SYSTEM
# -----------------------------------------------------------
def track_ftc(video_path, car_stop_time, fps, w, h, debug=False):
    """
    Analyzes Front Tire Changer motion after car stop.
    Returns timestamps for key crew events relative to car stop.
    """

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flow_scale = CONFIG["FLOW_RESCALE"]
    small_w, small_h = int(w * flow_scale), int(h * flow_scale)

    # Define ROIs relative to car position
    outside_roi = (int(w * 0.65), int(h * 0.5), int(w * 0.3), int(h * 0.4))
    inside_roi = (int(w * 0.05), int(h * 0.5), int(w * 0.3), int(h * 0.4))
    crossover_roi = (int(w * 0.35), int(h * 0.45), int(w * 0.3), int(h * 0.25))

    # Read first frame
    ret, first = cap.read()
    if not ret:
        return {k: None for k in [
            "Tire Drop", "Wheel Nut", "Tire Exchange 1",
            "Crossover", "Tire Exchange 2", "Car Drop"
        ]}

    prev_gray = cv2.cvtColor(cv2.resize(first, (small_w, small_h)), cv2.COLOR_BGR2GRAY)

    # FTC event times
    ftc_events = {k: None for k in [
        "Tire Drop", "Wheel Nut", "Tire Exchange 1",
        "Crossover", "Tire Exchange 2", "Car Drop"
    ]}
    event_triggered = set()

    frame_idx = 0
    flow_energy_out, flow_energy_in, flow_energy_cross = [], [], []
    last_active_zone = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate flow magnitude for each ROI
        def roi_energy(roi):
            x, y, w_, h_ = [int(i * flow_scale) for i in roi]
            return np.mean(mag[y:y + h_, x:x + w_])

        e_out = roi_energy(outside_roi)
        e_in = roi_energy(inside_roi)
        e_cross = roi_energy(crossover_roi)

        flow_energy_out.append(e_out)
        flow_energy_in.append(e_in)
        flow_energy_cross.append(e_cross)

        # Keep last ~2s of data
        max_len = int(fps * 2)
        if len(flow_energy_out) > max_len:
            flow_energy_out.pop(0)
            flow_energy_in.pop(0)
            flow_energy_cross.pop(0)

        t_sec = frame_idx / fps

        # ---------------------------------------------------
        # EVENT DETECTION LOGIC
        # ---------------------------------------------------
        # 1. Tire Drop — first strong motion in outside ROI
        if ftc_events["Tire Drop"] is None and e_out > CONFIG["FTC_ACTIVITY_THRESHOLD"]:
            ftc_events["Tire Drop"] = round(t_sec - car_stop_time, 2)
            last_active_zone = "outside"

        # 2. Wheel Nut — burst of energy soon after Tire Drop
        if ftc_events["Tire Drop"] and ftc_events["Wheel Nut"] is None and e_out > CONFIG["FTC_ACTIVITY_THRESHOLD"] * 1.3:
            ftc_events["Wheel Nut"] = round(t_sec - car_stop_time, 2)
            last_active_zone = "outside"

        # 3. Tire Exchange 1 — drop in flow after sustained activity
        if ftc_events["Wheel Nut"] and ftc_events["Tire Exchange 1"] is None:
            recent_avg = np.mean(flow_energy_out[-int(fps * 0.5):])
            if recent_avg < CONFIG["FTC_ACTIVITY_THRESHOLD"] * 0.7:
                ftc_events["Tire Exchange 1"] = round(t_sec - car_stop_time, 2)
                last_active_zone = "outside"

        # 4. Crossover — motion in crossover ROI
        if ftc_events["Tire Exchange 1"] and ftc_events["Crossover"] is None and e_cross > CONFIG["FTC_ACTIVITY_THRESHOLD"]:
            ftc_events["Crossover"] = round(t_sec - car_stop_time, 2)
            last_active_zone = "crossover"

        # 5. Tire Exchange 2 — activity in inside ROI
        if ftc_events["Crossover"] and ftc_events["Tire Exchange 2"] is None and e_in > CONFIG["FTC_ACTIVITY_THRESHOLD"]:
            ftc_events["Tire Exchange 2"] = round(t_sec - car_stop_time, 2)
            last_active_zone = "inside"

        # 6. Car Drop — final burst (air wand pull / car down)
        if ftc_events["Tire Exchange 2"] and ftc_events["Car Drop"] is None and e_in > CONFIG["FTC_ACTIVITY_THRESHOLD"] * 1.2:
            ftc_events["Car Drop"] = round(t_sec - car_stop_time, 2)
            last_active_zone = "inside"

        # ---------------------------------------------------
        # DEBUG VISUALIZATION
        # ---------------------------------------------------
        if debug:
            overlay = frame.copy()
            draw_roi(overlay, outside_roi, (255, 255, 0), "Outside")
            draw_roi(overlay, crossover_roi, (0, 165, 255), "Crossover")
            draw_roi(overlay, inside_roi, (255, 255, 0), "Inside")

            active_color = (0, 255, 255)
            if last_active_zone == "outside":
                draw_roi(overlay, outside_roi, active_color)
            elif last_active_zone == "crossover":
                draw_roi(overlay, crossover_roi, active_color)
            elif last_active_zone == "inside":
                draw_roi(overlay, inside_roi, active_color)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return ftc_events


# -----------------------------------------------------------
# STREAMLIT USER INTERFACE + COMBINED ANALYSIS LOGIC
# -----------------------------------------------------------
st.set_page_config(page_title="VSR Pit Stop Analyzer v12.8", layout="wide")
st.title("🏁 VSR Pit Stop Analyzer v12.8 (Precision + Crew Edition)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    upl = st.file_uploader("🎥 Upload Pit Stop Video", type=["mp4", "mov", "avi"])
    debug_mode = st.checkbox("🔍 Enable Debug Mode", value=False)
    calib_mode = st.checkbox("📏 Calibration Preview", value=False)
    start_btn = st.button("▶️ Start Analysis")
    progress_bar = st.progress(0.0)

# -----------------------------------------------------------
# CALIBRATION PREVIEW HANDLER
# -----------------------------------------------------------
if calib_mode and upl:
    st.subheader("Calibration Preview")
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(upl.read())
    cap = cv2.VideoCapture(tfile.name)
    ret, frame = cap.read()
    cap.release()
    if ret:
        preview = calibration_preview(frame)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                 caption="Verify ROI Alignment and Pit Line Position",
                 use_container_width=True)
    else:
        st.warning("⚠️ Unable to load first frame for calibration preview.")

# -----------------------------------------------------------
# MAIN ANALYSIS PIPELINE TRIGGER
# -----------------------------------------------------------
if start_btn and upl:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upl.read())
        tmp_path = tmp.name

    st.info("⏳ Analyzing video — please wait while optical flow is processed…")

    # Step 1 — Run car event detection
    car_results = analyze_video(tmp_path, upl.name, progress_bar, debug_mode)
    if not car_results:
        st.error("❌ Video analysis failed.")
        st.stop()

    # Step 2 — Run FTC tracking based on car stop time
    cap = cv2.VideoCapture(tmp_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    ftc_results = track_ftc(tmp_path, car_results["Car Stop Time (s)"], fps, w, h, debug_mode)
    car_results.update({f"FTC {k} (s)": v for k, v in ftc_results.items()})

    st.success("✅ Analysis Complete!")

    # -------------------------------------------------------
    # RESULTS DISPLAY
    # -------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚗 Car Events")
        for key in ["Car Stop Time (s)", "Car Up Time (s)",
                    "Car Down Time (s)", "Car Depart Time (s)",
                    "Pit Duration (s)"]:
            st.metric(key.replace(" (s)", ""), f"{car_results[key]} s")

    with col2:
        st.subheader("👨‍🔧 Front Tire Changer Performance")
        for k, v in ftc_results.items():
            label = k.replace("FTC ", "")
            st.metric(label, f"{v} s" if v else "—")

    # -------------------------------------------------------
    # VIDEO OUTPUT
    # -------------------------------------------------------
    st.subheader("🎬 Debug / Annotated Video")
    st.video(car_results["Debug Video"])
    with open(car_results["Debug Video"], "rb") as f:
        st.download_button("⬇️ Download Debug MP4",
                           data=f,
                           file_name=os.path.basename(car_results["Debug Video"]),
                           mime="video/mp4")

    st.caption("Tip: In Debug Mode you will see optical flow vectors, ROIs, and event labels in real time.")
# -----------------------------------------------------------
# FINALIZATION + CLEANUP UTILITIES
# -----------------------------------------------------------
def cleanup_temp_files():
    """Remove old temporary debug videos to free storage."""
    temp_dir = tempfile.gettempdir()
    for f in os.listdir(temp_dir):
        if f.endswith("_debug.mp4"):
            try:
                os.remove(os.path.join(temp_dir, f))
            except Exception:
                pass

def display_summary(car_results, ftc_results):
    """Prints a summary table in plain text for logs or console runs."""
    st.markdown("---")
    st.subheader("📋 Pit Stop Summary (Text Output)")
    summary = []
    for k in ["Car Stop Time (s)", "Car Up Time (s)", "Car Down Time (s)", "Car Depart Time (s)", "Pit Duration (s)"]:
        summary.append(f"{k}: {car_results.get(k, '—')}")
    for k, v in ftc_results.items():
        summary.append(f"FTC {k}: {v if v else '—'}")
    st.text("\n".join(summary))

# -----------------------------------------------------------
# ENTRY POINT HANDLER (FOR LOCAL DEV OR STREAMLIT CLOUD)
# -----------------------------------------------------------
if __name__ == "__main__":
    st.markdown("""
    <div style='background-color:#111;padding:15px;border-radius:10px;'>
        <h3 style='color:#00FFAA;'>🏁 VSR Pit Stop Analyzer v12.8 Initialized</h3>
        <p style='color:#CCCCCC;'>Ready for analysis. Upload an overhead pit stop video to begin.</p>
        <ul style='color:#AAAAAA;'>
            <li>Car detection uses optical flow with sustained motion filters.</li>
            <li>Front Tire Changer (FTC) stats calculated relative to Car Stop.</li>
            <li>Use <b>Debug Mode</b> to visualize flow vectors and ROIs.</li>
            <li>Use <b>Calibration Preview</b> to verify ROI scaling on new cameras.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Developed for precision IMSA-style pit stop analysis. Streamlit-ready.")
    cleanup_temp_files()
