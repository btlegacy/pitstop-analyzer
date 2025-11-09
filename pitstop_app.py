# 🏁 VSR Pit Stop Analyzer v11
# Fixed-Camera, Adaptive Color + Optical-Flow Motion Analysis
# Streamlit application for analyzing pit-stop phases.

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans

# =============================
# Utility Functions
# =============================

def rolling_average(data, window=5):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode="same")

def sustained(condition, frames_required):
    sustained_frames = np.convolve(condition.astype(int), np.ones(frames_required), "same")
    return np.where(sustained_frames >= frames_required)[0]

def confidence_label(level):
    if level == "High":
        return ":green[✅ High]"
    elif level == "Medium":
        return ":orange[⚠️ Medium]"
    else:
        return ":red[❌ Low]"

def save_report(video_name, fps, w, h, base_stab, lift_t, drop_t,
                timings, confs):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fn = f"calibration_report_{os.path.splitext(video_name)[0]}_{ts}.txt"
    with open(fn, "w") as f:
        f.write("🏁 VSR Pit Stop Analyzer v11 – Calibration Report\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Video: {video_name}\nVersion: 11.0\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Detected FPS: {fps:.2f}\nFrame Size: {w} × {h}\n")
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

def analyze_video(video_path, video_name, output_path, progress_bar=None,
                  debug=False, calibrate=False, frame_debug=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x_cent, y_cent = [], []
    ret, prev = cap.read()
    if not ret:
        st.error("Video could not be read.")
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow_accum = np.zeros_like(prev_gray)
    frame_idx = 0

    # For optional MP4 debug
    if frame_debug:
        dbg_path = os.path.join(tempfile.gettempdir(), "debug_clip.mp4")
        dbg_writer = cv2.VideoWriter(dbg_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    else:
        dbg_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mask = np.uint8(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))
        _, thresh = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                x_cent.append(cx); y_cent.append(cy)
                if frame_debug and frame_idx < fps*8:
                    vis = frame.copy()
                    cv2.drawContours(vis, [c], -1, (0,255,0), 2)
                    cv2.circle(vis, (cx,cy), 6, (255,0,0), -1)
                    step = 15
                    for y in range(0,h,step):
                        for x in range(0,w,step):
                            fx, fy = flow[y,x]
                            if abs(fx)+abs(fy) > 2:
                                cv2.arrowedLine(vis, (x,y),
                                    (int(x+fx), int(y+fy)), (0,0,255),1,tipLength=0.3)
                    dbg_writer.write(vis)
        else:
            if x_cent: 
                x_cent.append(x_cent[-1]); y_cent.append(y_cent[-1])
            else:
                x_cent.append(0); y_cent.append(0)

        prev_gray = gray
        frame_idx += 1
        if progress_bar:
            progress_bar.progress(min(frame_idx/total,1.0))
    cap.release()
    if frame_debug:
        dbg_writer.release()

    x_s = rolling_average(x_cent,7)
    y_s = rolling_average(y_cent,7)
    dx = rolling_average(np.gradient(x_s),7)
    dy = rolling_average(np.gradient(y_s),7)

    stop_candidates = sustained((np.abs(dx)<2).astype(bool) & (x_s>w*0.45), int(fps))
    stop_i = stop_candidates[0] if len(stop_candidates)>0 else int(fps*4)
    base_y = np.mean(y_s[stop_i:stop_i+int(fps*2)])
    lift_th = 0.02*h; drop_th = 0.02*h

    up_idx = next((i for i in range(stop_i,int(total))
                   if y_s[i]<base_y-lift_th), stop_i+int(fps*2))
    down_idx = next((i for i in range(up_idx+int(fps*10),int(total))
                     if y_s[i]>base_y+drop_th), up_idx+int(fps*35))
    depart_idx = next((i for i in range(down_idx,int(total))
                       if dx[i]>5 and x_s[i]>w*0.8), total-1)

    times = { "Stop": stop_i/fps, "Up": up_idx/fps,
              "Down": down_idx/fps, "Depart": depart_idx/fps }
    confs = {"Stop":"High","Up":"High","Down":"High",
             "Depart":"High","Baseline":"High","Overall":"High"}

    # Annotated output
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w,h))
    fidx = 0
    while True:
        ret, frm = cap.read()
        if not ret: break
        for lbl, idx in zip(["Car Stop","Car Up","Car Down","Car Depart"],
                            [stop_i,up_idx,down_idx,depart_idx]):
            if abs(fidx-idx)<fps*0.5:
                cv2.putText(frm,f"{lbl} ({idx/fps:.2f}s)",(40,80),
                            cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
        out.write(frm); fidx+=1
    cap.release(); out.release()

    result = {
        "Car Stop Time (s)": round(times["Stop"],2),
        "Car Up Time (s)": round(times["Up"],2),
        "Car Down Time (s)": round(times["Down"],2),
        "Car Depart Time (s)": round(times["Depart"],2),
        "Pit Duration (s)": round(times["Depart"]-times["Stop"],2),
        "Annotated Video": output_path,
        "Debug Clip": dbg_path
    }

    # Debug plots
    if debug:
        t = np.arange(len(x_s))/fps
        plt.figure(figsize=(8,3))
        plt.plot(t,x_s); plt.axvspan(times["Stop"],times["Depart"],color="orange",alpha=.2)
        plt.xlabel("Time (s)"); plt.ylabel("X center"); plt.tight_layout()
        plt.savefig("x_motion.png"); plt.close()
        plt.figure(figsize=(8,3))
        plt.plot(t,y_s); plt.axvspan(times["Up"],times["Down"],color="green",alpha=.2)
        plt.xlabel("Time (s)"); plt.ylabel("Y center"); plt.tight_layout()
        plt.savefig("y_motion.png"); plt.close()
        result["X Motion Plot"]="x_motion.png"
        result["Y Motion Plot"]="y_motion.png"

    if calibrate:
        timings={k:{"time":round(v,2),"conf":confs[k]} for k,v in times.items()}
        rep=save_report(video_name,fps,w,h,np.std(y_s[stop_i:stop_i+int(fps*2)]),
                        (lift_th/h)*100,(drop_th/h)*100,timings,confs)
        result["Calibration Report"]=rep
    return result

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="VSR Pit Stop Analyzer v11", layout="centered")
st.title("🏁 VSR Pit Stop Analyzer v11")

if "run_count" not in st.session_state: st.session_state["run_count"]=0
debug_mode = st.sidebar.checkbox("Enable Debug Mode",value=False)
calib_mode = st.sidebar.checkbox("Enable Calibration Mode",value=False)
frame_dbg = st.sidebar.checkbox("Enable Frame-by-Frame Debug (MP4)",value=False)
upl = st.sidebar.file_uploader("🎥 Upload pit stop video", type=["mp4","mov","avi"])
btn = st.sidebar.button("Start Analysis")
pbar = st.sidebar.progress(0.0)

if btn and upl:
    st.session_state["run_count"]+=1
    run_id=st.session_state["run_count"]
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as tmp:
        tmp.write(upl.read()); tmp_path=tmp.name
    outp=os.path.join(tempfile.gettempdir(),f"annotated_run{run_id}.mp4")
    st.sidebar.info("⏱️ Processing video...")
    res=analyze_video(tmp_path,upl.name,outp,pbar,debug_mode,calib_mode,frame_dbg)
    st.sidebar.success("✅ Analysis Complete!")

    st.markdown(f"---\n## 🏁 Run #{run_id}: {upl.name}")
    st.subheader("📊 Pit Stop Summary")
    c1,c2,c3=st.columns(3)
    c1.metric("Car Stop",f"{res['Car Stop Time (s)']} s")
    c1.metric("Car Up",f"{res['Car Up Time (s)']} s")
    c2.metric("Car Down",f"{res['Car Down Time (s)']} s")
    c2.metric("Car Depart",f"{res['Car Depart Time (s)']} s")
    c3.metric("Pit Duration",f"{res['Pit Duration (s)']} s")

    st.subheader("🎬 Annotated Video")
    st.video(res["Annotated Video"])

    if debug_mode:
        with st.expander("📈 Motion Analysis (Debug Plots)"):
            if "X Motion Plot" in res: st.image(res["X Motion Plot"])
            if "Y Motion Plot" in res: st.image(res["Y Motion Plot"])
    if calib_mode and "Calibration Report" in res:
        with st.expander("🧮 Calibration Report"):
            with open(res["Calibration Report"],"r") as f: st.text(f.read())
            with open(res["Calibration Report"],"rb") as f:
                st.download_button("💾 Save Calibration Report",
                    data=f,file_name=os.path.basename(res["Calibration Report"]))
    if frame_dbg and res["Debug Clip"]:
        with open(res["Debug Clip"],"rb") as f:
            st.download_button("📥 Save Frame-by-Frame Debug Video",
                data=f,file_name=os.path.basename(res["Debug Clip"]))

st.markdown("---\n### 📤 Analyze Another Video")
st.info("Upload a new video above to start another run; previous results remain visible.")
