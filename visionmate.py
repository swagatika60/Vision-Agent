import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import bcrypt
import base64
from io import BytesIO
import tempfile
import time
from datetime import datetime
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import gc

nest_asyncio.apply()
from ui_components import VisionMateUI
load_dotenv()

from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, gemini, ultralytics as agent_ultralytics

try:
    from ultralytics import YOLO
    import torch
    from gtts import gTTS
except ImportError as e:
    st.error(f"Missing required dependency: {e}. Please run pip install -r requirements.txt")
    st.stop()

# ==========================================
# STATE INITIALIZATION
# ==========================================

def init_session_state():
    defaults = {
        'authenticated': False,
        'username': None,
        'user_name': None,
        'audio_enabled': True,
        'models_loaded': False,
        'current_page': 'home',
        'welcome_played': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
USER_DATA_FILE = "users.json"

# ==========================================
# AUTHENTICATION LAYER
# ==========================================

def get_db():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_db(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def hash_pw(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_pw(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def handle_registration(username, name, password):
    db = get_db()
    if username in db:
        return False, "Username already exists."
    db[username] = {
        'password': hash_pw(password),
        'name': name,
        'created_at': datetime.now().isoformat()
    }
    save_db(db)
    return True, "Registration successful."

def handle_login(username, password):
    db = get_db()
    if username not in db:
        return False, "Invalid credentials.", None
    
    user_data = db[username]
    hashed_password = user_data.get('password') if isinstance(user_data, dict) else user_data
    display_name = user_data.get('name', username) if isinstance(user_data, dict) else username
    
    if verify_pw(password, hashed_password):
        return True, "Login successful.", display_name
    return False, "Invalid credentials.", None

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def generate_audio_feedback(text):
    """Generates base64 encoded audio player for UI feedback."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        return f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None

# ==========================================
# CORE VISION ENGINE
# ==========================================

@st.cache_resource
def initialize_vision_model():
    """Caches YOLOv8s model to prevent memory leaks during standard ops."""
    try:
        return YOLO('yolov8s.pt')
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        return None

def resize_maintaining_aspect(image, target_width=640):
    """Dynamically scales frame to target width while preserving aspect ratio for detection accuracy."""
    h, w = image.shape[:2]
    if w > target_width:
        ratio = target_width / w
        return cv2.resize(image, (target_width, int(h * ratio)))
    return image

def process_bounding_boxes(image, results):
    """Draws boxes and calculates dynamic proximity hazards based on vertical span."""
    annotated = image.copy()
    img_h = image.shape[0]
    
    if results.boxes.id is None and len(results.boxes) == 0:
        return annotated

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_idx = int(box.cls[0])
        name = results.names[cls_idx]
        
        # Spatial calculation: object height relative to frame height
        vertical_span = (y2 - y1) / img_h
        
        if vertical_span > 0.4:
            status, color, thickness = "HAZARD", (0, 0, 255), 4
        elif vertical_span > 0.2:
            status, color, thickness = "MID", (0, 255, 255), 2
        else:
            status, color, thickness = "FAR", (0, 255, 0), 2
            
        label = f"{name} [{status}]"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, thickness-1))
        
    return annotated

# ==========================================
# UI ROUTING & PAGES
# ==========================================

def render_auth():
    VisionMateUI.load_css()
    VisionMateUI.welcome_banner()
    tab1, tab2 = VisionMateUI.auth_tabs()
    
    with tab1:
        st.subheader("Account Access")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                success, msg, name = handle_login(u, p)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = u
                    st.session_state.user_name = name
                    st.rerun()
                else: st.error(msg)
                
    with tab2:
        st.subheader("New Registration")
        with st.form("reg_form"):
            nu = st.text_input("Username")
            nn = st.text_input("Display Name")
            np_pw = st.text_input("Password", type="password")
            if st.form_submit_button("Register"):
                success, msg = handle_registration(nu, nn, np_pw)
                if success: st.success(msg)
                else: st.error(msg)

def render_dashboard():
    VisionMateUI.load_css()
    VisionMateUI.welcome_banner(st.session_state.user_name)
    
    if not st.session_state.welcome_played and st.session_state.audio_enabled:
        st.markdown(generate_audio_feedback(f"Welcome back, {st.session_state.user_name}"), unsafe_allow_html=True)
        st.session_state.welcome_played = True
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(VisionMateUI.mode_card("Image Processing", "Static analysis", "📷"), unsafe_allow_html=True)
        if st.button("Launch Image Engine", use_container_width=True):
            st.session_state.current_page = 'image'; st.rerun()
    with col2:
        st.markdown(VisionMateUI.mode_card("Live Protocol", "Agent WebRTC", "🎥"), unsafe_allow_html=True)
        if st.button("Launch Agent Room", use_container_width=True):
            st.session_state.current_page = 'live'; st.rerun()
    with col3:
        st.markdown(VisionMateUI.mode_card("Video Pipeline", "Live telemetry scan", "🎬"), unsafe_allow_html=True)
        if st.button("Launch Video Pipeline", use_container_width=True):
            st.session_state.current_page = 'video'; st.rerun()

def view_image_analysis():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Image Processing", "High-Fidelity Frame Inspection", "📷")
    if st.button("← Return to Dashboard"): 
        st.session_state.current_page = 'home'; st.rerun()
    
    upload = st.file_uploader("Select media", type=['jpg', 'png', 'jpeg'])
    if upload:
        img = Image.open(upload).convert("RGB")
        _, col_main, _ = st.columns([1, 2, 1])
        with col_main:
            st.image(img, use_container_width=True)
        
        if st.button("Execute Deep Scan", type="primary", use_container_width=True):
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_cv = resize_maintaining_aspect(img_cv, 800)
            h = img_cv.shape[0]
            
            with st.spinner("Processing structural tensors..."):
                results = st.session_state.yolo_model(img_cv, conf=0.25, iou=0.45)[0]
                annotated = process_bounding_boxes(img_cv, results)
                
                hazards = []
                freq_map = {}
                for box in results.boxes:
                    obj_class = results.names[int(box.cls[0])]
                    freq_map[obj_class] = freq_map.get(obj_class, 0) + 1
                    if (int(box.xyxy[0][3]) - int(box.xyxy[0][1])) / h > 0.4:
                        hazards.append(obj_class)
            
            if hazards:
                alert = f"HAZARD DETECTED: {', '.join(set(hazards))} proximity threshold breached."
                st.error(alert)
                if st.session_state.audio_enabled: st.markdown(generate_audio_feedback(alert), unsafe_allow_html=True)
            
            VisionMateUI.image_comparison(img, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Source", "Processed")
            str_counts = ", ".join([f"{v} {k}" for k, v in freq_map.items()])
            desc = f"Scan complete. Classified elements: {str_counts}." if freq_map else "No structured elements classified."
            
            st.markdown(VisionMateUI.info_card("Telemetry", desc, "success"), unsafe_allow_html=True)
            if st.session_state.audio_enabled and not hazards:
                st.markdown(generate_audio_feedback(desc), unsafe_allow_html=True)

async def init_agent_webrtc(user_name):
    """Establishes stream edge connection for native low-latency agent."""
    processor = agent_ultralytics.YOLOPoseProcessor(model_path="yolov8s-pose.pt")
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="VisionMate", id="sys_agent"),
        instructions=f"Provide spatial awareness and mobility assistance for {user_name}.",
        llm=gemini.Realtime(), 
        processors=[processor] 
    )
    await agent.create_user()
    call = await agent.create_call("default", "visionmate-demo-room")
    async with agent.join(call):
        await agent.finish()

def view_live_protocol():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Live Protocol", "Real-Time Agent Interface", "🎥")
    if st.button("← Return to Dashboard"): 
        st.session_state.current_page = 'home'; st.rerun()
    
    st.markdown("---")
    st.subheader("Native Hardware Test")
    st.info("Execute proximity calculations via standard browser camera access.")
    
    cam_buffer = st.camera_input("Initialize hardware stream")
    if cam_buffer:
        with st.spinner("Processing telemetry..."):
            img = Image.open(cam_buffer).convert("RGB")
            cv_img = resize_maintaining_aspect(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 640)
            h = cv_img.shape[0]
            
            res = st.session_state.yolo_model(cv_img, conf=0.3)[0]
            ann = process_bounding_boxes(cv_img, res)
            
            hazards = [res.names[int(b.cls[0])] for b in res.boxes if (int(b.xyxy[0][3]) - int(b.xyxy[0][1])) / h > 0.4]
            freq = {}
            for b in res.boxes: freq[res.names[int(b.cls[0])]] = freq.get(res.names[int(b.cls[0])], 0) + 1
                    
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption="Processed Buffer", use_container_width=True)
            desc = "Detected: " + ", ".join([f"{v} {k}" for k, v in freq.items()]) if freq else "Clear."
            
            st.markdown(VisionMateUI.info_card("Environment Data", desc, "success"), unsafe_allow_html=True)
            if hazards: st.error(f"CRITICAL: {', '.join(set(hazards))} breaching safe zone.")

    st.markdown("---")
    st.subheader("WebRTC Edge Network")
    st.caption("Bypasses standard HTTP latency. Uses Stream's protocol for <30ms response times.")
    
    webrtc_uri = "https://demo.visionagents.ai/join?call_id=visionmate-demo-room"
    st.markdown(f"**[Authenticate WebRTC Client]({webrtc_uri})**")
    
    if st.button("Initialize Agent Socket"):
        with st.spinner("Establishing WebSocket connection..."):
            try: asyncio.run(init_agent_webrtc(st.session_state.user_name))
            except Exception as e: st.error(f"Socket binding failed: {e}")

def view_video_pipeline():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Video Pipeline", "Time-Series Hazard Tracking", "🎬")
    if st.button("← Return to Dashboard"): 
        st.session_state.current_page = 'home'; st.rerun()
    
    st.markdown("---")
    upload = st.file_uploader("Select environment recording", type=['mp4', 'avi', 'mov'])
    
    if upload:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp.write(upload.read())
        path = tmp.name
        
        _, col_vid, _ = st.columns([1, 2, 1])
        with col_vid: st.video(path)
        
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Duration", f"{frames / max(fps, 1):.1f}s")
        with c2: st.metric("Total Frames", frames)
        with c3: st.metric("Framerate", fps)
            
        st.markdown("---")
        
        if st.button("Execute Live Pipeline", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(path)
            pb = st.progress(0)
            sys_log = st.empty()
            
            # --- SYNCHRONOUS LIVE DASHBOARD LAYOUT ---
            col_stream, col_telemetry = st.columns([2, 1])
            
            with col_stream:
                stream_disp = st.empty()
                hud_disp = st.empty()
            
            with col_telemetry:
                st.markdown("### 📡 Live Telemetry")
                metric_disp = st.empty()
                timeline_disp = st.empty()
                
            hazards, safe, cache = set(), set(), set()
            timeline = []
            
            step_rate = max(1, fps // 5) 
            idx = 0
            
            sys_log.info("Pipeline initialized. Compiling tensors...")
            
            while cap.isOpened():
                ret, frm = cap.read()
                if not ret: break
                
                if idx % step_rate == 0:
                    frm = resize_maintaining_aspect(frm, 640)
                    h = frm.shape[0]
                    
                    res = st.session_state.yolo_model.track(frm, persist=True, verbose=False, conf=0.25, iou=0.45)[0]
                    ann = frm.copy()
                    frame_cache = {}
                    
                    if res.boxes.id is not None or len(res.boxes) > 0:
                        for b in res.boxes:
                            cls = res.names[int(b.cls[0])]
                            cache.add(cls)
                            frame_cache[cls] = frame_cache.get(cls, 0) + 1
                            
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            if (y2 - y1) / h > 0.4:
                                hazards.add(cls)
                                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                cv2.putText(ann, f"HAZARD: {cls}", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
                            else:
                                safe.add(cls)
                                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(ann, cls, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    
                    # Store Timeline Data
                    if frame_cache:
                        timeline.append({'ts': idx / max(fps, 1), 'data': frame_cache})
                        hud_str = " | ".join([f"{v} {k}" for k, v in frame_cache.items()])
                    else:
                        hud_str = "Null output"
                        
                    # Video HUD Draw
                    h_hud = int(h * 0.08)
                    cv2.rectangle(ann, (0, 0), (ann.shape[1], h_hud), (0, 0, 0), -1)
                    cv2.putText(ann, f"TELEMETRY: {hud_str}", (10, int(h_hud*0.7)), cv2.FONT_HERSHEY_SIMPLEX, h*0.0015, (255, 255, 255), max(1, int(h*0.003)))
                    
                    # --- LIVE UI UPDATES ---
                    stream_disp.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    hud_disp.markdown(f"**<div align='center'>Network Data: {hud_str}</div>**", unsafe_allow_html=True)
                    
                    with metric_disp.container():
                        m1, m2 = st.columns(2)
                        m1.metric("Unique Types", len(cache))
                        m2.metric("Total Hazards", len(hazards))
                    
                    log_txt = "#### Time-Series Log\n"
                    # Display the 6 most recent logs during playback
                    for t in reversed(timeline[-6:]): 
                        ts = f"{int(t['ts']//60):02d}:{int(t['ts']%60):02d}"
                        obj = ", ".join([f"{v} {k}" for k, v in t['data'].items()])
                        log_txt += f"🔴 **{ts}** - {obj}\n\n"
                    
                    timeline_disp.markdown(log_txt)

                idx += 1
                pb.progress(min(idx / frames, 1.0))
                
            cap.release()
            pb.progress(1.0)
            sys_log.success("Pipeline execution complete.")
            hud_disp.empty()
            
            # --- FINAL SUMMARY ---
            st.markdown("---")
            st.markdown("### Aggregated Telemetry")
            if cache:
                st.write("**Classes Detected:** ", ", ".join(list(cache)))
            
            if hazards:
                warn = f"CRITICAL: {', '.join(hazards)} recorded within minimum safe distance."
                st.markdown(VisionMateUI.info_card("Security Audit", warn, "warning"), unsafe_allow_html=True)
            else:
                warn = "Audit passed. Zero hazards recorded."
                st.markdown(VisionMateUI.info_card("Security Audit", warn, "success"), unsafe_allow_html=True)
            
            if st.session_state.audio_enabled: st.markdown(generate_audio_feedback(warn), unsafe_allow_html=True)

            rpt = {
                "status": "CRITICAL" if hazards else "NOMINAL",
                "hazards": list(hazards),
                "entities": list(cache),
                "timestamp": datetime.now().isoformat()
            }
            st.download_button("Export System Report (.json)", data=json.dumps(rpt, indent=4), file_name="visionmate_audit.json", mime="application/json")
            gc.collect()

# ==========================================
# BOOTSTRAP
# ==========================================

def main():
    st.set_page_config(page_title="VisionMate Core", layout="wide")
    
    if not st.session_state.authenticated:
        render_auth()
    else:
        with st.sidebar:
            VisionMateUI.sidebar_header(st.session_state.user_name)
            st.session_state.audio_enabled = st.checkbox("System Audio", value=True)
            if st.button("Dashboard"): st.session_state.current_page = 'home'; st.rerun()
            if st.button("Terminate Session"): 
                st.session_state.authenticated = False; st.rerun()

        if not st.session_state.models_loaded:
            with st.spinner("Initializing structural arrays..."):
                st.session_state.yolo_model = initialize_vision_model()
                st.session_state.models_loaded = True
        
        pg = st.session_state.current_page
        if pg == 'home': render_dashboard()
        elif pg == 'image': view_image_analysis()
        elif pg == 'live': view_live_protocol()
        elif pg == 'video': view_video_pipeline()

if __name__ == "__main__":
    main()
