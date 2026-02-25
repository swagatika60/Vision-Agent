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

# Allow asyncio to run inside Streamlit's event loop
nest_asyncio.apply()

from ui_components import VisionMateUI
load_dotenv()

from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, gemini, ultralytics as agent_ultralytics, moondream, smart_turn

try:
    from ultralytics import YOLO
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    from gtts import gTTS
except ImportError as e:
    st.error(f"Missing library: {e}")
    st.stop()

# --- Session State ---
for key, val in {
    'authenticated': False, 'username': None, 'user_name': None,
    'audio_enabled': True, 'models_loaded': False, 
    'current_page': 'home', 'welcome_played': False
}.items():
    if key not in st.session_state: st.session_state[key] = val

USER_DATA_FILE = "users.json"

# --- Auth Module ---
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f: return json.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f: json.dump(users, f, indent=4)

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def login_user(username, password):
    users = load_users()
    if username not in users: return False, "User not found", None
    user_data = users[username]
    hashed = user_data['password'] if isinstance(user_data, dict) else user_data
    name = user_data.get('name', username) if isinstance(user_data, dict) else username
    if verify_password(password, hashed): return True, "Login successful", name
    return False, "Incorrect password", None

# --- Audio Engine ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        return f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: return None

# --- Vision AI System ---
@st.cache_resource
def load_models():
    try:
        yolo = YOLO('yolov8m.pt') 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        return yolo, processor, model, device
    except Exception as e:
        st.error(f"Model Error: {e}"); return None, None, None, None

def draw_boxes(image, results):
    annotated = image.copy()
    h_img, w_img, _ = image.shape
    
    # Draw a subtle "Safe Lane" overlay for the demo
    cv2.rectangle(annotated, (int(w_img/3), 0), (int(2*w_img/3), h_img), (255, 255, 255), 1)
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        box_h = (y2 - y1) / h_img
        
        # Color coding: Red=Danger, Yellow=Caution, Green=Safe
        dist, color = ("CLOSE", (0,0,255)) if box_h > 0.5 else (("MID", (0,255,255)) if box_h > 0.2 else ("FAR", (0,255,0)))
        
        # Spatial Position
        center_x = (x1 + x2) / 2
        pos = "on your LEFT" if center_x < w_img/3 else ("on your RIGHT" if center_x > 2*w_img/3 else "AHEAD")
        
        label = f"{results.names[cls]} [{dist}]"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated 

def create_detection_description(results):
    if not results.boxes: return "No objects detected."
    counts = {}
    for box in results.boxes:
        name = results.names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1
    # This specifically creates the "4 bikes, 2 cars" style speech
    return "Detected: " + ", ".join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()])

# --- Page Logic ---
def auth_page():
    VisionMateUI.load_css()
    VisionMateUI.welcome_banner()
    tab1, tab2 = VisionMateUI.auth_tabs()
    with tab1:
        st.subheader("🔐 Login")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("🚀 Login", type="primary"):
                success, msg, name = login_user(u, p)
                if success:
                    st.session_state.authenticated, st.session_state.username, st.session_state.user_name = True, u, name
                    st.rerun()
                else: st.error(msg)
    with tab2:
        st.subheader("✨ Register")
        with st.form("reg_form"):
            nu, nn, np = st.text_input("Username"), st.text_input("Name"), st.text_input("Password", type="password")
            if st.form_submit_button("📝 Create Account"):
                # Register logic here
                st.info("Registration is handled by your users.json logic.")

def home_page():
    VisionMateUI.load_css()
    VisionMateUI.welcome_banner(st.session_state.user_name)
    if not st.session_state.welcome_played and st.session_state.audio_enabled:
        st.markdown(text_to_speech(f"Welcome back, {st.session_state.user_name}"), unsafe_allow_html=True)
        st.session_state.welcome_played = True
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(VisionMateUI.mode_card("Image Analysis", "Analyze Photos", "📷"), unsafe_allow_html=True)
        if st.button("Start Image Mode", use_container_width=True):
            st.session_state.current_page = 'image'; st.rerun()
    with col2:
        st.markdown(VisionMateUI.mode_card("Live Agent", "Real-time Call", "🎥"), unsafe_allow_html=True)
        if st.button("Start Live Mode", use_container_width=True):
            st.session_state.current_page = 'live'; st.rerun()
    with col3:
        st.markdown(VisionMateUI.mode_card("Video Analysis", "Upload MP4s", "🎬"), unsafe_allow_html=True)
        if st.button("Start Video Mode", use_container_width=True):
            st.session_state.current_page = 'video'; st.rerun()

def process_image_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Image Analysis", "Depth-Aware Vision", "📷")
    if st.button("⬅️ Back"): st.session_state.current_page = 'home'; st.rerun()
    
    upload = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if upload:
        img = Image.open(upload).convert("RGB")
        st.image(img, use_container_width=True)
        if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results = st.session_state.yolo_model(img_cv)[0]
            
            # Distance Alert
            hazards = [results.names[int(b.cls[0])] for b in results.boxes if (int(b.xyxy[0][3]) - int(b.xyxy[0][1]))/img_cv.shape[0] > 0.5]
            
            VisionMateUI.image_comparison(img, cv2.cvtColor(draw_boxes(img_cv, results), cv2.COLOR_BGR2RGB), "Original", "AI Vision")
            
            # Speak exact counts!
            obj_summary = create_detection_description(results)
            full_speech = ""
            if hazards:
                full_speech += f"Warning! {', '.join(set(hazards))} is very close. "
                st.error(f"⚠️ HAZARD: {', '.join(set(hazards))} is close!")
            
            full_speech += obj_summary
            st.markdown(VisionMateUI.info_card("AI Report", full_speech, "success"), unsafe_allow_html=True)
            if st.session_state.audio_enabled:
                st.markdown(text_to_speech(full_speech), unsafe_allow_html=True)

def process_video_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("High-Speed Navigation", "Optimized Spatial Analysis", "🎬")
    if st.button("⬅️ Home"): st.session_state.current_page = 'home'; st.rerun()
    
    vid = st.file_uploader("Upload Travel Video", type=['mp4'])
    if vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(vid.read()); path = tfile.name
        st.video(path)
        
        if st.button("🎬 High-Speed Analysis", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            win = st.empty(); prog = st.progress(0)
            
            unique_ids = set()
            final_counts = {}
            hazards = []
            
            # --- SPEED OPTIMIZATION SETTINGS ---
            frame_skip = 5  # Analyze 1 frame, skip 4. (Massive speed boost)
            count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Only run the AI on specific intervals
                if count % frame_skip == 0:
                    # stream=True allows YOLO to process more efficiently in memory
                    res = st.session_state.yolo_model.track(frame, persist=True, verbose=False, stream=False)[0]
                    h_img, w_img, _ = frame.shape
                    
                    if res.boxes.id is not None:
                        ids = res.boxes.id.int().cpu().tolist()
                        cls = res.boxes.cls.int().cpu().tolist()
                        boxes = res.boxes.xyxy.int().cpu().tolist()
                        
                        for obj_id, obj_cls, box in zip(ids, cls, boxes):
                            name = res.names[obj_cls]
                            if obj_id not in unique_ids:
                                unique_ids.add(obj_id)
                                final_counts[name] = final_counts.get(name, 0) + 1
                            
                            # Depth/Hazard Logic
                            if (box[3] - box[1]) / h_img > 0.5:
                                center_x = (box[0] + box[2]) / 2
                                pos = "LEFT" if center_x < w_img/3 else ("RIGHT" if center_x > 2*w_img/3 else "CENTER")
                                hazards.append(f"{name} at your {pos}")

                    # Update the display only on analyzed frames to save UI resources
                    win.image(cv2.cvtColor(draw_boxes(frame, res), cv2.COLOR_BGR2RGB), use_container_width=True)
                
                count += 1
                if count % 10 == 0: # Update progress bar less frequently to save time
                    prog.progress(min(count/total, 1.0))
            
            cap.release()
            
            # --- ACCESSIBILITY REPORT ---
            st.subheader("🏁 Safety Summary")
            unique_summary = ", ".join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in final_counts.items()])
            danger_zone = list(set(hazards))
            
            if danger_zone:
                alert = f"⚠️ IMMEDIATE HAZARD: {', '.join(danger_zone)}."
                st.error(alert)
                if st.session_state.audio_enabled: st.markdown(text_to_speech(alert), unsafe_allow_html=True)
            
            density = "Low" if len(unique_ids) < 5 else ("Moderate" if len(unique_ids) < 15 else "High")
            msg = f"Traffic Density is {density}. Objects identified: {unique_summary}."
            st.markdown(VisionMateUI.info_card("Final Scan", msg, "success"), unsafe_allow_html=True)
            if st.session_state.audio_enabled: st.markdown(text_to_speech(msg), unsafe_allow_html=True)
# --- Main App ---
def main_app():
    with st.sidebar:
        VisionMateUI.sidebar_header(st.session_state.user_name)
        st.session_state.audio_enabled = st.checkbox("🔊 Audio Feedback", value=st.session_state.audio_enabled)
        if st.button("🏠 Home", use_container_width=True): st.session_state.current_page = 'home'; st.rerun()
        if st.button("📷 Images", use_container_width=True): st.session_state.current_page = 'image'; st.rerun()
        if st.button("🎥 Live Call", use_container_width=True): st.session_state.current_page = 'live'; st.rerun()
        if st.button("🎬 Video", use_container_width=True): st.session_state.current_page = 'video'; st.rerun()
        if st.button("🚪 Logout", use_container_width=True): st.session_state.authenticated = False; st.rerun()

    if not st.session_state.models_loaded:
        with st.spinner("Loading AI Brain..."):
            y, p, m, d = load_models()
            st.session_state.yolo_model, st.session_state.blip_processor, st.session_state.blip_model, st.session_state.device = y, p, m, d
            st.session_state.models_loaded = True

    if st.session_state.current_page == 'home': home_page()
    elif st.session_state.current_page == 'image': process_image_page()
    elif st.session_state.current_page == 'live':
        VisionMateUI.page_header("Live Call", "Edge Agent Protocol", "🎥")
        if st.button("🚀 Start WebRTC Call", type="primary"):
            asyncio.run(Agent(edge=getstream.Edge(), agent_user=User(name="VisionMate", id="vm"), instructions="Assist the blind.", llm=gemini.Realtime(), processors=[agent_ultralytics.YoloProcessor(model="yolov8n.pt"), moondream.MoondreamProcessor()], turn_detection=smart_turn.SmartTurn()).start())
    elif st.session_state.current_page == 'video': process_video_page()

def main():
    st.set_page_config(page_title="VisionMate Hackathon", layout="wide")
    if not st.session_state.authenticated: 
        auth_page() # Removed the "from visionmate import..." line
    else: 
        main_app()

if __name__ == "__main__":
    main()
