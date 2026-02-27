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

# Import UI components
from ui_components import VisionMateUI

# Load environment variables
load_dotenv()

# Hackathon Agent Libraries
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, gemini, ultralytics as agent_ultralytics, moondream, smart_turn

# Legacy AI Libraries
try:
    from ultralytics import YOLO
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    from gtts import gTTS
except ImportError as e:
    st.error(f"Missing required library: {e}. Please install all dependencies.")
    st.stop()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'welcome_played' not in st.session_state:
    st.session_state.welcome_played = False

USER_DATA_FILE = "users.json"

# ============== AUTHENTICATION MODULE ==============

def load_users():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(username, name, password):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        'password': hash_password(password),
        'name': name,
        'created_at': datetime.now().strftime('%Y-%m-%d')
    }
    save_users(users)
    return True, "Registration successful"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "User not found", None
    user_data = users[username]
    hashed_password = user_data['password'] if isinstance(user_data, dict) else user_data
    user_name = user_data.get('name', username) if isinstance(user_data, dict) else username
    if verify_password(password, hashed_password):
        return True, "Login successful", user_name
    return False, "Incorrect password", None

# ============== AUDIO UTILITIES ==============

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        return audio_html
    except Exception as e:
        st.error(f"Speech error: {e}")
        return None

# ============== VISION AI ENGINE ==============

@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO('yolov8m.pt') # Upgraded to Medium for Hackathon accuracy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        return yolo_model, blip_processor, blip_model, device
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def detect_objects(image, yolo_model):
    results = yolo_model(image)
    return results[0]

def draw_boxes(image, results):
    annotated_image = image.copy()
    h_img, _, _ = image.shape
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Distance Estimation Logic
        box_height = (y2 - y1) / h_img
        if box_height > 0.5:
            dist_tag, color = "CLOSE", (0, 0, 255) # Red
        elif box_height > 0.2:
            dist_tag, color = "MID", (0, 255, 255) # Yellow
        else:
            dist_tag, color = "FAR", (0, 255, 0) # Green
            
        label = f"{results.names[cls]} [{dist_tag}]"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_image

def create_detection_description(results):
    if len(results.boxes) == 0:
        return "No objects detected."
    counts = {}
    for box in results.boxes:
        name = results.names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1
    return ", ".join([f"{c} {n}{'s' if c > 1 else ''}" for n, c in counts.items()])

def generate_caption(image_pil, blip_processor, blip_model, device):
    try:
        inputs = blip_processor(image_pil, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_length=50)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except:
        return "Scene description unavailable."

# ============== PAGES ==============

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
                success, msg = register_user(nu, nn, np)
                if success: st.success(msg)
                else: st.error(msg)

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
            st.session_state.current_page = 'image'
            st.rerun()
    with col2:
        st.markdown(VisionMateUI.mode_card("Live Agent", "Real-time Call", "🎥"), unsafe_allow_html=True)
        if st.button("Start Live Mode", use_container_width=True):
            st.session_state.current_page = 'live'
            st.rerun()
    with col3:
        st.markdown(VisionMateUI.mode_card("Video Analysis", "Upload MP4s", "🎬"), unsafe_allow_html=True)
        if st.button("Start Video Mode", use_container_width=True):
            st.session_state.current_page = 'video'
            st.rerun()

def process_image_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Image Analysis", "Depth-Aware Vision", "📷")
    if st.button("⬅️ Home"): st.session_state.current_page = 'home'; st.rerun()
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            h_img = image_cv.shape[0]
            results = detect_objects(image_cv, st.session_state.yolo_model)
            annotated = draw_boxes(image_cv, results)
            
            # Distance Alert
            close_items = []
            for box in results.boxes:
                if (int(box.xyxy[0][3]) - int(box.xyxy[0][1])) / h_img > 0.5:
                    close_items.append(results.names[int(box.cls[0])])
            
            if close_items:
                alert = f"⚠️ WARNING: {', '.join(set(close_items))} is very close!"
                st.error(alert)
                if st.session_state.audio_enabled: st.markdown(text_to_speech(alert), unsafe_allow_html=True)
            
            VisionMateUI.image_comparison(image, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Original", "Analysis")
            desc = f"I see {create_detection_description(results)}. {generate_caption(image, st.session_state.blip_processor, st.session_state.blip_model, st.session_state.device)}"
            st.markdown(VisionMateUI.info_card("AI Insights", desc, "success"), unsafe_allow_html=True)
            if st.session_state.audio_enabled and not close_items:
                st.markdown(text_to_speech(desc), unsafe_allow_html=True)
async def launch_hackathon_agent(user_name):
    # Notice 'model' is changed to 'model_path'
    yolo_processor = agent_ultralytics.YOLOPoseProcessor(model_path="yolov8n.pt")
    
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="VisionMate", id="vision_agent"),
        instructions=f"You are a mobility assistant for {user_name}. Warn of obstacles and describe the scene concisely.",
        llm=gemini.Realtime(), 
        processors=[yolo_processor, moondream.MoondreamProcessor()], 
        turn_detection=smart_turn.SmartTurn()
    )
    await agent.start()
def process_live_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Live Agent", "Ultra-Low Latency Call", "🎥")
    if st.button("⬅️ Home"): st.session_state.current_page = 'home'; st.rerun()
    st.info("Initiating secure WebRTC connection via Stream Edge...")
    if st.button("🚀 Launch Call", type="primary", use_container_width=True):
        try: asyncio.run(launch_hackathon_agent(st.session_state.user_name))
        except Exception as e: st.error(f"Error: {e}")

def process_video_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Video Analysis", "Spatial Intelligence", "🎬")
    if st.button("⬅️ Home"): st.session_state.current_page = 'home'; st.rerun()
    
    uploaded_video = st.file_uploader("Upload Video", type=['mp4'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read()); video_path = tfile.name
        st.video(video_path)
        
        if st.button("🎬 Run AI Analysis", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_window = st.empty()
            progress = st.progress(0)
            logs = []
            
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if count % 10 == 0:
                    results = st.session_state.yolo_model(frame, verbose=False)[0]
                    frame_window.image(cv2.cvtColor(draw_boxes(frame, results), cv2.COLOR_BGR2RGB), use_container_width=True)
                    for b in results.boxes: logs.append(results.names[int(b.cls[0])])
                count += 1
                progress.progress(min(count/total_frames, 1.0))
            cap.release()
            
            unique_counts = {x: logs.count(x) for x in set(logs)}
            summary = "Analysis Complete: " + ", ".join([f"{v} {k}s" for k, v in unique_counts.items()])
            st.markdown(VisionMateUI.info_card("Report", summary, "success"), unsafe_allow_html=True)
            
            # Hackathon Export
            st.download_button("📥 Export JSON Report", data=json.dumps(unique_counts), file_name="report.json", mime="application/json")

# ============== MAIN APP ==============

def main():
    st.set_page_config(page_title="VisionMate Hackathon", layout="wide")
    if not st.session_state.authenticated:
        auth_page()
    else:
        with st.sidebar:
            VisionMateUI.sidebar_header(st.session_state.user_name)
            st.session_state.audio_enabled = st.checkbox("🔊 Audio Feedback", value=True)
            if st.button("🏠 Home"): st.session_state.current_page = 'home'; st.rerun()
            if st.button("🚪 Logout"): 
                st.session_state.authenticated = False; st.rerun()

        if not st.session_state.models_loaded:
            with st.spinner("🤖 Loading AI Brain..."):
                yolo, b_p, b_m, dev = load_models()
                st.session_state.yolo_model, st.session_state.blip_processor, st.session_state.blip_model, st.session_state.device = yolo, b_p, b_m, dev
                st.session_state.models_loaded = True
        
        if st.session_state.current_page == 'home': home_page()
        elif st.session_state.current_page == 'image': process_image_page()
        elif st.session_state.current_page == 'live': process_live_page()
        elif st.session_state.current_page == 'video': process_video_page()

if __name__ == "__main__":
    main()





   
