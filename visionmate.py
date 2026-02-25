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

# Import UI components (Keeping your custom UI!)
from ui_components import VisionMateUI

# Load environment variables for the Hackathon API Keys
load_dotenv()

# Hackathon Agent Libraries
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, gemini, ultralytics as agent_ultralytics, moondream, smart_turn

# Legacy AI Libraries for Image/Video pages
try:
    from ultralytics import YOLO
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    from gtts import gTTS
except ImportError as e:
    st.error(f"Missing required library: {e}. Please install all dependencies.")
    st.stop()

# Initialize session state (Kept exactly as you had it)
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'welcome_played' not in st.session_state:
    st.session_state.welcome_played = False

USER_DATA_FILE = "users.json"

# ============== AUTHENTICATION MODULE ==============
# (Kept exactly the same)

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
    if isinstance(user_data, str):
        hashed_password = user_data
        user_name = username
    else:
        hashed_password = user_data['password']
        user_name = user_data.get('name', username)
    
    if verify_password(password, hashed_password):
        return True, "Login successful", user_name
    return False, "Incorrect password", None

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        return audio_html
    except Exception as e:
        st.error(f"Speech error: {e}")
        return None

def auth_page():
    VisionMateUI.load_css()
    VisionMateUI.welcome_banner()
    tab1, tab2 = VisionMateUI.auth_tabs()
    
    with tab1:
        st.subheader("🔐 Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            col1, col2 = st.columns([3, 1])
            with col1:
                login_submitted = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
            with col2:
                demo_submitted = st.form_submit_button("👁️ Demo Mode", use_container_width=True)
            
            if login_submitted:
                if username and password:
                    success, message, user_name = login_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_name = user_name
                        VisionMateUI.success_message(message)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        VisionMateUI.error_message(message)
                else:
                    VisionMateUI.warning_message("Please enter both username and password")
            
            if demo_submitted:
                st.session_state.authenticated = True
                st.session_state.username = "demo_user"
                st.session_state.user_name = "Demo User"
                VisionMateUI.success_message("Welcome to Demo Mode!")
                time.sleep(0.5)
                st.rerun()
    
    with tab2:
        st.subheader("✨ Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username", key="reg_username")
            new_name = st.text_input("Your Name", key="reg_name")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            register_submitted = st.form_submit_button("📝 Create Account", type="primary", use_container_width=True)
            
            if register_submitted:
                if new_username and new_name and new_password and confirm_password:
                    if new_password != confirm_password:
                        VisionMateUI.error_message("Passwords do not match")
                    elif len(new_password) < 6:
                        VisionMateUI.error_message("Password must be at least 6 characters")
                    else:
                        success, message = register_user(new_username, new_name, new_password)
                        if success:
                            VisionMateUI.success_message(message)
                            VisionMateUI.info_message("You can now login with your credentials")
                        else:
                            VisionMateUI.error_message(message)
                else:
                    VisionMateUI.warning_message("Please fill all fields")
    
    VisionMateUI.footer(version="2.0", show_time=True)

# ============== VISION AI SYSTEM ==============

@st.cache_resource

@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO('yolov8m.pt')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Using the larger model for better accuracy!
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
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{results.names[cls]} {conf:.2f}"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return annotated_image

def generate_caption(image_pil, blip_processor, blip_model, device):
    try:
        inputs = blip_processor(image_pil, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Caption error: {e}"

def create_detection_description(results):
    if len(results.boxes) == 0:
        return "No objects detected in the scene."
    object_counts = {}
    for box in results.boxes:
        cls = int(box.cls[0])
        name = results.names[cls]
        object_counts[name] = object_counts.get(name, 0) + 1
    description = f"Detected {len(results.boxes)} object{'s' if len(results.boxes) > 1 else ''}: "
    items = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in object_counts.items()]
    description += ", ".join(items)
    return description

def home_page():
    VisionMateUI.load_css()
    VisionMateUI.welcome_banner(st.session_state.user_name)
    
    if not st.session_state.welcome_played:
        welcome_msg = f"Welcome to Vision Mate, {st.session_state.user_name}"
        if st.session_state.audio_enabled:
            audio_html = text_to_speech(welcome_msg)
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)
        st.session_state.welcome_played = True
    
    st.markdown("### 🌟 Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(VisionMateUI.feature_card("Object Detection", "Real-time identification of objects in your environment", "🎯"), unsafe_allow_html=True)
    with col2:
        st.markdown(VisionMateUI.feature_card("Scene Description", "AI-powered descriptions of visual scenes", "🖼️"), unsafe_allow_html=True)
    with col3:
        st.markdown(VisionMateUI.feature_card("Audio Feedback", "Voice announcements for accessibility", "🔊"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📱 Choose Your Mode")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(VisionMateUI.mode_card("Image Analysis", "Analyze uploaded images", "📷", "Upload photos for detailed analysis"), unsafe_allow_html=True)
        if st.button("Start Image Mode", key="btn_image_mode", use_container_width=True):
            st.session_state.current_page = 'image'
            st.rerun()
    with col2:
        st.markdown(VisionMateUI.mode_card("Live Camera (Hackathon Agent)", "Real-time WebRTC AI", "🎥", "Connect directly to the Stream Vision Edge"), unsafe_allow_html=True)
        if st.button("Start Live Mode", key="btn_live_mode", use_container_width=True):
            st.session_state.current_page = 'live'
            st.rerun()
    with col3:
        st.markdown(VisionMateUI.mode_card("Video Analysis", "Analyze video files", "🎬", "Upload videos for processing"), unsafe_allow_html=True)
        if st.button("Start Video Mode", key="btn_video_mode", use_container_width=True):
            st.session_state.current_page = 'video'
            st.rerun()
def process_image_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Image Analysis", "Upload and analyze images", "📷")
    if st.button("⬅️ Back to Home", key="back_from_image"):
        st.session_state.current_page = 'home'
        st.rerun()
    st.markdown("---")
    
    uploaded_file = VisionMateUI.file_uploader("Upload an image for analysis", file_types=['jpg', 'jpeg', 'png'], help_text="Supported formats: JPG, JPEG, PNG", key="image_upload_main")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if VisionMateUI.custom_button("🔍 Analyze Image", "analyze_img_btn", "primary"):
            with st.spinner("🤖 Analyzing image..."):
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                h_img = image_cv.shape[0]
                
                # 1. Detection and Annotation
                results = detect_objects(image_cv, st.session_state.yolo_model)
                annotated = draw_boxes(image_cv, results)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # 2. Extract distance-aware labels for the Danger Alert
                detection_list_with_dist = []
                for box in results.boxes:
                    label = results.names[int(box.cls[0])]
                    # Calculate height to determine distance (consistent with draw_boxes)
                    box_height = (int(box.xyxy[0][3]) - int(box.xyxy[0][1])) / h_img
                    dist = "CLOSE" if box_height > 0.5 else ("MID" if box_height > 0.2 else "FAR")
                    detection_list_with_dist.append(f"{label} ({dist})")

                # 3. Filter for close objects
                close_objects = [obj for obj in detection_list_with_dist if "CLOSE" in obj]
                
                # 4. Generate Descriptions
                detection_desc = create_detection_description(results)
                caption = generate_caption(image, st.session_state.blip_processor, st.session_state.blip_model, st.session_state.device)
                full_description = f"{detection_desc}. Scene description: {caption}"

                # --- UI DISPLAY ---
                st.markdown("### 📊 Analysis Results")
                
                # Show Danger Alert if objects are close
                if close_objects:
                    danger_msg = f"⚠️ WARNING: {', '.join(close_objects)} are very close to you!"
                    st.error(danger_msg)
                    if st.session_state.audio_enabled:
                        st.markdown(text_to_speech(danger_msg), unsafe_allow_html=True)

                VisionMateUI.image_comparison(image, annotated_rgb, "Original", "Detected Objects")
                st.markdown(VisionMateUI.info_card("Analysis Result", full_description, "success"), unsafe_allow_html=True)
                
                # Play general description after the danger warning
                if st.session_state.audio_enabled and not close_objects:
                    audio_html = text_to_speech(full_description)
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)

# Then play the general scene description
st.markdown(VisionMateUI.info_card("Analysis Result", full_description, "success"), unsafe_allow_html=True)
# ============== HACKATHON LIVE CAMERA AGENT ==============

async def launch_hackathon_agent(user_name):
    """The core requirement for the Vision Possible Hackathon"""
    yolo_processor = agent_ultralytics.YoloProcessor(model="yolov8n.pt")
    moondream_processor = moondream.MoondreamProcessor()
    turn_detector = smart_turn.SmartTurn()

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="VisionMate", id="visionmate_agent"),
        instructions=(
            f"You are a real-time mobility assistant for {user_name}, who is visually impaired. "
            "You receive obstacle data from YOLO and scene descriptions from Moondream. "
            "Warn the user of immediate physical obstacles and answer their questions about their surroundings. "
            "Keep your audio responses extremely concise, calm, and highly descriptive."
        ),
        llm=gemini.Realtime(), 
        processors=[yolo_processor, moondream_processor], 
        turn_detection=turn_detector 
    )
    await agent.start()

def process_live_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Live Vision Agent", "Ultra-Low Latency WebRTC Assistant", "🎥")
    
    if st.button("⬅️ Back to Home", key="back_from_live"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    st.markdown(VisionMateUI.info_card(
        "Hackathon Agent Protocol Active",
        "This mode connects directly to Stream's edge network for sub-30ms audio and video latency.",
        "success"
    ), unsafe_allow_html=True)
    
    st.write("Clicking the button below will open a secure WebRTC video call. The AI Agent will join the call with you, watch your camera, and speak to you in real-time.")

    if st.button("🚀 Launch VisionMate Agent", type="primary", use_container_width=True, key="start_webrtc_btn"):
        st.info("Initiating connection to Stream Edge Network... A new browser tab should open shortly.")
        try:
            # Run the asynchronous hackathon agent
            asyncio.run(launch_hackathon_agent(st.session_state.user_name))
        except Exception as e:
            st.error(f"Failed to launch Agent: {e}. Check your .env API keys!")

# =========================================================

def process_video_page():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Video Analysis", "Deep Spatial Intelligence", "🎬")
    
    if st.button("⬅️ Back to Home", key="back_from_video"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'], key="video_upload_main")
    
    if uploaded_video:
        # 1. Save and Display Original Video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        st.video(video_path)
        
        # 2. Extract Metadata
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Duration", f"{duration:.1f}s")
        with col2: st.metric("Total Frames", total_frames)
        with col3: st.metric("FPS", fps)
        
        # 3. The Processing Engine
        if st.button("🎬 Start AI Video Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            video_frame_window = st.empty() # This is where the AI "sees" the video
            
            cap = cv2.VideoCapture(video_path)
            detections_log = []
            
            # For a hackathon demo, we process every 5th frame to keep it snappy
            # but still highly accurate.
            frame_skip = 5 
            current_frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame_idx % frame_skip == 0:
                    # RUN DETECTION
                    results = st.session_state.yolo_model(frame, verbose=False)[0]
                    
                    # DRAW BOXES
                    annotated_frame = draw_boxes(frame, results)
                    
                    # Convert BGR to RGB for Streamlit
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_frame_window.image(rgb_frame, use_container_width=True)
                    
                    # LOG DATA
                    for box in results.boxes:
                        label = results.names[int(box.cls[0])]
                        detections_log.append(label)
                
                current_frame_idx += 1
                progress_bar.progress(min(current_frame_idx / total_frames, 1.0))
                status_text.text(f"Scanning Frame {current_frame_idx} of {total_frames}...")
            
            cap.release()
            status_text.success("✅ Analysis Complete!")
            
            # 4. Generate AI Summary
            if detections_log:
                counts = {x: detections_log.count(x) for x in set(detections_log)}
                summary_items = [f"{count} instances of {name}" for name, count in counts.items()]
                final_summary = "Video Analysis Report: " + ", ".join(summary_items)
                
                st.markdown(VisionMateUI.info_card("AI Insights", final_summary, "success"), unsafe_allow_html=True)
                
                if st.session_state.audio_enabled:
                    audio_html = text_to_speech(final_summary)
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)

def main_app():
    VisionMateUI.load_css()
    with st.sidebar:
        VisionMateUI.sidebar_header(st.session_state.user_name)
        st.markdown("---")
        VisionMateUI.audio_status_indicator(st.session_state.audio_enabled)
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        if st.button("🏠 Home", use_container_width=True, key="nav_home_sb"):
            st.session_state.current_page = 'home'
            st.rerun()
        if st.button("📷 Image Analysis", use_container_width=True, key="nav_image_sb"):
            st.session_state.current_page = 'image'
            st.rerun()
        if st.button("🎥 Live Agent Call", use_container_width=True, key="nav_live_sb"):
            st.session_state.current_page = 'live'
            st.rerun()
        if st.button("🎬 Video Analysis", use_container_width=True, key="nav_video_sb"):
            st.session_state.current_page = 'video'
            st.rerun()
            
        st.markdown("---")
        st.session_state.audio_enabled = st.checkbox("🔊 Audio Enabled", value=st.session_state.audio_enabled, key="audio_toggle_sb")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, key="logout_sb"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_name = None
            st.session_state.welcome_played = False
            st.session_state.current_page = 'home'
            st.rerun()
            
        st.markdown("---")
        st.caption(f"👤 {st.session_state.user_name}")
        st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")

    if not st.session_state.models_loaded:
        with st.spinner("🤖 Loading legacy AI models for Image/Video pages..."):
            yolo_model, blip_processor, blip_model, device = load_models()
            if yolo_model and blip_model:
                st.session_state.yolo_model = yolo_model
                st.session_state.blip_processor = blip_processor
                st.session_state.blip_model = blip_model
                st.session_state.device = device
                st.session_state.models_loaded = True
                VisionMateUI.success_message("✅ Models loaded successfully!")
            else:
                VisionMateUI.error_message("Failed to load models.")
                return

    if st.session_state.current_page == 'home': home_page()
    elif st.session_state.current_page == 'image': process_image_page()
    elif st.session_state.current_page == 'live': process_live_page()
    elif st.session_state.current_page == 'video': process_video_page()
    
    VisionMateUI.footer(version="2.0", show_time=True)

def main():
    st.set_page_config(page_title="VisionMate - Hackathon Edition", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")
    if not st.session_state.authenticated:
        auth_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
