import streamlit as st
import json
import os
import bcrypt
import base64
from io import BytesIO
import tempfile
import asyncio
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
from ui_components import VisionMateUI
load_dotenv()

# --- THE SDK WAY: Only import the Agent and its tools ---
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, gemini, ultralytics as agent_ultralytics

try:
    from gtts import gTTS
    from PIL import Image
except ImportError as e:
    st.error(f"Missing required dependency: {e}. Please run pip install -r requirements.txt")
    st.stop()

# ==========================================
# STATE INITIALIZATION & AUTH
# ==========================================
def init_session_state():
    defaults = {
        'authenticated': False,
        'username': None,
        'user_name': None,
        'audio_enabled': True,
        'current_page': 'home',
        'welcome_played': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
USER_DATA_FILE = "users.json"

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
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        return f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        return None

# ==========================================
# AGENT ORCHESTRATION (THE NEW SDK WAY)
# ==========================================
def get_vision_agent():
    return Agent(
        agent_user=User(name="VisionMate", id="sys_agent"),
        llm=gemini.Realtime()
    )

HAZARD_PROMPT = """
You are a spatial safety assistant. Analyze this media and identify all objects. 
If any object's vertical height appears to exceed 40% of the total frame height, classify it as a 'HAZARD'.
Return ONLY a valid JSON object in this exact format, with no markdown formatting:
{
    "detected_objects": ["list", "of", "all", "objects"],
    "hazards": ["list", "of", "hazard", "objects"],
    "summary": "A short 1-sentence summary of the scene."
}
"""

def parse_agent_response(response_text):
    try:
        clean_text = response_text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception:
        return {"detected_objects": [], "hazards": [], "summary": "Failed to parse agent response."}

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
        st.markdown(VisionMateUI.mode_card("Image Processing", "Agent analysis", "📷"), unsafe_allow_html=True)
        if st.button("Launch Image Engine", use_container_width=True):
            st.session_state.current_page = 'image'; st.rerun()
    with col2:
        st.markdown(VisionMateUI.mode_card("Live Protocol", "Agent WebRTC", "🎥"), unsafe_allow_html=True)
        if st.button("Launch Agent Room", use_container_width=True):
            st.session_state.current_page = 'live'; st.rerun()
    with col3:
        st.markdown(VisionMateUI.mode_card("Video Pipeline", "Agent video scan", "🎬"), unsafe_allow_html=True)
        if st.button("Launch Video Pipeline", use_container_width=True):
            st.session_state.current_page = 'video'; st.rerun()

def view_image_analysis():
    VisionMateUI.load_css()
    VisionMateUI.page_header("Image Processing", "Agent-Driven Frame Inspection", "📷")
    if st.button("← Return to Dashboard"): 
        st.session_state.current_page = 'home'; st.rerun()
    
    upload = st.file_uploader("Select media", type=['jpg', 'png', 'jpeg'])
    if upload:
        img = Image.open(upload).convert("RGB")
        _, col_main, _ = st.columns([1, 2, 1])
        with col_main:
            st.image(img, use_container_width=True)
        
        if st.button("Ask Agent to Scan", type="primary", use_container_width=True):
            with st.spinner("Agent is analyzing spatial data..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    img.save(tmp.name)
                    temp_path = tmp.name
                
                agent = get_vision_agent()
                response = agent.chat(HAZARD_PROMPT, media=temp_path)
                data = parse_agent_response(response.text)
                
                if data.get("hazards"):
                    alert = f"HAZARD DETECTED: {', '.join(data['hazards'])}."
                    st.error(alert)
                    if st.session_state.audio_enabled: st.markdown(generate_audio_feedback(alert), unsafe_allow_html=True)
                else:
                    st.success(data.get("summary", "Scene is clear."))
                    if st.session_state.audio_enabled: st.markdown(generate_audio_feedback("Scene is clear."), unsafe_allow_html=True)
                
                st.markdown("### Agent Telemetry")
                st.json(data)

async def init_agent_webrtc(user_name):
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
    VisionMateUI.page_header("Video Pipeline", "Agent Time-Series Analysis", "🎬")
    if st.button("← Return to Dashboard"): 
        st.session_state.current_page = 'home'; st.rerun()
    
    st.markdown("---")
    upload = st.file_uploader("Select environment recording", type=['mp4', 'avi', 'mov'])
    
    if upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(upload.read())
            temp_path = tmp.name
        
        _, col_vid, _ = st.columns([1, 2, 1])
        with col_vid: st.video(temp_path)
        
        if st.button("Ask Agent to Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Agent is processing video timeline. This may take a moment..."):
                agent = get_vision_agent()
                response = agent.chat(HAZARD_PROMPT, media=temp_path)
                data = parse_agent_response(response.text)
                
                st.markdown("### 📡 Agent Telemetry Report")
                st.json(data)
                
                if data.get("hazards"):
                    warn = f"CRITICAL: Hazards recorded: {', '.join(data['hazards'])}."
                    st.error(warn)
                else:
                    warn = "Audit passed. Zero hazards recorded."
                    st.success(warn)
                
                if st.session_state.audio_enabled: st.markdown(generate_audio_feedback(warn), unsafe_allow_html=True)

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
        
        pg = st.session_state.current_page
        if pg == 'home': render_dashboard()
        elif pg == 'image': view_image_analysis()
        elif pg == 'live': view_live_protocol()
        elif pg == 'video': view_video_pipeline()

if __name__ == "__main__":
    main()
