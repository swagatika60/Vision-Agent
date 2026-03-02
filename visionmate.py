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
# STATE INITIALIZATION & AUTH (Unchanged)
# ==========================================
def init_session_state():
    defaults = {'authenticated': False, 'username': None, 'user_name': None, 'audio_enabled': True, 'current_page': 'home', 'welcome_played': False}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
USER_DATA_FILE = "users.json"

def get_db():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f: return json.load(f)
        except json.JSONDecodeError: return {}
    return {}

def save_db(data):
    with open(USER_DATA_FILE, 'w') as f: json.dump(data, f, indent=4)

def hash_pw(password): return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
def verify_pw(password, hashed): return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def handle_login(username, password):
    db = get_db()
    if username not in db: return False, "Invalid credentials.", None
    user_data = db[username]
    if verify_pw(password, user_data.get('password')): return True, "Login successful.", user_data.get('name')
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
        return None

# ==========================================
# AGENT ORCHESTRATION (THE NEW WAY)
# ==========================================
@st.cache_resource
def get_vision_agent():
    """Initializes the Agent once. We don't load YOLO manually anymore."""
    return Agent(
        agent_user=User(name="VisionMate", id="sys_agent"),
        llm=gemini.Realtime(), # The Agent uses Gemini's vision reasoning under the hood
    )

# The master prompt that replaces all your math and CV2 code
HAZARD_PROMPT = """
You are a spatial safety assistant. Analyze this media and identify all objects. 
Calculate their bounding boxes internally. If any object's vertical height exceeds 40% of the total frame height, classify it as a 'HAZARD'.
Return ONLY a valid JSON object in this exact format:
{
    "detected_objects": ["list", "of", "all", "objects"],
    "hazards": ["list", "of", "hazard", "objects"],
    "summary": "A short 1-sentence summary of the scene."
}
"""

def parse_agent_response(response_text):
    """Safely extracts the JSON from the Agent's text response."""
    try:
        # Strip markdown code blocks if the LLM includes them
        clean_text = response_text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"detected_objects": [], "hazards": [], "summary": "Failed to parse agent response."}

# ==========================================
# UI ROUTING & PAGES
# ==========================================
def render_auth():
    # ... (Keep your existing auth UI code here) ...
    pass

def render_dashboard():
    # ... (Keep your existing dashboard routing code here) ...
    pass

def view_image_analysis():
    VisionMateUI.page_header("Image Processing", "Agent-Driven Frame Inspection", "📷")
    if st.button("← Return to Dashboard"): st.session_state.current_page = 'home'; st.rerun()
    
    upload = st.file_uploader("Select media", type=['jpg', 'png', 'jpeg'])
    if upload:
        img = Image.open(upload).convert("RGB")
        st.image(img, use_container_width=True)
        
        if st.button("Ask Agent to Scan", type="primary", use_container_width=True):
            with st.spinner("Agent is analyzing spatial data..."):
                # 1. Save upload to a temp file for the SDK
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    img.save(tmp.name)
                    temp_path = tmp.name
                
                # 2. Let the Agent do ALL the work (No YOLO, no math!)
                agent = get_vision_agent()
                response = agent.chat(HAZARD_PROMPT, media=temp_path)
                
                # 3. Handle the JSON output
                data = parse_agent_response(response.text)
                
                if data.get("hazards"):
                    alert = f"HAZARD DETECTED: {', '.join(data['hazards'])}."
                    st.error(alert)
                    if st.session_state.audio_enabled: st.markdown(generate_audio_feedback(alert), unsafe_allow_html=True)
                else:
                    st.success(data.get("summary"))
                    if st.session_state.audio_enabled: st.markdown(generate_audio_feedback("Scene is clear."), unsafe_allow_html=True)
                
                st.json(data) # Show the raw telemetry to the user

def view_video_pipeline():
    VisionMateUI.page_header("Video Pipeline", "Agent Time-Series Analysis", "🎬")
    if st.button("← Return to Dashboard"): st.session_state.current_page = 'home'; st.rerun()
    
    upload = st.file_uploader("Select environment recording", type=['mp4', 'avi', 'mov'])
    if upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(upload.read())
            temp_path = tmp.name
        
        st.video(temp_path)
        
        if st.button("Ask Agent to Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Agent is processing video timeline..."):
                
                # We just pass the whole video to the agent with the same prompt!
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

async def init_agent_webrtc(user_name):
    """Establishes stream edge connection for native low-latency agent. (UNCHANGED)"""
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
    # ... (Keep your WebRTC UI code here) ...
    pass

def main():
    st.set_page_config(page_title="VisionMate Core", layout="wide")
    if not st.session_state.authenticated:
        # render_auth()
        pass
    else:
        pg = st.session_state.current_page
        if pg == 'home': render_dashboard()
        elif pg == 'image': view_image_analysis()
        elif pg == 'live': view_live_protocol()
        elif pg == 'video': view_video_pipeline()

if __name__ == "__main__":
    main()
