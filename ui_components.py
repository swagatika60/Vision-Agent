import streamlit as st
from datetime import datetime

class VisionMateUI:
    """Modern, High-Tech UI components for VisionMate Agent Protocol"""
    
    @staticmethod
    def load_css():
        """Load ultra-modern CSS styles with glassmorphism"""
        st.markdown("""
<style>
:root {
    --bg-main: #0B0E14; 
    --bg-card: rgba(22, 27, 34, 0.7); 
    --bg-soft: rgba(30, 41, 59, 0.5);
    --text-main: #F3F4F6;
    --text-muted: #9CA3AF;
    --primary: #6366F1; 
    --primary-glow: rgba(99, 102, 241, 0.4);
    --success: #10B981;
    --warning: #EF4444;
    --accent: #8B5CF6; 
    --border-soft: rgba(255, 255, 255, 0.08);
}
html, body, .stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.main-header {
    font-size: 3.5rem; font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.5rem; letter-spacing: -1px;
}
.sub-header {
    font-size: 1.2rem; color: var(--text-muted); text-align: center;
    margin-bottom: 3rem; font-weight: 300; letter-spacing: 0.5px;
}
.feature-card, .mode-card {
    background: var(--bg-card); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-soft); padding: 2rem; border-radius: 16px;
    margin: 1rem 0; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative; overflow: hidden;
}
.feature-card::before, .mode-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    opacity: 0; transition: opacity 0.3s ease;
}
.feature-card:hover, .mode-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px var(--primary-glow);
    border-color: rgba(255, 255, 255, 0.15);
}
.feature-card:hover::before, .mode-card:hover::before { opacity: 1; }
.feature-icon, .mode-icon {
    font-size: 2.8rem; margin-bottom: 1.2rem; display: inline-block;
    filter: drop-shadow(0 0 8px var(--primary-glow));
}
.mode-card h3 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.5rem; }
.mode-card p { color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; }
.welcome-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(139, 92, 246, 0.2); padding: 2.5rem; border-radius: 20px;
    text-align: center; margin-bottom: 3rem; position: relative;
}
.welcome-header h2 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }
.info-box, .warning-box, .success-box {
    padding: 1.2rem 1.5rem; border-radius: 12px; margin: 1rem 0;
    display: flex; flex-direction: column; border: 1px solid transparent;
}
.info-box { background: rgba(99, 102, 241, 0.1); border-color: rgba(99, 102, 241, 0.2); border-left: 4px solid var(--primary); }
.success-box { background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.2); border-left: 4px solid var(--success); }
.warning-box { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.2); border-left: 4px solid var(--warning); }
</style>
""", unsafe_allow_html=True)

    @staticmethod
    def welcome_banner(user_name=None):
        if user_name:
            st.markdown(f"<div class='welcome-header'><h2>Welcome back, {user_name}!</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='welcome-header'><h2>Welcome to VisionMate</h2><p>Your Real-time AI Mobility Assistant</p></div>", unsafe_allow_html=True)

    @staticmethod
    def auth_tabs():
        return st.tabs(["Login", "Register"])

    @staticmethod
    def success_message(msg):
        st.success(msg)

    @staticmethod
    def error_message(msg):
        st.error(msg)

    @staticmethod
    def warning_message(msg):
        st.warning(msg)

    @staticmethod
    def info_message(msg):
        st.info(msg)

    @staticmethod
    def footer(version="2.0", show_time=True):
        st.markdown("---")
        time_str = f" | {datetime.now().strftime('%H:%M:%S')}" if show_time else ""
        st.markdown(f"<div style='text-align: center; color: gray;'>VisionMate Protocol v{version}{time_str}</div>", unsafe_allow_html=True)

    @staticmethod
    def feature_card(title, desc, icon):
        return f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """

    @staticmethod
    def mode_card(title, desc, icon, hover_text=""):
        return f"""
        <div class="mode-card" title="{hover_text}">
            <div class="mode-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """

    @staticmethod
    def page_header(title, desc, icon):
        st.markdown(f"<h1 class='main-header'>{icon} {title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='sub-header'>{desc}</p>", unsafe_allow_html=True)

    @staticmethod
    def file_uploader(label, file_types, help_text, key):
        return st.file_uploader(label, type=file_types, help=help_text, key=key)

    @staticmethod
    def custom_button(label, key, type="secondary"):
        return st.button(label, key=key, type=type, use_container_width=True)

    @staticmethod
    def image_comparison(img1, img2, label1, label2):
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption=label1, use_container_width=True)
        with col2:
            st.image(img2, caption=label2, use_container_width=True)

    @staticmethod
    def info_card(title, desc, type_class="info"):
        return f"""
        <div class="{type_class}-box">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """

    @staticmethod
    def sidebar_header(user_name):
        st.markdown(f"## 👁️ VisionMate\nHello, {user_name if user_name else 'Guest'}!")

    @staticmethod
    def audio_status_indicator(is_enabled):
        status = "🔊 Audio ON" if is_enabled else "🔇 Audio OFF"
        st.markdown(f"**Status:** {status}")
