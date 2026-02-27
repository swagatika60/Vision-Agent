import streamlit as st
from datetime import datetime

class VisionMateUI:
    """Ultra-Modern Cyberpunk UI for VisionMate Mobility Assistant"""
    
    @staticmethod
    def load_css():
        st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    :root {
        --primary: #6366F1;
        --secondary: #A855F7;
        --bg-dark: #0F172A;
        --glass: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --glow: rgba(99, 102, 241, 0.3);
    }

    /* Main Container Cleanup */
    .stApp {
        background: radial-gradient(circle at top right, #1E1B4B, #0F172A);
        font-family: 'Inter', sans-serif;
    }

    /* High-Tech Header */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 15px var(--glow));
    }

    .sub-header {
        color: #94A3B8;
        text-align: center;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 3rem;
    }

    /* Animated Mode Cards */
    .mode-card {
        background: var(--glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }

    .mode-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: var(--primary);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px var(--glow);
    }

    .mode-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 10px var(--glow));
    }

    /* Welcome Banner with Floating Effect */
    .welcome-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid var(--glass-border);
        border-radius: 30px;
        padding: 3rem;
        margin-bottom: 3rem;
        animation: float 6s ease-in-out infinite;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* Modern Info/Success Boxes */
    .info-box, .success-box, .warning-box {
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid;
        background: rgba(255,255,255,0.03);
    }
    .info-box { border-color: var(--primary); color: #E2E8F0; }
    .success-box { border-color: #10B981; color: #E2E8F0; }
    .warning-box { border-color: #F59E0B; color: #E2E8F0; }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
</style>
        """, unsafe_allow_html=True)

    @staticmethod
    def welcome_banner(user_name=None):
        if user_name:
            st.markdown(f"""
                <div class='welcome-header'>
                    <p style='color: var(--primary); font-weight: 600; margin-bottom: 0;'>SYSTEMS ONLINE</p>
                    <h2 style='margin-top: 0;'>Hello, {user_name}</h2>
                    <p style='color: #94A3B8;'>VisionMate is ready to assist your mobility.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='welcome-header'>
                    <h2 style='margin-bottom: 0.2rem;'>VisionMate AI</h2>
                    <p style='color: #94A3B8; font-size: 1.2rem;'>Next-Gen Mobility Assistance Protocol</p>
                </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def mode_card(title, desc, icon):
        return f"""
        <div class="mode-card">
            <div class="mode-icon">{icon}</div>
            <h3 style='color: white; margin-bottom: 0.5rem;'>{title}</h3>
            <p style='color: #94A3B8; font-size: 0.9rem;'>{desc}</p>
        </div>
        """

    @staticmethod
    def page_header(title, desc, icon):
        st.markdown(f"<h1 class='main-header'>{icon} {title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='sub-header'>{desc}</p>", unsafe_allow_html=True)

    @staticmethod
    def sidebar_header(user_name):
        st.sidebar.markdown(f"""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: var(--primary); margin-bottom: 0;'>👁️ VisionMate</h2>
                <p style='color: gray; font-size: 0.8rem;'>Operator: {user_name if user_name else 'Guest'}</p>
                <hr style='border-color: var(--glass-border);'>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def info_card(title, desc, type_class="info"):
        css_class = f"{type_class}-box"
        return f"""
        <div class="{css_class}">
            <h4 style='margin-bottom: 0.5rem; color: white;'>{title}</h4>
            <p style='margin-bottom: 0; font-size: 0.95rem;'>{desc}</p>
        </div>
        """

    # --- THESE WERE THE MISSING FUNCTIONS! ---
    @staticmethod
    def auth_tabs():
        return st.tabs(["🔐 Login", "✨ Register"])

    @staticmethod
    def image_comparison(img1, img2, label1, label2):
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption=label1, use_container_width=True)
        with col2:
            st.image(img2, caption=label2, use_container_width=True)
