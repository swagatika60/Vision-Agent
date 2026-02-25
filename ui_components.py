import streamlit as st
from typing import Optional, Dict, List
from datetime import datetime

class VisionMateUI:
    """Modern, High-Tech UI components for VisionMate Agent Protocol"""
    
    # ===================== MODERN CSS STYLES =====================
    
    @staticmethod
    def load_css():
        """Load ultra-modern CSS styles with glassmorphism"""
        st.markdown("""
<style>

/* ================================
   THEME VARIABLES (DARK MODE OPTIMIZED)
================================ */
:root {
    --bg-main: #0B0E14; /* Deep, techy background */
    --bg-card: rgba(22, 27, 34, 0.7); /* Glassmorphism base */
    --bg-soft: rgba(30, 41, 59, 0.5);

    --text-main: #F3F4F6;
    --text-muted: #9CA3AF;

    --primary: #6366F1; /* Vibrant Indigo */
    --primary-glow: rgba(99, 102, 241, 0.4);
    
    --success: #10B981;
    --warning: #EF4444;
    --accent: #8B5CF6; /* Purple accent for AI feel */

    --border-soft: rgba(255, 255, 255, 0.08);
}

/* ================================
   GLOBAL TEXT & BACKGROUND
================================ */
html, body, .stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* ================================
   HEADERS & TYPOGRAPHY
================================ */
.main-header {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.sub-header {
    font-size: 1.2rem;
    color: var(--text-muted);
    text-align: center;
    margin-bottom: 3rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ================================
   GLASSMORPHISM CARDS
================================ */
.feature-card, .mode-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-soft);
    padding: 2rem;
    border-radius: 16px;
    margin: 1rem 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

/* AI Glow Effect */
.feature-card::before, .mode-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.feature-card:hover, .mode-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px var(--primary-glow);
    border-color: rgba(255, 255, 255, 0.15);
}

.feature-card:hover::before, .mode-card:hover::before {
    opacity: 1;
}

.feature-icon, .mode-icon {
    font-size: 2.8rem;
    margin-bottom: 1.2rem;
    display: inline-block;
    filter: drop-shadow(0 0 8px var(--primary-glow));
}

.mode-card h3 {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.mode-card p {
    color: var(--text-muted);
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ================================
   WELCOME HEADER
================================ */
.welcome-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(139, 92, 246, 0.2);
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 3rem;
    position: relative;
}

.welcome-header h2 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* ================================
   INFO BOXES (STATUS MESSENGERS)
================================ */
.info-box, .warning-box, .success-box {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    display: flex;
    flex-direction: column;
    border: 1px solid transparent;
}

.info-box {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.2);
    border-left: 4px solid var(--primary);
}

.success-box {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.2);
    border-left: 4px solid var(--success);
}

.warning-box {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.2);
    border