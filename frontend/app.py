# frontend/app.py
import streamlit as st
import requests
import os
import time

st.set_page_config(page_title="Autonomous SDLC Agent", page_icon="🚀", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .success-box {
        padding: 1em;
        background-color: #d1fae5;
        color: #065f46;
        border-radius: 8px;
        margin-bottom: 1em;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2em;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
def get_backend_url():
    try:
        return st.secrets["backend_url"]
    except:
        return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

BACKEND_URL = get_backend_url()

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "preview_url" not in st.session_state:
    st.session_state.preview_url = None

# --- Header ---
st.markdown("<h1 class='main-header'>🚀 Autonomous SDLC Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Turning prompt into production-ready software automatically.</p>", unsafe_allow_html=True)

# --- Main Interface ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    prompt = st.text_area("What should I build for you?", height=150, placeholder="E.g., A beauty parlor booking app with admin dashboard...")
    
    if st.button("✨ Start Build"):
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Initializing Agent..."):
                try:
                    resp = requests.post(f"{BACKEND_URL}/sdlc/build", json={"prompt": prompt}, timeout=10)
                    if resp.ok:
                        data = resp.json()
                        st.session_state.job_id = data.get("job_id")
                        st.rerun()
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Backend unreachable: {e}")

# --- Status & Actions (If Job Active) ---
if st.session_state.job_id:
    st.markdown("---")
    
    # Poll Status
    status_data = {}
    try:
        r = requests.get(f"{BACKEND_URL}/sdlc/status", params={"job_id": st.session_state.job_id}, timeout=5)
        if r.ok:
            status_data = r.json()
    except: pass

    # Layout
    col_info, col_preview = st.columns([1, 2])
    
    with col_info:
        st.subheader("🛠️ Build Status")
        status = status_data.get("status", "unknown")
        phases = status_data.get("phases", {})
        
        # Phase Indicators
        for p in ["requirements", "planning", "design", "frontend", "backend"]:
            s = phases.get(p, "waiting")
            icon = "⚪"
            if s == "running": icon = "🔄"
            elif s == "completed": icon = "✅"
            elif s == "failed": icon = "❌"
            st.write(f"{icon} **{p.capitalize()}**")

        if status == "running":
            with st.spinner("Agent is working..."):
                time.sleep(2)
                st.rerun()
        
        if status == "completed":
            st.success("Build Completed Successfully!")
            
            st.markdown("### Actions")
            if st.button("📂 Open Run Folder"):
                try:
                    run_dir = status_data.get("run_dir")
                    requests.post(f"{BACKEND_URL}/open-folder", json={"path": run_dir})
                except: st.error("Failed to open folder")

    with col_preview:
        st.subheader("📱 App Preview")
        
        if status == "completed":
            if st.button("▶️ Launch Preview"):
                with st.spinner("Starting Dev Server..."):
                    try:
                        r = requests.post(f"{BACKEND_URL}/sdlc/preview", json={"job_id": st.session_state.job_id})
                        if r.ok:
                            url = r.json().get("url")
                            st.session_state.preview_url = url
                        else:
                            st.error(f"Preview Failed: {r.text}")
                    except Exception as e:
                        st.error(str(e))
        
        if st.session_state.preview_url:
            st.success(f"Live at: {st.session_state.preview_url}")
            st.components.v1.iframe(st.session_state.preview_url, height=600, scrolling=True)
        else:
            st.info("Preview available after build completes.")
