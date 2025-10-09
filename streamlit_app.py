import streamlit as st
import requests
import json
import os
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="SDLC Agent Interface",
    page_icon="🚀",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8080"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_providers():
    """Get available API providers"""
    try:
        response = requests.get(f"{API_BASE_URL}/providers", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def trigger_sdlc_build(prompt):
    """Trigger SDLC build with the given prompt"""
    try:
        payload = {"prompt": prompt}
        response = requests.post(
            f"{API_BASE_URL}/sdlc/build",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error triggering build: {str(e)}")
        return None

def poll_sdlc_status(job_id, max_wait_sec: int = 600, interval_sec: float = 2.0):
    """Poll /sdlc/status until job completes or fails, or timeout."""
    start = time.time()
    last_status = None
    while time.time() - start < max_wait_sec:
        try:
            r = requests.get(f"{API_BASE_URL}/sdlc/status", params={"job_id": job_id}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                last_status = data
                status = (data or {}).get("status") or (data or {}).get("latest", {}).get("status")
                if status in ("completed", "failed"):
                    return data
        except Exception:
            pass
        time.sleep(interval_sec)
    return last_status or {"status": "timeout", "job_id": job_id}

def fetch_sdlc_report(job_id: str):
    try:
        r = requests.get(f"{API_BASE_URL}/sdlc/report", params={"job_id": job_id}, timeout=15)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        st.error(f"Error fetching report: {str(e)}")
        return None

def get_logs():
    """Get build logs"""
    try:
        response = requests.get(f"{API_BASE_URL}/logs", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Main Streamlit App
st.title("🚀 SDLC Agent Interface")
st.markdown("**Autonomous Software Development Lifecycle Agent**")

# Sidebar for API Status
with st.sidebar:
    st.header("API Status")
    if check_api_health():
        st.success("✅ API is running")
        st.info(f"Connected to: {API_BASE_URL}")
    else:
        st.error("❌ API is not running")
        st.warning("Please start the API server first:")
        st.code("python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload")
    
    # Show available providers
    providers = get_providers()
    if providers:
        st.header("Available Providers")
        for provider, available in providers.get("providers", {}).items():
            if available:
                st.success(f"✅ {provider}")
            else:
                st.error(f"❌ {provider}")

# Main Interface
if check_api_health():
    # Prompt Input
    st.header("📝 Project Prompt")
    prompt = st.text_area(
        "Describe your project:",
        placeholder="e.g., Build a task planner with login and CRUD operations",
        height=100
    )
    
    # Build Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Build Project", type="primary", disabled=not prompt.strip()):
            with st.spinner("Starting build..."):
                start_res = trigger_sdlc_build(prompt)
            if not start_res or not isinstance(start_res, dict) or start_res.get("status") != "build_started":
                st.error("❌ Failed to start build.")
            else:
                job_id = start_res.get("job_id")
                st.info(f"Build started. Job ID: {job_id}")
                status_area = st.empty()
                with st.spinner("Building your project in the background... (polling status)"):
                    while True:
                        status = poll_sdlc_status(job_id, max_wait_sec=2, interval_sec=0.1)  # single quick poll iteration
                        # Render current status
                        if isinstance(status, dict):
                            status_area.write(json.dumps(status, indent=2))
                            state = status.get("status")
                        else:
                            state = None
                        if state in ("completed", "failed", "timeout"):
                            break
                        time.sleep(2)
                        # continue polling until terminal
                        status = poll_sdlc_status(job_id, max_wait_sec=10, interval_sec=1.0)
                        if isinstance(status, dict):
                            status_area.write(json.dumps(status, indent=2))
                            state = status.get("status")
                        if state in ("completed", "failed", "timeout"):
                            break

                if state == "completed":
                    st.success("✅ Project built successfully!")
                    report = fetch_sdlc_report(job_id)
                    if report:
                        st.header("📊 Build Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Summary", report.get("summary", "N/A"))
                        with col2:
                            st.metric("Run Directory", report.get("run_dir", "N/A").split("\\")[-1])

                        artifacts = report.get("artifacts", {})
                        if artifacts:
                            st.subheader("📁 Generated Artifacts")
                            for category, details in artifacts.items():
                                with st.expander(f"📂 {category.title()}"):
                                    if isinstance(details, dict):
                                        for key, value in details.items():
                                            if isinstance(value, list):
                                                st.write(f"**{key}**:")
                                                for item in value:
                                                    st.write(f"  - {item}")
                                            else:
                                                st.write(f"**{key}**: {value}")
                                    else:
                                        st.write(details)

                        commands = report.get("commands", {})
                        if commands:
                            st.subheader("⚡ Next Steps")
                            for category, cmd_list in commands.items():
                                with st.expander(f"🔧 {category.title()}"):
                                    for cmd in cmd_list:
                                        st.code(cmd, language="bash")

                        logs = report.get("logs", [])
                        if logs:
                            st.subheader("📋 Build Logs")
                            for log in logs[-5:]:
                                status_icon = "✅" if log.get("success") else "❌"
                                st.write(f"{status_icon} **{log.get('stage', 'Unknown')}**: {log.get('message', 'No message')}")
                    else:
                        st.warning("Build completed but no report found yet. Try refreshing status or check /runs.")
                elif state == "failed":
                    st.error("❌ Build failed. Check logs for details.")
                else:
                    st.warning("⏱️ Build did not finish in time. Try polling again or check /runs.")
    
    # Logs Section
    st.header("📋 Recent Logs")
    if st.button("🔄 Refresh Logs"):
        logs = get_logs()
        if logs and logs.get("items"):
            for log in logs["items"][:10]:  # Show last 10 logs
                status = "✅" if log.get("success") else "❌"
                timestamp = log.get("created_at", "Unknown time")
                stage = log.get("stage", "Unknown")
                message = log.get("message", "No message")
                
                with st.expander(f"{status} {stage} - {timestamp}"):
                    st.write(f"**Message**: {message}")
                    if log.get("provider"):
                        st.write(f"**Provider**: {log.get('provider')}")
                    if log.get("model"):
                        st.write(f"**Model**: {log.get('model')}")
        else:
            st.info("No logs available")

else:
    st.error("🚫 Cannot connect to API server")
    st.markdown("""
    ### To start the API server:
    
    1. Open a new PowerShell window
    2. Navigate to your project directory:
       ```bash
       cd C:\\Users\\karis\\Desktop\\sdlc_agent_cursor
       ```
    3. Start the API server:
       ```bash
       python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
       ```
    4. Come back here and refresh this page
    """)

# Footer
st.markdown("---")
st.markdown("**SDLC Agent** - Autonomous Software Development Lifecycle Agent")
