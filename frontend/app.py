# frontend/app.py
import streamlit as st
import requests
import os
from typing import Any

st.set_page_config(page_title="Autonomous SDLC Agent", page_icon="🤖", layout="centered")

st.title("Autonomous SDLC Agent")
st.caption("Describe what to build, then see generated folders and commands.")

def get_backend_default() -> str:
    try:
        return st.secrets["backend_url"]
    except Exception:
        return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- session_state defaults (avoid assigning widget returns directly to session_state) ---
if "backend_url" not in st.session_state:
    st.session_state.backend_url = get_backend_default()
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_kind" not in st.session_state:
    st.session_state.agent_kind = None
if "req_title" not in st.session_state:
    st.session_state.req_title = ""
if "req_text" not in st.session_state:
    st.session_state.req_text = ""
if "req_list" not in st.session_state:
    st.session_state.req_list = []
if "req_id" not in st.session_state:
    st.session_state.req_id = 0
if "agent_file_choice" not in st.session_state:
    st.session_state.agent_file_choice = None

# --- Sidebar settings (explicit widget key "backend_url") ---
with st.sidebar:
    st.subheader("Settings")
    backend_url = st.text_input("Backend URL", key="backend_url")
    # keep a local variable for quick checks
    try:
        r = requests.get(f"{backend_url}/health", timeout=5)
        status = "online" if r.ok else "unreachable"
    except Exception:
        status = "offline"
    st.caption(f"API status: {status}")

    # API list in sidebar
    try:
        idx = requests.get(f"{backend_url}/", timeout=10)
        if idx.ok:
            data = idx.json()
            eps = data.get("endpoints", [])
            st.markdown("**APIs**")
            for it in eps:
                st.markdown(f"- `{it.get('method','GET')}` [{it.get('path','')}]({backend_url}{it.get('path','')})")
            st.markdown(f"[Swagger]({backend_url}/docs) · [ReDoc]({backend_url}/redoc)")
    except Exception:
        pass

    # Provider order & availability
    try:
        p = requests.get(f"{backend_url}/providers", timeout=10)
        prov = p.json().get("providers", {}) if p.ok else {}
    except Exception:
        prov = {}
    order = [("Gemini","gemini"),("Perplexity","perplexity"),("Hugging Face","hf"),("Mistral","mistral"),("Groq","groq"),("OpenAI","openai"),("v0.dev","v0")]
    st.markdown("**Providers (order used)**")
    for label,key in order:
        ok = prov.get(key)
        color = "#22c55e" if ok else "#ef4444"
        st.markdown(f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:999px;border:1px solid {color};background:{color}20;font-size:12px'>{label}: {'available' if ok else 'unavailable'}</span>", unsafe_allow_html=True)

    # Latest run folder + commands in sidebar
    st.markdown("---")
    st.markdown("**Latest Run**")
    try:
        s = requests.get(f"{backend_url}/sdlc/status", timeout=10)
        latest = (s.json().get("latest") if s.ok else None)
        if latest and latest.get("run_dir"):
            run_dir = latest.get("run_dir")
            st.caption(f"Job {latest.get('job_id')} · {latest.get('status')}")
            st.code(run_dir)
            rep = requests.get(f"{backend_url}/sdlc/report", params={"run_dir": run_dir}, timeout=10)
            if rep.ok:
                report = rep.json()
                artifacts = (report or {}).get("artifacts", {})
                commands = (report or {}).get("commands", {})
                for key,val in artifacts.items():
                    with st.expander(f"📁 {key}"):
                        if isinstance(val, dict):
                            for v in val.values():
                                if isinstance(v, str):
                                    st.code(v)
                        elif isinstance(val, list):
                            for v in val:
                                st.code(str(v))
                        else:
                            st.code(str(val))
                if commands:
                    st.markdown("**Commands**")
                    for section in ["backend","frontend","tests","docs","deploy"]:
                        cmds = commands.get(section)
                        if cmds:
                            st.caption(section.capitalize())
                            st.code("\n".join(cmds))
        else:
            st.caption("No runs yet. Use Build tab.")
    except Exception:
        st.caption("Run info unavailable.")

tab_build, tab_api = st.tabs(["Build", "APIs & Runs"])

with tab_build:
    st.subheader("Build a Project")
    build_prompt = st.text_area("Describe what to build", placeholder="Build a full-stack app that ...", key="build_prompt")
    colB1, colB2 = st.columns(2)
    with colB1:
        if st.button("Start Build", disabled=(status != "online")):
            if not build_prompt.strip():
                st.warning("Enter a prompt.")
            else:
                try:
                    resp = requests.post(f"{backend_url}/sdlc/build", json={"prompt": build_prompt}, timeout=30)
                    if resp.ok:
                        out = resp.json()
                        st.session_state["last_job_id"] = out.get("job_id")
                        with st.status("Building… this may take a few minutes", expanded=True) as status_box:
                            import time
                            while True:
                                time.sleep(2)
                                s = requests.get(f"{backend_url}/sdlc/status", timeout=10)
                                if not s.ok:
                                    st.write("Status unavailable…")
                                    continue
                                j = s.json().get("latest", {})
                                st.write(f"Status: {j.get('status')} started: {j.get('started_at')} finished: {j.get('finished_at')}")
                                if j.get("status") in ("completed", "failed"):
                                    break
                            status_box.update(label="Build finished", state="complete")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(f"Cannot reach backend. {e}")
    with colB2:
        if st.button("Refresh Status", disabled=(status != "online")):
            try:
                s = requests.get(f"{backend_url}/sdlc/status", timeout=15)
                st.json(s.json() if s.ok else {"error": s.text})
            except Exception as e:
                st.error(f"Cannot reach backend. {e}")

    st.markdown("<div style='margin-top: 8px'></div>", unsafe_allow_html=True)

# ----------------
# APIs & Runs tab
# ----------------
with tab_api:
    st.subheader("Available APIs")
    try:
        r = requests.get(f"{backend_url}/", timeout=15)
        p = requests.get(f"{backend_url}/providers", timeout=15)
        providers = (p.json().get("providers", {}) if p.ok else {}) if p is not None else {}
        if r.ok:
            api_index = r.json()
            endpoints = api_index.get("endpoints", [])
            # Render as cards
            if endpoints:
                cols = st.columns(2)
                for i, it in enumerate(endpoints):
                    with cols[i % 2]:
                        st.markdown(
                            f"<div style='border:1px solid #2d2d2d;border-radius:8px;padding:12px;margin-bottom:8px;background:#0f1116;'>"
                            f"<div style='font-size:12px;text-transform:uppercase;color:#9aa4b2'>{it.get('method','GET')}</div>"
                            f"<div style='font-family:monospace'>{it.get('path','')}</div>"
                            f"<div style='color:#9aa4b2;font-size:13px;margin-top:4px'>{it.get('desc','')}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            docs = api_index.get("docs", {})
            if docs:
                st.markdown(f"Docs: [Swagger]({backend_url}/docs) · [ReDoc]({backend_url}/redoc)")
        else:
            st.info("API index unavailable.")
    except Exception as e:
        st.info(f"API index error: {e}")

    st.markdown("---")
    st.subheader("Provider Order & Availability")
    try:
        p = requests.get(f"{backend_url}/providers", timeout=15)
        prov = p.json().get("providers", {}) if p.ok else {}
    except Exception:
        prov = {}
    desired_order = [
        ("Gemini", "gemini"),
        ("Perplexity", "perplexity"),
        ("Hugging Face", "hf"),
        ("Mistral", "mistral"),
        ("Groq", "groq"),
        ("OpenAI", "openai"),
        ("v0.dev (frontend)", "v0"),
    ]
    badges = []
    for label, key in desired_order:
        available = prov.get(key)
        color = "#22c55e" if available else "#ef4444"
        badges.append(f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;background:{color}20;border:1px solid {color};margin:4px 6px 4px 0;font-size:12px'>{label}: {'available' if available else 'unavailable'}</span>")
    st.markdown("".join(badges), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Latest Run — Folders and Commands")
    try:
        s = requests.get(f"{backend_url}/sdlc/status", timeout=15)
        if s.ok:
            data = s.json()
            latest = data.get("latest") if isinstance(data, dict) else None
            if latest:
                st.caption(f"Job ID: {latest.get('job_id')} · Status: {latest.get('status')}")
                run_dir = latest.get("run_dir")
                if run_dir:
                    st.markdown(f"📁 **Run Folder**")
                    st.code(run_dir)
                    # Fetch report for artifacts and commands
                    try:
                        rep = requests.get(f"{backend_url}/sdlc/report", params={"run_dir": run_dir}, timeout=20)
                        if rep.ok:
                            report = rep.json()
                            artifacts = (report or {}).get("artifacts", {})
                            commands = (report or {}).get("commands", {})
                            if artifacts:
                                st.markdown("#### Folders/Artifacts")
                                # Render as expandable folders
                                for key, val in artifacts.items():
                                    with st.expander(f"📁 {key}"):
                                        if isinstance(val, dict):
                                            for k2, v in val.items():
                                                if isinstance(v, str):
                                                    st.code(v)
                                        elif isinstance(val, list):
                                            for v in val:
                                                st.code(str(v))
                                        else:
                                            st.code(str(val))
                            if commands:
                                st.markdown("#### Commands to Run")
                                for section in ["backend", "frontend", "tests", "docs", "deploy"]:
                                    cmds = commands.get(section)
                                    if cmds:
                                        st.markdown(f"**{section.capitalize()}**")
                                        cmd_text = "\n".join(cmds)
                                        st.code(cmd_text)
                                        st.button("Copy", key=f"copy_{section}", on_click=lambda t=cmd_text: st.session_state.update({f"_copied_{section}": t}))
                                        if st.session_state.get(f"_copied_{section}"):
                                            st.caption("Copied to clipboard — select and Ctrl+C")
                        else:
                            st.info("No report available yet. Try again once the build completes.")
                    except Exception as e:
                        st.info(f"Report error: {e}")
            else:
                st.info("No builds yet. Use the Build tab or POST /sdlc/build to start one.")
        else:
            st.info("SDLC status unavailable.")
    except Exception as e:
        st.info(f"Status error: {e}")

# ----------------
# Recent Logs (optional)
# ----------------
with st.expander("Recent Logs (optional)"):
    try:
        resp = requests.get(f"{backend_url}/logs", timeout=30)
        if resp.ok:
            logs = resp.json().get("items", [])
            st.table([{k: v for k, v in item.items() if k in ("id", "stage", "provider", "model", "success", "created_at", "message")} for item in logs])
        else:
            st.info("No logs available.")
    except Exception:
        st.info("Logs unavailable.")
