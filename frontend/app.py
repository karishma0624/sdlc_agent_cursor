# frontend/app.py
import streamlit as st
import requests
import os
from typing import Any

st.set_page_config(page_title="Autonomous SDLC Agent", page_icon="🤖", layout="centered")

st.title("Autonomous SDLC Agent")
st.write("Give a natural language task to build, or try the demos below.")

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

tab_agent, tab_build, tab_predict, tab_text = st.tabs(
    ["Agent", "Build", "Predict (demo)", "Text Task"]
)

# ----------------
# Agent tab
# ----------------
with tab_agent:
    st.subheader("Agent — Code View by default")
    prompt = st.text_area("Task prompt", placeholder="Describe what to build or generate", key="agent_prompt")

    colA, colB, colC, colD = st.columns(4)

    # Dispatch
    with colA:
        if st.button("Dispatch", disabled=(status != "online")):
            try:
                resp = requests.post(f"{backend_url}/dispatch", json={"prompt": prompt, "free_only": True}, timeout=120)
                if resp.ok:
                    out = resp.json()
                    st.session_state.agent_result = out.get("result")
                    st.session_state.agent_kind = out.get("kind")
                    st.success(f"Dispatched: kind={out.get('kind')} tool={out.get('tool')}")
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(f"Cannot reach backend. {e}")

    # Preview / materialize
    with colB:
        preview_enabled = st.session_state.agent_kind in ("FRONTEND_ONLY", "BUILD_FULLSTACK")
        if st.button("Preview (materialize)", disabled=(status != "online" or not preview_enabled)):
            try:
                resp = requests.post(f"{backend_url}/materialize", json={"prompt": prompt, "free_only": True}, timeout=180)
                if resp.ok:
                    mat = resp.json()
                    st.info("Materialized at: " + mat.get("outdir", ""))
                    st.code("\n".join(mat.get("commands", [])))
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(f"Cannot reach backend. {e}")

    # Run diagnostics
    with colC:
        if st.button("Run Tests (diagnostics)", disabled=(status != "online")):
            try:
                resp = requests.post(f"{backend_url}/diagnostics", timeout=60)
                if resp.ok:
                    st.json(resp.json())
                else:
                    st.json({"error": resp.text})
            except Exception as e:
                st.error(f"Cannot reach backend. {e}")

    # Save to repo
    with colD:
        if st.button("Save to Repo (materialize)", disabled=(status != "online")):
            try:
                resp = requests.post(f"{backend_url}/materialize", json={"prompt": prompt, "free_only": True}, timeout=180)
                if resp.ok:
                    st.json(resp.json())
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(f"Cannot reach backend. {e}")

    st.markdown("---")
    st.caption("Code View")

    files = None
    if st.session_state.agent_result and isinstance(st.session_state.agent_result, dict):
        files = st.session_state.agent_result.get("files")
    if files:
        paths = sorted(files.keys())
        choice = st.selectbox("File", paths, key="agent_file_choice")
        st.code(files.get(choice, ""))
    else:
        text = None
        if st.session_state.agent_result and isinstance(st.session_state.agent_result, dict):
            text = st.session_state.agent_result.get("text")
        if text:
            st.code(text)
        else:
            st.info("Dispatch to see code or files here.")

# ----------------
# Editable Requirements / Spec (DB-backed)
# ----------------
st.markdown("---")
st.caption("Editable Requirements / Spec (DB-backed)")

col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    # define text_input with key "req_title" (syncs to st.session_state.req_title)
    new_title = st.text_input("Title", value=st.session_state.get("req_title", ""), key="req_title")

# Create button (inline is fine — we only read session_state here)
with col_r2:
    if st.button("Create", disabled=(status != "online")):
        try:
            payload = {"title": st.session_state.get("req_title", "Untitled"),
                       "content": {"text": st.session_state.get("req_text", "")}}
            r = requests.post(f"{backend_url}/requirements", json=payload, timeout=60)
            if r.ok:
                st.success(f"Created requirement id={r.json().get('id')}")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(f"Cannot reach backend. {e}")

# List requirements
with col_r3:
    if st.button("List", disabled=(status != "online")):
        try:
            r = requests.get(f"{backend_url}/requirements", timeout=60)
            if r.ok:
                items = r.json().get("items", [])
                st.session_state.req_list = items
                st.table([{k: v for k, v in it.items() if k in ("id", "title", "created_at")} for it in items])
            else:
                st.error(r.text)
        except Exception as e:
            st.error(f"Cannot reach backend. {e}")

# ID input
with col_r4:
    selected_id = st.number_input("ID", min_value=0, value=st.session_state.get("req_id", 0), step=1, key="req_id")

# Content text area (widget uses key "req_text" and syncs with session_state)
req_text = st.text_area("Content", value=st.session_state.get("req_text", ""), height=200, key="req_text")

# --- helper callbacks for safe session_state mutation (use on_click) ---
def _load_requirement() -> None:
    try:
        rid = int(st.session_state.get("req_id", 0))
        if rid == 0:
            st.session_state._req_error = "Please enter a non-zero ID."
            return
        r = requests.get(f"{st.session_state.get('backend_url')}/requirements/{rid}", timeout=60)
        if r.ok:
            data = r.json()
            st.session_state.req_title = data.get("title", "")
            st.session_state.req_text = (data.get("content", {}) or {}).get("text", "")
            st.session_state._req_msg = f"Loaded requirement id={rid}"
        else:
            st.session_state._req_error = r.text
    except Exception as e:
        st.session_state._req_error = str(e)

def _update_requirement() -> None:
    try:
        rid = int(st.session_state.get("req_id", 0))
        if rid == 0:
            st.session_state._req_error = "Please enter a non-zero ID."
            return
        payload = {"title": st.session_state.get("req_title", "Untitled"),
                   "content": {"text": st.session_state.get("req_text", "")}}
        r = requests.put(f"{st.session_state.get('backend_url')}/requirements/{rid}", json=payload, timeout=60)
        if r.ok:
            st.session_state._req_msg = f"Updated requirement id={rid}"
            st.session_state._req_update_resp = r.json()
        else:
            st.session_state._req_error = r.text
    except Exception as e:
        st.session_state._req_error = str(e)

col_u1, col_u2 = st.columns(2)
with col_u1:
    st.button("Load", on_click=_load_requirement, disabled=(status != "online"))
with col_u2:
    st.button("Update", on_click=_update_requirement, disabled=(status != "online"))

# show any messages from callbacks
if st.session_state.get("_req_msg"):
    st.success(st.session_state.get("_req_msg"))
    # clear it so subsequent runs won't keep showing forever
    st.session_state.pop("_req_msg", None)
if st.session_state.get("_req_error"):
    st.error(st.session_state.get("_req_error"))
    st.session_state.pop("_req_error", None)
if st.session_state.get("_req_update_resp"):
    st.json(st.session_state.pop("_req_update_resp", None))

# ----------------
# Build tab
# ----------------
with tab_build:
    build_prompt = st.text_area("Describe what to build", placeholder="Build a full-stack app that ...")
    if st.button("Build Project", disabled=(status != "online")):
        if not build_prompt.strip():
            st.warning("Enter a prompt.")
        else:
            try:
                resp = requests.post(f"{backend_url}/build", json={"prompt": build_prompt}, timeout=120)
                if resp.ok:
                    st.success("Build orchestrated. Run report:")
                    st.json(resp.json())
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(f"Cannot reach backend. Check the Backend URL in Settings and ensure the API is running. {e}")

# ----------------
# Predict tab (image demo)
# ----------------
with tab_predict:
    image = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    notes = st.text_input("Notes (optional)")
    if st.button("Classify", disabled=(status != "online")):
        if image is None:
            st.warning("Please upload an image.")
        else:
            try:
                files = {"file": (image.name, image.read(), image.type)}
                data = {"notes": notes}
                resp = requests.post(f"{backend_url}/predict", files=files, data=data, timeout=60)
                if resp.ok:
                    out = resp.json()
                    st.success(f"Label: {out.get('label')} (conf {out.get('confidence', 0):.2f})")
                    st.json(out.get("run_report"))
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Cannot reach backend. {e}")

# ----------------
# Text Task tab
# ----------------
with tab_text:
    text_prompt = st.text_area("Enter a text/code task prompt")
    if st.button("Run Text Task", disabled=(status != "online")):
        if not text_prompt.strip():
            st.warning("Enter a prompt.")
        else:
            try:
                resp = requests.post(f"{backend_url}/task", json={"prompt": text_prompt}, timeout=60)
                if resp.ok:
                    st.json(resp.json())
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Cannot reach backend. {e}")

# ----------------
# Recent Logs
# ----------------
st.header("Recent Logs")
try:
    resp = requests.get(f"{backend_url}/logs", timeout=30)
    if resp.ok:
        logs = resp.json().get("items", [])
        st.table([{k: v for k, v in item.items() if k in ("id", "stage", "provider", "model", "success", "created_at", "message")} for item in logs])
    else:
        st.info("No logs available.")
except Exception:
    st.info("Logs unavailable.")
