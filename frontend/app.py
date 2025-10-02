import streamlit as st
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Autonomous SDLC Agent", page_icon="🤖", layout="centered")

st.title("Autonomous SDLC Agent")
st.write("Give a natural language task to build, or try the demos below.")

def get_backend_default():
	try:
		return st.secrets["backend_url"]
	except Exception:
		return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

if "backend_url" not in st.session_state:
	st.session_state.backend_url = get_backend_default()

with st.sidebar:
	st.subheader("Settings")
	st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
	backend_url = st.session_state.backend_url
	status = "unknown"
	try:
		r = requests.get(f"{backend_url}/health", timeout=5)
		status = "online" if r.ok else "unreachable"
	except Exception:
		status = "offline"
	st.caption(f"API status: {status}")

tab_build, tab_predict, tab_text = st.tabs(["Build", "Predict (demo)", "Text Task"]) 


with tab_build:
	prompt = st.text_area("Describe what to build", placeholder="Build a full-stack app that ...")
	if st.button("Build Project", disabled=(status != "online")):
		if not prompt.strip():
			st.warning("Enter a prompt.")
		else:
			try:
				resp = requests.post(f"{backend_url}/build", json={"prompt": prompt}, timeout=120)
				if resp.ok:
					st.success("Build orchestrated. Run report:")
					st.json(resp.json())
				else:
					st.error(resp.text)
			except Exception as e:
				st.error("Cannot reach backend. Check the Backend URL in Settings and ensure the API is running.")

with tab_predict:
	image = st.file_uploader("Image", type=["jpg", "jpeg", "png"]) 
	notes = st.text_input("Notes (optional)")
	if st.button("Classify", disabled=(status != "online")):
		if image is None:
			st.warning("Please upload an image.")
		else:
			files = {"file": (image.name, image.read(), image.type)}
			data = {"notes": notes}
			try:
				resp = requests.post(f"{backend_url}/predict", files=files, data=data, timeout=60)
				if resp.ok:
					out = resp.json()
					st.success(f"Label: {out['label']} (conf {out['confidence']:.2f})")
					st.json(out["run_report"]) 
				else:
					st.error(f"Error: {resp.text}")
			except Exception:
				st.error("Cannot reach backend.")

with tab_text:
	prompt = st.text_area("Enter a text/code task prompt")
	if st.button("Run Text Task", disabled=(status != "online")):
		if not prompt.strip():
			st.warning("Enter a prompt.")
		else:
			try:
				resp = requests.post(f"{backend_url}/task", json={"prompt": prompt}, timeout=60)
				if resp.ok:
					st.json(resp.json())
				else:
					st.error(f"Error: {resp.text}")
			except Exception:
				st.error("Cannot reach backend.")

st.header("Recent Logs")
try:
	resp = requests.get(f"{backend_url}/logs", timeout=30)
	if resp.ok:
		logs = resp.json().get("items", [])
		st.table([{k: v for k, v in item.items() if k in ("id", "stage", "provider", "model", "success", "created_at", "message")} for item in logs])
	else:
		st.info("No logs available.")
except Exception as e:
	st.info("Logs unavailable.")


