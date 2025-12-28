# Autonomous SDLC Builder Agent

# Autonomous SDLC Builder Agent

**v3.0.0 - Autonomous SDLC Orchestrator**

An intelligent, autonomous software development agent that builds full-stack web applications (FastAPI + React) from natural language prompts. It features a robust, self-healing architecture with persistent session management, intelligent AI provider routing, and a professional workspace UI.

## 🚀 Key Features

### 🧠 Intelligent Core
*   **Multi-Provider Fallback**: Automatically retries generation with different AI models (OpenAI, Gemini, Mistral, Groq, HF) if the primary provider fails.
*   **Full 6-Phase SDLC**: Executes Planning, Design, Backend, Frontend, Testing (Real Execution), and Deployment Preparation.
*   **State-Aware Resumability**: Sketches `status.json` checkpoints. If a phase fails, fixing the error and continuing resumes exactly where it left off.

### 🛡️ Robust Architecture
*   **"One Job = One Session"**: Strict session binding ensures that a chat session never loses contact with its underlying build job.
*   **Disk-Based Persistence**: The `runs/` directory is the single source of truth. The server scans this directory on startup, meaning **zero data loss** if the backend restarts.
*   **Real Artifacts**: Generates working `pytest` suites and `Dockerfile`/`docker-compose.yml` for immediate deployment.

### 💻 Professional Workspace
*   **Chat Interface**: A ChatGPT-like experience that supports follow-up prompts.
*   **Live Build Context**: Real-time status indicators, **Live Terminal Output**, and Provider Usage Badges.
*   **Visual Artifacts**: Renders Mermaid architecture diagrams directly in the UI.

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   An API Key for at least one provider (OpenAI, Gemini, etc.) set in `.env`.

### 1. Backend Setup (FastAPI)
```bash
cd backend
# Create virtual environment (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup (React + Vite)
```bash
cd frontend
# Install dependencies
npm install

# Run the Development Server
npm run dev
```

### 3. Access the Agent
Open your browser to: **http://localhost:5173**

---

## 📖 Usage Guide

### Starting a New Build
1.  Click **"New Project"** in the sidebar.
2.  Type your request: *"Build a personal finance tracker with a dashboard."*
3.  The Agent will:
    *   Create a persistent Session ID.
    *   Plan the architecture.
    *   Generate a `FastAPI` backend (saved to `runs/<timestamp-slug>/backend`).
    *   Generate a `React` frontend (saved to `runs/<timestamp-slug>/frontend`).
    *   Update the status live in the UI.

### Continuing a Conversation
The agent supports context-aware follow-ups.
*   *User*: "The text is too small."
*   *Agent*: Triggers a continuation build, modifying the frontend code to increase font size.

### Running Generated Apps
Each generated project is a standalone codebase located in the `runs/` folder.
To run a specific project:
```bash
cd runs/20251225-123456-my-app
# Run Backend
python -m uvicorn backend.main:app --reload
# Run Frontend
cd frontend && npm install && npm run dev
```

---

## 🔧 Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **"Job not found"** | URL `job_id` mismatch or server cache issue. | **Fixed in v2.0**: The server now checks the disk. Refresh the page. |
| **"Failed to fetch"** | Backend server is not running. | Ensure `uvicorn` is active on port 8000. |
| **Input Disabled** | Agent thinks a build is in progress. | The UI unlocks automatically when the backend reports `idle` or `completed`. |
| **Partial Failure** | One AI provider failed. | Type "Retry generation" in the chat to trigger a new attempt. |

---

## 📂 Project Structure

```
sdlc_agent_cursor/
├── backend/
│   ├── main.py              # Core API & Persistence Logic
│   ├── simple_builder.py    # AI Orchard & Fallback Logic
│   └── services/            # Provider Adapters
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx  # Main Interaction UI
│   │   │   └── Sidebar.jsx        # History Management
│   │   └── App.jsx          # Router & Session Manager
│   └── package.json
└── runs/                    # ALL GENERATED USER PROJECTS LIVE HERE
```
