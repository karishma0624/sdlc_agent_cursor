
# How to Run SDLC Agent

## Option 1: Automatic (New Windows)
Run the helper script which opens new windows for you:
```powershell
.\start_dev.ps1
```

## Option 2: Manual (Integrated Terminal)
If you prefer running everything in your VS Code terminal (e.g., using Split Terminal):

### Terminal 1: Backend (API)
```powershell
cd backend
# Make sure your virtual env is active if you have one, or just:
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*Note: You MUST be inside the `backend` folder for this to work!*

### Terminal 2: Frontend (UI)
```powershell
cd frontend
npm run dev
```

## Why did I see "Could not import module main"?
You were running `python -m uvicorn main:app` from the **root** folder. The `main.py` file is inside the `backend` folder. By changing directory (`cd backend`) first, Python can find the file correctly.
