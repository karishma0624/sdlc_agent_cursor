
_TEMPLATE_PACKAGE_JSON = """{
  "name": "generated-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 5173",
    "start": "vite preview --host 0.0.0.0 --port 5173"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "lucide-react": "^0.300.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.18",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.3",
    "vite": "^5.0.0"
  }
}
"""

_TEMPLATE_TAILWIND_CONFIG = """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""

_TEMPLATE_POSTCSS = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""

_TEMPLATE_VITE = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { host: '0.0.0.0', port: 5173 }
})
"""

_TEMPLATE_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Generated App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
  </html>
"""

_TEMPLATE_MAIN_JSX = """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"""

_TEMPLATE_INDEX_CSS = """@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root { height: 100%; }
body { @apply bg-gray-50 text-gray-900; }
"""

_TEMPLATE_APP_JSX = """import { useState } from 'react'
import { Layout, Palette, Code, Sparkles } from 'lucide-react'

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center p-4 overflow-hidden">
      <div className="bg-white/10 backdrop-blur-lg border border-white/20 p-8 rounded-3xl shadow-2xl max-w-lg w-full text-center text-white relative z-10">
        <Sparkles className="w-16 h-16 mx-auto mb-6 text-yellow-300 animate-pulse" />
        <h1 className="text-4xl font-extrabold mb-4 drop-shadow-md tracking-tight">Beautiful UI Generated!</h1>
        <p className="text-lg text-white/90 mb-8 font-medium leading-relaxed">
          The AI providers successfully laid out your infrastructure. This is your high-performance React + Vite foundation.
        </p>
        
        <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/5 border border-white/10 p-4 shadow-inner rounded-2xl hover:bg-white/20 transition-all cursor-pointer">
                <Palette className="w-8 h-8 mb-2 mx-auto text-pink-300" />
                <h3 className="font-bold tracking-wide">Tailwind CSS</h3>
                <span className="text-xs text-white/70">Fully Configured & Ready</span>
            </div>
            <div className="bg-white/5 border border-white/10 p-4 shadow-inner rounded-2xl hover:bg-white/20 transition-all cursor-pointer">
                <Code className="w-8 h-8 mb-2 mx-auto text-indigo-300" />
                <h3 className="font-bold tracking-wide">React + Vite</h3>
                <span className="text-xs text-white/70">HMR Active</span>
            </div>
        </div>
      </div>
      
      {/* Decorative background shapes */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob"></div>
      <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob animation-delay-2000"></div>
    </div>
  )
}
"""

_TEMPLATE_BACKEND_REQUIREMENTS = """fastapi==0.109.2
uvicorn==0.27.1
sqlalchemy==2.0.27
pydantic==2.6.1
python-multipart==0.0.9
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.1
"""

_TEMPLATE_BACKEND_MAIN = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Generated API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
"""

_TEMPLATE_BACKEND_DATABASE = """from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
"""

