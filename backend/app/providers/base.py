"""
Base provider interface for SDLC Builder Agent
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import shutil
import subprocess
import asyncio

logger = logging.getLogger(__name__)

class Provider(ABC):
    """Base class for all AI providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    @abstractmethod
    async def generate_application(self, prompt: str, output_dir: str) -> Dict[str, Any]:
        """
        Generate an application based on the given prompt.
        
        Args:
            prompt: The natural language description of the application
            output_dir: Directory where the generated application should be saved
            
        Returns:
            Dict containing metadata about the generated application
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the provider."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this provider."""
        pass


class ProviderFactory:
    """Factory for creating provider instances."""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class):
        """Register a provider class."""
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_provider(cls, name: str, **kwargs) -> Provider:
        """Get a provider instance by name."""
        name = name.lower()
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        return cls._providers[name](**kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers."""
        return list(cls._providers.keys())


class MockProvider(Provider):
    """Mock provider for testing and development."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.name = "mock"
    
    async def generate_application(self, prompt: str, output_dir: str) -> Dict[str, Any]:
        """Generate a mock application."""
        output_path = Path(output_dir)
        
        # Create a simple FastAPI backend
        backend_dir = output_path / "backend"
        backend_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple React frontend
        frontend_dir = output_path / "frontend"
        frontend_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple README
        with open(output_path / "README.md", "w") as f:
            f.write(f"# Generated Application\n\n")
            f.write(f"This application was generated from the prompt:\n\n")
            f.write(f"> {prompt}\n\n")
            f.write("## Getting Started\n\n")
            f.write("### Backend\n\n")
            f.write("```bash\n")
            f.write("cd backend\n")
            f.write("pip install -r requirements.txt\n")
            f.write("uvicorn main:app --reload\n")
            f.write("```\n\n")
            f.write("### Frontend\n\n")
            f.write("```bash\n")
            f.write("cd frontend\n")
            f.write("npm install\n")
            f.write("npm run dev\n")
            f.write("```\n")
        
        # Create backend files
        (backend_dir / "main.py").write_text("""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello from your generated API!"}

@app.get("/api/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
""")
        
        (backend_dir / "requirements.txt").write_text("""
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
""")
        
        # Create frontend files
        (frontend_dir / "package.json").write_text(json.dumps({
            "name": "generated-app",
            "private": True,
            "version": "0.1.0",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "axios": "^1.3.4"
            },
            "devDependencies": {
                "@vitejs/plugin-react": "^3.1.0",
                "vite": "^4.2.0"
            }
        }, indent=2))
        
        (frontend_dir / "index.html").write_text("""
<!DOCTYPE html>
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
""")
        
        src_dir = frontend_dir / "src"
        src_dir.mkdir(exist_ok=True)
        
        (src_dir / "main.jsx").write_text("""
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
""")
        
        (src_dir / "App.jsx").write_text("""
import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok')
        }
        return response.json()
      })
      .then(data => {
        setData(data)
        setLoading(false)
      })
      .catch(error => {
        setError(error.toString())
        setLoading(false)
      })
  }, [])

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>

  return (
    <div className="app">
      <header className="app-header">
        <h1>Welcome to Your Generated App</h1>
        <p>Message from the backend: {data?.message || 'No message received'}</p>
      </header>
      <main>
        <section>
          <h2>Getting Started</h2>
          <p>This is a basic template generated from your prompt. You can now start customizing it!</p>
        </section>
      </main>
      <footer>
        <p>Generated with SDLC Builder Agent</p>
      </footer>
    </div>
  )
}

export default App
""")
        
        (src_dir / "index.css").writeText("""
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  line-height: 1.6;
  color: #333;
}

.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.app-header {
  text-align: center;
  margin-bottom: 3rem;
  padding: 2rem 0;
  border-bottom: 1px solid #eee;
}

h1 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: #2c3e50;
}

h2 {
  font-size: 1.8rem;
  margin: 2rem 0 1rem;
  color: #34495e;
}

p {
  margin-bottom: 1rem;
}

footer {
  margin-top: 4rem;
  padding: 2rem 0;
  text-align: center;
  border-top: 1px solid #eee;
  color: #7f8c8d;
  font-size: 0.9rem;
}
""")
        
        (src_dir / "App.css").writeText("""
/* Add your component styles here */
""")
        
        # Create a simple Vite config
        (frontend_dir / "vite.config.js").write_text("""
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
""")
        
        # Create logs directory
        logs_dir = output_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Log the generation
        with open(logs_dir / "generation.log", "w") as f:
            f.write(f"Application generated at: {output_dir}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("Status: Success\n")
        
        return {
            "status": "success",
            "message": "Application generated successfully",
            "artifacts": [
                {"type": "frontend", "path": str(frontend_dir.relative_to(output_path))},
                {"type": "backend", "path": str(backend_dir.relative_to(output_path))},
                {"type": "logs", "path": str(logs_dir.relative_to(output_path))}
            ]
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "available",
            "name": "mock",
            "capabilities": ["web", "api", "frontend", "backend"],
            "limitations": ["demo_only", "no_ai"]
        }
    
    def get_capabilities(self) -> List[str]:
        return ["web", "api", "frontend", "backend"]


# Register the mock provider
ProviderFactory.register_provider("mock", MockProvider)
