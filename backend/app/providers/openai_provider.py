"""
OpenAI provider implementation for SDLC Builder Agent
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
import shutil
import subprocess
import asyncio
from ..models import BuildArtifact, BuildStatus
from .base import Provider

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class OpenAIProvider(Provider):
    """OpenAI provider for generating applications."""
    
    name = "OpenAI"
    requires_key = True
    supports_streaming = True
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-1106-preview"):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable.
            model: The model to use for generation.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate_application(self, prompt: str, output_dir: str) -> Dict[str, Any]:
        """Generate an application using OpenAI's API.
        
        Args:
            prompt: The natural language description of the application.
            output_dir: Directory where the generated application should be saved.
            
        Returns:
            Dict containing metadata about the generated application.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a system prompt for the AI
        system_prompt = """You are an expert full-stack developer. Your task is to generate a complete, 
        working web application based on the user's description. The application should be production-ready 
        and follow best practices for the chosen technologies.
        
        The application should include:
        1. A backend API (FastAPI/Node.js/Express)
        2. A frontend UI (React/Vue/Svelte)
        3. A database schema if needed
        4. Configuration files (Docker, package.json, requirements.txt, etc.)
        5. A README with setup instructions
        
        Generate the complete file structure with all necessary code. For each file, include the 
        content in a code block with the file path as a comment at the top.
        """
        
        # Generate the application using OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a web application with the following requirements:\n\n{prompt}"}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            
            # Process the response and extract files
            generated_text = response.choices[0].message.content
            files = self._extract_files_from_response(generated_text)
            
            # Save the files to disk
            artifacts = []
            for file_path, content in files.items():
                file_full_path = output_path / file_path
                file_full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Add to artifacts
                artifact_type = self._determine_artifact_type(file_path)
                artifacts.append(BuildArtifact(
                    type=artifact_type,
                    path=str(file_path),
                    size=len(content.encode('utf-8'))
                ))
            
            # Create a README if not already created
            readme_path = output_path / "README.md"
            if not readme_path.exists():
                with open(readme_path, 'w') as f:
                    f.write(f"# Generated Application\n\n")
                    f.write(f"This application was generated from the prompt:\n\n")
                    f.write(f"> {prompt}\n\n")
                    f.write("## Getting Started\n\n")
                    f.write("Follow the instructions below to run the application.\n\n")
                    
                    if (output_path / "backend").exists():
                        f.write("### Backend Setup\n\n")
                        f.write("```bash\n")
                        f.write(f"cd {output_dir}/backend\n")
                        f.write("pip install -r requirements.txt\n")
                        f.write("uvicorn main:app --reload\n")
                        f.write("```\n\n")
                    
                    if (output_path / "frontend").exists():
                        f.write("### Frontend Setup\n\n")
                        f.write("```bash\n")
                        f.write(f"cd {output_dir}/frontend\n")
                        f.write("npm install\n")
                        f.write("npm run dev\n")
                        f.write("```\n")
                
                artifacts.append(BuildArtifact(
                    type="documentation",
                    path="README.md",
                    size=os.path.getsize(readme_path)
                ))
            
            return {
                "status": "success",
                "message": "Application generated successfully",
                "artifacts": artifacts,
                "provider": "openai",
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating application with OpenAI: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate application: {str(e)}",
                "provider": "openai",
                "error": str(e)
            }
    
    def _extract_files_from_response(self, response_text: str) -> Dict[str, str]:
        """Extract files and their content from the AI's response."""
        files = {}
        current_file = None
        current_content = []
        
        for line in response_text.split('\n'):
            if line.startswith('```') and '```' in line[3:]:
                # End of a code block
                if current_file:
                    files[current_file] = '\n'.join(current_content)
                    current_file = None
                    current_content = []
            elif line.startswith('```') and current_file is None:
                # Start of a new code block with filename
                file_path = line[3:].strip()
                if file_path:  # If there's a file path
                    current_file = file_path
                    current_content = []
            elif current_file is not None:
                # Inside a code block
                current_content.append(line)
        
        # Add the last file if the response ended unexpectedly
        if current_file and current_content:
            files[current_file] = '\n'.join(current_content)
        
        return files
    
    def _determine_artifact_type(self, file_path: str) -> str:
        """Determine the type of artifact based on the file path."""
        file_path = file_path.lower()
        
        if file_path.startswith('frontend/'):
            return 'frontend'
        elif file_path.startswith('backend/'):
            return 'backend'
        elif file_path.endswith(('.md', '.txt', '.rst')):
            return 'documentation'
        elif file_path.endswith(('.yaml', '.yml', '.json', '.toml', '.ini', '.cfg')):
            return 'configuration'
        elif file_path.endswith(('.py')):
            return 'python'
        elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
            return 'javascript'
        elif file_path.endswith(('.css', '.scss', '.sass', '.less')):
            return 'stylesheet'
        elif file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico')):
            return 'image'
        else:
            return 'other'
    
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this provider."""
        return [
            "web_application",
            "api",
            "frontend",
            "backend",
            "documentation",
            "code_generation",
            "text_completion"
        ]

    async def check_health(self) -> Dict[str, Any]:
        """Check if the provider is healthy."""
        try:
            # Make a simple API call to check if the provider is working
            await self.client.models.list()
            return {
                "status": "healthy",
                "model": self.model,
                "provider": "openai"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "openai"
            }
