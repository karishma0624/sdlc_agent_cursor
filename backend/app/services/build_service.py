"""
Build Service - Handles the core build logic for SDLC Builder Agent
"""
import os
import shutil
import uuid
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from ..models.build import Build, BuildStatus, BuildArtifact
from ..providers.base import ProviderFactory

logger = logging.getLogger(__name__)

class BuildService:
    def __init__(self, base_dir: str = "runs"):
        """Initialize the build service with a base directory for storing builds."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.active_builds: Dict[str, asyncio.Task] = {}
        self.builds: Dict[str, Build] = {}
        
    async def create_build(self, prompt: str, provider: str = "openai") -> Build:
        """Create a new build and start processing it asynchronously."""
        build_id = str(uuid.uuid4())
        build_dir = self.base_dir / build_id
        build_dir.mkdir(exist_ok=True)
        
        build = Build(
            id=build_id,
            prompt=prompt,
            provider=provider,
            status=BuildStatus.QUEUED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            artifacts=[],
            logs=[{"timestamp": datetime.utcnow().isoformat(), "message": "Build created and queued"}]
        )
        
        # Save build metadata
        self.builds[build_id] = build
        self._save_build_metadata(build)
        
        # Start the build process in the background
        self.active_builds[build_id] = asyncio.create_task(self._process_build(build))
        
        return build
    
    async def get_build(self, build_id: str) -> Optional[Build]:
        """Retrieve a build by its ID."""
        if build_id in self.builds:
            return self.builds[build_id]
        
        # Try to load from disk if not in memory
        build_path = self.base_dir / build_id / "build.json"
        if build_path.exists():
            try:
                with open(build_path, 'r') as f:
                    data = json.load(f)
                    build = Build(**data)
                    self.builds[build_id] = build
                    return build
            except Exception as e:
                logger.error(f"Error loading build {build_id}: {e}")
        
        return None
    
    async def _process_build(self, build: Build) -> None:
        """Process a build asynchronously."""
        build_dir = self.base_dir / build.id
        
        try:
            # Update status to processing
            build.status = BuildStatus.PROCESSING
            build.updated_at = datetime.utcnow()
            build.logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Starting build processing"
            })
            self._save_build_metadata(build)
            
            # Get the appropriate provider
            provider = ProviderFactory.get_provider(build.provider)
            
            # Generate the application
            result = await provider.generate_application(
                prompt=build.prompt,
                output_dir=str(build_dir)
            )
            
            # Update build with results
            build.status = BuildStatus.COMPLETED
            build.updated_at = datetime.utcnow()
            build.artifacts = [
                BuildArtifact(
                    type="frontend",
                    path=f"{build.id}/frontend",
                    size=self._get_directory_size(build_dir / "frontend") if (build_dir / "frontend").exists() else 0
                ),
                BuildArtifact(
                    type="backend",
                    path=f"{build.id}/backend",
                    size=self._get_directory_size(build_dir / "backend") if (build_dir / "backend").exists() else 0
                ),
                BuildArtifact(
                    type="logs",
                    path=f"{build.id}/logs",
                    size=self._get_directory_size(build_dir / "logs") if (build_dir / "logs").exists() else 0
                )
            ]
            
            build.logs.extend([
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Build completed successfully with {len(build.artifacts)} artifacts"
                }
            ])
            
        except Exception as e:
            build.status = BuildStatus.FAILED
            build.updated_at = datetime.utcnow()
            build.logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Build failed: {str(e)}",
                "error": str(e)
            })
            logger.exception(f"Build {build.id} failed")
            
        finally:
            # Clean up
            self._save_build_metadata(build)
            if build.id in self.active_builds:
                del self.active_builds[build.id]
    
    def _save_build_metadata(self, build: Build) -> None:
        """Save build metadata to disk."""
        build_dir = self.base_dir / build.id
        build_dir.mkdir(exist_ok=True, parents=True)
        
        # Convert the build object to a dictionary
        build_dict = build.dict()
        build_dict["created_at"] = build.created_at.isoformat() if build.created_at else None
        build_dict["updated_at"] = build.updated_at.isoformat() if build.updated_at else None
        
        # Save to file
        with open(build_dir / "build.json", 'w') as f:
            json.dump(build_dict, f, indent=2, default=str)
    
    def _get_directory_size(self, path: Path) -> int:
        """Calculate the total size of a directory in bytes."""
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    
    async def list_builds(self, limit: int = 10, offset: int = 0) -> List[Build]:
        """List all builds with pagination."""
        # Load builds from disk if not already in memory
        for build_dir in self.base_dir.iterdir():
            if build_dir.is_dir() and build_dir.name not in self.builds:
                build_file = build_dir / "build.json"
                if build_file.exists():
                    try:
                        with open(build_file, 'r') as f:
                            data = json.load(f)
                            self.builds[build_dir.name] = Build(**data)
                    except Exception as e:
                        logger.error(f"Error loading build {build_dir.name}: {e}")
        
        # Return sorted by updated_at (newest first)
        return sorted(
            self.builds.values(),
            key=lambda x: x.updated_at or datetime.min,
            reverse=True
        )[offset:offset+limit]

# Singleton instance
build_service = BuildService()
