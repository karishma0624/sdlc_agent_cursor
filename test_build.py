#!/usr/bin/env python3
"""Test script for the dynamic build system"""

import sys
import os
sys.path.append('backend')

from services.sdlc_builder import SDLCBuilder
import json

def test_build_system():
    """Test the build system with a sample prompt"""
    print("Testing dynamic build system...")
    
    # Test the build system
    builder = SDLCBuilder(fast_mode=True)  # Use fast mode for testing
    result = builder.build('Build a personal finance tracker with charts and user login')
    
    print('Build completed successfully!')
    print(f'Run directory: {result["run_dir"]}')
    print(f'Summary: {result["summary"]}')
    print(f'Artifacts generated: {len(result["artifacts"])}')
    
    # Check if key files were created
    run_dir = result['run_dir']
    key_files = [
        'backend/main.py',
        'backend/requirements.txt', 
        'backend/tests/test_health.py',
        'frontend/package.json',
        'frontend/src/App.jsx',
        'docker-compose.yml',
        'README.md',
        'status.json'
    ]
    
    print('\nChecking generated files:')
    for file_path in key_files:
        full_path = os.path.join(run_dir, file_path)
        exists = os.path.exists(full_path)
        print(f'  {file_path}: {"✓" if exists else "✗"}')
    
    # Check status.json content
    status_path = os.path.join(run_dir, 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            status = json.load(f)
        print(f'\nStatus.json content:')
        print(f'  Build info: {status["build_info"]["status"]}')
        print(f'  Files generated: {status["build_info"]["files_generated"]}')
        print(f'  Backend status: {status["modules"]["backend"]["status"]}')
        print(f'  Frontend status: {status["modules"]["frontend"]["status"]}')
    
    # Check backend main.py content
    main_path = os.path.join(run_dir, 'backend/main.py')
    if os.path.exists(main_path):
        with open(main_path, 'r') as f:
            main_content = f.read()
        print(f'\nBackend main.py contains:')
        print(f'  CORS middleware: {"✓" if "CORSMiddleware" in main_content else "✗"}')
        print(f'  Health endpoint: {"✓" if "/health" in main_content else "✗"}')
        print(f'  Status endpoint: {"✓" if "/status" in main_content else "✗"}')
        print(f'  Providers endpoint: {"✓" if "/providers" in main_content else "✗"}')
        print(f'  Structured logging: {"✓" if "JSON" in main_content and "logging" in main_content else "✗"}')
    
    # Check frontend App.jsx content
    app_path = os.path.join(run_dir, 'frontend/src/App.jsx')
    if os.path.exists(app_path):
        with open(app_path, 'r') as f:
            app_content = f.read()
        print(f'\nFrontend App.jsx contains:')
        print(f'  API integration: {"✓" if "fetch" in app_content else "✗"}')
        print(f'  Loading states: {"✓" if "loading" in app_content else "✗"}')
        print(f'  Error handling: {"✓" if "error" in app_content else "✗"}')
        print(f'  Dynamic prompt: {"✓" if "personal finance tracker" in app_content else "✗"}')
    
    # Check docker-compose.yml
    compose_path = os.path.join(run_dir, 'docker-compose.yml')
    if os.path.exists(compose_path):
        with open(compose_path, 'r') as f:
            compose_content = f.read()
        print(f'\nDocker Compose contains:')
        print(f'  Backend service: {"✓" if "backend:" in compose_content else "✗"}')
        print(f'  Frontend service: {"✓" if "frontend:" in compose_content else "✗"}')
        print(f'  Health checks: {"✓" if "healthcheck" in compose_content else "✗"}')
        print(f'  Service dependencies: {"✓" if "depends_on" in compose_content else "✗"}')
    
    print(f'\n✅ Build system test completed!')
    print(f'Generated application is ready in: {run_dir}')
    
    return result

if __name__ == "__main__":
    test_build_system()
