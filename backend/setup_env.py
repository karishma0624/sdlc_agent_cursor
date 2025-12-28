import os

def create_env_file():
    env_content = """# Database
DATABASE_URL=sqlite:///./app.db

# JWT
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS (comma-separated origins)
FRONTEND_URL=http://localhost:5173
"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"Created {env_path} file with default settings.")
        print("Please review and update the settings as needed.")
    else:
        print(f"{env_path} already exists. No changes made.")

if __name__ == "__main__":
    create_env_file()
