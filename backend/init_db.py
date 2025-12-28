import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import app modules
sys.path.append(str(Path(__file__).parent))

from app.database import SessionLocal, engine, Base
from app.models import User
from app.core.security import get_password_hash

def init_db():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create a default admin user if it doesn't exist
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            hashed_password = get_password_hash("admin123")
            admin_user = User(
                username="admin",
                email="admin@example.com",
                hashed_password=hashed_password,
                is_active=True,
                is_superuser=True
            )
            db.add(admin_user)
            db.commit()
            print("Created default admin user")
        else:
            print("Admin user already exists")
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialization complete")
