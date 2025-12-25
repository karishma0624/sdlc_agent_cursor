# Autonomous SDLC Agent - Backend

This is the backend service for the Autonomous SDLC Agent, built with FastAPI and SQLAlchemy.

## Features

- JWT Authentication
- User management
- Secure password hashing with Argon2
- CORS enabled for frontend
- SQLite database (can be configured to use PostgreSQL)
- Health check endpoint

## Prerequisites

- Python 3.8+
- pip
- SQLite (or PostgreSQL if using a different database)

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example environment file and update it with your settings:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration.

## Running the Application

```bash
# Start the development server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

```bash
# Run tests
pytest
```

## Production Deployment

For production, consider using:
- Gunicorn with Uvicorn workers
- PostgreSQL or MySQL
- Environment variables for sensitive data
- HTTPS with a reverse proxy (Nginx, Traefik, etc.)

## License

MIT
