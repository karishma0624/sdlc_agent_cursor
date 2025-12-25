import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code} - {response.text}")

def test_register():
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }
    response = requests.post(
        f"{BASE_URL}/register",
        json=user_data
    )
    print(f"Register user: {response.status_code} - {response.text}")

def test_login():
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    response = requests.post(
        f"{BASE_URL}/token",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if response.status_code == 200:
        token = response.json()["access_token"]
        print(f"Login successful. Token: {token[:20]}...")
        return token
    else:
        print(f"Login failed: {response.status_code} - {response.text}")
        return None

def test_protected_route(token):
    if not token:
        print("No token available. Please login first.")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{BASE_URL}/users/me",
        headers=headers
    )
    print(f"Protected route: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("Testing API endpoints...\n")
    
    # Test health check
    test_health()
    
    # Test registration
    test_register()
    
    # Test login
    token = test_login()
    
    # Test protected route
    if token:
        test_protected_route(token)
    
    print("\nTesting complete!")
