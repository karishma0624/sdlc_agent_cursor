
import os
import sys
from dotenv import load_dotenv

# Explicitly load from possible locations
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("backend/.env"):
    load_dotenv("backend/.env")

try:
    from backend.services.adapters import call_gemini, call_v0
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "backend"))
    from services.adapters import call_gemini, call_v0

print("Checking Keys...")
print(f"GEMINI_API_KEY: {'Found' if os.getenv('GEMINI_API_KEY') else 'Missing'}")
print(f"V0_API_KEY: {'Found' if os.getenv('V0_API_KEY') else 'Missing'}")

print("\n--- Testing Gemini ---")
try:
    # prompt for a tiny file
    res = call_gemini("Generate a simple json with one file 'test.txt' containing 'hello'")
    print("Gemini Success:", res.keys())
except Exception as e:
    print("Gemini Failed:", e)

print("\n--- Testing v0 ---")
try:
    res = call_v0("Generate a simple json with one file 'test.txt' containing 'hello'")
    print("v0 Success:", res.keys())
except Exception as e:
    print("v0 Failed:", e)
