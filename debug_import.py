import sys
import os
import traceback

print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    print("Attempting: import backend.simple_builder")
    import backend.simple_builder
    print("Success: backend.simple_builder")
except ImportError:
    print("Failed: backend.simple_builder")
    traceback.print_exc()

print("-" * 20)

try:
    print("Attempting: from simple_builder import SDLCBuilder")
    from simple_builder import SDLCBuilder
    print("Success: simple_builder")
except ImportError:
    print("Failed: from simple_builder")
    traceback.print_exc()
