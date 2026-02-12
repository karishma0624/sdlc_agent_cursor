
import os
import json
from datetime import datetime

runs_dir = r"c:\Users\karis\Desktop\sdlc_agent_cursor\backend\runs"
latest_time = None
latest_run = None

for run_id in os.listdir(runs_dir):
    status_path = os.path.join(runs_dir, run_id, "status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                data = json.load(f)
                updated_str = data.get("last_updated")
                if updated_str:
                    updated = datetime.fromisoformat(updated_str)
                    if latest_time is None or updated > latest_time:
                        latest_time = updated
                        latest_run = run_id
        except Exception:
            continue

if latest_run:
    print(f"Latest Run: {latest_run}")
    status_path = os.path.join(runs_dir, latest_run, "status.json")
    with open(status_path, "r") as f:
        print(f.read())
else:
    print("No runs found.")
