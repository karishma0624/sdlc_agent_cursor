import os
import json
import shutil

RUNS_DIR = "runs"

def cleanup_failed():
    if not os.path.exists(RUNS_DIR):
        print("No runs directory found.")
        return

    deleted_count = 0
    for job_id in os.listdir(RUNS_DIR):
        job_path = os.path.join(RUNS_DIR, job_id)
        if not os.path.isdir(job_path):
            continue

        status_path = os.path.join(job_path, "status.json")
        should_delete = False
        
        if os.path.exists(status_path):
            try:
                with open(status_path, "r") as f:
                    data = json.load(f)
                    if data.get("status") == "failed":
                        should_delete = True
            except:
                # If JSON is corrupt, usually safe to delete or ignore. 
                # Let's be conservative and delete corrupt ones if they are interfering.
                # But safer to only delete explicitly failed ones for now unless user asks for more.
                pass
        else:
            # No status.json? It might be a broken run. 
            # Check if it's empty or just has a timestamp folder name (legacy).
            # The current system uses UUIDs.
            # Let's stick to explicit "failed" status for safety first.
            pass

        if should_delete:
            print(f"Deleting failed run: {job_id}")
            try:
                shutil.rmtree(job_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {job_id}: {e}")

    print(f"Cleanup complete. Removed {deleted_count} failed sessions.")

if __name__ == "__main__":
    cleanup_failed()
