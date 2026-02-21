import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:8000/sdlc/preview', 
    data=json.dumps({"job_id":"b601d5017f6943ebbcb417360557597c"}).encode(),
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req) as response:
        print("Response:", response.read().decode())
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code, e.reason, e.read().decode())
except Exception as e:
    print("Error:", e)
