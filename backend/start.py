import uvicorn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    print("Starting FastAPI Backend Server on http://localhost:8000 ...", flush=True)
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
