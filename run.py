#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_OUT = os.path.join(FRONTEND_DIR, "out")

print("\n" + "=" * 60)
print("  AI SIGN LANGUAGE TRANSLATION PLATFORM LAUNCHER")
print("=" * 60 + "\n")

# Mode selection: Default to "1" (Instant 1-Click Launch with 0 prompts)
mode = "1"
if "--dev" in sys.argv:
    mode = "2"

processes = []

try:
    if mode == "1":
        # Option 3: Single Server Mode
        if not os.path.exists(FRONTEND_OUT):
            print("\n[INFO] Compiling Next.js static UI build first...")
            subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, shell=True, check=True)
            
        print("\n[STARTING] Launching Single Server (Backend + UI) on http://localhost:8000 ...")
        backend_proc = subprocess.Popen(
            [sys.executable, "backend/start.py"],
            cwd=PROJECT_ROOT
        )
        processes.append(backend_proc)
        
        # Wait 3 seconds for server startup & open browser
        time.sleep(3)
        print("\n[SUCCESS] Server online! Opening http://localhost:8000 in your browser...")
        webbrowser.open("http://localhost:8000")
        
    else:
        # Option 1: Development Concurrent Mode
        print("\n[STARTING] Launching Backend Server on http://localhost:8000 ...")
        backend_proc = subprocess.Popen(
            [sys.executable, "backend/start.py"],
            cwd=PROJECT_ROOT
        )
        processes.append(backend_proc)
        
        print("\n[STARTING] Launching Next.js Dev Server on http://localhost:3000 ...")
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
            shell=True
        )
        processes.append(frontend_proc)
        
        # Wait 4 seconds & open browser
        time.sleep(4)
        print("\n[SUCCESS] All servers online! Opening http://localhost:3000 in your browser...")
        webbrowser.open("http://localhost:3000")

    print("\n" + "=" * 60)
    print("  APPLICATION IS RUNNING! Press CTRL+C in this window to stop.")
    print("=" * 60 + "\n")
    
    # Keep main process alive
    for proc in processes:
        proc.wait()

except KeyboardInterrupt:
    print("\n\n[STOPPING] Shutting down all application servers gracefully...")
    for proc in processes:
        try:
            proc.terminate()
        except Exception:
            pass
    print("[DONE] All servers stopped cleanly. Goodbye!")
