"""
Unified Server Launcher for Fintech Credit Risk Scoring.
Starts both the FastAPI REST backend and the Streamlit dashboard.
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def run_fastapi(host: str = "127.0.0.1", port: int = 8000):
    print(f"[*] Starting FastAPI service at http://{host}:{port} ...")
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", host, "--port", str(port), "--reload"]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))

def run_streamlit(port: int = 8501):
    print(f"[*] Starting Streamlit Dashboard at http://localhost:{port} ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", str(port), "--server.headless", "true"]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))

def main():
    parser = argparse.ArgumentParser(description="Fintech Credit Risk Scoring Service Launcher")
    parser.add_argument("--mode", choices=["all", "api", "dashboard"], default="all", help="Service to run")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI port")
    parser.add_argument("--ui-port", type=int, default=8501, help="Streamlit port")
    args = parser.parse_args()

    procs = []
    try:
        if args.mode in ["all", "api"]:
            p_api = run_fastapi(port=args.api_port)
            procs.append(p_api)
            time.sleep(2)

        if args.mode in ["all", "dashboard"]:
            p_ui = run_streamlit(port=args.ui_port)
            procs.append(p_ui)

        print("\n" + "=" * 70)
        print("🎉 CREDIT RISK SCORING PLATFORM IS LIVE!")
        if args.mode in ["all", "dashboard"]:
            print(f"  👉 Streamlit Dashboard: http://localhost:{args.ui_port}")
        if args.mode in ["all", "api"]:
            print(f"  👉 FastAPI Documentation (Swagger): http://localhost:{args.api_port}/docs")
            print(f"  👉 FastAPI Alternative (ReDoc):     http://localhost:{args.api_port}/redoc")
        print("=" * 70)
        print("Press Ctrl+C to terminate services gracefully.\n")

        # Keep alive
        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        print("\nShutting down services...")
        for p in procs:
            p.terminate()
            p.wait()
        print("Services stopped.")

if __name__ == "__main__":
    main()
