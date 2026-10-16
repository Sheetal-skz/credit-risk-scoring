"""
Unified Server Launcher for Fintech Credit Risk Scoring.
Starts both the FastAPI REST backend and the Streamlit dashboard,
automatically handling port conflicts and opening the browser.
"""

import sys
import time
import socket
import webbrowser
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def kill_process_on_port_windows(port: int):
    """Gracefully terminate any stale process holding a port on Windows."""
    try:
        cmd = f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"'
        subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
    except Exception:
        pass


def run_fastapi(host: str = "127.0.0.1", port: int = 8000):
    if is_port_in_use(port, host):
        print(f"[*] Port {port} is busy. Freeing port {port}...")
        kill_process_on_port_windows(port)
        time.sleep(1)

    print(f"[*] Launching FastAPI backend at http://{host}:{port} ...")
    cmd = [
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", host,
        "--port", str(port)
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))


def run_streamlit(port: int = 8501):
    if is_port_in_use(port):
        print(f"[*] Port {port} is busy. Freeing port {port}...")
        kill_process_on_port_windows(port)
        time.sleep(1)

    print(f"[*] Launching Streamlit Dashboard at http://localhost:{port} ...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false"
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Fintech Credit Risk Scoring Service Launcher")
    parser.add_argument("--mode", choices=["all", "api", "dashboard"], default="all", help="Service to run")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI port")
    parser.add_argument("--ui-port", type=int, default=8501, help="Streamlit port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    procs = []
    try:
        # 1. Start FastAPI
        if args.mode in ["all", "api"]:
            p_api = run_fastapi(port=args.api_port)
            procs.append(p_api)
            time.sleep(1.5)

        # 2. Start Streamlit
        if args.mode in ["all", "dashboard"]:
            p_ui = run_streamlit(port=args.ui_port)
            procs.append(p_ui)
            time.sleep(2.0)

        print("\n" + "=" * 75)
        print("🎉 CREDIT RISK SCORING & EXPLAINABLE AI PLATFORM IS LIVE!")
        print("=" * 75)
        if args.mode in ["all", "dashboard"]:
            print(f"  📊 Streamlit Web UI:   http://localhost:{args.ui_port}")
        if args.mode in ["all", "api"]:
            print(f"  ⚡ FastAPI Swagger UI: http://127.0.0.1:{args.api_port}/docs")
            print(f"  📖 FastAPI ReDoc:      http://127.0.0.1:{args.api_port}/redoc")
        print("=" * 75)
        print("  Press Ctrl+C in this terminal window anytime to shut down servers.\n")

        # Automatically open browser to the Streamlit UI
        if not args.no_browser and args.mode in ["all", "dashboard"]:
            try:
                webbrowser.open(f"http://localhost:{args.ui_port}")
            except Exception:
                pass

        # Keep alive
        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        print("\n[!] Shutting down services gracefully...")
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        print("[✓] All services stopped.")


if __name__ == "__main__":
    main()
