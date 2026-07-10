"""
launch.py — DRVE Quantum Simulation launcher.

Starts the aiohttp server on localhost:8765 and opens the browser.
Package with:  pyinstaller --onefile --noconsole --add-data "static;static" launch.py -n DRVE
"""
import sys
import os
import threading
import time
import webbrowser

HOST = "127.0.0.1"
PORT = 8765
URL  = f"http://{HOST}:{PORT}/"


def check_deps():
    missing = []
    for pkg in ["aiohttp", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[DRVE] Missing packages: {', '.join(missing)}")
        print(f"[DRVE] Install with:  pip install {' '.join(missing)}")
        input("Press Enter to exit...")
        sys.exit(1)


def open_browser():
    time.sleep(1.2)  # give server a moment to bind
    webbrowser.open(URL)


if __name__ == "__main__":
    check_deps()

    print(f"[DRVE] Starting simulation server at {URL}")
    print(f"[DRVE] Close this window to stop.")

    threading.Thread(target=open_browser, daemon=True).start()

    from server import run
    run(host=HOST, port=PORT)
