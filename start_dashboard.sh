#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LIMONS Anti-Drone Telemetry Defense System
# Automated 1-Click Startup Script for Linux / macOS / Raspberry Pi
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==============================================================================="
echo "           LIMONS ANTI-DRONE TELEMETRY DEFENSE SYSTEM"
echo "                   Autonomous Surveillance Suite"
echo "==============================================================================="
echo ""

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH!"
    echo "Please install Python 3 (3.10 or 3.11 recommended): sudo apt install python3 python3-pip"
    exit 1
fi

PY_VER=$(python3 --version 2>&1)
echo "[OK] Detected $PY_VER"

# 2. Check Dependencies
echo "[*] Checking required dependencies..."
if ! python3 -c "import flask, cv2, yaml, numpy, ultralytics" &>/dev/null; then
    echo ""
    echo "==============================================================================="
    echo " [FIRST TIME SETUP] Missing required libraries detected!"
    echo " Automatically installing dependencies from requirements.txt..."
    echo " Please wait (2-3 minutes)..."
    echo "==============================================================================="
    echo ""
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    echo "[OK] Dependencies installed successfully!"
else
    echo "[OK] All dependencies verified and ready."
fi

# 3. Launch Dashboard & Trigger Browser
echo ""
echo "==============================================================================="
echo " Starting Tactical Defense Dashboard at http://127.0.0.1:5000 ..."
echo " (Press CTRL+C anytime to stop)"
echo "==============================================================================="
echo ""

(sleep 2 && (xdg-open "http://127.0.0.1:5000" 2>/dev/null || open "http://127.0.0.1:5000" 2>/dev/null || true)) &

python3 dashboard/app.py
