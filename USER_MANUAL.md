# 🛰️ LIMONS Anti-Drone Telemetry & Autonomous Defense System
## Complete Operator & End-User Manual
**Version:** 2.4.0 (Production Release)  
**Classification:** Operational Security & Tactical Defense  
**Target Hardware:** Windows 10/11, Ubuntu Linux, macOS, Raspberry Pi 4/5, Jetson Orin  

---

## 📋 Table of Contents
1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Hardware & Software Prerequisites](#2-hardware--software-prerequisites)
3. [1-Click Quick Startup (Windows & Linux)](#3-1-click-quick-startup-windows--linux)
4. [First-Time Camera Setup & Compass Calibration](#4-first-time-camera-setup--compass-calibration)
5. [C2 Command & Control Dashboard Walkthrough](#5-c2-command--control-dashboard-walkthrough)
   - [5.1 Tactical Operations & Live HUD](#51-tactical-operations--live-hud)
   - [5.2 360° PPI Airspace Radar Scope](#52-360-ppi-airspace-radar-scope)
   - [5.3 Real-Time Telemetry Data Feed](#53-real-time-telemetry-data-feed)
   - [5.4 Early Warning Alerts & Siren System](#54-early-warning-alerts--siren-system)
   - [5.5 Dynamic Analytics & Trend Visualizations](#55-dynamic-analytics--trend-visualizations)
   - [5.6 Tactical Defense Audit Reports](#56-tactical-defense-audit-reports)
   - [5.7 Forensic Video Evidence Vault](#57-forensic-video-evidence-vault)
6. [PTZ Camera Controls & ONVIF Steer](#6-ptz-camera-controls--onvif-steer)
7. [System Shutdown & Maintenance](#7-system-shutdown--maintenance)
8. [Troubleshooting & Frequently Asked Questions (FAQ)](#8-troubleshooting--frequently-asked-questions-faq)

---

## 1. System Overview & Architecture

The **LIMONS Anti-Drone Telemetry Defense System** is an enterprise, military-grade airspace surveillance and counter-UAS platform. It autonomously detects, tracks, classifies, and geolocates hostile commercial and tactical drones in real-time.

### Key Capabilities:
* **High-Accuracy AI Detection:** Powered by **YOLO11** neural network with automated multi-scale inference.
* **DeepSORT Multi-Object Tracking:** Temporal tracking with Kalman filtering and appearance-based re-ID to eliminate false positives from birds or clouds.
* **Decoupled Asynchronous AI Engine:** Video capture and browser streaming run at **25 to 30 FPS** buttery-smooth performance by decoupling heavy neural network math onto a dedicated background worker thread.
* **True Compass & Spatial Telemetry:** Computes relative azimuth, elevation angle, true compass bearing (0°–360°), slant range, ground distance, drone altitude, speed (km/h), and **Base Arrival ETA**.
* **Strict 1-Clip Forensic Video Evidence:** When a target is locked and confirmed, the system captures **strictly one (1)** verified 5-second MP4 evidence video per intrusion cycle.
* **Zero Baseline Post-Configuration:** Reconfiguring the camera automatically purges stale dummy data, resets all analytics, graphs, and counters to **0**, allowing genuine telemetry to populate purely from live detections.

---

## 2. Hardware & Software Prerequisites

### Required Software:
1. **Python 3.10 or 3.11** (Download from [python.org](https://www.python.org/downloads/)).
   > ⚠️ **CRITICAL:** During Python installation, you **MUST** check the box **"Add python.exe to PATH"**.

### Recommended Hardware:
* **Processor:** Intel Core i5/i7 (8th Gen+) or AMD Ryzen 5/7 (or Apple Silicon M-series).
* **RAM:** 8 GB minimum (16 GB recommended).
* **GPU (Optional):** NVIDIA GTX/RTX GPU with CUDA support for 50+ FPS inference; CPU runs at a smooth 25–30 FPS via our asynchronous pipeline.
* **Cameras Supported:**
  - Standard USB Webcams (Index 0, 1, 2)
  - RTSP/HTTP IP Security Cameras (Hikvision, Dahua, Axis, Hanwha, Uniview)
  - ONVIF PTZ Pan-Tilt-Zoom Speed Domes

---

## 3. 1-Click Quick Startup (Windows & Linux)

The system includes fully automated single-click startup scripts. You **do not** need to type manual terminal commands.

### On Windows (Single Click):
1. Navigate to the project root folder.
2. **Double-click** on **`start_dashboard.bat`**.
3. **What happens automatically:**
   - **Python Check:** Verifies Python is installed and configured on PATH.
   - **Auto-Dependency Installation:** If running for the first time, it automatically installs all required dependencies (`pip install -r requirements.txt`).
   - **Fast Startup:** On subsequent runs, it skips installation and boots in under 2 seconds.
   - **Browser Auto-Launch:** Your default web browser (Chrome, Edge, Brave, etc.) will automatically pop open to `http://127.0.0.1:5000`.

### On Linux / macOS / Raspberry Pi:
Open a terminal in the folder and run:
```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

---

## 4. First-Time Camera Setup & Compass Calibration

When a new operator launches the system for the first time, the hardware camera remains **offline by design** to prevent accidental surveillance of your workspace. The dashboard will automatically guide you to the **"Camera Setup"** tab.

```
+-------------------------------------------------------------------------+
| [SENSOR UNCONFIGURED // HARDWARE OFFLINE]                               |
| -> Please select your camera hardware in the Camera Setup tab.          |
+-------------------------------------------------------------------------+
```

### Step-by-Step Configuration:

1. **Select Camera Connection Mode:**
   - **USB Camera:** Select your webcam index (usually `0` for built-in webcam, or `1`/`2` for external USB lens).
   - **RTSP IP Camera:** Enter the camera's local IP address (e.g., `192.168.1.101`), RTSP port (`554`), username, password, and RTSP stream path (e.g., `/stream1` or `/h264Preview_01_main`).
   - **PTZ ONVIF Camera:** Enter the ONVIF IP and port (`80`) to enable software pan/tilt/zoom steering.

2. **Calibrate Compass Direction (Azimuth Bearing):**
   - Identify the real-world direction the camera lens is physically facing:
     - `0.0°` = Due **North**
     - `90.0°` = Due **East**
     - `180.0°` = Due **South**
     - `270.0°` = Due **West**
   - Enter the exact angle in degrees (e.g., `90.0` for East). The interactive compass rose will rotate in real-time to match your alignment.

3. **Field of View (FOV):**
   - Select your camera lens horizontal FOV (standard optical cameras typically range between `85°` and `90°`).

4. **Geographic Base Station Coordinates (GPS):**
   - Enter your base station Latitude and Longitude.
   - *Tip:* Click the **"Auto-Detect GPS"** button to instantly query your browser's geolocation sensor.

5. **Click "Save & Apply Camera Configuration":**
   - The settings are persistently written to disk (`camera_int/cameras_config.json`).
   - The live feed immediately hot-reloads and connects to your hardware.
   - All graphs, telemetry history, and evidence vault drop to clean zero (**0**) baseline, ready to log genuine drone incursions.

---

## 5. C2 Command & Control Dashboard Walkthrough

### 5.1 Tactical Operations & Live HUD
* **High-FPS Video Stream:** Displays live video at 25–30 FPS with optical boresight crosshairs and cardinal bearing indicators.
* **AI Target Locking:** When a drone enters the surveillance sector:
  - **Cyan Box (`ACQUIRING`):** Initial visual confirmation.
  - **Yellow Box (`CONFIRMING`):** DeepSORT track verification across consecutive frames.
  - **Red Box (`TARGET LOCKED`):** Hostile target confirmed. Corner brackets lock on, bearing ray line connects from center boresight to target, and full telemetry banner displays above the target.

### 5.2 360° PPI Airspace Radar Scope
* Located on the right-hand side of the dashboard.
* Emulates a military Plan Position Indicator (PPI) radar display with a continuous sweeping beam.
* **Range Selector:** Toggle detection radius between **50m**, **100m**, **200m**, and **500m**.
* Blips display exact polar coordinates (`Range` and `Bearing`) with target ID tags and altitude indicators.

### 5.3 Real-Time Telemetry Data Feed
The telemetry table updates sub-second via Server-Sent Events (SSE):
| Metric | Description |
| :--- | :--- |
| **Track ID** | Unique track identifier (e.g., `TGT-#01`) assigned by DeepSORT. |
| **Bearing & Sector** | True geographic compass heading (e.g., `104.2° ESE`). |
| **Elevation / Azimuth** | Sensor-relative angle offsets from center boresight. |
| **Distance (Slant & Ground)** | Real-time distance estimated via calibrated optical trigonometry. |
| **Speed (km/h)** | Temporal kinematic velocity calculated across video frames. |
| **Motion State & ETA** | Classifies target as `INBOUND`, `OUTBOUND`, `CROSSING`, or `HOVERING`. For inbound threats, calculates estimated seconds to base perimeter. |
| **Target GPS** | Estimated real-time Latitude and Longitude coordinates of the drone. |

### 5.4 Early Warning Alerts & Siren System
* **Tactical Alert Banner:** Drops down with hostile target coordinates when locked.
* **Audio Warning Siren:** Synthesized tactical alert sounds for 30 seconds (with manual silence toggle).
* **RF Comset Nodes:** The 5 perimeter radio units flash red to indicate active jamming/warning readiness.

### 5.5 Dynamic Analytics & Trend Visualizations
All charts start empty (**0**) upon camera configuration and dynamically plot live data:
1. **Main Aerial Incursion Trajectory:** Weekly, Monthly, and Yearly line/bar charts showing incursion frequency over time.
2. **Threat Composition Donut:** Visual breakdown of drone categories (Quadcopter vs. Fixed-Wing vs. FPV Micro-drone).
3. **360° Polar Bearing Radar Chart:** Shows which directional vectors (North, East, South, West) hostiles are approaching from.
4. **24-Hour Peak Incursion Heatmap:** Highlights high-risk diurnal hours to assist patrol scheduling.

### 5.6 Tactical Defense Audit Reports
* Select your reporting window (**Weekly**, **Monthly**, or **Yearly**) and sensor scope.
* Generates formal Intelligence Defense Reports complete with:
  - Classification stamp and coverage window.
  - High-threat counts, neutralization success rates, and mean intercept ranges.
  - Incident breakdown tables and automated tactical recommendations.
* **Export Options:** One-click **Download CSV** and **Download JSON** for military audits and archival.

### 5.7 Forensic Video Evidence Vault
* **Strict 1-Clip Rule:** The system records **only one (1) verified 5-second MP4 evidence clip** per incursion session to preserve storage and avoid duplicate files.
* **Video Lightbox Player:** Click any card in the vault to inspect recorded video evidence.
* **Download Clip:** Click the **MP4** button to download full-resolution forensic clips stamped with case IDs, confidence scores, and bearing metadata.

---

## 6. PTZ Camera Controls & ONVIF Steer

If using an ONVIF-compatible PTZ camera:
1. Navigate to the **Camera Setup** tab.
2. Under **Connection Mode**, select **PTZ Camera (ONVIF)**.
3. Use the directional control pad in the dashboard to:
   - **Pan Left / Right**
   - **Tilt Up / Down**
   - **Zoom In (+) / Zoom Out (-)**
   - Adjust motor movement speed (`0.1` to `1.0`).

---

## 7. System Shutdown & Maintenance

To safely stop the system:
* In the console window running `start_dashboard.bat`, press **`CTRL + C`** or click the window's close (`X`) button.
* Port `5000` will be cleanly released.
* All recorded evidence videos in `logs/evidence_videos/` and configuration settings in `camera_int/cameras_config.json` remain safely preserved for your next session.

---

## 8. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: `start_dashboard.bat` opens and closes immediately. What should I do?
* **Solution:** Verify Python is installed and added to PATH. Open Command Prompt and type `python --version`. If an error appears, re-run the Python installer, select **Modify**, and make sure **Add python.exe to PATH** is checked.

### Q2: Why is the video feed showing "SENSOR UNCONFIGURED"?
* **Solution:** This is normal on first launch. Navigate to the **Camera Setup** tab, verify your USB camera index or IP camera credentials, and click **Save & Apply Camera Configuration**.

### Q3: How do I get 25–30 FPS on a standard laptop without a dedicated GPU?
* **Solution:** The system is already optimized with our decoupled asynchronous engine. Video capture runs at 25–30 FPS, while YOLO AI detection processes in the background. If using high-resolution 4K IP cameras, ensure your RTSP stream is set to 1080p or 720p for optimal CPU performance.

### Q4: How do I test the system without a physical drone?
* **Solution:** You can show an image or video of a drone on your smartphone screen to the camera, or place test video clips into the `test_videos/` folder. The AI will detect, track, estimate bearing, and record video evidence.

### Q5: Can multiple computers view the dashboard simultaneously?
* **Solution:** Yes! From any computer or tablet on the same local network (Wi-Fi/LAN), open a browser and navigate to:
  `http://<SERVER_IP>:5000` (e.g., `http://192.168.0.103:5000`).

---

*Developed by Google Advanced Agentic Coding for the LIMONS Anti-Drone Telemetry Defense System.*  
*For questions or updates, consult the repository at [GitHub](https://github.com/ShakeelRana624/anti-drone-telemetry-defense-system).*
