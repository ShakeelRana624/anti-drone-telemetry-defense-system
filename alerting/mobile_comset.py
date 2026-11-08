"""
mobile_comset.py — Wireless Mobile Tactical Alert & ADB Orchestrator
=====================================================================
Manages wireless pairing with Android devices (e.g. Android 11+ Wireless Debugging),
auto-detects ADB ports, launches the dedicated tactical alert screen on the phone,
and triggers automated screen wakeup & Urdu voice warnings on drone detection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/mobile_config.json")

# Ensure portable tools (bin/platform-tools) are in PATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PORTABLE_BIN_PATHS = [
    _PROJECT_ROOT / "bin" / "platform-tools",
    _PROJECT_ROOT / "bin",
]
for _p in _PORTABLE_BIN_PATHS:
    if (_p / "adb.exe").exists() or (_p / "adb").exists():
        _p_str = str(_p)
        if _p_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p_str + os.pathsep + os.environ.get("PATH", "")
        break


def get_local_ip(target_ip: Optional[str] = None) -> str:
    """
    Detect the local Wi-Fi / Hotspot LAN IP of the host machine.
    Works 100% OFFLINE without any active internet connection.
    Accurately matches whether the phone is connected via Local Wi-Fi or Windows Mobile Hotspot.
    """
    if target_ip and ":" in target_ip:
        target_ip = target_ip.split(":")[0]

    host_ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                host_ips.append(ip)
    except Exception:
        pass

    # 1. If target phone IP is provided, match host interface on the same subnet (/24)
    if target_ip:
        prefix = ".".join(target_ip.split(".")[:3]) + "."
        for hip in host_ips:
            if hip.startswith(prefix):
                return hip

        # 2. Try OS routing table lookup directly to the target phone
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0.2)
                s.connect((target_ip, 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except Exception:
            pass

    # 3. Default fallback: return first active LAN interface or localhost
    if host_ips:
        # Prefer 192.168.137.1 (Hotspot) or 10.x / 192.168.x LAN interfaces
        for hip in host_ips:
            if hip.startswith("192.168.137."):
                return hip
        return host_ips[0]

    return "127.0.0.1"


class MobileComset:
    """Orchestrates Android Wireless ADB pairing, connection, and alert screen launching."""

    _instance: Optional[MobileComset] = None
    _lock = threading.Lock()
    _launch_lock = threading.Lock()
    _last_launch_time: float = 0.0
    LAUNCH_COOLDOWN_SECONDS: float = 45.0

    def __init__(self) -> None:
        self.config: Dict[str, Any] = self._load_config()
        self._connected: bool = False
        self._model: str = "Unknown"

    @classmethod
    def get_instance(cls) -> MobileComset:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _load_config(self) -> Dict[str, Any]:
        default_cfg = {
            "enabled": True,
            "ip": "10.196.162.171",
            "pair_port": 39527,
            "conn_port": 36777,
            "device_model": "Android Device",
            "auto_launch_on_alert": True,
            "alert_url": f"http://{get_local_ip()}:5000/alert",
        }
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    default_cfg.update(data)
            except Exception as e:
                logger.warning("Failed to load mobile_config.json: %s", e)
        return default_cfg

    def save_config(self, new_cfg: Dict[str, Any]) -> None:
        self.config.update(new_cfg)
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error("Failed to save mobile_config.json: %s", e)

    @staticmethod
    def _get_adb_cmd() -> str:
        found = shutil.which("adb")
        if found:
            return found
        for p in _PORTABLE_BIN_PATHS:
            for fname in ("adb.exe", "adb"):
                cand = p / fname
                if cand.is_file():
                    return str(cand)
        return "adb"

    def _run_adb(self, args: list[str], timeout: int = 10) -> Tuple[int, str]:
        cmd = [self._get_adb_cmd()] + args
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False
            )
            out = (res.stdout + "\n" + res.stderr).strip()
            return res.returncode, out
        except subprocess.TimeoutExpired:
            return -1, "ADB command timed out"
        except Exception as exc:
            return -1, str(exc)

    def auto_scan_port(self, ip: str) -> Optional[int]:
        """Scan ports 30000-50000 on target phone to find active Wireless Debugging port."""
        def check_port(p: int) -> Optional[int]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.12)
                if s.connect_ex((ip, p)) == 0:
                    return p
            return None

        with ThreadPoolExecutor(max_workers=200) as executor:
            results = executor.map(check_port, range(30000, 50000))
            for port in results:
                if port:
                    return port
        return None

    def pair(self, pair_addr: str, pair_code: str, conn_port: Optional[str] = None) -> Tuple[bool, str]:
        """
        Pair device via Android 11+ Wireless Debugging pairing code.
        Format of pair_addr: 'IP:PORT' e.g. '10.196.162.171:39527'.
        """
        pair_addr = pair_addr.strip().replace("..", ".").replace(" ", "")
        # Normalize if user entered ip.port instead of ip:port
        if pair_addr.count(".") == 4 and ":" not in pair_addr:
            parts = pair_addr.rsplit(".", 1)
            pair_addr = f"{parts[0]}:{parts[1]}"

        pair_code = str(pair_code).strip()
        code, out = self._run_adb(["pair", pair_addr, pair_code], timeout=15)
        logger.info("ADB PAIR [%s] -> %s", pair_addr, out)

        if "successfully paired" in out.lower():
            ip = pair_addr.split(":")[0]
            # Try to connect
            target_port = None
            if conn_port and str(conn_port).strip():
                try:
                    target_port = int(conn_port.strip())
                except ValueError:
                    target_port = None

            if not target_port:
                target_port = self.auto_scan_port(ip)

            if target_port:
                connect_ok, connect_msg = self.connect(ip, target_port)
                if connect_ok:
                    self.save_config({
                        "ip": ip,
                        "conn_port": target_port,
                        "device_model": self._model,
                    })
                    return True, f"Successfully paired and connected to {self._model} ({ip}:{target_port})!"
                return True, f"Paired successfully! Connect manually using port {target_port}."

            return True, "Paired successfully! Please check connection port on phone."
        else:
            return False, f"Pairing failed: {out}"

    def connect(self, ip: str, port: int) -> Tuple[bool, str]:
        """Connect to device on specified IP and port."""
        target = f"{ip}:{port}"
        code, out = self._run_adb(["connect", target], timeout=10)
        if "connected to" in out.lower() and "cannot" not in out.lower():
            self._connected = True
            # Get device model
            m_code, m_out = self._run_adb(["-s", target, "shell", "getprop", "ro.product.model"])
            if m_code == 0 and m_out.strip():
                self._model = m_out.strip()
            else:
                self._model = "Android Device"

            self.save_config({
                "ip": ip,
                "conn_port": port,
                "device_model": self._model,
            })
            return True, f"Connected to {self._model} ({target})"
        return False, f"Could not connect: {out}"

    def get_status(self) -> Dict[str, Any]:
        """Check live connection status and details."""
        ip = self.config.get("ip", "10.196.162.171")
        port = self.config.get("conn_port", 36777)
        target = f"{ip}:{port}"

        # Quick check if device is listed in adb devices
        code, out = self._run_adb(["devices"])
        is_online = target in out and "offline" not in out

        if not is_online:
            # Attempt quick auto-reconnect
            c_code, c_out = self._run_adb(["connect", target], timeout=2)
            if "connected to" in c_out.lower() and "cannot" not in c_out.lower():
                is_online = True

        device_model = self.config.get("device_model", "Android Device")
        if is_online:
            m_code, m_out = self._run_adb(["-s", target, "shell", "getprop", "ro.product.model"], timeout=2)
            if m_code == 0 and m_out.strip():
                device_model = m_out.strip()
                self._model = device_model
                self.config["device_model"] = device_model

        local_ip = get_local_ip(ip)
        alert_url = f"http://{local_ip}:5000/alert"

        return {
            "connected": is_online,
            "device_model": device_model,
            "target": target,
            "ip": ip,
            "port": port,
            "local_ip": local_ip,
            "alert_url": alert_url,
            "auto_launch": self.config.get("auto_launch_on_alert", True),
        }

    def launch_alert_screen(self) -> Tuple[bool, str]:
        """Wake up phone screen, unlock keyguard, and launch Chrome with the alert URL."""
        status = self.get_status()
        target = status["target"]
        url = status["alert_url"]

        if not status["connected"]:
            # One more connection attempt
            c_code, c_out = self._run_adb(["connect", target], timeout=3)
            if "connected to" not in c_out.lower() or "cannot" in c_out.lower():
                return False, f"Mobile device ({target}) is offline or disconnected. Re-pair or check IP/port."

        # 1. Wake up screen and dismiss keyguard
        self._run_adb(["-s", target, "shell", "input", "keyevent", "224"])
        self._run_adb(["-s", target, "shell", "wm", "dismiss-keyguard"])
        self._run_adb(["-s", target, "shell", "input", "keyevent", "82"])

        # 2. Launch Chrome specifically with VIEW intent
        code, out = self._run_adb([
            "-s", target, "shell", "am", "start",
            "-a", "android.intent.action.VIEW",
            "-d", url,
            "com.android.chrome"
        ])

        if code == 0 and "error" not in out.lower():
            return True, f"Alert screen launched on {status['device_model']}!"

        # Fallback 1: Direct Component launch
        c2_code, c2_out = self._run_adb([
            "-s", target, "shell", "am", "start",
            "-n", "com.android.chrome/com.google.android.apps.chrome.Main",
            "-a", "android.intent.action.VIEW",
            "-d", url
        ])
        if c2_code == 0 and "error" not in c2_out.lower():
            return True, f"Alert screen launched on {status['device_model']}!"

        # Fallback 2: Generic VIEW intent for any default browser
        f_code, f_out = self._run_adb([
            "-s", target, "shell", "am", "start",
            "-a", "android.intent.action.VIEW",
            "-d", url
        ])
        if f_code == 0 and "error" not in f_out.lower():
            return True, "Alert screen launched in default browser!"

        return False, f"Failed to launch browser: {out}"

    def reset_cooldown(self) -> None:
        """Reset launch cooldown when alert is dismissed."""
        with self._launch_lock:
            self._last_launch_time = 0.0

    def on_drone_detected(self, context: Optional[dict] = None) -> None:
        """Called automatically whenever a drone threat is confirmed by AI."""
        if not self.config.get("auto_launch_on_alert", True):
            return

        import time
        now = time.time()
        with self._launch_lock:
            # Prevent rapid reloading: Only launch Chrome ONCE every 45 seconds!
            if now - self._last_launch_time < self.LAUNCH_COOLDOWN_SECONDS:
                return
            self._last_launch_time = now

        def _bg_task():
            try:
                self.launch_alert_screen()
            except Exception as e:
                logger.error("Auto-launch on drone detection failed: %s", e)

        threading.Thread(target=_bg_task, daemon=True).start()

