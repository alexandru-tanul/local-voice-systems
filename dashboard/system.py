"""Mic to Voice's services, started and stopped together: speech recognition, the voice engine,
and the push-to-talk helper. Also System Check."""

import base64
import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import asr, config, engine, jobs, models, wsl


# Held while a startup step checks that its run is still current and starts a service, so
# Stop Everything cannot fall between the check and the start.
SYSTEM_LOCK = threading.Lock()
PTT_CONDITION = threading.Condition()
PTT_STATE = {
    "binding": "ShiftLeft",
    "held": False,
    "sequence": 0,
}
SYSTEM_STATE = {
    "status": "stopped",
    "phase": "stopped",
    "message": "System is stopped.",
    "error": "",
}
SYSTEM_THREAD = None
SYSTEM_RUN_ID = 0


def set_system_state(status, phase, message, error=""):
    with jobs.STATE_LOCK:
        SYSTEM_STATE.update(
            {
                "status": status,
                "phase": phase,
                "message": message,
                "error": error,
            }
        )
    jobs.log(message)


class TcpInitialRtoParameters(ctypes.Structure):
    _fields_ = [("Rtt", ctypes.c_ushort), ("MaxSynRetransmissions", ctypes.c_ubyte)]


def port_open(port, timeout=0.2):
    """True when something accepts connections on 127.0.0.1:port.

    Windows retries a refused connection until the timeout, so each check of a stopped service
    took the full timeout. With SYN retransmissions turned off, a refusal returns at once."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "nt":
            # SIO_TCP_INITIAL_RTO with TCP_INITIAL_RTO_UNSPECIFIED_RTT and
            # TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS, from mstcpip.h. If the call fails, the
            # connect still works and only waits longer.
            params = TcpInitialRtoParameters(0xFFFF, 0xFE)
            ctypes.windll.ws2_32.WSAIoctl(
                ctypes.c_size_t(sock.fileno()), ctypes.c_ulong(0x98000011),
                ctypes.byref(params), ctypes.c_ulong(ctypes.sizeof(params)),
                None, ctypes.c_ulong(0), ctypes.byref(ctypes.c_ulong()), None, None,
            )
        sock.settimeout(timeout)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def ptt_helper_ready():
    proc = jobs.JOBS.get("ptt-helper")
    return bool(proc and proc.process and proc.process.poll() is None)


def start_ptt_helper():
    if ptt_helper_ready():
        jobs.log("Global PTT helper is already running.")
        return True
    helper_path = config.ROOT / "ptt_helper.py"
    if not helper_path.exists():
        raise RuntimeError("Global PTT helper is missing.")
    proc = jobs.ManagedProcess(
        "Global PTT helper",
        [sys.executable, str(helper_path), "--dashboard", f"http://127.0.0.1:{config.DASHBOARD_PORT}"],
        cwd=config.ROOT,
    )
    with jobs.STATE_LOCK:
        jobs.JOBS["ptt-helper"] = proc
    proc.start()
    jobs.log("Global PTT helper started.")
    return True


def wait_until(check, timeout, run_id, description, proc=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if run_id != SYSTEM_RUN_ID:
            raise RuntimeError("Startup was cancelled.")
        if check():
            return
        if proc and proc.process and proc.process.poll() is not None:
            # The process ended before it became ready. Its last lines usually name the cause.
            output = "\n".join(proc.last_lines(6))
            raise RuntimeError(
                f"{description} exited with code {proc.process.poll()} before it became ready."
                + (f"\n{output}" if output else "")
            )
        time.sleep(0.5)
    raise RuntimeError(f"{description} did not become ready within {int(timeout)} seconds.")


def startup_step(run_id, phase, message):
    # Call with SYSTEM_LOCK held, so Stop Everything cannot cancel the run between this
    # check and the service the step starts.
    if run_id != SYSTEM_RUN_ID:
        raise RuntimeError("Startup was cancelled.")
    set_system_state("starting", phase, message)


def _start_system_worker(gpt_path, sovits_path, run_id, spoken="en", speaks="en"):
    try:
        result = wsl.run_wsl("true", timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "WSL failed to start.").replace("\x00", "").strip()
            raise RuntimeError(detail)
        gpt_path, sovits_path = models.model_pair(gpt_path, sovits_path)
        if not gpt_path or not sovits_path:
            raise RuntimeError("No compatible GPT and SoVITS model pair was found.")

        with SYSTEM_LOCK:
            startup_step(run_id, "asr", "Starting local speech recognition...")
            asr.start_relay_asr()
        wait_until(asr.relay_asr_ready, 120, run_id, "Speech recognition", jobs.JOBS.get("relay-asr"))
        with SYSTEM_LOCK:
            translated = asr.written(spoken) != asr.written(speaks)
            startup_step(run_id, "asr", "Loading the speech recognition and translation models..." if translated
                         else "Loading the speech recognition model...")
        asr.warm_up_relay_asr(spoken, speaks)

        with SYSTEM_LOCK:
            startup_step(run_id, "tts", "Starting the selected voice engine...")
            engine_running = engine.api_ready()
            if not engine_running:
                engine.start_api(gpt_path, sovits_path)
        if engine_running:
            # Only applies the selected models, which can take minutes, so it runs outside the lock.
            engine.start_api(gpt_path, sovits_path)
        wait_until(engine.api_ready, 240, run_id, "Voice engine", engine.API_PROCESS)

        with SYSTEM_LOCK:
            startup_step(run_id, "ptt", "Starting global push-to-talk...")
            start_ptt_helper()
        wait_until(ptt_helper_ready, 15, run_id, "Global push-to-talk")

        with SYSTEM_LOCK:
            if run_id != SYSTEM_RUN_ID:
                raise RuntimeError("Startup was cancelled.")
            set_system_state("ready", "ready", "System ready. Press your push-to-talk key.")
    except Exception as exc:
        with SYSTEM_LOCK:
            if run_id == SYSTEM_RUN_ID:
                set_system_state("failed", "failed", "System startup failed.", str(exc))


def start_system(gpt_path, sovits_path, binding="ShiftLeft", spoken="en", speaks="en"):
    """Start Mic to Voice's services. spoken and speaks are the languages of its Settings, whose
    models load while it starts."""
    global SYSTEM_THREAD, SYSTEM_RUN_ID
    with SYSTEM_LOCK:
        # A startup that Stop Everything cancelled can still be waiting, but it can no longer
        # change the status or start a service, so only the current run's state matters here.
        if SYSTEM_STATE["status"] == "starting":
            raise RuntimeError("The system is already starting.")
        if SYSTEM_STATE["status"] == "stopping":
            raise RuntimeError("The system is still stopping. Try again in a moment.")
        with PTT_CONDITION:
            PTT_STATE["binding"] = binding or "ShiftLeft"
            PTT_CONDITION.notify_all()
        SYSTEM_RUN_ID += 1
        run_id = SYSTEM_RUN_ID
        set_system_state("starting", "wsl", "Starting the local model environment...")
        SYSTEM_THREAD = threading.Thread(
            target=_start_system_worker,
            args=(gpt_path, sovits_path, run_id, spoken, speaks),
            daemon=True,
        )
        SYSTEM_THREAD.start()


def run_powershell(script, timeout):
    # -EncodedCommand takes the script as base64 UTF-16, so its quotes and backslashes need no escaping.
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        text=True,
        capture_output=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def kill_port_listener(port, label, marker):
    # JOBS only knows processes this dashboard instance launched; a service started
    # by an earlier instance survives a dashboard restart, so stop it by port owner.
    # Only an owner whose command line contains the marker is stopped, so another
    # program that uses the same port is left alone.
    try:
        result = run_powershell(
            f"Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | "
            "ForEach-Object { Get-CimInstance Win32_Process -Filter \"ProcessId=$($_.OwningProcess)\" } | "
            f"Where-Object {{ $_.CommandLine -like '*{marker}*' }} | Select-Object -ExpandProperty ProcessId",
            timeout=10,
        )
        pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
        for pid in pids:
            if pid <= 0 or pid == os.getpid():
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=6,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            jobs.log(f"Stopped orphaned {label} process {pid}.")
    except Exception as exc:
        jobs.log(f"Could not stop orphaned {label}: {exc}")


def kill_orphan_ptt_helpers():
    """Stop push-to-talk helpers that an earlier run of this dashboard left. Only Python processes
    whose command line ends with this dashboard's address match, as start_ptt_helper starts them,
    so editors, shells and other dashboards that name ptt_helper.py are left alone."""
    marker = f"--dashboard http://127.0.0.1:{config.DASHBOARD_PORT}"
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        f"Where-Object {{ $_.CommandLine -like '*ptt_helper.py*' -and $_.CommandLine.TrimEnd().EndsWith('{marker}') }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    try:
        run_powershell(script, timeout=8)
    except (OSError, subprocess.SubprocessError) as exc:
        jobs.log(f"Could not stop orphaned PTT helpers: {exc}")


def stop_system():
    global SYSTEM_RUN_ID
    with SYSTEM_LOCK:
        SYSTEM_RUN_ID += 1
        stop_id = SYSTEM_RUN_ID
        set_system_state("stopping", "stopping", "Stopping managed services...")
    try:
        asr_job = jobs.JOBS.get("relay-asr")
        if asr_job:
            asr_job.stop()
        if asr.relay_asr_ready():
            kill_port_listener(config.RELAY_ASR_PORT, "speech recognition", "relay_asr_server.py")
            if asr.relay_asr_ready():
                jobs.log(f"Another program listens on port {config.RELAY_ASR_PORT}. It was left running.")
        helper = jobs.JOBS.get("ptt-helper")
        if helper:
            helper.stop()
        kill_orphan_ptt_helpers()
        engine.stop_api()
    finally:
        with PTT_CONDITION:
            PTT_STATE["held"] = False
            PTT_STATE["sequence"] += 1
            PTT_CONDITION.notify_all()
        with SYSTEM_LOCK:
            # A second Stop that started meanwhile reports the result itself.
            if SYSTEM_RUN_ID == stop_id:
                set_system_state("stopped", "stopped", "System is stopped.")


def system_snapshot():
    snapshot = dict(SYSTEM_STATE)
    snapshot["components"] = {
        "asr": "ready" if asr.relay_asr_ready() else "off",
        "tts": "ready" if engine.api_ready() else "off",
        "ptt": "ready" if ptt_helper_ready() else "off",
    }
    if snapshot["status"] == "ready" and "off" in snapshot["components"].values():
        snapshot["status"] = "degraded"
        snapshot["message"] = "A required service stopped unexpectedly."
    return snapshot


def setup_status():
    available = models.model_files()
    def safe_exists(path):
        try:
            return path.exists()
        except OSError:
            return False

    gpu = {"status": "missing", "detail": "Voice engine is not installed."}
    try:
        runtime = wsl.voice_runtime()
        probe = (
            "import torch; "
            f"x = torch.ones((8, 8), device={runtime['device']!r}, "
            f"dtype=torch.{'float16' if runtime['is_half'] else 'float32'}); "
            "assert (x @ x).isfinite().all().item(), 'Try tts_precision: fp32'; "
            f"print(torch.cuda.get_device_name(0) if {runtime['device'] == 'cuda'} else 'CPU')"
        )
        result = wsl.run_wsl(f"{wsl.shquote(config.WSL_PYTHON)} -c {wsl.shquote(probe)}", timeout=60)
        if result.returncode:
            # wsl.exe reports its own failures on stdout in UTF-16, so fall back to that.
            message = result.stderr.strip() or result.stdout.replace("\x00", "").strip().split("\n")[0]
            raise RuntimeError(message or "The voice engine check failed.")
        precision = "FP16" if runtime["is_half"] else "FP32"
        gpu = {"status": "ready", "detail": f"{result.stdout.strip()} ({precision})"}
    except Exception as exc:
        gpu["detail"] = str(exc)

    return {
        "config": {
            "status": "missing" if not config.CONFIG_PATH.exists() else ("needs_configuration" if config.CONFIG_ERROR else "ready"),
            "detail": config.CONFIG_ERROR or str(config.CONFIG_PATH),
        },
        "dashboard_python": {
            "status": "ready" if Path(sys.executable).exists() else "missing",
            "detail": sys.executable,
        },
        "wsl_files": {
            "status": "needs_configuration" if config.WSL_PATHS_ERROR else ("ready" if safe_exists(config.WSL_GSV_WIN) else "missing"),
            "detail": config.WSL_PATHS_ERROR or str(config.WSL_GSV_WIN),
        },
        "gpt_models": {
            "status": "ready" if available["gpt"] else "missing",
            "detail": f"{len(available['gpt'])} model(s) found",
        },
        "sovits_models": {
            "status": "ready" if available["sovits"] else "missing",
            "detail": f"{len(available['sovits'])} model(s) found",
        },
        "asr_environment": {
            "status": "ready" if config.RELAY_ASR_PYTHON.exists() else "missing",
            "detail": str(config.RELAY_ASR_PYTHON),
        },
        "asr_model_cache": {
            "status": "ready" if config.RELAY_ASR_MODEL_DIR.exists() else "needs_configuration",
            "detail": str(config.RELAY_ASR_MODEL_DIR),
        },
        "ptt_helper": {
            "status": "ready" if (config.ROOT / "ptt_helper.py").exists() else "missing",
            "detail": str(config.ROOT / "ptt_helper.py"),
        },
        "gpu": gpu,
        "vb_cable": {
            "status": "needs_browser_check",
            "detail": "Audio devices are checked in the browser after microphone permission is granted.",
        },
    }
