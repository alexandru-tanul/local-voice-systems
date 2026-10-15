"""Background work: the jobs the page shows, run as programs (ManagedProcess) or as threads of the
dashboard (ThreadJob), the activity log, and the rule that keeps GPU jobs from running together."""

import contextlib
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from . import config, engine, system, wsl


# Jobs that use the GPU, and with it the memory the voice engine also needs.
GPU_JOBS = ("dataset-prepare", "train-sovits", "train-gpt")
APP_LOG = []
JOBS = {}
STATE_LOCK = threading.Lock()


def log(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    with STATE_LOCK:
        APP_LOG.append(line)
        del APP_LOG[:-500]
    print(line, flush=True)


class ManagedProcess:
    def __init__(self, name, command, stop_script=None, cwd=config.ROOT, env=None):
        self.name = name
        self.command = command
        self.stop_script = stop_script
        self.cwd = cwd
        self.env = env
        self.process = None
        self.returncode = None
        self.lines = []
        self.status = "idle"
        self.reader = None
        # Called with each line of output, as TrainingProgress is.
        self.watch = None

    def start(self):
        if self.process and self.process.poll() is None:
            return False
        self.returncode = None
        self.status = "running"
        self.lines = []
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=self.env,
            )
        except OSError as exc:
            self.status = "failed"
            self.lines.append(f"Could not start {self.name}: {exc}")
            raise
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        threading.Thread(target=self._wait, daemon=True).start()
        return True

    def last_lines(self, count):
        # Output can still be in the pipe when the process ends, so read it first.
        if self.reader and not self.running():
            self.reader.join(timeout=1)
        return self.lines[-count:]

    def _read_output(self):
        if not self.process or not self.process.stdout:
            return
        with self.process.stdout:
            for line in self.process.stdout:
                line = line.rstrip()
                with STATE_LOCK:
                    self.lines.append(line)
                    del self.lines[:-1200]
                if self.watch:
                    self.watch(line)

    def _wait(self):
        if not self.process:
            return
        self.returncode = self.process.wait()
        if self.status in ("stopping", "stopped"):
            self.status = "stopped"
            return
        self.status = "done" if self.returncode == 0 else "failed"
        log(f"{self.name} {self.status} with exit code {self.returncode}.")

    def running(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        running = self.running()
        if running:
            # Marked before the stop script, as the program it ends would otherwise count as failed.
            self.status = "stopping"
        if self.stop_script:
            wsl.stop_in_wsl(self.stop_script)
        if not running:
            # Already ended: keep "done" or "failed" instead of reporting "stopped".
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.status = "stopped"
        log(f"{self.name} stopped.")

    def snapshot(self):
        running = self.process is not None and self.process.poll() is None
        return {
            "name": self.name,
            "status": "running" if running else self.status,
            "returncode": self.returncode,
            "lines": self.lines[-250:],
            "progress": self.watch.state() if self.watch else None,
        }


class JobStopped(Exception):
    pass


class ThreadJob:
    """A job that runs a function in a thread of the dashboard, shown and stopped like a
    ManagedProcess. The function reports progress with say(), runs programs with run(), and
    calls check_stop() between steps."""

    def __init__(self, name, target):
        self.name = name
        self.target = target
        self.lines = []
        self.status = "idle"
        self.returncode = None
        self.stopping = False
        self.thread = None
        self.child = None
        self.child_stop_script = None
        self.uses_gpu = False

    def start(self):
        self.lines = []
        self.returncode = None
        self.stopping = False
        self.status = "running"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def _run(self):
        try:
            self.target(self)
            self.returncode = 0
            self.status = "stopped" if self.stopping else "done"
        except JobStopped:
            self.returncode = 1
            self.status = "stopped"
        except Exception as exc:
            self.returncode = 1
            self.status = "stopped" if self.stopping else "failed"
            if not self.stopping:
                self.say(f"Failed: {exc}")
        log(f"{self.name} {self.status}.")

    def say(self, line):
        with STATE_LOCK:
            self.lines.append(line)
            del self.lines[:-1200]

    def check_stop(self):
        if self.stopping:
            raise JobStopped()

    def run(self, command, timeout=None):
        """Run a program and return its finished process, with its output as text."""
        self.check_stop()
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as child:
            self.child = child
            if self.stopping:
                child.kill()
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
                raise RuntimeError(f"{Path(command[0]).name} did not finish within {timeout} seconds.")
            finally:
                self.child = None
        self.check_stop()
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)

    def stream(self, command, env=None, stop_script=None):
        """Run a program and show its output as it comes. A stop ends it, and a program in WSL
        also by stop_script, as ending wsl.exe leaves the Linux process running."""
        self.check_stop()
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as child:
            self.child, self.child_stop_script = child, stop_script
            if self.stopping:
                child.kill()
                if stop_script:
                    wsl.stop_in_wsl(stop_script)
            try:
                for line in child.stdout:
                    line = line.rstrip()
                    # Progress bars redraw many times a second; their last state says nothing new.
                    if line and not re.search(r"\d+%\||it/s\]|s/it\]|GetSegmentId", line):
                        self.say(line)
                child.wait()
            finally:
                self.child = self.child_stop_script = None
        self.check_stop()
        return child.returncode

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def last_lines(self, count):
        return self.lines[-count:]

    def stop(self):
        if not self.running():
            return
        self.stopping = True
        child, stop_script = self.child, getattr(self, "child_stop_script", None)
        if child and child.poll() is None:
            child.kill()
        if stop_script:
            wsl.stop_in_wsl(stop_script)
        self.thread.join(timeout=15)
        log(f"{self.name} stopped.")

    def snapshot(self):
        return {
            "name": self.name,
            "status": "running" if self.running() else self.status,
            "returncode": self.returncode,
            "lines": self.lines[-250:],
            "dataset": getattr(self, "dataset_id", ""),
            "results": getattr(self, "results", None),
        }


GPU_RESERVATIONS = 0


@contextlib.contextmanager
def gpu_reserved():
    """Count a GPU job as running while its request is handled, before its process exists."""
    global GPU_RESERVATIONS
    with STATE_LOCK:
        GPU_RESERVATIONS += 1
    try:
        yield
    finally:
        with STATE_LOCK:
            GPU_RESERVATIONS -= 1


def gpu_busy():
    """True while a job uses, or is about to use, the GPU that the voice engine runs on."""
    try:
        if wsl.voice_runtime()["device"] != "cuda":
            return False
    except ValueError:
        return False
    with STATE_LOCK:
        return GPU_RESERVATIONS > 0 or any(
            job.running() and (key in GPU_JOBS or getattr(job, "uses_gpu", False)) for key, job in JOBS.items()
        )


def separation_running():
    """True while Restore audio removes music or echo, on the GPU in WSL."""
    with STATE_LOCK:
        job = JOBS.get("restore")
    return bool(job and job.running() and getattr(job, "uses_gpu", False))


def refuse_gpu_conflicts(starting):
    """Training, preparation, music removal and Compare epochs each need the GPU or the voice
    engine for a long time, and together they can run out of memory or stop each other."""
    if separation_running():
        raise RuntimeError(f"Removing music or echo is using the GPU. Start {starting} when it ends.")
    if job_running("compare"):
        raise RuntimeError(f"Compare epochs is using the voice engine. Start {starting} when it ends, or stop it.")


def job_running(key):
    with STATE_LOCK:
        job = JOBS.get(key)
        return bool(job and job.running())


def free_gpu_for(job_name):
    """Stop the voice before a GPU job. Both on one GPU can run out of memory, and the voice
    starts again when it is used."""
    if wsl.voice_runtime()["device"] != "cuda":
        return
    with STATE_LOCK:
        status = system.SYSTEM_STATE["status"]
    if status not in {"stopped", "stopping"}:
        log(f"Stopping Mic to Voice to free GPU memory for {job_name}.")
        system.stop_system()
    elif engine.api_ready() or (engine.API_PROCESS and engine.API_PROCESS.running()):
        log(f"Stopping the voice engine to free GPU memory for {job_name}.")
        engine.stop_api()


def jobs_now():
    with STATE_LOCK:
        return list(JOBS.items())
