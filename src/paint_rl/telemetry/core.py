import uuid
import platform
import subprocess
import json
import os
import sys

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def get_hardware_fingerprint():
    fingerprint = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    try:
        import torch
        fingerprint["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            fingerprint["gpu_count"] = torch.cuda.device_count()
            fingerprint["gpu_name"] = torch.cuda.get_device_name(0)
    except:
        fingerprint["cuda_available"] = False
    return fingerprint

class ExperimentLogger:
    STATES = ["CREATED", "RUNNING", "PAUSED", "COMPLETED", "FAILED", "INTERRUPTED", "INVALIDATED"]

    def __init__(self, config_hash: str):
        self.run_id = f"run_{uuid.uuid4().hex[:8]}"
        self.config_hash = config_hash
        self.commit = get_git_commit()
        self.fingerprint = get_hardware_fingerprint()
        self.state = "CREATED"
        os.makedirs("artifacts/logs", exist_ok=True)
        self.log_file = os.path.join("artifacts/logs", f"{self.run_id}.jsonl")
        
        # In a real system, we'd log this to JSON Lines or a DB.
        print(f"[{self.run_id}] Experiment started. Commit: {self.commit} | Config Hash: {self.config_hash}")
        self._write_log({"event": "init", "commit": self.commit, "hash": self.config_hash})
        
    def log_step(self, step: int, metrics: dict):
        # Structured logging
        payload = {
            "run_id": self.run_id,
            "step": step,
            **metrics
        }
        print(f"STEP_METRICS: {json.dumps(payload)}")
        self._write_log(payload)

    def log_error(self, error_class: str, details: str):
        payload = {
            "run_id": self.run_id,
            "error_class": error_class,
            "details": details
        }
        print(f"ERROR: {json.dumps(payload)}", file=sys.stderr)
        self._write_log(payload)
        self.set_state("FAILED")

    def set_state(self, new_state: str):
        if new_state in self.STATES:
            self.state = new_state
            self._write_log({"event": "state_change", "state": new_state})
    
    def _write_log(self, data: dict):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
