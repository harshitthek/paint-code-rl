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
    def __init__(self, config_hash: str):
        self.run_id = f"run_{uuid.uuid4().hex[:8]}"
        self.config_hash = config_hash
        self.commit = get_git_commit()
        self.fingerprint = get_hardware_fingerprint()
        
        # In a real system, we'd log this to JSON Lines or a DB.
        print(f"[{self.run_id}] Experiment started. Commit: {self.commit} | Config Hash: {self.config_hash}")
        
    def log_step(self, step: int, metrics: dict):
        # Structured logging
        payload = {
            "run_id": self.run_id,
            "step": step,
            **metrics
        }
        print(f"STEP_METRICS: {json.dumps(payload)}")

    def log_error(self, error_class: str, details: str):
        payload = {
            "run_id": self.run_id,
            "error_class": error_class,
            "details": details
        }
        print(f"ERROR: {json.dumps(payload)}", file=sys.stderr)
