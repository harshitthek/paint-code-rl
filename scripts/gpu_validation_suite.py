import os
import json
import time
import platform
import subprocess

REPORT = {}

def run_phase_a():
    print("PHASE A: Hardware & Environment")
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda else 0
        REPORT["phase_a"] = {
            "os": platform.system(),
            "cpu": platform.processor(),
            "cuda_available": cuda,
            "gpu_count": gpu_count,
            "torch_version": torch.__version__,
        }
        if cuda:
            REPORT["phase_a"]["gpus"] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            REPORT["phase_a"]["vram_gb"] = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(gpu_count)]
    except ImportError:
        REPORT["phase_a"] = {"error": "PyTorch not installed"}

def main():
    run_phase_a()
    
    cuda_available = REPORT.get("phase_a", {}).get("cuda_available", False)
    
    if not cuda_available:
        print("BLOCKED: No CUDA GPU available. Cannot proceed with Phases B-O.")
        REPORT["final_status"] = "BLOCKED"
        REPORT["block_reason"] = "No GPU detected in current environment."
    else:
        # We would run real GPU tests here
        pass
        
    with open("artifacts/gpu_validation_report_raw.json", "w") as f:
        json.dump(REPORT, f, indent=2)

if __name__ == '__main__':
    os.makedirs("artifacts", exist_ok=True)
    main()
