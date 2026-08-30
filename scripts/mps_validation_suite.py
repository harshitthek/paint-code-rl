import os
import json
import torch
import platform
import psutil

REPORT = {}

def get_mac_hardware():
    return {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "mps_built": torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    }

def main():
    hw = get_mac_hardware()
    REPORT["hardware"] = hw
    
    if not hw["mps_available"]:
        REPORT["status"] = "BLOCKED"
        REPORT["reason"] = "No MPS backend available on this machine."
    else:
        # Placeholder for actual tensor tests
        pass
        
    with open("artifacts/mps_validation_report_raw.json", "w") as f:
        json.dump(REPORT, f, indent=2)

if __name__ == '__main__':
    os.makedirs("artifacts", exist_ok=True)
    main()
