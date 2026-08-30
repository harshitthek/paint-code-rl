import platform
import json
import os
import psutil

def get_compute_capabilities():
    caps = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
    }
    
    try:
        import torch
        caps["torch_version"] = torch.__version__
        
        # CUDA Detection
        caps["cuda_available"] = torch.cuda.is_available()
        if caps["cuda_available"]:
            caps["gpu_count"] = torch.cuda.device_count()
            caps["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(caps["gpu_count"])]
            caps["vram_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2) for i in range(caps["gpu_count"])]
        
        # MPS Detection
        if hasattr(torch.backends, 'mps'):
            caps["mps_built"] = torch.backends.mps.is_built()
            caps["mps_available"] = torch.backends.mps.is_available()
        else:
            caps["mps_built"] = False
            caps["mps_available"] = False
            
    except ImportError:
        caps["cuda_available"] = False
        caps["mps_available"] = False
        caps["torch_version"] = "missing"
    
    return caps

if __name__ == "__main__":
    caps = get_compute_capabilities()
    print("=== Compute Capabilities ===")
    print(json.dumps(caps, indent=2))
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/compute_capabilities.json", "w") as f:
        json.dump(caps, f, indent=2)
