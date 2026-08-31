import os
import sys
import json
import platform
import psutil
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REPORT = {}

def get_mac_hardware():
    hw = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "available_ram_gb": round(psutil.virtual_memory().available / 1e9, 2),
    }
    try:
        import torch
        hw["torch_version"] = torch.__version__
        hw["mps_built"] = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
        hw["mps_available"] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    except ImportError:
        hw["torch_version"] = "missing"
        hw["mps_built"] = False
        hw["mps_available"] = False
    return hw

def test_mps_tensors():
    import torch
    print("Testing MPS Tensor Allocation & Math Operations...")
    device = torch.device("mps")
    
    # 1. Allocation
    x = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    y = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    
    # 2. Matmul
    start = time.time()
    z = torch.matmul(x, y)
    torch.mps.synchronize()
    elapsed_ms = (time.time() - start) * 1000
    
    # 3. Check for NaNs or Infs
    has_nan = torch.isnan(z).any().item()
    has_inf = torch.isinf(z).any().item()
    
    return {
        "tensor_allocation": "PASS",
        "matmul_1024x1024_ms": round(elapsed_ms, 2),
        "nan_check": "PASS" if not has_nan else "FAIL",
        "inf_check": "PASS" if not has_inf else "FAIL",
        "status": "PASS" if (not has_nan and not has_inf) else "FAIL"
    }

def evaluate_model_feasibility(ram_gb):
    # Unified memory budget: Reserve ~4GB for macOS and background apps
    usable_vram = max(0, ram_gb - 4.0)
    
    candidates = [
        {"name": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "params_b": 0.5, "est_train_mem_gb": 3.2, "est_inference_mem_gb": 1.2},
        {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "params_b": 1.5, "est_train_mem_gb": 6.8, "est_inference_mem_gb": 3.4},
        {"name": "Qwen/Qwen2.5-Coder-3B-Instruct", "params_b": 3.0, "est_train_mem_gb": 13.5, "est_inference_mem_gb": 6.5},
        {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "params_b": 7.0, "est_train_mem_gb": 28.0, "est_inference_mem_gb": 14.5},
    ]
    
    eval_results = []
    for c in candidates:
        can_train = c["est_train_mem_gb"] <= usable_vram
        can_infer = c["est_inference_mem_gb"] <= usable_vram
        eval_results.append({
            "model": c["name"],
            "params": f"{c['params_b']}B",
            "inference_feasible": can_infer,
            "training_feasible": can_train,
            "estimated_train_memory_gb": c["est_train_mem_gb"],
            "estimated_infer_memory_gb": c["est_inference_mem_gb"]
        })
    return {
        "total_ram_gb": ram_gb,
        "estimated_usable_for_models_gb": round(usable_vram, 2),
        "candidates": eval_results
    }

def main():
    print("==================================================")
    print("      APPLE SILICON / MPS VALIDATION SUITE        ")
    print("==================================================")
    hw = get_mac_hardware()
    REPORT["hardware"] = hw
    print(f"OS: {hw['os']} {hw['release']} ({hw['machine']})")
    print(f"Total Unified Memory: {hw['ram_gb']} GB (Available: {hw['available_ram_gb']} GB)")
    print(f"PyTorch Version: {hw['torch_version']}")
    print(f"MPS Available: {hw['mps_available']}")
    
    if not hw["mps_available"]:
        REPORT["status"] = "BLOCKED"
        REPORT["reason"] = "No MPS backend available on this PyTorch build."
        print("\n❌ RESULT: BLOCKED - MPS is not available.")
    else:
        print("\n--- Phase 1: MPS Compute & Tensor Verification ---")
        try:
            tensor_res = test_mps_tensors()
            REPORT["tensor_benchmark"] = tensor_res
            print(f"✅ Tensor Allocation: {tensor_res['tensor_allocation']}")
            print(f"✅ Matmul Latency (1024x1024): {tensor_res['matmul_1024x1024_ms']} ms")
            print(f"✅ Numerical Integrity: {tensor_res['nan_check']}")
        except Exception as e:
            REPORT["tensor_benchmark"] = {"status": "FAIL", "error": str(e)}
            print(f"❌ Tensor Benchmark Failed: {e}")

        print("\n--- Phase 2: Memory Feasibility & Model Selection ---")
        feasibility = evaluate_model_feasibility(hw["ram_gb"])
        REPORT["memory_feasibility"] = feasibility
        print(f"Usable Budget for ML: ~{feasibility['estimated_usable_for_models_gb']} GB")
        print("\nModel Candidate Evaluation:")
        for c in feasibility["candidates"]:
            train_status = "✅ YES" if c["training_feasible"] else "❌ NO (OOM Risk)"
            infer_status = "✅ YES" if c["inference_feasible"] else "❌ NO (OOM Risk)"
            print(f" • {c['model']} ({c['params']}):")
            print(f"    - Inference Only: {infer_status} (~{c['estimated_infer_memory_gb']} GB)")
            print(f"    - Full GRPO RL Training: {train_status} (~{c['estimated_train_memory_gb']} GB)")
            
        REPORT["status"] = "COMPLETED"
        print("\n==================================================")
        print("✅ MPS VALIDATION FINISHED")
        print("Output saved to artifacts/mps_validation_report_raw.json")
        print("==================================================")
        
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/mps_validation_report_raw.json", "w") as f:
        json.dump(REPORT, f, indent=2)

if __name__ == '__main__':
    main()
