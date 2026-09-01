# Apple Silicon M4 (MPS) Hardware Validation & Training Guide

**Target Environment:** Apple Silicon M4 (16GB Unified RAM, macOS 15+ Sequoia).  
**Execution Mode:** `LOCAL` (Zero-cost, external APIs blocked by default).

---

## 1. Prerequisites

* **Hardware:** Apple Silicon M4 (or M3/M2 Pro), 16 GB unified memory.
* **OS:** macOS 15+ (Darwin arm64).
* **Python:** 3.11 with PyTorch 2.5.1+ compiled with MPS support (`torch.backends.mps.is_available() == True`).
* **Node.js:** `>=18.0.0` (with Chromium / Puppeteer).

---

## 2. Fast One-Click Training Launcher

We provide an automated, one-click launcher that sets all Apple Silicon memory watermarks, launches the Metal-accelerated WebGL daemon, and runs GRPO policy training:

```bash
# Make executable and launch
chmod +x scripts/run_m4.sh
./scripts/run_m4.sh
```

What `scripts/run_m4.sh` executes automatically:
1. Configures `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85` (prevents macOS OOM kills).
2. Configures `PYTORCH_ENABLE_MPS_FALLBACK=1` (routes non-MPS ops to CPU without crashing).
3. Starts the background Node.js WebGL renderer with native `--use-angle=metal` hardware acceleration.
4. Trains `Qwen2.5-Coder-1.5B-Instruct` via GRPO using Group Size $G=4$ and the 5-tier visual reward matrix.
5. Validates and saves final LoRA adapter checkpoints to `artifacts/checkpoints/`.

---

## 3. Step-by-Step Manual Validation Sequence

### Step 1: MPS Hardware & Tensor Benchmark
```bash
python scripts/mps_validation_suite.py
```
*Expected Output:*
* Allocates 1024x1024 FP32 tensors on `mps`.
* Matmul latency < 50ms.
* Checks for NaNs / Infs (Status: PASS).
* Generates report `artifacts/mps_validation_report_raw.json`.

### Step 2: Baseline Generation & Metal WebGL Render Test
```bash
python scripts/generate_baseline.py --num-samples 3 --output-dir artifacts/baseline_m4
```
*Expected Output:*
* Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct` in FP32/FP16 onto `mps`.
* Starts Puppeteer WebGL renderer with native Metal ANGLE backend.
* Generates 3 p5.js generative art scripts using few-shot templates.
* Renders canvases to `artifacts/baseline_m4/` in sub-180ms per canvas.

### Step 3: 1-Step Local GRPO Training
```bash
python scripts/train_grpo.py --mode one_step
```
*Expected Output:*
* Auto-selects `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
* Configures standard Causal SDPA attention without sliding window warnings.
* Evaluates 5-tier verifiable visual rewards (`Compile`, `PromptAlignment`, `VisualRichness`, `BrushUtilization`, `Aesthetic`).
* Executes 1 physical optimization step (forward $\rightarrow$ rewards $\rightarrow$ advantages $\rightarrow$ loss $\rightarrow$ backward $\rightarrow$ optimizer step).
* Validates and saves adapter to `artifacts/checkpoints/step_1_test/final_adapter/adapter_model.safetensors`.

---

### Step 4: Interactive Cyclic Training with Live Dashboard
```bash
# Squeeze maximum throughput on Apple Silicon M4 with live telemetry
python scripts/train_grpo.py --mode train --steps-per-cycle 25 --max --dashboard
```
* Saturated hardware `--max` utilizes unified memory bandwidth with $G=4 \rightarrow 6$ group generations.
* Emits human-readable **Diagnostic Scorecards** between cycles explaining visual texture and brush structure.
* Launches `artifacts/dashboard.html` for real-time loss and temperature curves.

---

## 4. Troubleshooting & Memory Management

* **Stale Renderer Daemons:** If port 3000 is occupied by a previous background daemon, kill it with:
  ```bash
  killall node 2>/dev/null || true
  ```
* **Memory Pressure on 16GB Mac:**
  * Usable ML budget on a 16GB Mac is ~13.2 GB after macOS window server allocations.
  * `Qwen2.5-Coder-1.5B` under LoRA training consumes ~6.5 GB.
  * Close heavy electron apps or Docker before starting long training runs.

---

## 5. Related Guides & References
- [User Guide: Interactive Cyclic Training](file:///docs/user_guide/INTERACTIVE_CYCLIC_TRAINING.md)
- [System Architecture Specification](file:///docs/architecture/SYSTEM_ARCHITECTURE.md)
- [ADR-009: Interactive Cyclic Training & Hardware Saturation](file:///docs/decisions/ADR-009-interactive-cyclic-training-and-hardware-saturation.md)
