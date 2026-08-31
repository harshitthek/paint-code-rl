# Apple Silicon M4 (MPS) Hardware Validation Guide

**Target Environment:** Apple Silicon M4 (16GB Unified RAM, macOS 15 Sequoia).  
**Execution Mode:** `LOCAL` (Zero-cost, external APIs blocked by default).

---

## 1. Prerequisites

* **Hardware:** Apple M4, 16 GB unified memory
* **OS:** macOS 15+ (Darwin arm64)
* **Python:** 3.11 with PyTorch 2.5.1 compiled with MPS support (`torch.backends.mps.is_available() == True`)
* **Node.js:** `>=18.0.0`

---

## 2. Installation & Setup

```bash
# 1. Fetch latest frozen release
cd paint-code-rl
git fetch --tags
git checkout v0.1.0-phase0

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Ensure dependencies are synced
pip install -e ".[dev]"

# 4. Install renderer dependencies
cd renderer && npm install && cd ..
```

---

## 3. Physical Validation Sequence

### Step 1: MPS Hardware & Tensor Benchmark
```bash
python scripts/mps_validation_suite.py
```
*Expected Output:*
* Allocates 1024x1024 FP32 tensors on `mps`
* Matmul latency < 50ms
* Checks for NaNs / Infs (Status: PASS)
* Memory budget evaluation saved to `artifacts/mps_validation_report_raw.json`

### Step 2: Baseline Generation & WebGL Render Test
```bash
python scripts/generate_baseline.py --num-samples 3 --output-dir artifacts/baseline_m4
```
*Expected Output:*
* Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct` in FP16 onto `mps`
* Starts Puppeteer WebGL renderer daemon on port 3000
* Generates 3 p5.js scripts using the few-shot template
* Renders canvases to `artifacts/baseline_m4/`
* Produces run manifest `artifacts/baseline_m4/baseline_*_manifest.json`

### Step 3: 1-Step Local GRPO Training
```bash
python scripts/train_grpo.py --mode one_step
```
*Expected Output:*
* Auto-selects `Qwen/Qwen2.5-Coder-1.5B-Instruct`
* Initializes LoRA adapter on `mps`
* Evaluates syntax and render validity reward functions
* Executes 1 physical optimization step (forward -> reward -> loss -> backward -> optimizer step)
* Validates and saves adapter to `artifacts/checkpoints/step_1_test/final_adapter/`

---

## 4. Troubleshooting & Memory Management

* **Zombie Node processes:** If port 3000 is occupied by a stale daemon, run:
  ```bash
  killall node 2>/dev/null
  ```
* **Memory Pressure / OOM:**
  * Usable ML budget on 16GB Mac is ~13.2 GB after OS overhead.
  * `Qwen2.5-Coder-1.5B` consumes ~6.8 GB under training.
  * Close heavy browser tabs or Docker before running GRPO.
