# CPU Development & Smoke Testing Deployment Guide

**Target Environment:** Local workstation (macOS / Linux / Windows) with CPU only.  
**Use Case:** Rapid unit testing, renderer verification, and small-model (0.5B) sanity checks.

---

## 1. Prerequisites

* **Python:** `3.10` or `3.11`
* **Node.js:** `>=18.0.0`
* **RAM:** Minimum 8 GB

---

## 2. Installation & Setup

```bash
# 1. Clone or checkout the frozen release
git clone https://github.com/harshitthek/paint-code-rl.git
cd paint-code-rl
git checkout v0.1.0-phase0

# 2. Install Python dependencies in virtualenv
python3 -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# 3. Install Node renderer dependencies
cd renderer
npm install
cd ..
```

---

## 3. Validation Sequence

### Step 1: Probe Capabilities
```bash
python scripts/benchmark.py
```
*Expected Output:* JSON output showing detected CPU cores and RAM in `artifacts/compute_capabilities.json`.

### Step 2: Run Full Pytest Suite
```bash
python -m pytest tests/ -v
```
*Expected Output:* `35 passed in ~5s`.

### Step 3: Run Renderer Security & Corpus Tests
```bash
cd renderer
node test_security.js
node test_corpus.js
cd ..
```
*Expected Output:* All 4 security smoke tests PASS, and all 10 corpus sketches PASS.

### Step 4: 1-Step Small-Model GRPO Sanity Check
```bash
python scripts/train_grpo.py --mode one_step
```
*Expected Output:* Auto-selects `Qwen/Qwen2.5-Coder-0.5B-Instruct` on CPU, executes 1 GRPO step, outputs metrics and saves adapter to `artifacts/checkpoints/step_1_test`.

---

## 4. Artifact Locations

* `artifacts/compute_capabilities.json` — Hardware capability dump
* `artifacts/checkpoints/step_1_test/` — LoRA adapter checkpoint
* `artifacts/renders/` — Generated canvas PNG outputs

---

## 5. Troubleshooting

* **Renderer fails to launch:** Ensure Chromium or Chrome is installed. On Linux, ensure `libgbm`, `libnss3`, `libasound2` are present (`apt-get install -y chromium-browser`).
* **PyTorch SDPA Warning:** Harmless warning regarding Sliding Window Attention on SDPA. Suppress with `export TOKENIZERS_PARALLELISM=false`.
