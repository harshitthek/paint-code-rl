# Kaggle T4x2 GPU Hardware Validation Guide

**Target Environment:** Kaggle Notebook with 2x NVIDIA Tesla T4 GPUs (16GB VRAM each).  
**Execution Mode:** `FREE` (Zero-cost GPU compute).

---

## 1. Prerequisites

* Kaggle account with GPU accelerator enabled.
* Notebook Accelerator: **GPU T4 x 2** selected.
* Internet access: **Enabled** (for initial pip / model weight download).

---

## 2. Notebook Execution Workflow

The canonical entrypoint notebook is located at:
```
notebooks/Phase0_Kaggle_Validation.ipynb
```

### Direct Notebook Execution Cells

```python
# Cell 1: Install core packages
!pip install -q torch==2.5.1 transformers==4.49.0 trl==0.15.1 peft bitsandbytes accelerate pydantic safetensors datasets Pillow pyyaml tenacity requests psutil

# Cell 2: Install system renderer dependencies
!apt-get update -qq && apt-get install -y -qq chromium-browser
!npm install -g puppeteer

# Cell 3: Clone repository and run the validation driver
!git clone https://github.com/harshitthek/paint-code-rl.git
%cd paint-code-rl
!git checkout v0.1.0-phase0

# Cell 4: Execute Automated Validation Driver
!python scripts/kaggle_validation_driver.py
```

---

## 3. What the Kaggle Driver Tests

The automated driver (`scripts/kaggle_validation_driver.py`) executes sequentially:
1. **Hardware Detection:** Probes 2x T4 GPUs (`cuda:0`, `cuda:1`), memory, compute capability.
2. **Software Verification:** Verifies PyTorch 2.5.1, Transformers 4.49.0, TRL 0.15.1 pinned versions.
3. **Puppeteer WebGL Headless:** Tests Chromium launch and WebGL rendering in headless container.
4. **Cost Safety:** Verifies `allow_external_apis == false` in FREE mode.
5. **Policy Model Feasibility:** Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct` in BF16 onto `cuda:0`.
6. **VLM Feasibility:** Tests `Qwen/Qwen2.5-VL-7B-Instruct` placement on `cuda:1`.
7. **GRPO Step:** Executes 1 full GRPO step with Group Size $G=2$ and $G=4$.
8. **Checkpoint Reload:** Re-loads saved LoRA adapter from `/kaggle/working/artifacts`.

---

## 4. Artifact Outputs

All outputs are saved to `/kaggle/working/artifacts/`:
* `validation_results.json` — Phase-by-phase status and timings
* `compute_capabilities.json` — GPU specs and memory allocation
* `model_selection.json` — Measured inference speeds and peak VRAM
* `grpo_g2/` & `grpo_g4/` — Checkpoint directories
