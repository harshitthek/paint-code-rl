# Kaggle GPU Training & Hardware Validation Guide

**Target Environment:** Kaggle Notebook with 2x NVIDIA Tesla T4 GPUs (16GB VRAM each) or 1x P100 (16GB VRAM).  
**Execution Mode:** `FREE` (Zero-cost GPU compute).

---

## 1. Prerequisites

* Kaggle account with GPU accelerator enabled (Free 30h/week).
* Notebook Accelerator: **GPU T4 x 2** (or P100) selected.
* Internet access: **Enabled** (for pip / model weight downloads).

---

## 2. One-Click Notebook Execution

The self-contained, canonical Kaggle notebook is located at:
👉 [`notebooks/kaggle_paint_rl.ipynb`](../../notebooks/kaggle_paint_rl.ipynb)

### How to Run:
1. Open [Kaggle Notebooks](https://www.kaggle.com/code) $\rightarrow$ Click **New Notebook**.
2. Go to **File** $\rightarrow$ **Import Notebook** $\rightarrow$ Upload `notebooks/kaggle_paint_rl.ipynb`.
3. Set **Settings** $\rightarrow$ **Accelerator** $\rightarrow$ **GPU T4 x 2** (or GPU P100).
4. Turn on **Internet**.
5. Click **Run All**.

---

## 3. What Happens During Training

1. **Environment Setup:** Installs Chromium, Node.js, and PyTorch dependencies.
2. **WebGL Daemon:** Starts the headless Node.js renderer on port 3000 in the background.
3. **GRPO Policy Training:** 
   * Loads `Qwen2.5-Coder-1.5B-Instruct` in FP16 on `cuda:0` with standard Causal SDPA.
   * Generates groups of $G=4$ completions per prompt.
   * Computes the 5-tier visual reward matrix (`Compile`, `PromptAlignment`, `VisualRichness`, `BrushUtilization`, `Aesthetic`).
   * Optimizes LoRA adapter weights via GRPO backprop.
4. **Interactive Artwork Showcase:** Automatically renders generated p5.brush watercolors and displays them inline in the notebook.

---

## 4. Artifact Outputs

All training outputs are persisted to `/kaggle/working/artifacts/`:
* `artifacts/checkpoints/` — Saved LoRA adapter safetensors checkpoints
* `artifacts/renders/` — Rendered high-resolution PNG artworks
* `artifacts/logs/` — JSONL experiment telemetry and loss/reward trajectories
* `artifacts/dashboard.html` — Live HTML visual dashboard with Chart.js curves

---

## 5. Running Unattended Cyclic Training via Terminal / Kaggle Script
```bash
python scripts/train_grpo.py --mode train --steps-per-cycle 50 --max-steps 500 --unattended --max --dashboard
```

### Publishing Model Checkpoints to Hugging Face
To publish checkpoints to Hugging Face from Kaggle, save your token in **Add-ons $\rightarrow$ Secrets** as `HF_TOKEN`, then run:
```bash
python scripts/upload_model.py \
  --destination hf \
  --repo-id YOUR_HF_USERNAME/paint-code-rl-lora \
  --token "$HF_TOKEN"
```

---

## 6. Related Documentation
- [User Guide: Interactive Cyclic Training](../../docs/user_guide/INTERACTIVE_CYCLIC_TRAINING.md)
- [System Architecture Specification](../../docs/architecture/SYSTEM_ARCHITECTURE.md)
- [ADR-009: Interactive Cyclic Training & Hardware Saturation](../../docs/decisions/ADR-009-interactive-cyclic-training-and-hardware-saturation.md)
