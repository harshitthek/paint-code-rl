# ADR-009: Interactive Cyclic Continuous Training, Explainable Scorecards & Hardware Saturation

**Status:** Accepted  
**Date:** 2026-09-02  
**Deciders:** Core Engineering & Research Team  

---

## Context & Motivation

Initial reinforcement learning iterations for paint code generation operated with two significant workflow bottlenecks:

1. **Monolithic Training Runs with Black-Box Rewards:**  
   Training executed in rigid batch runs where failure modes (e.g. blank canvases, blurry compositions, text-in-canvas exploits) were only visible as scalar reward drops without human-interpretable explanations.
2. **Disk I/O and Sequential Rendering Bottlenecks:**  
   Writing PNG files to disk for every completion candidate and rendering GRPO group items sequentially ($G=4$) introduced 4–6 seconds of latency per step.
3. **Static Resource Allocations:**  
   Hardware configurations on Apple Silicon M4 (16GB unified memory) and multi-GPU cloud instances (Kaggle 2x T4 / A100) were under-utilizing physical compute cores, memory bandwidth, and group generation capacity.

---

## Decisions

### 1. In-Memory Base64 Streaming & Batch Rendering Endpoint
* **Base64 Streaming:** [`renderer/sandbox.js`](renderer/sandbox.js) supports `options.return_base64`, capturing screenshots directly into memory and bypassing disk I/O during RL training.
* **Concurrent Batching:** [`renderer/server.js`](renderer/server.js) provides `POST /render_batch`, utilizing `Promise.all` across isolated browser contexts to render entire GRPO generation groups concurrently.

### 2. Interactive Cyclic Continuous Training Engine
* Implemented `PaintGRPOTrainer.train_cyclic()` in [`src/paint_rl/trainer/grpo.py`](src/paint_rl/trainer/grpo.py).
* Retains model, tokenizer, and optimizer permanently in GPU/MPS memory across cycles to eliminate reloading latency.
* Prompts user between cycles: `[y (1 cycle) / n (save & quit) / <number> (N cycles)]`. Supports `--unattended` for headless cloud execution.

### 3. Closed-Form Laplacian Edge Variance & Entropy Metrics
* Added 3x3 Laplacian edge variance filter ($\nabla^2 I$) in [`src/paint_rl/rewards/aesthetic.py`](src/paint_rl/rewards/aesthetic.py) to reward sharp textures without VRAM overhead.
* Combined with 4-bit quantized color histogram entropy and active canvas coverage ratios.

### 4. Explainable Diagnostic Scorecards
* Structured critiques generated for every component:
  * `Execution Gate:` `[PASS] Valid WebGL canvas rendered` vs `[FAIL] Timeout / Syntax error`.
  * `Semantic Alignment:` `[GOOD] Strong semantic match (CLIP sim +0.28)` vs `[POOR] Weak match`.
  * `Visual Richness:` `[EXCELLENT] Rich color palette (8 hues) & sharp texture` vs `[CRITIQUE] Canvas empty/flat`.
  * `Natural Media:` `[GOOD] Natural media fills detected (wash/fill/bleed)` vs `[CRITIQUE] Missing brush.scaleBrushes()`.
  * `Anti-Cheat:` `[CLEAN] Valid procedural art` vs `[CHEAT DETECTED] Text primitive text() attempted`.
* Aggregated via `RewardComposer.generate_scorecard()`.

### 5. Dynamic Resource Saturation (`--max`)
* [`src/paint_rl/config/core.py`](src/paint_rl/config/core.py) provides `apply_max_hardware_config()` which probes system RAM, CPU cores, and GPU VRAM, saturating thread pools and expanding `group_size` ($G=4 \rightarrow 6/8$) and token budgets.

### 6. Live Auto-Refreshing HTML Visual Dashboard
* [`src/paint_rl/telemetry/dashboard.py`](src/paint_rl/telemetry/dashboard.py) generates `artifacts/dashboard.html` with Chart.js loss & temperature curves, live artwork gallery, and full XSS entity escaping.

### 7. Dynamic Temperature Annealing
* Exponential annealing schedule $T(\text{step}) = 0.55 + 0.30 \cdot \exp(-\text{step} / 100)$ transitioning from exploration to precision.

---

## Consequences & Verification

- **Group Render Latency:** Reduced from $4 \times 1.5\text{s} = 6.0\text{s}$ down to $<300\text{ms}$.
- **Diagnostic Transparency:** Engineers and researchers can immediately inspect scorecards in the terminal and live dashboard.
- **Hardware Throughput:** Squeezes maximum memory bandwidth and GPU compute on Apple Silicon M4 and Kaggle 2x T4 instances.
- **Verification:** 100% test pass rate across 96 automated tests (78 Python + 18 Node.js).
