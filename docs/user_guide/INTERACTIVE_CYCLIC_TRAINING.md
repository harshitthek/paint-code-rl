# User Guide: Interactive Cyclic Training, Scorecards & Telemetry

> Step-by-step guide for configuring, running, and monitoring continuous reinforcement learning on `paint-code-rl`.

---

## 1. Quick Start Commands

### A. Local Interactive Training with Dashboard & Saturated Hardware
```bash
python scripts/train_grpo.py --mode train --steps-per-cycle 25 --max --dashboard
```

### B. Unattended Headless Training (Kaggle / Cloud GPU)
```bash
python scripts/train_grpo.py --mode train --steps-per-cycle 50 --max-steps 500 --unattended --max
```

### C. 1-Step Hardware Sanity Check
```bash
python scripts/train_grpo.py --mode one_step
```

---

## 2. CLI Options Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `str` | `one_step` | Execution mode: `one_step` (hardware check) or `train` (cyclic continuous). |
| `--steps-per-cycle` | `int` | `25` | Number of training steps executed per interactive feedback cycle. |
| `--max-steps` | `int` | `None` | Total training step budget (stops automatically when reached). |
| `--unattended` | `flag` | `False` | Disables interactive terminal prompts; runs autonomously until `--max-steps`. |
| `--max` | `flag` | `False` | Probes system RAM, CPU cores, and GPU VRAM to maximize thread pools and group sizes ($G=4 \rightarrow 6/8$). |
| `--dashboard` | `flag` | `False` | Generates and updates `artifacts/dashboard.html` in real-time. |
| `--checkpoint-dir` | `str` | `artifacts/checkpoints` | Target directory for saving LoRA adapter weights. |
| `--no-renderer` | `flag` | `False` | Runs syntax-only rewards without launching the headless WebGL renderer daemon. |

---

## 3. Interactive Cyclic Training Loop

When running in interactive mode (`--mode train` without `--unattended`), the training loop pauses at the end of each cycle and presents the prompt:

```text
============================================================
  CYCLE 1 Complete (Steps 1-25)
  Loss: 0.2842 | Temperature: 0.812
============================================================

  Continue training? [y (1 cycle) / n (save & quit) / <number> (N cycles)]: 
```

### User Input Options:
- **`y` or `[Enter]`**: Executes exactly **1 more cycle** and prompts again.
- **`n` or `quit`**: Safely saves the LoRA adapter checkpoint to `artifacts/checkpoints/cyclic_run/final_adapter/`, validates the `adapter_model.safetensors` file, and exits.
- **`<number>` (e.g. `5` or `10`)**: Executes **$N$ cycles autonomously** without stopping, then prompts when the batch completes.

---

## 4. Reading Diagnostic Scorecards

Between steps and cycles, the reward engine emits an explainable **Diagnostic Scorecard**:

```text
============================================================
  DIAGNOSTIC SCORECARD
  Prompt: Paint a field of wildflowers with soft watercolor washes...
============================================================
  [GOOD] COMPILE              raw=1.000  weighted=0.100
  [GOOD] PROMPT_ALIGNMENT     raw=0.820  weighted=0.287
         [GOOD] Strong semantic alignment with prompt (CLIP sim: +0.284)
  [EXCELLENT] VISUAL_RICHNESS raw=0.880  weighted=0.220
         [EXCELLENT] Balanced coverage (42%) | Rich palette (8 hues) | Sharp edges (var 312.4)
  [GOOD] BRUSH_UTILIZATION    raw=0.850  weighted=0.128
         [GOOD] brush.scaleBrushes() called | [GOOD] Natural media fills detected
  [GOOD] AESTHETIC            raw=0.740  weighted=0.111
------------------------------------------------------------
  [EXCELLENT] TOTAL REWARD: 0.8460
============================================================
```

### Scorecard Quality Tiers:
- `[EXCELLENT]` ($\ge 0.70$): Exceptional composition, balanced negative space, rich watercolor washes, and sharp brush edges.
- `[GOOD]` ($0.50 - 0.69$): Valid artistic program with good execution and moderate texture richness.
- `[MEDIOCRE]` ($0.30 - 0.49$): Compiles and renders, but may lack brush variety or have sparse coverage.
- `[FAIL]` ($< 0.30$ or error): Blank canvas, timeout, syntax error, or anti-cheat text penalty.

---

## 5. Live Auto-Refreshing HTML Dashboard

When `--dashboard` is enabled, open `artifacts/dashboard.html` in your web browser:

```bash
# macOS
open artifacts/dashboard.html

# Linux
xdg-open artifacts/dashboard.html

# Windows
start artifacts/dashboard.html
```

### Features:
1. **Auto-Refresh:** Automatically refreshes every 10 seconds via meta refresh.
2. **Chart.js Trajectories:** Live interactive curve plotting training loss against exponential temperature decay.
3. **Artwork Gallery:** Shows the latest rendered artworks side-by-side with their prompt, reward score, diagnostic scorecard, and expandable p5.js source code.
4. **XSS Protection:** Full HTML entity escaping protects against code injection.
