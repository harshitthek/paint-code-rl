# Training AI to Paint with Code (`paint-code-rl`)

> A research platform for training language models to generate executable generative art programs (`p5.js` / `p5.brush`) using reinforcement learning (GRPO), a 5-tier verifiable visual reward matrix, and interactive cyclic continuous training.

---

## 🌟 Project Status

```text
STATUS:
Phase-0 Research Prototype — Production Hardened & Multi-Signal Shaped

Test Coverage:
143/143 TESTS PASSING (100% GREEN)
  - 104 Python Unit, Integration, Hardening & Telemetry Tests (pytest)
  - 8 Node.js Sandbox Security Smoke Tests (npm test)
  - 10 WebGL / p5.brush Visual Corpus Tests (node renderer/test_corpus.js)
  - 21 WebGL Adversarial Stress Tests (node renderer/test_adversarial_corpus.js)

Hardware Targets:
  - Apple Silicon M4 (16GB+): Physical MPS & Metal WebGL validation verified
  - Kaggle Dual GPU (2x Tesla T4): Multi-GPU cloud driver ready (notebooks/kaggle_paint_rl.ipynb)
  - Local CPU / CI: Verified 100% functional fallback
```

> [!IMPORTANT]
> **Scientific Integrity Notice:** The causal claim that pairwise visual judging solves mode collapse in code-based art generation has **not** been independently established by this repository yet. This platform is the experimental instrument constructed to rigorously test that hypothesis.

---

## 🏛 Architecture Overview

```mermaid
flowchart TD
    Prompt[Art Prompt from Dataset] --> Policy[Policy Model: Qwen2.5-Coder]
    Policy --> GenCode[Executable p5.js / p5.brush Code]
    GenCode --> BatchServer[Concurrent WebGL Batch Sandbox\nPOST /render_batch]
    BatchServer --> InMemStream[In-Memory Base64 Image Stream]
    
    InMemStream --> RewardEng[Hierarchical Reward Engine]
    Prompt --> RewardEng
    GenCode --> RewardEng
    
    subgraph Multi-Signal 5-Tier Reward Matrix
        CompScore[1. Compile Gate: 0.10\nExact Error Classification]
        PromptScore[2. Semantic Prompt CLIP: 0.35\nDifferential Cosine Sim]
        RichScore[3. Visual Richness: 0.25\nCoverage + Std + Palette + Laplacian]
        BrushScore[4. Brush Utilization: 0.15\np5.brush Natural Media + Anti-Cheat]
        AesScore[5. Global Aesthetic: 0.15\nComposition Harmony]
    end
    
    RewardEng --> CompScore
    RewardEng --> PromptScore
    RewardEng --> RichScore
    RewardEng --> BrushScore
    RewardEng --> AesScore
    
    CompScore --> TotalReward[Linear Weighted Bundle]
    PromptScore --> TotalReward
    RichScore --> TotalReward
    BrushScore --> TotalReward
    AesScore --> TotalReward
    
    TotalReward --> GRPOTrainer[TRL GRPOTrainer Engine]
    GRPOTrainer --> PolicyUpdate[LoRA Parameter Optimization]
    GRPOTrainer --> Scorecard[Diagnostic Scorecard Output]
    GRPOTrainer --> LiveDash[Live Auto-Refreshing HTML Dashboard]
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshitthek/paint-code-rl.git
cd paint-code-rl
```

### 2. Environment Setup
```bash
# Python Virtual Environment (3.10 or 3.11)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Node.js Renderer Runtime
cd renderer
npm install
cd ..
```

### 3. Run Automated Tests
```bash
# Full test suite (78 pytest tests: unit, rewards, edge cases, XSS protection)
python -m pytest tests/ -v

# Renderer security smoke tests and 10-sketch visual corpus (18 tests)
cd renderer
npm test
node test_corpus.js
cd ..
```

---

## 🎨 Interactive Cyclic Continuous Training

Run reinforcement learning in discrete, human-in-the-loop cycles with live diagnostic scorecards and hardware resource saturation:

```bash
# 1. Interactive cyclic continuous training with live dashboard & saturated hardware
python scripts/train_grpo.py --mode train --steps-per-cycle 25 --max --dashboard

# 2. Unattended continuous training (Kaggle / Cloud GPU headless mode)
python scripts/train_grpo.py --mode train --steps-per-cycle 50 --max-steps 500 --unattended --max

# 3. 1-Step hardware validation sanity check
python scripts/train_grpo.py --mode one_step
```

### Key CLI Flags:
* `--steps-per-cycle <int>`: Number of training steps executed per interactive feedback cycle (default: 25).
* `--max`: Automatically probes system RAM, CPU cores, and GPU VRAM, saturating thread pools and expanding group sizes ($G=4 \rightarrow 6/8$) and token budgets.
* `--dashboard`: Launches and continuously updates `artifacts/dashboard.html` with real-time Chart.js loss/temperature curves and rendered artwork gallery.
* `--unattended`: Runs continuously without interactive prompts for headless cloud runs.

---

## 📊 Live HTML Visual Dashboard

When `--dashboard` is enabled during training, open `artifacts/dashboard.html` in any browser:
* **Real-time Trajectories:** Visualizes training loss alongside dynamic exponential temperature annealing ($T=0.85 \rightarrow 0.55$).
* **Artwork Gallery:** Inspect latest generated artworks, reward breakdowns, and p5.js source code.
* **Auto-Refresh:** Updates automatically every 10 seconds.

---

## 📚 Complete Documentation Index

### Architecture & System Design
- [System Architecture Specification](docs/architecture/SYSTEM_ARCHITECTURE.md) — Comprehensive end-to-end subsystem specifications and dataflows.
- [Adversarial Red-Team & Architecture Synthesis](docs/research/ARCHITECTURAL_REDTEAM_AND_SYNTHESIS.md) — Analysis of 5 naive anti-patterns and battle-tested solutions.
- [Test Status & Verification Report](TEST_STATUS.md) — 96-test verification breakdown.
- [Final Implementation Status](FINAL_IMPLEMENTATION_STATUS.md) — Component status and verification matrix.

### Architecture Decision Records (ADRs)
- [ADR-001: Headless Chromium WebGL Architecture](docs/decisions/ADR-001-puppeteer-webgl.md)
- [ADR-002: Managing Stochasticity and GPU Seeding](docs/decisions/ADR-002-stochasticity.md)
- [ADR-003: Standalone Lightweight Implementation (De-vendor SOUP)](docs/decisions/ADR-003-soup-dependency.md)
- [ADR-004: Execution Modes and Cost Safety Guarantees](docs/decisions/ADR-004-execution-modes.md)
- [ADR-005: Pinned TRL & HuggingFace Dependencies](docs/decisions/ADR-005-trl-pin.md)
- [ADR-006: Deferred In-Memory Browser Page Pooling](docs/decisions/ADR-006-deferred-refresh.md)
- [ADR-007: Deferred Multi-Node Distributed Training](docs/decisions/ADR-007-deferred-distributed.md)
- [ADR-008: Multi-Signal Visual RL and Ephemeral Sandboxing](docs/decisions/ADR-008-multimodal-visual-rl-and-sandboxing.md)
- [ADR-009: Interactive Cyclic Training, Scorecards & Hardware Saturation](docs/decisions/ADR-009-interactive-cyclic-training-and-hardware-saturation.md)

### User & Deployment Guides
- [User Guide: Interactive Cyclic Training & Scorecards](docs/user_guide/INTERACTIVE_CYCLIC_TRAINING.md) — Step-by-step training and scorecard guide.
- [Apple Silicon M4 Deployment Guide](docs/deployment/M4_MPS.md) — Native Metal ANGLE acceleration and MPS memory management.
- [Kaggle 2x T4 GPU Deployment Guide](docs/deployment/KAGGLE.md) — Zero-cost cloud driver setup and Jupyter notebook workflow.
- [Local CPU Development Guide](docs/deployment/CPU.md) — Fast unit testing and functional validation.
- [Cloud GPU Deployment Guide](docs/deployment/PAID_CLOUD.md) — Docker and high-throughput A100 setups.

---

## ⚖ License

MIT License. See [LICENSE](LICENSE) for details.
