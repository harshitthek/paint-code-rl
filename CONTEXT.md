# System Context & Architecture Guide

## 1. Scientific Overview & Objective
**Paint-Code-RL** is a research platform designed to train Large Language Models to generate executable generative art programs using **p5.js** and **p5.brush** via **Group Relative Policy Optimization (GRPO)** and visual preference rewards.

The core research question investigates whether reinforcement learning with multi-modal visual feedback (aesthetic scoring + pairwise VLM preference judging) can guide code-generation models to produce high-aesthetic visual art without mode collapse.

---

## 2. Core Scientific & Technical Architecture

`	ext
┌─────────────────────────────────────────────────────────────┐
│                       Prompt Corpus                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Policy Model (Qwen)                      │
│      • 1.5B on Apple Silicon MPS / Local Development        │
│      • 7.0B on Multi-GPU CUDA (Kaggle T4x2 / Cloud)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Rollout Engine                        │
│            Generates G candidate p5.js programs             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                Browser Sandbox (Renderer)                   │
│        • Node.js + Express + Puppeteer + Chromium           │
│        • Hardware-accelerated WebGL canvas rendering        │
│        • Sandboxed execution & timeout protection           │
└──────────────────────────────┬──────────────────────────────┘
                               │ (PNG Images)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Reward Engine                         │
│  ┌───────────────────────┬───────────────────────────────┐  │
│  │ 1. Compile / Syntax   │ Execution validity & canvas   │  │
│  ├───────────────────────┼───────────────────────────────┤  │
│  │ 2. Aesthetic (HPSv3)  │ Human Preference Score v3     │  │
│  ├───────────────────────┼───────────────────────────────┤  │
│  │ 3. Pairwise VLM Judge │ Direction-invariant VLM       │  │
│  │    (Alignment)        │ comparison with reference pool│  │
│  └───────────────────────┴───────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Total Scalar Reward
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               GRPO Trainer (TRL 0.15.1 Core)                │
│       • Advantage normalization across group rollouts       │
│       • Policy gradient update with reference KL defense    │
└─────────────────────────────────────────────────────────────┘
`

---

## 3. Execution Modes & Cost Governance

The system enforces strict execution boundaries defined in configs/modes/:

| Mode | Target Hardware | External APIs | Cost Ceiling | Target Models |
| :--- | :--- | :--- | :--- | :--- |
| **FREE** | Kaggle Dual T4 (2x16GB), Google Colab | ❌ Strictly Prohibited | **.00** | Qwen2.5-Coder-7B (4-bit/8-bit QLoRA) + Local VLM |
| **LOCAL** | Apple Silicon (MPS), Local CUDA, CPU | ❌ Prohibited by default | **.00** | Qwen2.5-Coder-1.5B / 0.5B |
| **PAID** | RunPod, Lambda Labs, Vast.ai | ✅ Optional (OpenAI/Cloud APIs) | Configurable | Uncapped / 7B-32B Full Precision |
| **DRY_RUN** | Any CPU / Dev environment | ❌ No ML inference | **.00** | Mock models & pipeline validation |

---

## 4. Repository & Package Layout

`	ext
paint-code-rl/
├── src/paint_rl/                 # Production Python Package
│   ├── config/core.py            # Dynamic config loader & schema validator
│   ├── models/registry.py        # Model capability evaluation & selection
│   ├── rollout/                  # Generation batching & concurrency
│   ├── renderer/                 # Python client for WebGL sandbox
│   ├── rewards/
│   │   ├── api.py                # Main reward aggregation entrypoint
│   │   ├── validation.py         # NaN/Inf validation & reward contracts
│   │   ├── hpsv3_score.py        # Local aesthetic scoring interface
│   │   └── pairwise_vlm.py       # Pairwise judge provider with caching
│   ├── storage/cache.py          # SQLite judgment & render cache
│   ├── telemetry/core.py         # Experiment fingerprinting & logging
│   └── trainer/
│       ├── async_rollout.py      # Multi-worker async evaluation
│       ├── checkpoint_validator.py # Safetensors & state hash validator
│       └── train_grpo.py         # Main GRPO reinforcement learning loop
│
├── renderer/                     # Isolated WebGL Rendering Subsystem
│   ├── assets/                   # Vendored p5.min.js & p5.brush.min.js
│   ├── package.json              # Node.js dependencies (Puppeteer, Express)
│   ├── sandbox.js                # DOM/WebGL isolation & error traps
│   ├── server.js                 # HTTP rendering endpoint (:3000/render)
│   └── template.html             # Clean HTML5 canvas scaffolding
│
├── configs/                      # Externalized YAML Configurations
│   ├── base.yaml                 # Base scientific parameters
│   ├── modes/                    # free.yaml, local.yaml, paid.yaml
│   ├── providers/                # mps.yaml, kaggle.yaml, runpod.yaml, lambda.yaml
│   └── judges/                   # local.yaml, cloud.yaml
│
├── scripts/                      # Hardware Validation & Benchmarking
│   ├── benchmark.py              # System capability detector
│   ├── mps_validation_suite.py   # Apple Silicon tensor & memory profiler
│   ├── kaggle_validation_driver.py # End-to-end 22-phase CUDA validation
│   ├── generate_baseline.py      # Multi-device code generation baseline
│   └── test_reward_integration.py # Live render-to-reward pipeline test
│
├── tests/                        # Automated Pytest Suite
│   ├── test_hardening.py         # Cache, config, and checkpoint traps
│   ├── test_soup_integration.py  # Model selection, validation & async engine
│   └── test_rewards.py           # Unit tests for reward components
│
├── docs/                         # Scientific & Architectural Records
│   ├── decisions/                # Architecture Decision Records (ADR-001 - ADR-007)
│   └── research/                 # Deep research dossiers and hardware audits
│
├── datasets/                     # Reference image pools & prompt corpora
├── pyproject.toml                # Package definition & pytest config
└── requirements.txt              # Pinned dependencies
`

---

## 5. Verified Hardware Baselines

### Apple Silicon (M4 / M3 / M2 with 16GB Unified RAM)
* **MPS Acceleration:** Supported natively in 	orch.float16 / 	orch.float32.
* **Usable ML Budget:** ~13.18 GB (after macOS system allocation).
* **Recommended Policy Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct (Memory: ~3.4 GB inference, ~6.8 GB GRPO training).
* **Quantization:** itsandbytes (4-bit QLoRA) is not natively supported on MPS; use 1.5B half-precision.

### Kaggle Free Tier (Dual NVIDIA T4 x 2 - 32GB total)
* **Architecture:** 2 independent 16GB devices (GPU0: Policy Model, GPU1: Judge / Reward Model).
* **Recommended Policy Model:** Qwen/Qwen2.5-Coder-7B-Instruct with 4-bit LoRA.
