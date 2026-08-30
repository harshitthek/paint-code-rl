# Training AI to Paint with Code

> A research platform for training language models to generate executable p5.js/p5.brush programs using reinforcement learning and visual preference rewards.

## Project Status

**RESEARCH / PHASE-0 PROTOTYPE**

Architecture: **READY FOR HARDWARE VALIDATION**
Scientific claim: **NOT YET VALIDATED**
CUDA GRPO: **PENDING KAGGLE VALIDATION**
MPS GRPO: **PENDING M4 VALIDATION**

*Note: The original project's causal claim that pairwise judging solves mode collapse has NOT been independently established by this repository yet. The repository is designed to test that claim.*

## Architecture

Prompt
  ↓
Policy Model
  ↓
Rollout Engine
  ↓
Generated p5.js / p5.brush
  ↓
Puppeteer + Chromium + WebGL
  ↓
Rendered Image
  ↓
┌──────────────────┬────────────────────┐
│                  │                    │
HPSv3          Pairwise VLM        Execution
│                  │                validity
└──────────────────┴────────────────────┘
                  ↓
             Reward Engine
                  ↓
                GRPO
                  ↓
             Policy Update

## Execution Modes

This project supports multiple modes configured via configs/modes/:
* **FREE:** Strict configuration preventing external API costs. Requires local models or verified free Kaggle CUDA target.
* **LOCAL:** Local hardware execution (CPU, MPS, or CUDA). Does not automatically call paid APIs unless explicitly opted-in.
* **PAID:** Optional cloud scaling and remote API integration.
* **DRY_RUN:** Quick functional validation of the software stack without scientific weight updates.

## Quick Start

`ash
git clone https://github.com/harshitxdev/paint-code-rl.git
cd paint-code-rl

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd renderer
npm ci
cd ..

# Validate configuration
python scripts/preflight.py --mode dry_run

# Run CPU smoke tests
python scripts/generate_baseline.py --mode local

# Run Tests
pytest
`

## Deployment / Validation

### Kaggle (FREE CUDA)
Kaggle validation scripts exist to test the multi-GPU T4x2 environment constraint. 
1. Upload/clone repository to Kaggle
2. Enable T4x2
3. Install exact pinned dependencies
4. Execute 
otebooks/Phase0_Kaggle_Validation.ipynb

### MPS / Apple Silicon
Apple M4 validation scripts (scripts/mps_validation_suite.py) are strictly for local physical validation of TRL constraints on unified memory architectures. Do not assume 7B models can be loaded natively without validation.

## Scientific Roadmap

* **Phase 0:** Infrastructure validation
* **Phase 1:** Reward validation
* **Phase 2:** Pairwise vs absolute ablation
* **Phase 3:** Reference-pool experiments
* **Phase 4:** Reward-model distillation
* **Phase 5:** Scaling

## License and Citations

- License: MIT
- p5.js: https://p5js.org/
- p5.brush: https://github.com/acamposuribe/p5.brush
- HPSv3: https://github.com/tgxs002/HPSv3
- GRPO: https://arxiv.org/abs/2402.03300
