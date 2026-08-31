# Paid Cloud GPU Deployment Guide (RunPod / Lambda / Vast)

**Target Environment:** Cloud GPU Instance (e.g. 1x A100 80GB, 1x H100, or 1x RTX 4090 24GB).  
**Execution Mode:** `PAID` (External APIs and paid GPUs allowed).

---

## 1. Prerequisites

* Docker runtime or SSH access to cloud instance with Ubuntu 22.04 LTS / CUDA 12.4+
* Dedicated GPU with >= 24 GB VRAM (for 7B Policy + 2B VLM) or >= 80 GB (for full concurrent 7B + 7B)

---

## 2. Docker Container Deployment

The project provides a self-contained `Dockerfile` with Node.js, Chromium, and PyTorch dependencies:

```bash
# 1. Clone repository
git clone https://github.com/harshitthek/paint-code-rl.git
cd paint-code-rl
git checkout v0.1.0-phase0

# 2. Build Docker container
docker build -t paint-code-rl:phase0 .

# 3. Run container with GPU access
docker run --gpus all -it --shm-size=16g -p 3000:3000 -v $(pwd)/artifacts:/app/artifacts paint-code-rl:phase0 bash
```

---

## 3. Direct VM Setup & Run

```bash
# Install dependencies
pip install -e ".[dev]"
cd renderer && npm install && cd ..

# Launch training with 7B policy in PAID mode
export ENV=paid
python scripts/train_grpo.py --mode train --max-steps 100 --checkpoint-dir artifacts/checkpoints/runpod_7b
```

---

## 4. Configuration Overrides

Use `configs/modes/paid.yaml` or `configs/runpod.yaml`:
```yaml
safety:
  allow_external_apis: true
judge:
  provider: "openai" # Or "local" with Qwen2.5-VL-7B
  model: "gpt-4o-mini"
model:
  id: "Qwen/Qwen2.5-Coder-7B-Instruct"
training:
  batch_size: 16
  group_size: 4
```
