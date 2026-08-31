# Repository Freeze Manifest

**Release:** `v0.1.0-phase0`  
**Phase:** Phase-0 Hardware Validation Release  
**Status:** **FROZEN** (No further code changes prior to physical hardware validation)  
**Date:** 2026-09-01  

---

## 1. Repository Identification

* **Repository Name:** `paint-code-rl`
* **GitHub Remote:** `https://github.com/harshitthek/paint-code-rl`
* **Branch:** `main`
* **Commit SHA (Base):** `33bf78e8fda7ac900cc42c079be37ababba5e309`
* **Commit Date:** Tue Sep 1 02:15:38 2026 +0530
* **Release Tag:** `v0.1.0-phase0`

---

## 2. Runtime Environments

### Python Specification
* **Target Python Version:** `>=3.10, <3.12`
* **Development/Verification Python:** `Python 3.11.9` (Windows), `Python 3.11.0` (macOS Darwin arm64)
* **Packaging Specification:** `pyproject.toml` (PEP 621) & `requirements.txt`

### Node.js & Renderer Runtime
* **Target Node Version:** `>=18.0.0` (Tested: `v20.17.0`)
* **Package Manager:** `npm 10.8.2`
* **Browser Driver:** `puppeteer ^23.4.0` (Headless Chromium with WebGL2 / SwiftShader software backend)

---

## 3. Dependency Lock & Pinned Versions

### Python Dependencies (`pyproject.toml` & `requirements.txt`)

| Package | Pinned / Required Version | Role |
|---------|---------------------------|------|
| `torch` | `==2.5.1` | Core tensor math, autograd, device backends (CUDA/MPS/CPU) |
| `transformers` | `==4.49.0` | Policy model loading, tokenizers, CLIP aesthetic model |
| `trl` | `==0.15.1` | Canonical `GRPOTrainer`, `GRPOConfig` policy optimization |
| `peft` | `>=0.14.0` | LoRA parameter-efficient fine-tuning (`q_proj`, `v_proj`) |
| `accelerate` | `>=0.28.0` | Multi-GPU / device placement orchestration |
| `pydantic` | `>=2.0.0` | Strict configuration schema validation with type safety |
| `safetensors` | `>=0.4.0` | Checkpoint serialization and zero-copy tensor validation |
| `datasets` | `>=2.14.0` | HuggingFace dataset format for prompt rollouts |
| `Pillow` | `>=10.0.0` | Image processing and verification |
| `pyyaml` | `>=6.0` | Hierarchical YAML configuration parsing and merging |
| `tenacity` | `>=8.2.0` | Resilient network retries for judge API queries |
| `requests` | `>=2.31.0` | HTTP client for Renderer daemon RPC |
| `psutil` | `>=5.9.0` | Hardware memory and capability probing |
| `pytest` | `>=8.0.0` | Unit and integration test runner |

### Node.js Dependencies (`renderer/package.json`)

| Package | Version | Role |
|---------|---------|------|
| `express` | `^4.22.2` | Sandbox HTTP daemon (`/health`, `/render`) |
| `puppeteer` | `^23.4.0` | Headless browser management and screenshot capture |
| `pixelmatch` | `^7.2.0` | Visual regression and perceptual comparison |
| `pngjs` | `^7.0.0` | PNG byte validation and dimension inspection |

---

## 4. Renderer & Asset Integrity

| Asset | Path | SHA-256 Checksum |
|-------|------|------------------|
| **HTML Template** | `renderer/template.html` | `d85ce13db7a33c16ed1ba34fe41f479e106613a3a4aa5055c3908df485dd442a` |
| **Sandbox Engine** | `renderer/sandbox.js` | `06e5512d4582e96d0399281252d58e6c8ed3b5e7b12d3582bb949e36071858fc` |
| **Sandbox Server** | `renderer/server.js` | `c96edfd9fef1af75718256653ca7795daf72f3bd5758e8e1d5c9d8e70e776b35` |
| **p5.js Library** | `renderer/assets/p5.min.js` | `726ac96626b93f5bcaff83a910b6c60d3a9728f063e0eb73b5d0819ffc356915` |
| **p5.brush Library** | `renderer/assets/p5.brush.min.js` | `693aaa1bcb1feb942dc1066b8c106683f1b1432027363ac233b11ff9cf709dab` |

---

## 5. Model Registry & Precision Freeze

| Model Role | HuggingFace Model ID | Target Backend | Precision | Parameter Count | License |
|------------|----------------------|----------------|-----------|-----------------|---------|
| **Primary Local Policy** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `mps`, `cuda:0` | FP16 / BF16 | 1.54B | Apache-2.0 |
| **Large Cloud Policy** | `Qwen/Qwen2.5-Coder-7B-Instruct` | `cuda:0` (>=20GB) | BF16 / FP16 | 7.61B | Apache-2.0 |
| **CPU Fallback Policy** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | `cpu` | FP32 | 0.49B | Apache-2.0 |
| **Local VLM Judge** | `Qwen/Qwen2-VL-2B-Instruct` | `mps`, `cuda:1`, `cpu` | FP16 / BF16 | 2.21B | Apache-2.0 |
| **Aesthetic Scorer (CLIP)** | `openai/clip-vit-large-patch14` | `mps`, `cuda`, `cpu` | FP32 / FP16 | 428M | MIT |

---

## 6. Dataset Versioning & Checksums

* **Dataset Version:** `1.0.0`
* **Version Spec:** `datasets/VERSION.md` (`4ac419ea63e6edebe9645ee34b9dda381f3fa813e2ff5326be5b392a38eeb18c`)

| Split | File Path | Item Count | SHA-256 Checksum |
|-------|-----------|------------|------------------|
| **Train Prompts** | `datasets/prompts_v1.jsonl` | 56 | `951989b44771ed93a6c69901223d1304c4ac7c8760937bb5fc1c49fcfbd5f443` |
| **Validation Prompts** | `datasets/validation.jsonl` | 10 | `96dab2f87681a2011fcf25b7a244538d8436bc18c27fa022b55f793608cd52d1` |
| **Test Prompts** | `datasets/test.jsonl` | 5 | `3772f8234b5d4b111b4516a2cdb79248d751d166aaf50cf5054f431912eb4340` |

---

## 7. Scientific Configuration & Reward Formula

### Phase-0 Prototype Baseline Formula
The reward is composed linearly across three independent components:
$$\text{Reward}_{\text{Total}} = w_{\text{compile}} \cdot R_{\text{compile}} + w_{\text{aesthetic}} \cdot R_{\text{aesthetic}} + w_{\text{pairwise}} \cdot R_{\text{pairwise}}$$

Where:
* $w_{\text{compile}} = 0.10$ ($R_{\text{compile}} \in \{0.0, 1.0\}$)
* $w_{\text{aesthetic}} = 0.30$ ($R_{\text{aesthetic}} \in [0.0, 1.0]$ via CLIP visual cosine/aesthetic MLP)
* $w_{\text{pairwise}} = 0.60$ ($R_{\text{pairwise}} \in \{0.0, 0.5, 1.0\}$ via direction-invariant VLM comparison)

### Resolved Base Configuration (`configs/base.yaml`)
```yaml
run:
  experiment_name: "phase0-baseline"
  seed: 42
  log_level: "INFO"
model:
  id: "Qwen/Qwen2.5-Coder-7B-Instruct" # Overridden to 1.5B on MPS / local
  revision: "main"
  device_map: "auto"
training:
  batch_size: 16
  group_size: 4
  max_steps: 200
  checkpoint_freq: 50
generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
renderer:
  host: "127.0.0.1"
  port: 3000
  timeout_ms: 5000
  max_inflight_renders: 10
reward:
  weights:
    compile: 0.10
    hpsv3: 0.30
    pairwise: 0.60
judge:
  provider: "local"
  model: "Qwen/Qwen2-VL-2B-Instruct"
storage:
  type: "local"
  base_path: "artifacts"
  datasets_path: "datasets"
safety:
  allow_external_apis: false
aesthetic:
  provider: "clip"
  model: "openai/clip-vit-large-patch14"
  device: "auto"
```

---

## 8. Execution Modes & Cost Guard Contract

| Execution Mode | External API Calls | GPU Resource Allowed | Default Backend | Provider Boundary |
|----------------|-------------------|----------------------|-----------------|-------------------|
| `DRY_RUN` | **BLOCKED** | Mock/CPU | `cpu` | No tensor allocation |
| `FREE` | **BLOCKED** | Free T4x2 on Kaggle | `cuda:0` / `cuda:1` | Local models only |
| `LOCAL` | **BLOCKED by default** | Apple Silicon M4 / Local PC | `mps` / `cuda:0` / `cpu` | Local models only |
| `PAID` | Allowed if configured | Cloud GPU (RunPod/Lambda) | `cuda:0` | OpenAI / Paid VLMs |

**Cost Guard Enforcement:**
* `safety.allow_external_apis: false` is enforced at both the config schema validation level (`paint_rl/config/core.py`) and the judge provider initialization level (`paint_rl/judges/providers.py`).
* Attempting to instantiate `OpenAIJudgeProvider` while `allow_external_apis == false` raises an immediate `ValueError: ConfigurationError`.

---

## 9. Experiment Startup Contract

Every real experiment invocation must satisfy the pre-execution startup pipeline:
1. **Hardware Detection:** Probe CPU cores, RAM, and GPU device capability.
2. **Config Resolution:** Load and deep-merge environment YAML with `base.yaml`.
3. **Model Verification:** Verify model ID and device placement capability.
4. **Dataset Verification:** Confirm presence and SHA256 integrity of `datasets/prompts_v1.jsonl`.
5. **Renderer Asset Check:** Confirm local existence and hashes of `template.html`, `p5.min.js`, `p5.brush.min.js`.
6. **Config Hashing:** Compute SHA-256 hash of serialized JSON configuration.
7. **Run Registration:** Generate deterministic `run_id` (`run_{uuid}`).
8. **Logging:** Initialize structured JSONL run log under `artifacts/logs/{run_id}.jsonl`.

---

## 10. Hardware Validation Readiness

The repository code is verified, clean, committed, and frozen. No further architectural or implementation code edits may occur prior to running the physical test suites on physical Apple Silicon M4 and Kaggle T4x2 hardware.
