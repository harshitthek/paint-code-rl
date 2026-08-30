# Industry-Grade Engineering Hardening Plan

This document outlines the architectural transformation required to elevate the Phase-0 prototype into a portable, reproducible, multi-GPU-ready scientific research system.

---

## A. Hardcoded Values Audit
The following values are currently hardcoded in the Phase-0 repository and must be moved into the configuration system:

**`rewards/api.py`**
*   `url = "http://localhost:3000/render"` (Renderer API endpoint)
*   `timeout=15` (Renderer network timeout)
*   `RewardConfig` (Reward weights like `compile: 0.10, hpsv3: 0.30`) are hardcoded in the Python class rather than injected.

**`scripts/generate_baseline.py` & `trainer/train_grpo.py`**
*   `model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"`
*   `device_map="auto"`
*   `datasets/prompts_train.json` (Dataset path)
*   `datasets/reference_pool/references.json` (Reference path)
*   Generation parameters: `max_new_tokens=512`, `top_p=0.9`, `temperature=0.7`, `seed=42`.
*   `../logs/metrics.db` (Logging database path)

**`renderer/sandbox.js` & `server.js`**
*   `timeout: 2000` (Puppeteer navigation & render wait timeouts)
*   `../artifacts/renders` (Image output directory)
*   Browser launch arguments (`--use-gl=egl`, etc.) are not configurable.

---

## B. Configuration Architecture
All configurations must use typed YAML (e.g., via `OmegaConf` or `Pydantic Settings`) with strict schema validation. The resolution order is:
`base.yaml` -> `phase0.yaml` -> `provider_specific.yaml` -> `Environment Variables (Secrets)`.

```yaml
# Example: configs/base.yaml
project:
  experiment_id: "phase0-run-001"
model:
  path: "Qwen/Qwen2.5-Coder-7B-Instruct"
  revision: "main"
training:
  batch_size: 16
  group_size: 4
  generation:
    max_new_tokens: 512
    temperature: 0.7
renderer:
  host: "127.0.0.1"
  port: 3000
  timeout_ms: 2000
  output_dir: "/app/data/renders"
reward:
  weights:
    compile: 0.10
    hpsv3: 0.30
    pairwise: 0.60
judge:
  provider: "openai"
  model: "gpt-4o-mini"
storage:
  artifacts: "/app/data/logs"
```
**Immutability:** Once parsed, the `ResolvedConfig` is serialized to JSON and persisted with the run. Any missing keys must immediately throw an exception.

---

## C. Cloud/GPU Compatibility Matrix & Selection Report

*(Prices and availability verified via marketplace APIs and official pricing pages as of August 2026. Prices are per-GPU on-demand.)*

| Provider | GPU | VRAM | Current Price | Pros | Cons | Phase-0 Suitability |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Kaggle** | T4x2 | 16GB (x2) | **FREE** | Zero cost, pre-configured | 30h/week limit, ephemeral, T4 is slow | **Excellent** (for dry runs / tiny tests) |
| **RunPod (Comm.)** | RTX 3090 | 24GB | ~$0.22/hr | Extremely cheap, 24GB fits 7B LoRA | Unreliable community hosts | **Best for budget iteration** |
| **Vast.ai** | RTX 4090 | 24GB | ~$0.37/hr | High throughput, very cheap | Dynamic pricing, interruptible | Good |
| **RunPod (Secure)** | A100 PCIe | 80GB | ~$1.39/hr | High VRAM, SOC2 reliable | More expensive | **Best for stable 7B training** |
| **Lambda Labs** | H100 PCIe | 80GB | $3.99/hr | Highest throughput, fast interconnects | Difficult to provision single-GPU on demand | Overkill for Phase-0 |

**Recommendation for Phase 0:**
*   **Cheapest viable:** RunPod Community RTX 3090 (~$0.22/hr). 24GB VRAM is tight but sufficient for a 7B model using LoRA + DeepSpeed ZeRO-2.
*   **Best price/performance & reliability:** RunPod Secure A100 80GB (~$1.39/hr). 80GB easily swallows GRPO rollouts and VRAM peaks.

---

## D. Portability Plan
The repository is completely containerized. A `ComputeBackend` configuration defines pathing:
*   **Kaggle Mode**: Storage abstraction routes to `/kaggle/working` for persistence and `/kaggle/input` for read-only datasets.
*   **RunPod/Vast Mode**: Routes storage to the `/workspace` network volume. 

The `Trainer` does not know where the GPU lives. The Dockerfile handles OS-level dependencies (Chromium, Node, PyTorch).

---

## E. Reliability Plan
1.  **Renderer Backpressure**: Implement a `queue_capacity` in the Node server. If inflight renders > CPU cores, return `HTTP 429` rather than crashing Puppeteer.
2.  **API Resilience**: The VLM Judge adapter must use `tenacity` for exponential backoff (retry on 429 and 502, fail on 400 or 401).
3.  **Browser Pooling**: Use `puppeteer-cluster` or a custom `PagePool` to reuse browser tabs, destroying tabs that exceed 100 renders to prevent memory leaks.
4.  **Idempotency & Caching**: Cache expensive `VLM` judgments by hashing `SHA256(image_hash + prompt + reference_id + judge_model)`.

---

## F. Observability Plan
1.  **Structured JSON Logging**: All Python and Node logs output structured JSON (`run_id`, `step`, `latency`, `error_class`).
2.  **Telemetry Database**: Expanding the current SQLite `metrics.db` to log:
    *   `tokens_per_sec`
    *   `gpu_utilization` (via `pynvml`)
    *   `vlm_api_failures`
3.  **Hardware Fingerprinting**: A startup script queries `nvidia-smi`, CPU info, and RAM, saving `compute_capabilities.json`.

---

## G. Security Plan
*   **Secret Management**: `OPENAI_API_KEY` is loaded exclusively from `.env` or cloud secret managers. Never hardcoded or logged.
*   **Renderer Sandbox**: Headless Chromium is fully disconnected from the network (`setRequestInterception` drops HTTP/HTTPS). 
*   **File Isolation**: The Node service only has read-access to vendored `assets/` and write-access to `artifacts/renders/`. It cannot traverse to `/etc/` or `.env`.

---

## H. Implementation Gaps
Before the first scientific run, the following architectural components must be built out of the current stubs:
1.  `configs/*.yaml` system (OmegaConf implementation).
2.  `CacheManager` (SQLite or Redis-based cache for VLM calls).
3.  `VisualRewardProvider` and `JudgeProvider` abstract base classes.
4.  Checkpoint / Resume logic for the `GRPOTrainer`.

---

## I. Industry-Grade Readiness Matrix

| Area | Current | Required | Status |
| :--- | :--- | :--- | :--- |
| **Configuration** | Hardcoded strings & dicts | YAML hierarchy, schema validation | 🔴 RED |
| **Reproducibility** | Local frozen env | Git commit + config hash + run_id | 🟡 YELLOW |
| **Security** | Smoke tests pass, offline mode active | Formal directory traversal protections | 🟢 GREEN |
| **Observability** | Basic stdout / SQLite schema | Structured JSON, hardware fingerprinting | 🟡 YELLOW |
| **Portability** | Dockerfile provided | `ComputeBackend` storage abstractions | 🟡 YELLOW |
| **Renderer** | Express server working | Browser Pooling, Backpressure (HTTP 429) | 🟡 YELLOW |
| **Reward / VLM** | Real APIs (Fail Closed) | `JudgeProvider` abstractions, Caching | 🔴 RED |
| **Training** | Mock stubs removed | ZeRO-2 Config, Checkpointing/Recovery | 🔴 RED |

---

## J. Final Verdict

### NOT READY FOR GPU VALIDATION

**Reasoning:** While the infrastructural *boundaries* and *interfaces* have been proven via the strict Mock-to-Real gates, deploying this to an expensive GPU instance right now would result in an unmanageable scientific environment. 

We cannot safely run GRPO until the **Configuration Architecture (A/B)**, **Caching/Idempotency (E)**, and **Checkpoint Recovery (H)** gaps are closed. Without these, a single API failure or RunPod interruption would destroy hours of expensive generation, and the results would be tied to hard-coded strings rather than reproducible, versioned configurations. 

The next step is to implement the Configuration and Provider Abstraction layers.
