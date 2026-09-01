# Hybrid Execution Architecture

## 1. Execution Modes

The scientific core (`Policy -> Rollout -> Render -> Reward -> GRPO`) runs identically across four strictly enforced execution modes:

### `FREE`
*   **Hardware**: Kaggle T4x2 / Colab T4.
*   **Safety Limits**: `max_external_api_cost = 0`, `max_gpu_cost = 0`.
*   **Components**: Open-weight policy (`Qwen2.5-Coder-7B-Instruct` heavily quantized/LoRA), Local VLM Judge (`Qwen2.5-VL-7B`), Local HPSv3.
*   **Behavior**: Checkpoints aggressively every 15 minutes to survive 9-12hr session limits. Uses `/kaggle/working` persistence.

### `LOCAL`
*   **Hardware**: User's local machine.
*   **Safety Limits**: External API calls are *optional* but monitored.
*   **Behavior**: Primarily for development, validation, and debugging. Relies on the same Provider Abstractions but targets `localhost`.

### `PAID`
*   **Hardware**: RunPod Secure / Vast.ai (A100 / RTX 4090).
*   **Safety Limits**: External APIs explicitly permitted.
*   **Behavior**: Focuses on maximum throughput, utilizing multi-GPU configurations (via VeRL/Ray migration paths) and Cloud VLM judges (e.g., `gpt-4o-mini`).

### `DRY_RUN`
*   **Safety Limits**: Zero generation. Zero rendering.
*   **Behavior**: Validates `config_hash`, schema constraints, database initialization, and capability checks. Marks all artifacts as `NON_SCIENTIFIC_DRY_RUN`.

## 2. Capability-Based Registry
Instead of hardcoding models, the `ComputeBackend` executes a hardware fingerprint (VRAM, CUDA, CPU).
*   If `VRAM < 20GB`, the registry auto-selects INT4/INT8 quantization paths or smaller policy models (e.g., `Qwen2.5-Coder-1.5B`).
*   If `API_KEYS` are missing in `FREE` mode, it auto-mounts the `LocalJudgeProvider` (e.g., `Pixtral-12B`).

## 3. Reward & Uncertainty Decoupling
Rewards are no longer a static sum. The architecture defines `RewardComponent` instances:
*   `CompileReward`
*   `HPSv3Reward`
*   `PairwisePreferenceReward`

The composed reward reports the expected value **and** its empirical variance ($Var[R]$). Because `p5.brush` injects environment noise, tracking uncertainty allows the GRPO optimizer to correctly discount highly variant signals.

## 4. Decoupled Rollout Engines
To prevent `Puppeteer` HTTP polling from bottlenecking PyTorch:
*   **Future Scale**: `Trainer` pushes sequences to a `RolloutQueue`. `RendererWorkers` and `RewardWorkers` consume and fulfill asynchronously.
*   **Phase 0 Implementation**: Retains in-process ThreadPool queues, ensuring the architecture supports an eventual migration to Redis/Ray without rewriting the `Trainer`.
