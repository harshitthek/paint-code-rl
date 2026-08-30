# SOUP TECHNICAL AUDIT & ARCHITECTURE COMPARISON

*Date: August 2026*

## 1. Repository Overview & Purpose
`Soup` (MakazhanAlpamys/Soup) is a highly optimized LLM training wrapper focusing on extreme VRAM efficiency (running 8B models on 4GB cards) and rapid experiment execution. It wraps Hugging Face `trl`, `peft`, and `axolotl` mechanics but builds bespoke runtime interceptions for memory management. 

## 2. GRPO Implementation Analysis
*   **Real vs Adapter**: Soup does not implement GRPO from scratch. It heavily relies on Hugging Face `trl.GRPOTrainer`.
*   **External Rewards**: It explicitly supports `reward_fn: custom.py` and verifiable rewards, preserving dataset columns to pass into the reward functions. This perfectly mirrors our required approach.
*   **Rollout vs Training**: Soup supports `async_grpo_prefetch` (overlapping rollout with training) and `vllm_sleep_mode` for between-rollouts vLLM standby, proving that async decoupling of the rollout engine from the trainer is industry-standard for GRPO.

## 3. Layer Streaming Deep Dive
Soup's flagship feature is **Layer Streaming**, keeping only a few transformer blocks in VRAM at any time while the rest reside in RAM/NVMe.
*   **The Excluded-GRPO Reality**: `src/soup_cli/utils/layer_stream.py` explicitly blacklists `grpo` and `ppo` tasks. 
*   **The Technical Reason**: Autoregressive generation (Rollouts). During SFT/DPO, the model does one giant forward/backward pass. Layer streaming amortizes the cost of moving a layer from RAM to VRAM over the entire sequence. However, in GRPO rollouts, the model must generate token-by-token. If the model streams layers, it would have to swap the *entire 8B model into and out of VRAM for every single token generated*. This destroys the PCIe bus and results in ~0.1 tokens/sec. 

### Could Layer Streaming Work For Us?
*   **No, not natively.** Our core constraint is generation speed during the Phase-0 GRPO rollouts.
*   **Alternative**: We must rely on QLoRA (quantized 4-bit resident base model + trainable LoRA adapters) combined with DeepSpeed ZeRO-2 CPU offloading if we want to run 7B/8B models on low-VRAM environments (like Kaggle's 16GB T4s).

## 4. Free Mode Impact
*   **Without Soup's streaming**: We are limited to models that can fit their KV-cache, 4-bit base weights, LoRA gradients, and optimizer states entirely in 16GB (Kaggle). `Qwen2.5-Coder-7B` fits comfortably in 4-bit, but leaves little room for a local VLM judge.
*   **With Soup-style techniques**: We cannot use Layer Streaming for GRPO. However, Soup's memory probe techniques (`stream_vram_probe`) and strict deterministic checkpoint formats (safetensors consolidation) are highly applicable.

## 5. Architectural Comparison

| Capability | Our System | Soup | Best Approach |
| :--- | :--- | :--- | :--- |
| **Configuration** | Pydantic/YAML | YAML (axolotl-style) | Pydantic (Ours is strongly typed) |
| **GRPO** | Real, tightly coupled | TRL wrapper with Async Prefetch | **Soup's Async Prefetch** concept |
| **External Environment** | Node JS Renderer (Real) | Text-only `openenv` strings | **Ours** (Multimodal/Visual real environment) |
| **Low-VRAM** | Naive LoRA | Layer Streaming (SFT/DPO only) | **Hybrid**: QLoRA + ZeRO-2 (Ours) |
| **Evaluation/Caching** | SQLite Deterministic | Offline verification | **Ours** (Inline VLM caching is critical for cost) |

## 6. Bugs & Pitfalls Discovered
Soup's README and issues highlight critical pitfalls we must avoid:
1.  **DPO Reference Model Duplication**: TRL often duplicates the base model in memory for the reference policy. Soup uses the *same base weights with adapters switched off*. We must ensure our GRPO implementation does this, otherwise a 7B model will instantly OOM on a 24GB card.
2.  **Streaming Safetensors Inertia**: Soup warns that saving streamed adapters can inject `.inner.` into the `state_dict` keys, breaking later inference. We must verify our LoRA save routines strip nested module prefixes.

## 7. Reusable Ideas (Classification)
*   **Async GRPO Prefetch**: *Adapt Concept*. Decoupling generation from backward passes.
*   **Layer Streaming**: *Study Only / Not Applicable*. Fails on autoregressive generation.
*   **VRAM Pre-flight Probe**: *Adapt Concept*. Doing a dummy forward/backward pass to catch OOMs before spending an hour generating rollouts.
*   **Reward Ensemble Validation**: *Reuse Directly*. Soup checks that reward functions return a finite float *before* casting to tensors to prevent NaN crashes.

## 8. Final Recommendation

**TAKE IDEAS** (Do not integrate as a dependency)

Soup is a brilliantly engineered wrapper, but its primary innovation (Layer Streaming) fundamentally conflicts with the autoregressive rollout bottleneck of GRPO. Incorporating Soup as a dependency would introduce massive complexity for a feature we cannot use. 

However, we must adapt its **VRAM Pre-flight Probing**, **Adapter-switching Reference Policies**, and **Async Prefetching** concepts to harden our own `Trainer` implementation.
