# MPS (APPLE SILICON) FEASIBILITY REPORT (FINAL CORRECTION)

*Date: August 2026*

This document outlines the theoretical feasibility and architectural boundaries for executing the Phase-0 GRPO pipeline on an Apple Silicon (MPS) target in `LOCAL` mode.

## 1. Zero-Cost Policy Enforcement (Unchanged)
The architecture rigorously enforces zero-cost defaults:
*   **FREE**: Zero paid APIs, Zero paid GPU.
*   **LOCAL**: Zero paid APIs by default. Hardware target adapts via MPS/CUDA/CPU detection.
*   **PAID**: External compute/API explicitly allowed.
The system will never automatically route to a paid API if the local VLM fails to load on the M4.

## 2. Memory Estimation (Theoretical Projections)
The `CapabilityEvaluator` formulas are explicitly treated as **ESTIMATES**, not exact requirements. The evaluation explicitly accounts for:
*   **Base Weights**
*   **Gradients**
*   **Optimizer Implementation** (AdamW FP32 master states)
*   **Activations**
*   **KV Cache**
*   **Framework/Runtime Overhead**

For a 7B policy model running natively in `bfloat16` with an AdamW optimizer, theoretical memory estimates easily exceed 24GB.

## 3. The `bitsandbytes` Reality
*   **Status**: `bitsandbytes` heavily relies on CUDA for its `load_in_4bit` (QLoRA/NF4/FP4) capabilities. Current stable and WIP support does not provide natively integrated 4-bit LoRA training via `TRL 0.15.1` on MPS. We must assume full 16-bit precision for the policy model on Apple Silicon until testing proves otherwise.

## 4. HPSv3 Status
*   **Status**: **UNVERIFIED**.
*   We cannot claim HPSv3 requires CUDA, nor can we claim it natively supports MPS. It must be physically tested on the M4 to determine if it achieves `MPS_NATIVE`, requires `PYTORCH_ENABLE_MPS_FALLBACK=1`, or is bound to `CPU_ONLY`.

## 5. MPS Graph Cache & Fallback
Apple's Metal Performance Shaders construct static graphs for tensor operations. Variable sequence length rollouts in GRPO can rapidly balloon the MPS graph cache, leading to severe unified memory leaks. We must explicitly test variable vs fixed-length batches and evaluate the impact of `torch.mps.empty_cache()` and `PYTORCH_ENABLE_MPS_FALLBACK=1` during real runs.

## 6. Resource Scheduling vs TRL 0.15.1 Lifecycle
Because unified memory pools CPU and GPU resources, we considered a sequential execution strategy (Unload Policy -> Load VLM -> Unload VLM -> Load Policy). However, we must explicitly distinguish:

### Theoretical Memory-Safe Schedule
A custom pipeline where the policy model is offloaded to disk/RAM during the reward phase, freeing unified memory for HPSv3 and the VLM Judge.

### Empirically Validated MPS Schedule
**UNVERIFIED**. The actual `TRL 0.15.1` `GRPOTrainer` lifecycle tightly couples these phases:
1.  Rollout Generation
2.  Reward Calculation
3.  Log-probability Calculation (requires Policy)
4.  Reference/KL Calculation (requires Reference Policy)
5.  Advantage Handling & Backward Pass (requires Policy)

**Crucially, the policy must remain resident immediately after reward calculation to compute log-probs and conduct the backward pass.** Unloading and reloading models between generation, rewarding, and the backward pass is unsupported natively by TRL and would require intrusive, custom core overrides. We will not change the scientific reward formulation or algorithm just to force MPS compatibility.

## Final Status Classification
Because the exact TRL 0.15.1 stack has not yet successfully executed a real optimizer update on an M4, **we do not mark MPS as supporting GRPO**.

### Status: BLOCKED — AWAITING PHYSICAL M4 TEST

The next action is physical execution on the M4. We will not redesign the architecture further.
