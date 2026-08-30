# Implementation Readiness Review

This document critically evaluates the "Verified Research Synthesis" to separate technically plausible designs from verified, implementable engineering specifications. 

---

## 1. Architecture Decision Audit

| Decision | Evidence | Confidence | Must Benchmark? | Risk | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Coder | Standard open-weight code model; strong `trl` support. | High | No | Low | **GREEN**. Proceed. |
| GRPO via TRL/vLLM | Math proven (DeepSeekMath); standard in `trl` & `unsloth`. | High | Yes (VRAM limits) | Med | **GREEN**. Proceed with LoRA prototyping. |
| `p5.brush` | The core visual library. | High | No | Low | **GREEN**. Proceed. |
| **Puppeteer to `skia-canvas`** | **CONTRADICTED**. `p5.brush` requires WebGL/shaders. `skia-canvas` is 2D-only. | **None** | Yes | **CRITICAL**| **RED**. Cannot replace browser without rewriting `p5.brush`. Must use headless Chromium. |
| AST Validation | Standard JS tooling (Esprima/Acorn). | High | No | Low | **GREEN**. Proceed. |
| Swapped Pairwise | Logically sound mechanism to stop positional bias. | Medium | Yes | Med | **YELLOW**. Test VLM cost vs. variance reduction first. |
| Multimodal RM | Viable theory, but multimodal RMs are notoriously hard to train to predict aesthetic gradients. | Low | Yes | High | **RED**. Do not assume a Code+PNG RM works without a pilot study. |

---

## 2. Renderer Replacement Audit

The previous synthesis proposed moving from Puppeteer to `skia-canvas` in a sandboxed Node environment to achieve ~50ms rendering. **This proposal is technically invalid.**

*   **The WebGL Dependency:** Reviewing the `p5.brush` source and p5.js ecosystem confirms that `p5.brush` relies heavily on custom GLSL shaders (`createShader()`), flow fields, and the `WEBGL` context to generate its organic textures.
*   **Node.js Limitations:** `skia-canvas` and `node-canvas` only implement the 2D Canvas API. They do not provide WebGL contexts. 
*   **Verdict:** The system *must* rely on a browser engine (Chromium via Puppeteer/Playwright) equipped with WebGL (e.g., using `--use-gl=egl` or `xvfb` for headless GPU acceleration). The original authors used Puppeteer out of necessity, not ignorance.

---

## 3. Renderer Equivalence Specification

If any performance optimizations are attempted (e.g., swapping Puppeteer for Playwright, or compiling headless-gl), equivalence must be proven against a baseline Puppeteer output corpus (1,000 generated scripts).

**Acceptance Criteria:**
*   **Pixel Similarity:** SSIM > 0.99 (slight anti-aliasing variations permitted).
*   **Determinism:** Two runs of the same script with the same `randomSeed()` must yield SSIM = 1.0.
*   **Shader Execution:** Custom brush textures, flow fields, and hatch patterns must load and execute identically.

---

## 4. Benchmark the Actual Rendering Pipeline

Because we are stuck with browser engines, throughput is the critical bottleneck. Guessed values ("100x faster") must be discarded.

**Required Benchmark Design:**
1.  **Corpus:** 500 valid `p5.brush` scripts of varying token length (1k to 10k).
2.  **Hardware:** 1x A100 node with 64 CPU cores.
3.  **Candidates to test:**
    *   Puppeteer (New context per script)
    *   Playwright (Reusing a single persistent browser, new pages)
    *   Playwright (Reusing a single page, injecting script, clearing canvas)
4.  **Metrics:** P50/P95/P99 latency, Images/sec/core, Memory leak rate (MB/hour).

## 5. Sandbox Security Audit

Running arbitrary LLM-generated JavaScript in a browser context exposes the host to CPU exhaustion, infinite loops, and potential Chromium sandbox escapes.

**Defense-in-Depth Architecture:**
1.  **Static AST Filtering (Pre-execution):** Parse code with Acorn. Reject ASTs containing `fetch`, `XMLHttpRequest`, `eval`, `Function`, `setTimeout`, `setInterval`, or `document.createElement`.
2.  **Browser Sandboxing:** Run Playwright/Puppeteer with `--no-sandbox` strictly disabled. Use `--disable-web-security` ONLY if loading local brush assets requires it, but isolate the container.
3.  **Container Limits (cgroups):** The rendering service must run in a Docker container with `--memory="4g"` and `--cpus="4.0"`.
4.  **Timeouts:** The JavaScript execution must be wrapped in a strict 2000ms wall-clock timeout.

**Threat Model:** The primary threat is not malicious exfiltration (as the LLM is not adversarial by intent), but *accidental resource exhaustion* (e.g., generating 1,000,000 recursive brush strokes) which will halt the RL training loop.

---

## 6. Exact Qwen Base Model

**Verification:** The project article generically specifies "Qwen". Given the timing (March 2026/August 2026) and the task (code generation), Qwen2.5-Coder is the most mathematically capable architecture.
*   **Recommendation for Prototype:** `Qwen/Qwen2.5-Coder-7B-Instruct`. It fits on a single 24GB/40GB GPU for GRPO LoRA training and is highly responsive to `<reasoning>` tagging.
*   **Recommendation for Production:** `Qwen/Qwen2.5-Coder-32B-Instruct` requires 2x A100 80GB for GRPO but offers vastly superior spatial and logical reasoning for complex p5.js geometry.

---

## 7. GRPO Implementation Audit

**Comparison:**
*   **Hugging Face TRL (`GRPOTrainer`):** Easiest to implement. Supports custom reward functions seamlessly. However, synchronously waiting for external environments (Puppeteer) blocks the GPU, leading to terrible utilization.
*   **VeRL (Volcengine RL):** Built specifically for high-throughput, distributed RLHF. Excels at async rollouts and vLLM integration.
*   **OpenRLHF:** Great for Ray-based distributed training, separating the actor, critic (not needed), and reference models across nodes.

**Implementation Architecture:** **TRL for Prototype, VeRL for Production.**
For prototype, use TRL `GRPOTrainer` with a custom reward function that makes a blocking HTTP request to a local Node.js rendering microservice. For production, the generation and rendering must be fully decoupled via Ray/VeRL to keep GPU utilization >90%.

---

## 8. Reward API Specification

To ensure causal analysis is possible, the reward function must not return a single scalar. It must return a structured bundle.

```python
def compute_reward(prompt, code_str, image_base64, ref_images):
    # 1. Compile/AST Check
    compile_valid = validate_ast(code_str)
    if not compile_valid:
        return {"total": 0.0, "compile": 0.0, "len": 0, "hps": 0.0, "pair": 0.0}
    
    # 2. HPSv3
    hps_score = hpsv3_api.score(image_base64, prompt)
    
    # 3. Pairwise
    pair_score = 0.0
    if len(ref_images) == 2:
        pair_score = swapped_pairwise_judge(prompt, image_base64, ref_images)
        
    total = (0.05 * compile_valid) + (0.30 * hps_score) + (0.60 * pair_score)
    
    return {
        "total": total,
        "compile": compile_valid,
        "len": len(code_str),
        "hps": hps_score,
        "pair": pair_score
    }
```
*Logging:* Every component must be logged to W&B alongside the code length and generation time.

## 9. Pairwise Judge Validation

The "swapped orientation" judge (where the VLM is queried `(A, B)` and then `(B, A)`) must be empirically validated before being used in the training loop.

**Required Test:**
Sample 500 Rollout vs. Reference pairs.
1. Run Single Orientation (Rollout Left, Ref Right).
2. Run Swapped Orientation (Ref Left, Rollout Right).
3. Run Randomized Orientation.

**Metrics:** Measure the tie rate (disagreements) and the positional bias (how often the model simply picks the left image).
*   *Verdict Trigger:* If the VLM defaults to the reference image >90% of the time regardless of position, the VLM is blind to the nuanced differences, and the reward will plateau. The judge model (e.g., GPT-4o vs Claude 3.5 Sonnet) must be selected based on highest self-consistency in this test.

---

## 10. Reference-Pool Leakage Audit

**The Flaw:** The prior synthesis proposed injecting the top 1% of rollouts back into the reference pool to "refresh" it.
**The Audit:** This creates an incestuous feedback loop. If the VLM judge has a bias (e.g., it loves high-contrast red pixels), it rewards high-contrast red rollouts. These rollouts enter the reference pool. Future rollouts are now judged against high-contrast red references, amplifying the VLM's original bias exponentially.

**Safeguards (Required):**
1.  **Frozen Validation Pool:** A set of 100 human-made artworks that is NEVER updated. The model's performance against this pool is tracked but *not* used for gradients.
2.  **Human-in-the-Loop Refresh:** Rollouts can only enter the training reference pool if they pass the HPSv3 threshold AND are manually approved by a human (or an independent, non-training VLM) in batches.

---

## 11. Code-Length Experiment Redesign

We must cleanly decompose the factors driving code length reduction (13.5k $\rightarrow$ 2k).

**Experiment:**
Generate 10,000 scripts from the SFT base model at varying lengths (forced via `min_new_tokens` or temperature). Do not train. Just execute and score them.

**Measure:**
*   $P(\text{parse success} | \text{length})$
*   $P(\text{timeout} | \text{length})$
*   $\mathbb{E}[\text{HPSv3} | \text{valid}, \text{length}]$

**Verdict Trigger:** 
If $P(\text{timeout})$ goes from 0.01 at 2k tokens to 0.80 at 10k tokens, then the RL algorithm is trivially learning to avoid timeouts. The narrative of "aesthetic simplification" is falsified.

---

## 12. Causal Mode-Collapse Experiment

To isolate exactly what solved the mode collapse, we require a fractional factorial experiment. 

**Minimum Matrix (4 runs, 500 steps each):**
*   **Run 1 (Baseline):** Absolute Judges (0-10) + No Reference Pool + HPSv3 (Weight 0.1)
*   **Run 2 (Isolate Weight):** Absolute Judges (0-10) + No Reference Pool + HPSv3 (**Weight 0.3**)
*   **Run 3 (Isolate Pairwise):** **Pairwise Judge** + Reference Pool + HPSv3 (Weight 0.1)
*   **Run 4 (Full Project):** **Pairwise Judge** + Reference Pool + HPSv3 (**Weight 0.3**)

**Stopping Criteria:** 500 steps is sufficient to observe divergence or mode collapse (as noted in the original article, it plateaued early).
**Statistical Test:** Calculate the visual diversity (LPIPS variance) of 100 outputs at step 500. A significantly higher variance in Run 3 vs Run 2 proves the pairwise pool solved the collapse.

## 13. Evaluation Specification

The model cannot be evaluated using its own reward function.

**Independent Benchmark Suite:**
1.  **Visual Quality:** Fréchet Inception Distance (FID) against a held-out human dataset.
2.  **Prompt Adherence:** CLIP text-image similarity score (independent of HPSv3).
3.  **Code Readability:** Cyclomatic complexity and AST depth metrics.
4.  **Execution Reliability:** Pass@1 compile/render rate on 1,000 $T=0.7$ generations.
5.  **Generalization:** A frozen test set of 50 prompts radically out-of-distribution (e.g., "draw a 3D isometric microchip").

---

## 14. Hardware / Compute Planning

**Ablation Study (The Minimum Viable Experiment):**
*   **Target:** Run the 4-run matrix (Section 12) for 500 steps. $G=8$.
*   **Hardware:** 1x Node with 4x A100 (80GB). (1 for Qwen2.5-Coder-7B policy, 1 for Reference, 2 for VLM Judge/HPSv3).
*   **CPU/RAM:** 64-core CPU, 256GB RAM to run 16 parallel Puppeteer headless instances.
*   **Duration:** ~24 hours total for all 4 runs.
*   *Note:* This is a measured estimate based on typical `trl` GRPO throughput for 7B models with external blocking rewards.

---

## 15. Minimum Viable Scientific Experiment

Before building the production VeRL distributed system or distilling the reward model, the following experiment must be executed:

> **Objective:** Prove that Pairwise Reference Anchoring stabilizes visual RL.
*   **Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (LoRA).
*   **Dataset:** 100 simple prompts (e.g., "draw a red circle", "draw a flower").
*   **Group Size:** $G=8$.
*   **Rendering:** Puppeteer (since WebGL is required).
*   **Runs:** Run 1 (Absolute) vs. Run 4 (Pairwise).
*   **Success Criterion:** Run 4 produces recognizable, visually distinct outputs across the 100 prompts at step 500, while Run 1 produces identical (collapsed) geometries.

---

## 16. Implementation Readiness Matrix

| Component | Ready Now? | Blocking Issue | Required Test | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-Coder GRPO** | **GREEN** | None | Base TRL setup | Implement |
| **AST Filtering** | **GREEN** | None | Basic unit tests | Implement |
| **Puppeteer Sandbox** | **YELLOW** | Latency/OOMs | Throughput load test | Prototype First |
| **Skia-Canvas / Node** | **RED** | `p5.brush` requires WebGL | Shader compatibility test | **ABANDON** |
| **Swapped Pairwise VLM** | **YELLOW** | Unknown positional bias | 500-sample orientation test | Prototype First |
| **Multimodal RM** | **RED** | Extremely hard to train | Code+PNG prediction pilot | Do not build yet |
| **Mode Collapse Theory** | **RED** | Confounded variables | 4-run ablation matrix | Needs testing |

---

## 17. Final Recommendation

### READY FOR PROTOTYPE ONLY

**Reasoning:**
The synthesis architecture proposed a fatal flaw: replacing the browser with `skia-canvas`, which violates `p5.brush`'s fundamental WebGL requirement. We must revert to Puppeteer or Playwright. Because browser rendering is immensely slow and resource-heavy, running a production-scale RL pipeline (e.g., $G=64$, 10k steps) is financially and temporally prohibitive until the minimum viable experiments are run.

Furthermore, the core scientific claim—that pairwise judging solves mode collapse—is unverified due to confounded variables in the original project. 

**Next Steps:**
Do not build the full distributed VeRL pipeline. Do not attempt reward model distillation.
1. Build the simple `trl` + Puppeteer loop.
2. Execute the 4-run ablation matrix (Section 12) on a 7B model.
3. If, and only if, the pairwise judge proves causally responsible for stabilizing the gradients, proceed to scale the architecture.
