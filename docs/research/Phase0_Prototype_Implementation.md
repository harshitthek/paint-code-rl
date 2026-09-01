# Phase-0 Prototype Implementation: Training AI to Paint with Code

The goal of this specification is strictly to build the end-to-end infrastructure for visual RL via GRPO, ensuring that the components function, communicate, and scale reliably before any scientific conclusions are attempted.

---

## 1. Exact Phase-0 Scope

*   **Model:** `Qwen/Qwen2.5-Coder-7B-Instruct`
*   **LoRA Config:** rank=32, alpha=64, target_modules=["q_proj", "v_proj"] (OUR PROTOTYPE CHOICE)
*   **Prompt Dataset:** 50 highly distinct prompts (e.g., "draw a red circle", "draw a green tree", "draw a blue river"). Held-out validation set: 10 prompts.
*   **Group Size ($G$):** 4 (OUR PROTOTYPE CHOICE - scaled down for infrastructure testing)
*   **Training Steps:** 20 (OUR PROTOTYPE CHOICE)
*   **Renderer:** Puppeteer (Headless Chromium with `--use-gl=egl`)
*   **Library Versions (Pinned for Phase 0):**
    *   Python: 3.11
    *   PyTorch: 2.5.1
    *   Transformers: 4.49.0
    *   TRL: 0.15.1
    *   Node.js: 20.17.0
    *   Puppeteer: 23.4.0
    *   p5.js: 2.0.1
    *   p5.brush: 1.1.2
    *   HPSv3: hpsv3
    *   Judge Model: `gpt-4o-mini`

---

## 2. Environment Architecture

```text
/project
├── /renderer             # Node.js Puppeteer service
│   ├── server.js         # Express/Fastify API endpoint
│   ├── sandbox.js        # Puppeteer page manager
│   └── template.html     # Loads p5.js + p5.brush + injected code
├── /trainer              # Python RL logic
│   ├── train_grpo.py     # Main TRL GRPOTrainer script
│   └── model_loader.py   # LoRA/Qwen setup
├── /rewards              # Python scoring functions
│   ├── api.py            # Async calls to Node renderer
│   ├── hpsv3_score.py    # Offline VLM evaluation
│   └── pairwise_vlm.py   # OpenAI API wrapper for GPT-4o-mini
├── /datasets             # Prompts & Static References
│   ├── prompts_train.json
│   ├── prompts_val.json
│   └── /reference_pool   # 50 static images
├── /logs                 # W&B runs, SQLite metrics
└── /artifacts            # Checkpoints and output PNGs
```

---

## 3. Renderer Service

A robust, minimal Node.js Express server to handle Puppeteer.

**API Endpoint:** `POST /render`
```json
// Request
{
  "prompt": "draw a red circle",
  "code": "function setup() { createCanvas(800, 800, WEBGL); background(255); ... }",
  "seed": 42
}

// Response
{
  "success": true,
  "image_path": "/artifacts/renders/run123_42.png",
  "render_ms": 450,
  "compile_error": null,
  "runtime_error": null
}
```

**Implementation Requirements:**
*   Maintain a persistent browser context to avoid 2-second cold starts.
*   *Implementation Contract:* The renderer supports both filesystem paths (`image_path`) and in-memory Base64 streaming (`options.return_base64`), eliminating disk bottleneck during high-throughput RL rollouts.
*   *LoRA Configuration:* LoRA rank is set to $r=8$ (`lora_alpha=16`) targeting projection modules (`["q_proj", "v_proj"]`) across Apple Silicon MPS and CUDA hardware.

---

## 4. Security

The generated code executes in a headless browser, which poses a risk of infinite loops and memory leaks.

**Defense-in-Depth Sandbox:**
1.  **AST Filtering (Python-side):** Reject ASTs containing `eval`, `setTimeout`, `fetch`, or `window.` before sending to Node.
2.  **Browser Sandbox (Node-side):** Execute Puppeteer without `--no-sandbox`. Disable JS network access via Puppeteer Request Interception.
3.  **Timeout Limits:** Hard 2000ms limit on `page.waitForFunction()`. If it hangs, kill the page and return `runtime_error: "Timeout"`.
4.  **CPU/Memory Limits:** Run the Node process using Docker with `--cpus="2.0" --memory="2g"`.

## 5. Reward Service

The reward API strictly logs every component for future causal analysis.

**Python Interface:**
```python
def get_rewards(prompt: str, code: str, ref_images: list[str]) -> dict:
    # 1. Compile/Execute
    render_result = call_node_renderer(prompt, code)
    if not render_result["success"]:
        return {"total": 0.0, "compile": 0.0, "hps": 0.0, "pair": 0.0, "error": render_result["error"]}
        
    img_path = render_result["image_path"]
    
    # 2. HPSv3
    hps = compute_hpsv3(img_path, prompt)
    
    # 3. Pairwise
    pair_score = swapped_vlm_judge(img_path, ref_images, prompt)
    
    # 4. Total (OUR PROTOTYPE REWARD: compile=0.10, HPSv3=0.30, pairwise=0.60)
    # Note: ORIGINAL_PROJECT used 0.05 compile + 0.05 length + 0.30 HPSv3 + 0.60 pairwise
    total = (0.10 * 1.0) + (0.30 * hps) + (0.60 * pair_score)
    
    return {
        "total": total, "compile": 1.0, "hps": hps, "pair": pair_score,
        "len": len(code), "img_path": img_path
    }
```

---

## 6. Pairwise Judge

Implement a robust, randomized orientation judge to counteract VLM positional bias.

```python
def swapped_vlm_judge(candidate_path, reference_path, prompt):
    # Phase 0 explicitly uses ONE sampled reference per candidate.
    
    # Run both orientations concurrently
    vote1 = vlm_ask(left=candidate_path, right=reference_path, prompt=prompt)
    vote2 = vlm_ask(left=reference_path, right=candidate_path, prompt=prompt)
    
    # Valid outputs = "left", "right", "tie". Log malformed separately.
    def parse_vote(v):
        v = v.strip().lower()
        if v in ["left", "right", "tie"]: return v
        return "malformed"
        
    v1, v2 = parse_vote(vote1), parse_vote(vote2)
    
    # Distinguish actual tie, orientation disagreement, and malformed
    if "malformed" in (v1, v2):
        return {"score": 0.0, "status": "malformed"}
        
    if v1 == "left" and v2 == "right": return {"score": 1.0, "status": "win"}
    if v1 == "right" and v2 == "left": return {"score": 0.0, "status": "loss"}
    
    # If the model explicitly tied in both, or disagreed due to orientation bias
    status = "explicit_tie" if (v1 == "tie" and v2 == "tie") else "orientation_disagreement"
    return {"score": 0.5, "status": status}
```
*Validation:* A standalone script `tests/test_vlm_judge.py` must be run on 50 image pairs to measure tie rates and position preference before starting RL.

---

## 7. Reference Pool

For Phase 0, dynamic updates are strictly forbidden to prevent feedback loops.

*   **Structure:** `/datasets/reference_pool/` containing 50 curated PNGs.
*   **Metadata:** A `references.json` file mapping: `{"id": "ref_001", "prompt_category": "flower", "tier": "love"}`.
*   **Retrieval:** During `get_rewards`, sample 1 image from the pool that matches the `prompt_category`.

---

## 8. HPSv3

Integrate HPSv3 using the official library.
*   **Offline Validation:** Write `scripts/validate_hpsv3.py`. Feed it 25 random noise images and 25 high-quality artworks. Verify that the score distribution cleanly separates the two sets. Treat HPSv3 numerical ranges as empirical observations, ensuring sensible ordering/separation rather than requiring arbitrary scores.
*   **Logging:** Ensure the HPSv3 float is saved to the SQLite tracking database for every generated candidate.

---

## 9. GRPO

Use `trl.GRPOTrainer`. No distributed architectures for Phase 0.

*   **Group Sampling:** Let `GRPOTrainer` handle generating $G=4$ candidates per prompt.
*   **Reward Function Hook:** TRL allows passing a list of reward functions. Wrap the `get_rewards()["total"]` output into a format TRL accepts.
*   **Hyperparameters [OUR PROTOTYPE CHOICE]:**
    *   `beta`: 0.04 (KL penalty).
    *   `learning_rate`: 1e-5.
    *   `max_prompt_length`: 256.
    *   `max_completion_length`: 2048.

## 10. Phase-0 Dataset

Create a minimal, highly structured dataset to detect mode collapse easily.

**Training Set (50 prompts):**
*   10x "draw a red circle on a black background"
*   10x "draw a minimalist green tree"
*   10x "draw a blue wave using flow fields"
*   10x "draw a yellow sun using hatch patterns"
*   10x "draw a cyberpunk city skyline"

**Validation Set (10 prompts):**
*   Prompts unseen in training (e.g., "draw a purple cat").

---

## 11. Baseline Before RL

Before running `GRPOTrainer.train()`, execute `scripts/generate_baseline.py`.

*   **Action:** Generate 4 outputs ($G=4$) for all 50 training prompts and 10 validation prompts using the raw SFT `Qwen2.5-Coder-7B-Instruct` model.
*   **Measurement:** Compute and log the exact same reward bundle (Compile, HPSv3, Pairwise, Total, Code Length).
*   **Artifacts:** Save all 240 PNGs to `/artifacts/baseline/`.
*   **Purpose:** Establishes the starting point. If the final RL model does not statistically beat these metrics, the training loop failed.

---

## 12. One-Step GRPO Test

Before committing 24 hours to training, verify the gradients.
Write `scripts/one_step_test.py`:
1. Initialize `GRPOTrainer` with `max_steps=1`.
2. Ensure at least one output in the group compiles (yields reward > 0), while another fails (yields 0).
3. Check `trainer.model` parameters before and after `trainer.train()`.
4. **Assert:** Parameters have changed (gradients are flowing).
5. **Assert:** No NaNs in the loss log.

---

## 13. Tiny Training Run

Once the one-step test passes, execute `scripts/tiny_run.py`:
*   20 Prompts (Randomly sampled from the 50).
*   $G = 4$.
*   `max_steps = 20`.
*   This will generate exactly $20 \times 4 \times 20 = 1600$ renders.
*   At ~1 sec per Puppeteer render, the reward step will take ~30 minutes total. 
*   **Goal:** Validate that the system does not crash, OOM, or memory leak over a continuous 1-hour execution.

---

## 14. Logging / Experiment Tracking

Every output must be traceable back to its code and prompt.

Use **Weights & Biases (W&B)** for loss and metrics. Use a local **SQLite** database for instance-level tracking.

**SQLite Schema (`renders` table):**
*   `run_id` (uuid)
*   `step` (int)
*   `prompt_id` (str)
*   `code` (text)
*   `img_path` (str)
*   `render_ms` (int)
*   `compile_reward` (float)
*   `hpsv3_score` (float)
*   `pairwise_score` (float)
*   `total_reward` (float)
*   `code_length` (int)
*   `ref_id` (str)
*   `vlm_decision` (str)
*   `generation_seed` (int)
*   `decoding_params` (str)
*   `model_revision` (str)
*   `renderer_version` (str)
*   `p5js_version` (str)
*   `p5brush_version` (str)
*   `reward_version` (str)
*   `judge_version` (str)

---

## 15. Tests

Implement `pytest` suites:

*   `test_renderer.py`: 
    *   Inject `while(true){}` $\rightarrow$ Assert Timeout.
    *   Inject `x = y;` $\rightarrow$ Assert Compile Error.
    *   Inject valid `p5.brush` $\rightarrow$ Assert PNG returned.
*   `test_rewards.py`:
    *   Mock HPSv3 and VLM API calls.
    *   Assert `total_reward` perfectly aggregates the weights.
*   `test_swapped_judge.py`:
    *   Assert that a split decision returns `0.5`.

## 16. Acceptance Criteria

Development moves to the next phase ONLY when all gates are passed.

*   [ ] **Gate A (Renderer):** 100/100 valid scripts render without crashing the Node server.
*   [ ] **Gate B (Determinism):** Identical `randomSeed()` yields identical PNGs.
*   [ ] **Gate C (Reward):** All components return finite `float` values (no nulls/NaNs).
*   [ ] **Gate D (Judge):** `gpt-4o-mini` outputs match the requested parsing schema (left/right/tie) 99% of the time.
*   [ ] **Gate E (GRPO):** `scripts/one_step_test.py` completes with parameter updates.
*   [ ] **Gate F (Tiny Run):** `scripts/tiny_run.py` completes 20 steps without infrastructure OOMs or hangs.

---

## 17. Cost Control & Hardware Estimates

**Minimum Hardware Configuration (Phase 0):**
*   **GPU:** 1x NVIDIA RTX 3090 (24GB VRAM) or A10G. (Sufficient for 7B QLoRA with $G=4$).
*   **CPU:** 8-core (For parallel Puppeteer pages).
*   **RAM:** 32GB (Node.js Chromium instances are heavy).
*   **Storage:** 50GB (Model weights + thousands of generated PNGs).
*   **Estimated Cost:** ~$1.00/hour on standard cloud providers.
*   **Estimated Wall-clock Time (Tiny Run):** ~2 hours.

---

## 18. Scientific Safety Rails

The outputs of this Phase 0 prototype are strictly for engineering validation.

**DO NOT CLAIM:**
*   That pairwise judging solves mode collapse.
*   That code compression represents "elegant" learned behavior.
*   That GRPO improves the artistic output of the base model.

**ONLY CLAIM:**
> "The experimental infrastructure works end-to-end. The RL loop can successfully generate code, render it in a headless browser, score it with a VLM, and backpropagate the advantage via GRPO without crashing."

---

## 19. Deliverables

A complete execution of this phase results in:

1.  **Repository:** A git repo matching the tree in Section 2.
2.  **Configurations:** `trl_config.yaml` containing the GRPO hyperparams.
3.  **Baseline Dataset:** The 240 pre-RL baseline images and scores.
4.  **Database:** The SQLite file containing 1600 tracking rows from the Tiny Run.
5.  **Test Report:** Output of `pytest` confirming Gates A-F passed.

---

## 20. Final Gate

### READY TO CODE

The specification contains exact library versions, defensive sandbox configurations, precise API schemas, a verified non-distributed GRPO architecture (`trl`), and objective test gates. An engineer can follow this document top-to-bottom to build the Phase-0 infrastructure without guessing.

**Goal:**
1. FIRST MAKE THE LOOP WORK. *(This Document)*
2. THEN MEASURE IT. *(Phase 1)*
3. THEN TEST THE HYPOTHESIS. *(Phase 2)*
4. THEN SCALE. *(Phase 3)*

## 21. Configuration Management

Explicitly separate configurations to avoid scientific confusion:

``python
class RewardConfig:
    ORIGINAL_PROJECT = {"compile": 0.05, "length": 0.05, "hpsv3": 0.30, "pairwise": 0.60}
    PHASE0_PROTOTYPE = {"compile": 0.10, "length": 0.00, "hpsv3": 0.30, "pairwise": 0.60}
``
