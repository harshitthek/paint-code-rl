# SYSTEM AUDIT: Phase-0 Research System

*Date: August 2026*

| Area | Current Problem | Severity | Recommended Fix | Evidence |
| ---- | --------------- | -------- | --------------- | -------- |
| **Model Selection** | Hardcoded to `Qwen2.5-Coder-7B-Instruct`. Cannot fit into Kaggle T4x2 alongside local VLMs without specific LoRA/ZeRO configs. | HIGH | Capability-based model registry to dynamically switch based on VRAM constraints. | `train_grpo.py`, `generate_baseline.py` |
| **VLM Judging** | Judge is hardcoded to OpenAI `gpt-4o-mini`. Cannot run on `$0` budget free compute nodes if API costs accrue. | CRITICAL | Implement Local VLM Judge (`Qwen2.5-VL-7B` or `Pixtral-12B`) inside the `JudgeProvider` abstraction. | `rewards/api.py` |
| **Pairwise Bias** | The pairwise swapping logic assumes swapped comparison (LR vs RL) eliminates bias, but doesn't track *Tie Rate* or *Positional Bias* metrics explicitly. | MEDIUM | Multi-judge evaluation tracking in the DB (Entropy, Positional Bias). | `rewards/pairwise_vlm.py` |
| **Reward Composition** | `compile`, `hpsv3`, `pairwise` are explicitly summed in `api.py`. Hard to version or swap without editing core code. | HIGH | Implement an explicit `RewardComponent` architecture composing independent normalized values. | `rewards/api.py` |
| **p5.brush Noise** | Perceptual differences remain due to unseeded `Math.random` inside shader initializations. | MEDIUM | Explicitly model this as Environment Uncertainty (Var[R]) and track confidence intervals per rollout. | `test_quantify.js` outputs |
| **Renderer Lifecycle** | Uses synchronous HTTP polling. Node `Express` limits concurrency, but Trainer is still blocked. | HIGH | Migrate to decoupled Async Job Worker pattern (Queue → RolloutEngine → RendererWorkers). | `renderer/server.js` |
| **Scientific Modes** | No clear boundaries between testing code, `$0` free execution, and paid execution. | HIGH | Introduce `FREE`, `LOCAL`, `PAID`, `DRY_RUN` orchestration. | Hardening Pass |
| **Kaggle Execution** | Kaggle sessions die after 9-12 hours. Current checkpointing doesn't proactively flush datasets to persistent `/kaggle/working`. | HIGH | Add aggressive checkpoint flushes and partial-run detection. | Cloud limits research |

### Priority Ordering for Resolution
1. **SCIENCE:** Capability-based model registry and Local VLM selection (ensures FREE mode works).
2. **CORRECTNESS:** Proper tracking of positional bias and reward uncertainty (Var[R]).
3. **REPRODUCIBILITY:** Mode separation (FREE vs DRY_RUN).
4. **PERFORMANCE:** Asynchronous rollout queue decoupling.
