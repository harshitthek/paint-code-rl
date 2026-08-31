# AI Agent & Developer Directives

This document outlines the operational rules, architectural invariants, and testing standards for all autonomous agents (e.g. Antigravity, Cursor, Claude Code, Copilot, Aider) and human contributors interacting with the **Paint-Code-RL** codebase.

---

## 1. Golden Rules for AI Agents

1. **Zero-Cost Invariant:**
   * Never modify FREE or LOCAL mode configurations to silently route inference or evaluation to paid external APIs (e.g. OpenAI GPT-4o, Anthropic, Gemini).
   * Presence of an OPENAI_API_KEY in the environment MUST NOT automatically trigger API expenditures unless the active execution mode is explicitly set to PAID.

2. **Multi-Device Compatibility:**
   * Never hardcode 	orch.cuda.is_available() as the sole check for hardware acceleration.
   * Always write device-agnostic logic using get_compute_device() or handle mps (Apple Silicon), cuda, and cpu gracefully:
     `python
     def get_device():
         if torch.cuda.is_available():
             return torch.device("cuda")
         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
             return torch.device("mps")
         return torch.device("cpu")
     `

3. **Strict Package Isolation (src/paint_rl/):**
   * All core library code belongs inside src/paint_rl/.
   * Do not place temporary scratch files, debug scripts, or notebooks in src/.
   * Keep the WebGL browser sandbox isolated in enderer/. Do not introduce Python runtime dependencies into enderer/.

4. **Reward Integrity & Numerical Safety:**
   * Rewards passed to GRPOTrainer must strictly pass alidate_reward_bundle().
   * Silent propagation of NaN, Inf, or uncaught exceptions during rollout/evaluation is forbidden. Return explicit error classifications (e.g. WORKER_CRASH, RENDERER_TIMEOUT, HPSV3_ERROR).

---

## 2. Directory & Coding Conventions

### Configuration Management
* Never hardcode hyperparameters (learning rate, batch size, group size, reward weights, model IDs) inside Python modules.
* All parameters must be defined in configs/base.yaml or layered configs in configs/modes/ and configs/providers/.
* Access configurations via paint_rl.config.core.ACTIVE_CONFIG.

### Checkpoint & Reproducibility Safety
* Every experiment run logs a unique un_id and an immutable config_hash derived from the merged YAML settings.
* When saving or loading checkpoints, always run CheckpointValidator.validate_safetensors() and CheckpointValidator.validate_experiment_state() to ensure adapter keys and configuration hashes match exactly.

### Renderer & Sandboxing
* Any p5.js/p5.brush rendering execution must happen via the HTTP bridge to enderer/server.js.
* Never execute un-sanitized generated JavaScript directly within the main Python training process.

---

## 3. Verification Protocol for Code Changes

Before committing or submitting a PR, agents must execute and pass:

1. **Automated Unit & Hardening Tests:**
   `ash
   pytest
   `
   *Expected result: 100% pass (7/7 tests).*

2. **Import Verification:**
   Ensure all imports resolve relative to src/ without requiring manual directory hopping.

3. **Secret Scan:**
   * Verify no API tokens (sk-..., KDAT_..., bearer tokens) or machine-specific absolute paths are committed.

4. **Hardware Validation (Target Dependent):**
   * On Apple Silicon: python3 scripts/mps_validation_suite.py
   * On CUDA / Kaggle: python3 scripts/kaggle_validation_driver.py

---

## 4. Git Commit Standards

* Follow the Conventional Commits format:
  * eat(scope): ... for new capabilities
  * ix(scope): ... for bug fixes and path resolution
  * docs(scope): ... for documentation updates
  * 	est(scope): ... for new test coverage
  * chore(scope): ... for repo maintenance
* Do not commit model weights (*.safetensors, *.bin), checkpoints, cache databases (*.db), or generated .png assets.
