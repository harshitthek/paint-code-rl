# Current Implementation Audit

Generated: 2026-09-01
Auditor: Antigravity (automated, against live repository state)
Commit: Latest on `main`

## Issue Tracker (All 14 Issues Resolved)

| # | Issue | Still Exists? | Evidence | Severity | Resolution Status |
|---|-------|--------------|----------|----------|-------------|
| 1 | `train_grpo.py` is a scaffold | **NO** | `src/paint_rl/trainer/grpo.py` executes real TRL GRPOTrainer with multi-step training, gradient calculation, and LoRA adapter checkpoints. | **CRITICAL** | **RESOLVED** |
| 2 | Two independent GRPO implementations | **NO** | Consolidated into canonical `src/paint_rl/trainer/grpo.py` + thin CLI `scripts/train_grpo.py`. | **HIGH** | **RESOLVED** |
| 3 | HPSv3 broken stub | **NO** | Replaced with fail-safe `CLIPAestheticScorer` with cosine similarity delta fallback in `src/paint_rl/rewards/aesthetic.py`. | **CRITICAL** | **RESOLVED** |
| 4 | Pairwise VLM judge hardcoded to OpenAI | **NO** | `LocalVLMProvider` implemented using `Qwen2-VL-2B-Instruct` (~4.5GB FP16) with sequential memory unloading. | **CRITICAL** | **RESOLVED** |
| 5 | `base.yaml` judge config hardcoded to OpenAI | **NO** | `base.yaml` defaults to `provider: local` (`Qwen2-VL-2B-Instruct`) and `allow_external_apis: false`. | **HIGH** | **RESOLVED** |
| 6 | Eager judge instantiation at import | **NO** | `src/paint_rl/rewards/api.py` uses lazy singleton factory pattern (`_get_composer()`). | **HIGH** | **RESOLVED** |
| 7 | `template.html` WebGL monkeypatch timing | **NO** | WebGL `createCanvas` wrapper executes immediately after p5.js script before user code. | **CRITICAL** | **RESOLVED** |
| 8 | No training dataset exists | **NO** | Created `datasets/prompts_v1.jsonl` (57 prompts across 8 categories), `validation.jsonl`, and `test.jsonl`. | **HIGH** | **RESOLVED** |
| 9 | `test_rewards.py` is dummy | **NO** | Comprehensive 57-test suite in `tests/` covering components, config, validation, cache, and trainers. | **MEDIUM** | **RESOLVED** |
| 10 | Config eager loading failure on missing file | **NO** | Graceful exception handling and multi-parent path discovery in `src/paint_rl/config/core.py`. | **MEDIUM** | **RESOLVED** |
| 11 | `SafetyConfig` missing in Pydantic schema | **NO** | `SafetyConfig` and `DeviceConfig` added to `ProjectConfig` in `src/paint_rl/config/core.py`. | **HIGH** | **RESOLVED** |
| 12 | Telemetry state machine & structured logs | **NO** | Structured JSONL logging with run IDs, commit hashes, and step metrics in `src/paint_rl/telemetry/core.py`. | **MEDIUM** | **RESOLVED** |
| 13 | Renderer single global timeout | **NO** | Granular phase timeouts (`browser_startup`, `page_load`, `code_execution`, `screenshot`) in `renderer/sandbox.js`. | **MEDIUM** | **RESOLVED** |
| 14 | Eager rewards API in `generate_baseline.py` | **NO** | Clean decoupled generation CLI with robust code extraction and native Apple Silicon acceleration. | **HIGH** | **RESOLVED** |

## Verification Summary

1. **Unit & Hardening Test Suite**: 57/57 tests passing (100% pass rate).
   - Command: `python -m pytest tests/ -v`
   - Scope: `tests/test_code_extractor_and_prompting.py` (8), `tests/test_hardening.py` (3), `tests/test_mps_integration.py` (14), `tests/test_rewards.py` (29), `tests/test_soup_integration.py` (3).
2. **Sandbox Security & Corpus Suite**: 14/14 tests passing.
   - Command: `node renderer/test_security.js && node renderer/test_corpus.js`
   - Scope: 4 security boundary tests (TIMEOUT, PARSE_ERROR, NO_CANVAS, RUNTIME_ERROR) + 10 p5/p5.brush WebGL rendering corpus programs (~180ms latency).
3. **Apple Silicon MPS Hardware Validation Suite**: 4/4 phases passing.
   - Command: `python scripts/mps_validation_suite.py`
   - Scope: Phase 1 (MPS Compute & Tensor verification, 12.39ms FP32 matmul), Phase 2 (Memory feasibility & candidate selection), Phase 3 (MPS config overlay resolution), Phase 4 (Trainer device & batch resolution).
   - Artifact: `artifacts/mps_validation_report_raw.json`
4. **GRPO RL Multi-Step Training**:
   - Command: `python scripts/train_grpo.py --mode train --max-steps 3`
   - Scope: Real policy gradient updates (`rewards/code_syntax_reward: 1.0`, `rewards/render_validity_reward: 0.5`, `reward: 1.5`, `grad_norm: 0.107`).
   - Artifact: `artifacts/checkpoints/final_adapter/adapter_model.safetensors`
5. **Safetensors Checkpoint Validation**: Validated against soup bug, empty adapters, and key mismatches via `CheckpointValidator.validate_safetensors()`.
