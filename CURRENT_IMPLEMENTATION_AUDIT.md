# Current Implementation Audit

Generated: 2026-09-01
Auditor: Antigravity (automated, against live repository state)
Commit: Latest on `main`

## Issue Tracker

| # | Issue | Still Exists? | Evidence | Severity | Planned Fix |
|---|-------|--------------|----------|----------|-------------|
| 1 | `train_grpo.py` is a scaffold — `run_one_step()` imports TRL but never calls trainer, `run_tiny_run()` fakes a step counter (`save_experiment_state(start_step + 20)`) | **YES** | `src/paint_rl/trainer/train_grpo.py:93` — `save_experiment_state(start_step + 20)` with no actual training | **CRITICAL** | Rewrite as canonical trainer with real TRL GRPOTrainer |
| 2 | Two independent GRPO implementations exist | **YES** | `src/paint_rl/trainer/train_grpo.py` (scaffold) AND `scripts/train_local_step.py` (actual TRL code, but standalone) | **HIGH** | Consolidate into `src/paint_rl/trainer/grpo.py` + thin CLI `scripts/train_grpo.py` |
| 3 | HPSv3 is a broken stub | **YES** | `src/paint_rl/rewards/hpsv3_score.py:7` — `import hpsv3` does not exist in requirements. Always crashes. | **CRITICAL** | Replace with real aesthetic scorer (LAION aesthetic predictor or ImageReward — HPSv3 itself requires 14GB+ Qwen2-VL-7B, infeasible on 16GB Mac) |
| 4 | Pairwise VLM judge hardcoded to OpenAI API | **YES** | `src/paint_rl/rewards/pairwise_vlm.py:28` — hardcoded `api.openai.com`. No local VLM alternative exists. | **CRITICAL** | Implement `LocalVLMProvider` using `Qwen2-VL-2B-Instruct` (~4.5GB), keep OpenAI as PAID-mode option |
| 5 | `base.yaml` judge config hardcoded to `openai` / `gpt-4o-mini` | **YES** | `configs/base.yaml:30-31` — violates FREE/LOCAL zero-cost invariant | **HIGH** | Change base default to `local`, add mode overlays |
| 6 | `rewards/api.py` eagerly instantiates `OpenAIJudgeProvider` at import time | **YES** | `src/paint_rl/rewards/api.py:28` — singleton created on module load, crashes if deps missing | **HIGH** | Use lazy initialization / factory pattern |
| 7 | `template.html` WebGL monkeypatch timing bug | **YES** | `renderer/template.html:23-31` — patches `createCanvas` on `window.load`, but p5.js `setup()` fires during script evaluation before `load` event | **CRITICAL** | Move patch before p5.js script tag, or use inline script block before library loads |
| 8 | No training dataset exists | **YES** | `datasets/` contains only `.gitkeep` and `README.md` | **HIGH** | Create `datasets/prompts_v1.jsonl` with versioned prompt corpus |
| 9 | `test_rewards.py` is a dummy | **YES** | `tests/test_rewards.py:1-2` — `def test_dummy(): assert True` | **MEDIUM** | Replace with real reward component tests |
| 10 | `config/core.py:126` eagerly loads config at import | **YES** | Singleton `ACTIVE_CONFIG = load_config(...)` at module level — breaks any import that touches this module if `configs/base.yaml` is not found | **MEDIUM** | Keep but add graceful fallback for test contexts |
| 11 | `modes/local.yaml` has `safety.allow_external_apis` field not in `ProjectConfig` schema | **YES** | `configs/modes/local.yaml:2` has `allow_external_apis: false` but `ProjectConfig` Pydantic model has no `safety` field — this config is silently ignored | **HIGH** | Add `SafetyConfig` to schema or enforce via provider layer |
| 12 | No experiment registry with proper state machine | **YES** | `telemetry/core.py` has basic `ExperimentLogger` with print statements only | **MEDIUM** | Enhance with structured JSONL logging and state tracking |
| 13 | Renderer has single global timeout | **YES** | `renderer/sandbox.js:29,96` — `setDefaultTimeout(5000)` and `waitForFunction(..., {timeout: 4000})` are flat values | **MEDIUM** | Make configurable per-phase timeouts |
| 14 | `generate_baseline.py` imports `rewards.api` which triggers OpenAI judge instantiation | **YES** | `scripts/generate_baseline.py:7` — `from paint_rl.rewards.api import get_rewards` triggers module-level singleton | **HIGH** | Fix lazy initialization in rewards/api.py |

## Repository Truth Scan

| Pattern | Location | Classification | Action |
|---------|----------|---------------|--------|
| `MOCK_MODE` | `hpsv3_score.py:15` | Production path references undefined flag | REMOVE — implement real scorer |
| `save_experiment_state(start_step + 20)` | `train_grpo.py:93` | Fake training progress — increments step counter without training | REMOVE — implement real training |
| `pass` in `interfaces.py` | Lines 6,11,16,21,25 | Abstract method bodies — legitimate ABC pattern | KEEP |
| `pass` in `manager.py` | Lines 25,103 | Exception swallowing in health check and shutdown | KEEP — intentional error tolerance |
| `assert True` | `test_rewards.py:1` | Fake test | REPLACE with real tests |

## Dependency Gaps

| Package | In `requirements.txt`? | In `pyproject.toml`? | Actually Used? | Status |
|---------|----------------------|---------------------|---------------|--------|
| `hpsv3` | ❌ | ❌ | `hpsv3_score.py:7` attempts import | BROKEN — package doesn't exist as named |
| `pydantic` | ❌ | ✅ | `config/core.py` | Missing from requirements.txt |
| `safetensors` | ❌ | ✅ | `checkpoint_validator.py` | Missing from requirements.txt |
| `datasets` (HuggingFace) | ❌ | ❌ | `train_local_step.py:5` | Missing from both |
| `qwen-vl-utils` | ❌ | ❌ | Needed for local VLM | Will add |
