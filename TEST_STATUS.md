# Test Status Manifest

**Release Version:** `v0.1.0-phase0`  
**Git Commit:** `33bf78e8fda7ac900cc42c079be37ababba5e309`  
**Date:** 2026-09-01  
**Environment:** Windows x64 / Python 3.11.9 / Node v20.17.0 / Chromium (SwiftShader WebGL2)  

---

## Test Execution Summary

| Suite / Component | Result | Passed | Failed | Not Executed | Environment | Evidence / Command |
|-------------------|--------|--------|--------|--------------|-------------|---------------------|
| **Unit Tests (pytest)** | **PASS** | 35 | 0 | 0 | CPU | `python -m pytest tests/ -v` |
| **Config Suite** | **PASS** | 6 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestConfig` |
| **Reward Validation** | **PASS** | 5 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestRewardValidation` |
| **Reward Components** | **PASS** | 3 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestRewardComponents` |
| **Reward Composer** | **PASS** | 2 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestRewardComposer` |
| **Storage & Cache** | **PASS** | 2 | 0 | 0 | CPU | `pytest tests/test_hardening.py::test_cache` |
| **Model Registry & Preflight**| **PASS** | 3 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestModelRegistry` |
| **Async Rollout Crash-Safety**| **PASS** | 2 | 0 | 0 | CPU | `pytest tests/test_soup_integration.py::test_async_rollout` |
| **Judge Providers & Cost Guard**| **PASS** | 3 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestJudgeProviders` |
| **Dataset Verification** | **PASS** | 5 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestDataset` |
| **Telemetry & Run ID** | **PASS** | 1 | 0 | 0 | CPU | `pytest tests/test_rewards.py::TestTelemetry` |
| **Checkpoint State & Hashing**| **PASS** | 1 | 0 | 0 | CPU | `pytest tests/test_hardening.py::test_checkpointing` |
| **Renderer Security Tests** | **PASS** | 4 | 0 | 0 | CPU/Node | `node renderer/test_security.js` |
| **Renderer 10-Script Corpus** | **PASS** | 10 | 0 | 0 | WebGL/Node | `node renderer/test_corpus.js` |
| **Import Verification Suite** | **PASS** | 41/41 files | 0 | 0 | CPU | `python -c "importlib test across all src/ and scripts/"` |
| **Real CLIP Aesthetic Inference** | **NOT EXECUTED** | - | - | 1 | CUDA/MPS | Deferred to hardware runtime (weights download on first run) |
| **Real Qwen2-VL-2B Inference** | **NOT EXECUTED** | - | - | 1 | CUDA/MPS | Deferred to hardware runtime (weights download on first run) |
| **Physical M4 MPS GRPO Step** | **PENDING** | - | - | 1 | Apple M4 | Command: `python scripts/train_grpo.py --mode one_step` |
| **Physical Kaggle T4x2 GRPO Step**| **PENDING** | - | - | 1 | Kaggle T4x2 | Command: `notebooks/Phase0_Kaggle_Validation.ipynb` |

---

## Detailed Test Logs

### Pytest Full Output (35/35 Passed)
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\user\.gemini\antigravity\brain\eabfab2e-f626-4128-9da1-6868c5d0f842\paint-code-rl
configfile: pyproject.toml

tests/test_hardening.py::test_cache PASSED                               [  2%]
tests/test_hardening.py::test_config PASSED                              [  5%]
tests/test_hardening.py::test_checkpointing PASSED                       [  8%]
tests/test_rewards.py::TestConfig::test_load_base_config PASSED          [ 11%]
tests/test_rewards.py::TestConfig::test_config_has_safety PASSED         [ 14%]
tests/test_rewards.py::TestConfig::test_config_has_aesthetic PASSED      [ 17%]
tests/test_rewards.py::TestConfig::test_config_hash_deterministic PASSED [ 20%]
tests/test_rewards.py::TestConfig::test_config_extra_keys_ignored PASSED [ 22%]
tests/test_rewards.py::TestConfig::test_kaggle_config_overlay PASSED     [ 25%]
tests/test_rewards.py::TestRewardValidation::test_valid_bundle PASSED    [ 28%]
tests/test_rewards.py::TestRewardValidation::test_nan_rejected PASSED    [ 31%]
tests/test_rewards.py::TestRewardValidation::test_inf_rejected PASSED    [ 34%]
tests/test_rewards.py::TestRewardValidation::test_none_rejected PASSED   [ 37%]
tests/test_rewards.py::TestRewardValidation::test_missing_total_rejected PASSED [ 40%]
tests/test_rewards.py::TestRewardComponents::test_compile_reward_success PASSED [ 42%]
tests/test_rewards.py::TestRewardComponents::test_compile_reward_failure PASSED [ 45%]
tests/test_rewards.py::TestRewardComponents::test_reward_result_fields PASSED [ 48%]
tests/test_rewards.py::TestRewardComposer::test_composer_with_compile_only PASSED [ 51%]
tests/test_rewards.py::TestRewardComposer::test_composer_skips_downstream_on_render_failure PASSED [ 54%]
tests/test_rewards.py::TestCache::test_cache_operations PASSED           [ 57%]
tests/test_rewards.py::TestModelRegistry::test_cuda_selection PASSED     [ 60%]
tests/test_rewards.py::TestModelRegistry::test_cpu_fallback PASSED       [ 62%]
tests/test_rewards.py::TestAsyncRollout::test_worker_crash_handling PASSED [ 65%]
tests/test_rewards.py::TestJudgeProviders::test_mock_judge PASSED        [ 68%]
tests/test_rewards.py::TestJudgeProviders::test_factory_returns_local_by_default PASSED [ 71%]
tests/test_rewards.py::TestJudgeProviders::test_openai_blocked_in_local_mode PASSED [ 74%]
tests/test_rewards.py::TestDataset::test_prompts_v1_exists PASSED        [ 77%]
tests/test_rewards.py::TestDataset::test_prompts_v1_valid_jsonl PASSED   [ 80%]
tests/test_rewards.py::TestDataset::test_validation_set_exists PASSED    [ 82%]
tests/test_rewards.py::TestDataset::test_test_set_exists PASSED          [ 85%]
tests/test_rewards.py::TestDataset::test_no_duplicate_prompt_ids PASSED  [ 88%]
tests/test_rewards.py::TestTelemetry::test_experiment_logger_creates_run_id PASSED [ 91%]
tests/test_soup_integration.py::test_preflight PASSED                    [ 94%]
tests/test_soup_integration.py::test_reward_validation PASSED            [ 97%]
tests/test_soup_integration.py::test_async_rollout PASSED                [100%]

============================= 35 passed in 5.00s ==============================
```

### Renderer Security Smoke Tests (4/4 Passed)
```
Running Security Smoke Test...
Testing: Infinite Loop -> PASS (TIMEOUT error caught)
Testing: Invalid JavaScript -> PASS (PARSE_ERROR caught)
Testing: Missing Canvas -> PASS (NO_CANVAS caught)
Testing: Runtime Exception -> PASS (RUNTIME_ERROR caught)
SUCCESS: All security smoke tests passed.
```

### Renderer 10-Script Test Corpus (10/10 Passed)
```
1. Plain p5 canvas with background color: PASS (Latency: 4166ms)
2. Colored rectangle: PASS (Latency: 1947ms)
3. Multiple circles with colors: PASS (Latency: 1766ms)
4. Lines and paths: PASS (Latency: 2032ms)
5. Transforms (translate, rotate): PASS (Latency: 1917ms)
6. Text rendering: PASS (Latency: 1419ms)
7. WebGL primitive (box, sphere): PASS (Latency: 1503ms)
8. p5.brush primitive: PASS (Latency: 1287ms)
9. Multi-operation p5.brush drawing with fill: PASS (Latency: 1586ms)
10. Intentionally broken code: PASS (Graceful RUNTIME_ERROR handling, Latency: 4589ms)
Done! Passed: 10, Failed: 0
```
