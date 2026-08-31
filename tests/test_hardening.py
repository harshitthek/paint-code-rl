import sys; import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
import os
import json
import hashlib
from paint_rl.storage.cache import CacheManager
from paint_rl.config.core import load_config
from paint_rl.trainer.train_grpo import save_experiment_state, resume_experiment_state
from paint_rl.config import core as config

def test_cache():
    if os.path.exists("artifacts/test_cache.db"): os.remove("artifacts/test_cache.db")
    cache = CacheManager("artifacts/test_cache.db")
    c_hash = "abc"
    r_hash = "def"
    prompt = "draw a circle"
    model = "gpt-4o-mini"
    version = "v1"
    
    key = cache.generate_vlm_key(c_hash, r_hash, prompt, model, version)
    
    assert cache.get_vlm(key) is None
    
    cache.set_vlm(key, "left", {"raw": True}, 1.5, model)
    
    res = cache.get_vlm(key)
    assert res is not None
    assert res["decision"] == "left"
    assert res["cache_hit"] is True

def test_config():
    c, c_json, c_hash = load_config("local")
    assert c.model.id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert c.storage.base_path == "artifacts"
    
    c_kaggle, _, _ = load_config("kaggle")
    assert c_kaggle.storage.base_path == "/kaggle/working/artifacts"

def test_checkpointing():
    # Setup test environment for checkpointing
    config.ACTIVE_CONFIG.storage.base_path = "artifacts/test_cp"
    config.CONFIG_HASH = "mock_hash"
    config.CONFIG_JSON = "{}"
    
    save_experiment_state(10)
    
    state = resume_experiment_state()
    assert state["step"] == 10
    
    # Test mismatch
    config.CONFIG_HASH = "different_hash"
    with pytest.raises(ValueError, match="CRITICAL: Attempting to resume with mismatched config hash"):
        resume_experiment_state()
