# pairwise_vlm.py
import os
import time
import requests
import base64
import hashlib
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from paint_rl.interfaces import JudgeProvider
from paint_rl.storage.cache import CacheManager

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        
def hash_image(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

class OpenAIJudgeProvider(JudgeProvider):
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.cache = CacheManager()
        self.api_key = os.environ.get("OPENAI_API_KEY")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _call_api(self, payload, headers):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    def compare(self, candidate_path: str, reference_path: str, prompt: str) -> dict:
        start_time = time.time()
        
        cand_hash = hash_image(candidate_path)
        ref_hash = hash_image(reference_path)
        
        # Check cache
        cache_key_lr = self.cache.generate_vlm_key(cand_hash, ref_hash, prompt, self.model_id, "v1_left_right")
        cache_key_rl = self.cache.generate_vlm_key(ref_hash, cand_hash, prompt, self.model_id, "v1_right_left")
        
        res_lr = self.cache.get_vlm(cache_key_lr)
        res_rl = self.cache.get_vlm(cache_key_rl)
        
        if not self.api_key and (not res_lr or not res_rl):
            return {"score": 0.0, "status": "VLM_API_ERROR", "error": "Missing OPENAI_API_KEY and cache miss.", "latency": time.time()-start_time}

        def fetch(left_path, right_path, c_key):
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model_id,
                "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "schema": {"type": "object", "properties": {"decision": {"type": "string", "enum": ["left", "right", "tie"]}}, "required": ["decision"], "additionalProperties": False}, "strict": True}},
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Prompt: {prompt}\nWhich image better matches this prompt?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(left_path)}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(right_path)}"}}
                    ]}
                ],
                "max_tokens": 50
            }
            try:
                raw = self._call_api(payload, headers)
                import json
                decision = json.loads(raw["choices"][0]["message"]["content"]).get("decision", "malformed")
                self.cache.set_vlm(c_key, decision, raw, time.time()-start_time, self.model_id)
                return decision
            except Exception as e:
                return "api_error"

        vote1 = res_lr["decision"] if res_lr else fetch(candidate_path, reference_path, cache_key_lr)
        vote2 = res_rl["decision"] if res_rl else fetch(reference_path, candidate_path, cache_key_rl)
        
        latency = time.time() - start_time
        
        if "api_error" in (vote1, vote2):
            return {"score": 0.0, "status": "VLM_API_ERROR", "latency": latency}
            
        if "malformed" in (vote1, vote2):
            return {"score": 0.0, "status": "VLM_MALFORMED_RESPONSE", "latency": latency}
            
        if vote1 == "left" and vote2 == "right": return {"score": 1.0, "status": "win", "latency": latency}
        if vote1 == "right" and vote2 == "left": return {"score": 0.0, "status": "loss", "latency": latency}
        
        status = "explicit_tie" if (vote1 == "tie" and vote2 == "tie") else "orientation_disagreement"
        return {"score": 0.5, "status": status, "latency": latency}
