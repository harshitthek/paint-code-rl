from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class RolloutEngine:
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def generate_and_evaluate(self, prompts, codes, reference_paths, seeds, reward_fn):
        """
        Submits jobs to the background queue and yields completed rewards.
        This allows overlapping rendering/VLM execution with policy forward passes if hardware permits.
        """
        futures = {}
        for idx, (prompt, code, ref_path, seed) in enumerate(zip(prompts, codes, reference_paths, seeds)):
            future = self.executor.submit(reward_fn, prompt, code, ref_path, seed)
            futures[future] = idx
            
        results = [None] * len(prompts)
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res = future.result()
                results[idx] = res
            except Exception as e:
                # Catch-all for async execution failures. Validation handles specific NaNs/types.
                print(f"Rollout Worker Error for index {idx}: {e}")
                results[idx] = {"total": 0.0, "error_class": "WORKER_CRASH"}
                
        return results

    def shutdown(self):
        self.executor.shutdown(wait=True)
