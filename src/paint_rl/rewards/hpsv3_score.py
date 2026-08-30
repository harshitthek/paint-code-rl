import time
from paint_rl.interfaces import VisualRewardProvider

class HPSv3Provider(VisualRewardProvider):
    def __init__(self):
        try:
            import hpsv3
            self.hpsv3 = hpsv3
            self.available = True
        except ImportError:
            self.available = False

    def compute(self, image_path: str, prompt: str) -> float:
        if not self.available:
            raise RuntimeError("HPSv3 library is not installed and MOCK_MODE is false. Failing closed.")
        # Real inference
        return self.hpsv3.score(image_path, prompt)

def compute_hpsv3(image_path: str, prompt: str) -> float:
    provider = HPSv3Provider()
    return provider.compute(image_path, prompt)
