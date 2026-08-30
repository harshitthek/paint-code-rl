from abc import ABC, abstractmethod

class Renderer(ABC):
    @abstractmethod
    def render(self, code: str, seed: int, run_id: str) -> dict:
        pass

class VisualRewardProvider(ABC):
    @abstractmethod
    def compute(self, image_path: str, prompt: str) -> float:
        pass

class JudgeProvider(ABC):
    @abstractmethod
    def compare(self, candidate_path: str, reference_path: str, prompt: str) -> dict:
        pass

class StorageBackend(ABC):
    @abstractmethod
    def save_artifact(self, path: str, content: bytes):
        pass
    
    @abstractmethod
    def read_artifact(self, path: str) -> bytes:
        pass
