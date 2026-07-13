from abc import ABC, abstractmethod
from PIL import Image

class BaseOCRModel(ABC):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        """Load model and processor/tokenizer into memory."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image, task: str, max_new_tokens: int = 1024) -> str:
        """Run prediction on the PIL image for the specified task."""
        pass
