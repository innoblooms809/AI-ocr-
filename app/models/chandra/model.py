import os
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from app.models.base_model import BaseOCRModel
from app.utils.logger import setup_logger

# Import chandra packages. We do a try-except to avoid failure if the package isn't loaded yet.
try:
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
except ImportError:
    generate_hf = None
    BatchInputItem = None
    parse_markdown = None

logger = setup_logger("chandra_model")

class ChandraOCRModel(BaseOCRModel):
    def load(self) -> None:
        logger.info(f"Loading Chandra OCR model from: {self.model_path}")
        
        # Determine device
        device_type = self.device
        if device_type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU.")
            device_type = "cpu"
            
        if device_type == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16
        
        # Resolve model path. If the local directory does not exist or is empty,
        # fallback to the Hugging Face Hub repository ID.
        actual_model_path = self.model_path
        if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
            fallback_repo = "datalab-to/chandra-ocr-2"
            logger.info(f"Local model path '{self.model_path}' not found or empty. Using HuggingFace Hub: '{fallback_repo}'")
            actual_model_path = fallback_repo
            
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            actual_model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device_type == "cuda" else None,
        )
        self.model.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(actual_model_path)
        self.processor.tokenizer.padding_side = "left"
        self.model.processor = self.processor  # Set processor attribute on model as required by generate_hf
        
        self.device = device_type
        logger.info("Chandra OCR model and processor loaded successfully.")

    def predict(self, image: Image.Image, task: str, max_new_tokens: int = 1024) -> str:
        if self.model is None or self.processor is None:
            self.load()
            
        if generate_hf is None or BatchInputItem is None or parse_markdown is None:
            raise ImportError(
                "chandra-ocr is not fully installed or imports are failing. "
                "Ensure 'chandra-ocr[hf]' is installed in the environment."
            )
            
        # Standardize task mapping. Chandra OCR uses 'ocr_layout' for layout-preserving document extraction.
        # Since 'ocr_layout' extracts tables, formulas, charts, and text all at once, we use it by default.
        prompt_type = "ocr_layout"
        
        logger.info(f"Running Chandra OCR with prompt_type: '{prompt_type}'")
        
        batch = [
            BatchInputItem(
                image=image,
                prompt_type=prompt_type
            )
        ]
        
        # Run inference using chandra's generate_hf
        with torch.no_grad():
            results = generate_hf(batch, self.model)
            
        if not results:
            raise RuntimeError("Chandra OCR generated no results.")
            
        result = results[0]
        
        # Parse output to markdown
        markdown_output = parse_markdown(result.raw)
        return markdown_output
