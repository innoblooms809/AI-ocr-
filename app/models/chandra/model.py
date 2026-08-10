import os
import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor
from app.models.base_model import BaseOCRModel
from app.utils.logger import setup_logger

# Import chandra packages
try:
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
except ImportError:
    generate_hf = None
    BatchInputItem = None
    parse_markdown = None

logger = setup_logger("chandra_model")

def preprocess_image_for_chandra(img: Image.Image, max_dimension: int = 1280) -> Image.Image:
    """Enhance image contrast/sharpness and auto-downscale ultra high-res images for fast inference."""
    try:
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)

        # Scale down giant images to prevent GPU out-of-memory or excessive CPU latency
        width, height = img.size
        if max(width, height) > max_dimension:
            scale = max_dimension / float(max(width, height))
            new_w = int(width * scale)
            new_h = int(height * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"⚡ Auto-scaled image for Chandra OCR from {width}x{height} → {new_w}x{new_h}")

        # Contrast & sharpness auto-conditioning for light/handwritten scans
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.1)
        return img
    except Exception as e:
        logger.warning(f"Image preprocessing warning: {e}")
        return img.convert("RGB")


class ChandraOCRModel(BaseOCRModel):
    def load(self) -> None:
        logger.info(f"Loading Chandra OCR model from: {self.model_path}")
        
        # Determine device & precision
        device_type = self.device
        if device_type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU.")
            device_type = "cpu"
            
        if device_type == "cpu":
            torch_dtype = torch.float32
            device_map = None
        else:
            torch_dtype = torch.bfloat16
            device_map = "auto"
        
        # Resolve model path. Fallback to HF Hub repo if local folder missing/empty
        actual_model_path = self.model_path
        if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
            fallback_repo = "datalab-to/chandra-ocr-2"
            logger.info(f"Local model path '{self.model_path}' not found or empty. Using HF Hub: '{fallback_repo}'")
            actual_model_path = fallback_repo
            
        # Load model & processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            actual_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(actual_model_path)
        self.processor.tokenizer.padding_side = "left"
        self.model.processor = self.processor  # Attribute required by generate_hf
        
        self.device = device_type
        logger.info(f"✅ Chandra OCR-2 model and processor loaded successfully (Device: {self.device}, Dtype: {torch_dtype}).")

    def predict(self, image: Image.Image, task: str = "ocr", max_new_tokens: int = 1024) -> str:
        if self.model is None or self.processor is None:
            self.load()
            
        # 1. Preprocess and enhance image
        processed_img = preprocess_image_for_chandra(image)
        
        # 2. Run Chandra inference
        prompt_type = "ocr_layout"
        logger.info(f"Running Chandra OCR prediction with prompt_type: '{prompt_type}'")

        if generate_hf is not None and BatchInputItem is not None and parse_markdown is not None:
            batch = [BatchInputItem(image=processed_img, prompt_type=prompt_type)]
            with torch.no_grad():
                results = generate_hf(batch, self.model)
                
            if not results:
                raise RuntimeError("Chandra OCR generated empty results.")
                
            raw_text = results[0].raw
            markdown_output = parse_markdown(raw_text)
        else:
            # Native Transformer fallback
            logger.info("Chandra package utils unavailable, using native transformers fallback...")
            inputs = self.processor(images=processed_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            markdown_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Free GPU memory after inference if on CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return markdown_output

