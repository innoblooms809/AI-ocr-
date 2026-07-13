from PIL import Image
from typing import Dict, Optional, Union
from app.models.base_model import BaseOCRModel
from app.ocr.loader import load_model
from app.preprocessing.image_loader import load_image
from app.preprocessing.resize import resize_image
from app.preprocessing.image_enhancer import enhance_contrast
from app.preprocessing.deskew import deskew_image
from app.preprocessing.denoise import denoise_image
from app.utils.logger import setup_logger
from app.utils.helpers import time_it

logger = setup_logger("predictor")

# Global cache of loaded model instances to avoid reloading weights
_LOADED_MODELS: Dict[str, BaseOCRModel] = {}

def get_cached_model(adapter_type: str, model_path: str, device: str = "cuda") -> BaseOCRModel:
    """Get loaded model from cache or load it if not present."""
    cache_key = f"{adapter_type}:{model_path}:{device}"
    if cache_key not in _LOADED_MODELS:
        logger.info(f"Model {cache_key} not in cache. Loading...")
        model_instance = load_model(adapter_type, model_path, device)
        _LOADED_MODELS[cache_key] = model_instance
    else:
        logger.info(f"Model {cache_key} fetched from cache.")
    return _LOADED_MODELS[cache_key]

def clear_model_cache() -> None:
    """Clear all loaded models from memory."""
    global _LOADED_MODELS
    _LOADED_MODELS.clear()
    logger.info("Model cache cleared.")

@time_it
def run_ocr_pipeline(
    image_source: Union[str, bytes],
    adapter_type: str,
    model_path: str,
    task: str,
    device: str = "cuda",
    max_new_tokens: int = 1024,
    do_resize: bool = True,
    max_dim: int = 768,
    do_deskew: bool = False,
    do_denoise: bool = False,
    contrast_factor: float = 1.0
) -> str:
    """
    Executes the entire image OCR processing pipeline.
    """
    logger.info("Starting OCR Pipeline execution.")
    
    # 1. Load image
    img = load_image(image_source)
    
    # 2. Preprocess image
    if do_resize:
        img = resize_image(img, max_dim=max_dim)
    if do_deskew:
        img = deskew_image(img)
    if do_denoise:
        img = denoise_image(img)
    if contrast_factor != 1.0:
        img = enhance_contrast(img, factor=contrast_factor)
        
    # 3. Retrieve model from cache
    model_instance = get_cached_model(adapter_type, model_path, device)
    
    # 4. Model Prediction
    logger.info(f"Invoking model prediction. Task: {task}")
    output = model_instance.predict(img, task=task, max_new_tokens=max_new_tokens)
    
    logger.info("OCR Pipeline execution completed.")
    return output
