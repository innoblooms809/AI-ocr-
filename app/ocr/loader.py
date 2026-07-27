import os
from typing import Dict, Type, List
from app.models.base_model import BaseOCRModel
from app.models.paddleocr_vl.model import PaddleOCRVLModel
from app.models.chandra.model import ChandraOCRModel
from app.utils.logger import setup_logger

logger = setup_logger("model_loader")

# Registry mapping model architecture/adapter types to classes
MODEL_REGISTRY: Dict[str, Type[BaseOCRModel]] = {
    "paddleocr_vl": PaddleOCRVLModel,
    "chandra": ChandraOCRModel,
}

def list_local_models(models_dir: str = "models") -> List[Dict[str, str]]:
    """
    Scan the models directory and return a list of dicts with path and name.
    """
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for d in os.listdir(models_dir):
        full_path = os.path.join(models_dir, d)
        if os.path.isdir(full_path):
            # If it has config.json, it's a HuggingFace-style model
            if os.path.exists(os.path.join(full_path, "config.json")):
                models.append({
                    "name": d,
                    "path": full_path.replace("\\", "/")
                })
    return models

def load_model(adapter_type: str, model_path: str, device: str = "cuda") -> BaseOCRModel:
    """
    Factory function to load a model given the adapter type and its local path.
    """
    if adapter_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown adapter type: '{adapter_type}'. Available options: {list(MODEL_REGISTRY.keys())}")
        
    logger.info(f"Instantiating model adapter '{adapter_type}' for path '{model_path}' on device '{device}'")
    model_class = MODEL_REGISTRY[adapter_type]
    model_instance = model_class(model_path=model_path, device=device)
    model_instance.load()
    return model_instance
