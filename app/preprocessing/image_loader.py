import io
import requests
from PIL import Image
from app.utils.logger import setup_logger

logger = setup_logger("image_loader")

def load_image(source: io.BytesIO | bytes | str) -> Image.Image:
    """
    Load an image from bytes, file path, or URL.
    Returns:
        PIL.Image.Image in RGB format.
    """
    if isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    elif hasattr(source, "read"):  # handles file-like objects (BytesIO, etc.)
        image = Image.open(source)
    elif isinstance(source, str):
        if source.startswith(("http://", "https://")):
            logger.info(f"Loading image from URL: {source}")
            resp = requests.get(source, timeout=15)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
        else:
            logger.info(f"Loading image from file path: {source}")
            image = Image.open(source)
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")
    
    return image.convert("RGB")
