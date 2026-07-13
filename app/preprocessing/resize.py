from PIL import Image
from app.utils.logger import setup_logger

logger = setup_logger("resize")

def resize_image(image: Image.Image, max_dim: int = 1536) -> Image.Image:
    """
    Resize the image so its maximum dimension is at most `max_dim`,
    preserving aspect ratio.
    """
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
        
    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
        
    logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
