from PIL import Image
from app.utils.logger import setup_logger

logger = setup_logger("denoise")

def denoise_image(image: Image.Image) -> Image.Image:
    """
    Placeholder function to denoise the image.
    To implement denoising:
    - Apply GaussianBlur or MedianFilter using PIL's ImageFilter.
    - Or apply bilateral filtering/non-local means denoising via OpenCV (cv2).
    """
    logger.info("Skipping denoising (not implemented/stubbed).")
    return image
