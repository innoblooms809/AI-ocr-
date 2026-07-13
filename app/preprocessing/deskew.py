from PIL import Image
from app.utils.logger import setup_logger

logger = setup_logger("deskew")

def deskew_image(image: Image.Image) -> Image.Image:
    """
    Placeholder function to deskew the image.
    To implement auto-deskewing:
    1. Convert PIL image to grayscale numpy array.
    2. Threshold / detect text lines (e.g. using Hough Transform).
    3. Calculate skew angle.
    4. Rotate the image back.
    """
    logger.info("Skipping deskewing (not implemented/stubbed).")
    return image
