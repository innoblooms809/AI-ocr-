from PIL import Image, ImageEnhance

def enhance_contrast(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Enhance the contrast of the PIL image."""
    if factor == 1.0:
        return image
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def enhance_brightness(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Enhance the brightness of the PIL image."""
    if factor == 1.0:
        return image
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def enhance_sharpness(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Enhance the sharpness of the PIL image."""
    if factor == 1.0:
        return image
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)
