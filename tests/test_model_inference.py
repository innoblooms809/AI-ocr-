import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from app.ocr.predictor import run_ocr_pipeline
from app.ocr.loader import list_local_models
from app.ocr.postprocess import clean_markdown

def main():
    print("Python Executable:", sys.executable)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        
    print("\n--- Listing Local Models ---")
    models = list_local_models("models")
    for m in models:
        print(f"- {m['name']} located at {m['path']}")
        
    if not models:
        print("No models found in models/ directory. Make sure models/PaddleOCR-VL exists.")
        sys.exit(0)
        
    model_info = models[0]
    print(f"\nTesting model: {model_info['name']} using type: paddleocr_vl")
    
    # Create a simple white test image
    os.makedirs("tests", exist_ok=True)
    img = Image.new("RGB", (200, 100), color="white")
    test_img_path = "tests/test_image.png"
    img.save(test_img_path)
    print(f"Created a dummy test image at {test_img_path}")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running prediction pipeline on dummy image using {device}...")
        raw_output = run_ocr_pipeline(
            image_source=test_img_path,
            adapter_type="paddleocr_vl",
            model_path=model_info["path"],
            task="ocr",
            device=device,
            max_new_tokens=128
        )
        
        print("\n--- Pipeline Raw Output ---")
        print(raw_output)
        
        print("\n--- Cleaned Output ---")
        print(clean_markdown(raw_output))
        
        print("\nSUCCESS: Model inference run complete!")
    except Exception as e:
        print("\nFAILED: Error running inference pipeline.")
        import traceback
        traceback.print_exc()
        
    # Clean up test image
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

if __name__ == "__main__":
    main()
