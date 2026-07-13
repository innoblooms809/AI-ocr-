import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    processor = AutoProcessor.from_pretrained("models/PaddleOCR-VL", trust_remote_code=True)
    image = Image.new("RGB", (200, 100), color="white")
    prompt = "OCR:"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Method 1: apply_chat_template(tokenize=True)
    try:
        inputs_m1 = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        print("M1 keys:", list(inputs_m1.keys()))
        if "pixel_values" in inputs_m1:
            print("M1 pixel_values shape:", inputs_m1["pixel_values"].shape)
        else:
            print("M1 does NOT have pixel_values!")
    except Exception as e:
        print("M1 failed:", e)

    # Method 2: apply_chat_template(tokenize=False) + processor()
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("M2 text:", repr(text))
        inputs_m2 = processor(text=[text], images=[image], return_tensors="pt")
        print("M2 keys:", list(inputs_m2.keys()))
        if "pixel_values" in inputs_m2:
            print("M2 pixel_values shape:", inputs_m2["pixel_values"].shape)
    except Exception as e:
        print("M2 failed:", e)

if __name__ == "__main__":
    main()
