import os
import sys

# Add project root to sys.path so 'app' can be found when executed directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.ocr.predictor import run_ocr_pipeline, get_cached_model
from app.ocr.parser import parse_ocr_text
from app.ocr.postprocess import clean_markdown
from app.extraction.question_extractor import extract_questions
from app.extraction.answer_extractor import extract_answers
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("api_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Pre-loading default model '{settings.DEFAULT_MODEL_NAME}' at startup...")
    try:
        get_cached_model(settings.DEFAULT_MODEL_NAME, settings.DEFAULT_MODEL_PATH, settings.DEVICE)
        logger.info("✅ Default model weights loaded successfully into memory.")
    except Exception as e:
        logger.warning(f"⚠️ Startup model pre-loading deferred: {e}. Model will load on first /predict request.")
    yield

app = FastAPI(
    title="Modular OCR API",
    description="FastAPI service for document extraction using PaddleOCR-VL and other VLMs",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "default_model": settings.DEFAULT_MODEL_NAME,
        "default_path": settings.DEFAULT_MODEL_PATH
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    adapter_type: str = Form(settings.DEFAULT_MODEL_NAME),
    model_path: str = Form(settings.DEFAULT_MODEL_PATH),
    task: str = Form("ocr"),
    max_new_tokens: int = Form(1024),
    device: str = Form(settings.DEVICE),
    do_resize: bool = Form(True),
    do_deskew: bool = Form(False),
    do_denoise: bool = Form(False),
    contrast_factor: float = Form(1.0)
):
    # Validate image file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # Execute pipeline
        raw_output = run_ocr_pipeline(
            image_source=image_bytes,
            adapter_type=adapter_type,
            model_path=model_path,
            task=task,
            device=device,
            max_new_tokens=max_new_tokens,
            do_resize=do_resize,
            do_deskew=do_deskew,
            do_denoise=do_denoise,
            contrast_factor=contrast_factor
        )
        
        # Post-process and parse structure
        cleaned_text = clean_markdown(raw_output)
        parsed_data = parse_ocr_text(cleaned_text)
        
        # QA extraction
        questions = extract_questions(cleaned_text)
        qa_pairs = extract_answers(cleaned_text, questions)
        
        return {
            "status": "success",
            "task": task,
            "adapter_type": adapter_type,
            "model_path": model_path,
            "raw_output": raw_output,
            "cleaned_text": cleaned_text,
            "parsed_structure": parsed_data,
            "extracted_qa": qa_pairs
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
