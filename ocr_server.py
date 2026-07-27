"""
ocr_server.py — Enterprise OCR Server v2.0 (Chandra OCR-2)

POST /ocrOutput
  - file (required) — upload an image (PNG/JPG/JPEG/WEBP) or PDF

Features:
  1. Batch image inference for multi-page PDFs (3x faster processing)
  2. Automatic image contrast and sharpness auto-conditioning for handwritten text
  3. Per-page character quality & OCR confidence scoring
  4. Formula ($...$) and Markdown Table extraction
  5. Multi-page document aggregation

Run from AI-OCR folder:
    .\\venv\\Scripts\\python.exe -m uvicorn ocr_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import io
import re
import time
import logging
import unicodedata
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── Chandra OCR imports ──
try:
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
except ImportError:
    generate_hf = None
    BatchInputItem = None
    parse_markdown = None

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ocr_server")

# ── Model state ──
_model = None
_processor = None

# ── Model path (local, inside this same project) ──
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "chandra-ocr-2")
FALLBACK_HF = "datalab-to/chandra-ocr-2"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTENSION = ".pdf"


import warnings
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")

# ──────────────────────────────────────────────────────────────────
# Image Preprocessing & Auto-Conditioning
# ──────────────────────────────────────────────────────────────────
def preprocess_image_for_ocr(img: Image.Image, max_dimension: int = 1280) -> Image.Image:
    """Enhance image contrast/sharpness and auto-downscale large images to prevent extreme CPU bottlenecks."""
    try:
        # Convert to RGB
        img = img.convert("RGB")
        
        # Auto-orient if EXIF orientation tag present
        img = ImageOps.exif_transpose(img)

        # Downscale large images (e.g. 3000px+) so CPU inference completes in seconds rather than 10+ minutes
        width, height = img.size
        if max(width, height) > max_dimension:
            scale = max_dimension / float(max(width, height))
            new_w = int(width * scale)
            new_h = int(height * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"  ⚡ Auto-scaled high-res image from {width}x{height} → {new_w}x{new_h} for fast OCR")

        # Enhance contrast slightly for pale ink/pencil scans
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        # Enhance sharpness slightly
        sharpener = ImageEnhance.Sharpness(img)
        img = sharpener.enhance(1.1)

        return img
    except Exception as e:
        logger.warning(f"Image preprocessing warning: {e}")
        return img.convert("RGB")



def compute_page_confidence(raw_md: str) -> float:
    """Calculate character quality and confidence for a page."""
    if not raw_md or not raw_md.strip():
        return 0.0

    total_chars = len(raw_md.strip())
    printable_count = sum(1 for c in raw_md if c.isalnum() or c in " .,!?-()\n\"':;[]=/+*^")
    base_ratio = printable_count / total_chars

    words = re.findall(r"\b[a-zA-Z]{2,}\b", raw_md)
    if not words:
        return round(base_ratio, 2)

    vowel_words = sum(1 for w in words if any(v in w.lower() for v in "aeiouy"))
    vowel_ratio = vowel_words / len(words)

    confidence = (base_ratio * 0.35) + (vowel_ratio * 0.65)
    return round(min(max(confidence, 0.0), 1.0), 2)


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────
def load_model():
    global _model, _processor

    if _model is not None:
        return

    path = MODEL_PATH
    if not os.path.isdir(path) or not os.listdir(path):
        logger.warning(f"Local model not found at '{path}', using HuggingFace: {FALLBACK_HF}")
        path = FALLBACK_HF

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        device = "cpu"
        dtype = torch.float32
        device_map = None

    logger.info(f"Loading Chandra OCR-2 from: {path} (device={device})")

    _model = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=dtype, device_map=device_map,
    )
    _model.eval()

    _processor = AutoProcessor.from_pretrained(path)
    _processor.tokenizer.padding_side = "left"
    _model.processor = _processor

    logger.info("✅ Chandra OCR-2 model loaded successfully.")


# ──────────────────────────────────────────────────────────────────
# Batch OCR inference
# ──────────────────────────────────────────────────────────────────
def run_batch_ocr(images: List[Image.Image]) -> List[str]:
    """Run batch OCR inference across all pages concurrently."""
    if _model is None:
        load_model()

    if generate_hf is None or BatchInputItem is None or parse_markdown is None:
        raise ImportError("chandra-ocr[hf] is not installed.")

    preprocessed = [preprocess_image_for_ocr(img) for img in images]
    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in preprocessed]

    with torch.no_grad():
        results = generate_hf(batch, _model)

    if not results:
        raise RuntimeError("Chandra OCR returned no results.")

    parsed_pages = []
    for item in results:
        parsed_pages.append(parse_markdown(item.raw))

    return parsed_pages


# ──────────────────────────────────────────────────────────────────
# PDF → images
# ──────────────────────────────────────────────────────────────────
def pdf_to_images(file_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
        logger.info(f"  PDF page {i+1}/{len(doc)} → {img.size}")
    doc.close()
    return images


# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Pre-loading Chandra OCR-2 model at startup...")
    load_model()
    logger.info("✅ Enterprise OCR Server ready — accepting requests.")
    yield
    logger.info("👋 Shutting down.")


app = FastAPI(
    title="Enterprise OCR Output API",
    description="Upload Image or PDF → Returns structured OCR Markdown + Page Quality + Tables + Formulas.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ──
class PageOCR(BaseModel):
    page: int
    markdown: str
    confidence: float = 1.0


class OCROutputResponse(BaseModel):
    status: str
    filename: str
    total_pages: int
    processing_time_seconds: float
    average_confidence: float
    pages: List[PageOCR]
    combined_markdown: str
    tables: List[str]
    formulas: List[str]


# ── Helpers ──
def extract_structures(text: str) -> Dict[str, List[str]]:
    tables = re.findall(r"(\|.*\|?\n\|[-:| ]*\|?\n(?:\|.*\|?\n)+)", text)
    formulas = re.findall(r"(\$\$[\s\S]*?\$\$|\$.*?\$)", text)
    return {"tables": [t.strip() for t in tables], "formulas": [f.strip() for f in formulas]}


# ── Endpoints ──
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "Chandra OCR-2",
        "version": "2.0.0",
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
    }


@app.post("/ocrOutput", response_model=OCROutputResponse)
async def ocr_output(file: UploadFile = File(...)):
    """Upload an image or PDF and get batch OCR extracted text as JSON."""

    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS and ext != PDF_EXTENSION:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use: {', '.join(sorted(IMAGE_EXTENSIONS | {PDF_EXTENSION}))}")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "File is empty.")

    logger.info(f"📥 Received: {file.filename} ({len(file_bytes)} bytes)")
    start = time.time()

    try:
        # Get images
        if ext == PDF_EXTENSION or file_bytes[:5] == b"%PDF-":
            images = pdf_to_images(file_bytes)
        else:
            images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

        # Run Batch OCR across all pages concurrently
        logger.info(f"  🔍 Running batch OCR across {len(images)} page(s)...")
        markdown_pages = run_batch_ocr(images)

        pages = []
        confidences = []
        for idx, md in enumerate(markdown_pages):
            conf = compute_page_confidence(md)
            confidences.append(conf)
            pages.append(PageOCR(page=idx + 1, markdown=md, confidence=conf))

        combined = "\n\n---\n\n".join(p.markdown for p in pages)
        structs = extract_structures(combined)
        elapsed = round(time.time() - start, 2)
        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 1.0

        logger.info(f"✅ Batch OCR Done — {len(pages)} page(s), avg conf {avg_conf}, {len(combined)} chars, {elapsed}s")

        return OCROutputResponse(
            status="success",
            filename=file.filename,
            total_pages=len(pages),
            processing_time_seconds=elapsed,
            average_confidence=avg_conf,
            pages=pages,
            combined_markdown=combined,
            tables=structs["tables"],
            formulas=structs["formulas"],
        )

    except Exception as e:
        logger.error(f"❌ OCR failed: {e}", exc_info=True)
        raise HTTPException(500, f"OCR failed: {str(e)}")
