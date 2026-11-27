"""
ocr_server.py — Enterprise OCR Server v3.1 (Chandra OCR-2)
Optimized for: Dual Intel Xeon Platinum 8173M | 56 Physical Cores / 112 Threads | 256 GB RAM
               Windows Server 2022 Standard x64 (Build 20348) — and the shared 80 GB GPU.

POST /v1/ocr   (same as the original POST /ocrOutput, which stays)
  - file (required) — an image (PNG/JPG/JPEG/WEBP) or a PDF

What a page goes through:
  1. PDF pages rendered at OCR_PDF_DPI (200) in parallel
  2. EXIF rotation, then deskew (phone photos of sheets are often tilted a few degrees)
  3. Blank pages detected and skipped (no model time on an empty back page)
  4. Resized to OCR_MAX_DIM — 2048 px on GPU. Chandra is built for pages up to
     3072×2048 (its own scale_to_fit); the old 1024 px cut lost handwriting detail.
  5. Mild contrast / sharpness (pale ink, pencil)
  6. Chandra OCR-2 in mini-batches, up to OCR_MAX_OUTPUT_TOKENS per page. A page that
     hits the limit (cut off mid-answer) is read again alone with twice the budget; a
     page that comes back empty is read again without the contrast boost.
  7. combined_markdown joins pages with "--- Page N ---" lines, so the evaluator knows
     where each answer is and an answer running onto the next page stays together.

Production guards: INTERNAL_API_TOKEN (X-Internal-Token header), at most
OCR_MAX_CONCURRENT documents at once and OCR_MAX_WAITING queued (else 503 +
Retry-After), size / page limits, GET /health (live) and GET /ready (model loaded).

Run (dev):
    .\\venv\\Scripts\\python.exe -m uvicorn ocr_server:app --host 0.0.0.0 --port 8000

Run (production — no file watch overhead):
    .\\venv\\Scripts\\python.exe run_production.py
"""

import os
import sys
import io
import re
import hmac
import time
import logging
import asyncio
import json
import threading
import unicodedata
import warnings
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor


def _load_local_env(path: str) -> None:
    """KEY=VALUE lines of the git-ignored .env.local (secrets such as the LangSmith key) into the
    environment; a variable already set in the real environment wins. Read before langsmith loads."""
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip().removeprefix("export ").strip()
                if key and key not in os.environ:
                    os.environ[key] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[ocr_server] Could not read {path}: {e}", file=sys.stderr)


_load_local_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Intel MKL / OpenMP / KMP tuning — MUST be set BEFORE torch import
#
# Dual Xeon Platinum 8173M:  2 sockets × 28 physical cores = 56 physical cores
#                            112 logical threads (HyperThreading)
#
# Benchmark result on this exact hardware:
#   threads=112 → GEMM 0.46s   (cross-socket NUMA lock contention)
#   threads= 56 → GEMM 0.31s
#   threads= 28 → GEMM 0.13s  ← OPTIMAL (single-socket, avoids QPI bus traffic)
#   threads= 14 → GEMM 0.13s  (same, not faster)
#   threads=  8 → GEMM 0.22s
#
# We use 28 intra-op threads (one full NUMA socket of physical cores) for the
# model forward pass, and reserve the remaining 28+ cores for:
#   - Parallel PDF page rasterization (ThreadPoolExecutor)
#   - Parallel image preprocessing  (ThreadPoolExecutor)
#   - Uvicorn async I/O             (asyncio event loop)
# ─────────────────────────────────────────────────────────────────────────────
CPU_LOGICAL   = os.cpu_count() or 112          # 112 logical threads
CPU_PHYSICAL  = CPU_LOGICAL // 2               # 56 physical cores (HT x2)
NUMA_SOCKETS  = 2
CORES_PER_SOCKET = max(1, CPU_PHYSICAL // NUMA_SOCKETS)  # 28 per socket

# Optimal intra-op threads = one NUMA socket of physical cores
TORCH_INTRAOP_THREADS = CORES_PER_SOCKET          # 28
TORCH_INTEROP_THREADS = max(NUMA_SOCKETS, 4)      # 4  (pipeline parallelism)

# Workers for parallel I/O tasks (PDF rendering, image preprocessing)
# Use the *other* NUMA socket's cores so they don't compete with torch GEMM
IO_WORKERS = min(CORES_PER_SOCKET, 28)            # 28

# Apply BEFORE torch / MKL init
os.environ["MKL_NUM_THREADS"]      = str(TORCH_INTRAOP_THREADS)
os.environ["OMP_NUM_THREADS"]      = str(TORCH_INTRAOP_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(TORCH_INTRAOP_THREADS)
os.environ["KMP_AFFINITY"]         = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"]        = "1"          # ms before idle spin → sleep
os.environ["KMP_SETTINGS"]         = "0"          # suppress KMP info spam

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# LangSmith traces: off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set (.env.local).
# The same module as the AI evaluator's; both report to one agent, told apart by `service`.
import langsmith_tracing as lst  # noqa: E402
lst.configure("ocr-server", "3.1.0")
# Per-document figures (pages, tokens, time, GPU) for the AI server's dashboard: GET /telemetry/documents.
import ocr_telemetry as ot  # noqa: E402

# ─── Suppress noisy warnings ──────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*processor.*")

# ─── FastAPI / Pydantic ───────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ─── PyTorch ─────────────────────────────────────────────────────────────────
import torch

# Set thread counts AFTER import
torch.set_num_threads(TORCH_INTRAOP_THREADS)
try:
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
except RuntimeError:
    pass   # Already set by another module; harmless

# ─── Vision / PDF ────────────────────────────────────────────────────────────
import numpy as np
import fitz              # PyMuPDF
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor

# ─── Chandra OCR helpers ──────────────────────────────────────────────────────
try:
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
except ImportError:
    generate_hf  = None
    BatchInputItem = None
    parse_markdown = None

# Optional: psutil for RAM telemetry in /health
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ocr_server")

logger.info(
    f"🖥️  Hardware profile: {CPU_LOGICAL} logical CPUs | {CPU_PHYSICAL} physical cores "
    f"| {NUMA_SOCKETS} NUMA sockets | {CORES_PER_SOCKET} cores/socket"
)
logger.info(
    f"⚙️  PyTorch threads: intra-op={TORCH_INTRAOP_THREADS} | inter-op={TORCH_INTEROP_THREADS}"
)
logger.info(f"🔄 I/O ThreadPool workers: {IO_WORKERS}")

# ─────────────────────────────────────────────────────────────────────────────
# Global model state & thread-safety
# ─────────────────────────────────────────────────────────────────────────────
_model:     Optional[Any] = None
_processor: Optional[Any] = None
_ocr_model_lock = threading.Lock()
_model_error: str = ""

# Shared executor for PDF rasterization + image preprocessing
_io_executor = ThreadPoolExecutor(max_workers=IO_WORKERS, thread_name_prefix="ocr-io")

# ─────────────────────────────────────────────────────────────────────────────
# Model paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "chandra-ocr-2")
FALLBACK_HF = "datalab-to/chandra-ocr-2"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTENSION    = ".pdf"

IS_GPU = torch.cuda.is_available()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# Batch size: 8 pages per CUDA batch (GPU) | 1 page per CPU pass (avoids padding & L3 thrashing)
OCR_BATCH_SIZE = _env_int("OCR_BATCH_SIZE", 8 if IS_GPU else 1)
# Longest side a page is sent at. Chandra resizes to at most 3072×2048 itself and is made
# for pages at least 1536 px; 2048 keeps handwriting legible. CPU: smaller, for speed.
OCR_MAX_DIM = _env_int("OCR_MAX_DIM", 2048 if IS_GPU else 1280)
# Output budget per page. A dense handwritten page in markdown easily passes 1536 tokens
# (the old limit), which cut answers off mid-page. Chandra's own default is 12384.
OCR_MAX_OUTPUT_TOKENS = _env_int("OCR_MAX_OUTPUT_TOKENS", 6144 if IS_GPU else 3072)
OCR_MAX_OUTPUT_TOKENS_RETRY = _env_int("OCR_MAX_OUTPUT_TOKENS_RETRY", 12288)
OCR_PDF_DPI = _env_int("OCR_PDF_DPI", 200)
OCR_DESKEW = os.getenv("OCR_DESKEW", "true").lower() != "false"
OCR_MAX_FILE_BYTES = int(float(os.getenv("OCR_MAX_FILE_MB", "30")) * 1024 * 1024)
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 40)
# Back-pressure: documents read at once, and how many more may wait (then 503 + Retry-After).
OCR_MAX_CONCURRENT = max(1, _env_int("OCR_MAX_CONCURRENT", 2 if IS_GPU else 1))
OCR_MAX_WAITING = max(0, _env_int("OCR_MAX_WAITING", 8))
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "").strip()
OPEN_PATHS = {"/health", "/ready"}


# ─────────────────────────────────────────────────────────────────────────────
# Page conditioning (CPU-parallel via _io_executor)
# ─────────────────────────────────────────────────────────────────────────────
def is_blank_page(img: Image.Image, ink_ratio: float = 0.0012) -> bool:
    """A page with (almost) no ink — an empty back page, a blank scan."""
    g = img.convert("L")
    g.thumbnail((500, 500))
    arr = np.asarray(g, dtype=np.uint8)
    if arr.size == 0:
        return True
    dark = float((arr < 140).mean())
    return dark < ink_ratio


def estimate_skew(img: Image.Image, max_angle: float = 5.0, step: float = 0.5) -> float:
    """Angle (degrees, counter-clockwise) that makes the text lines horizontal.

    Projection profile: rotate a small binarised copy and keep the angle where the row
    sums vary most (rows of text and gaps between them line up). 0.0 when the page is
    straight, empty, or the gain is too small to trust."""
    g = img.convert("L")
    g.thumbnail((900, 900))
    arr = np.asarray(g, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    threshold = min(170.0, float(arr.mean()) - 0.5 * float(arr.std()))
    ink = (arr < threshold).astype(np.uint8) * 255
    if (ink > 0).mean() < 0.002:
        return 0.0
    ink_img = Image.fromarray(ink, mode="L")

    def score(angle: float) -> float:
        rows = np.asarray(ink_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0), dtype=np.float32).sum(axis=1)
        return float(np.var(rows))

    base = score(0.0)
    best_angle, best = 0.0, base
    angle = -max_angle
    while angle <= max_angle + 1e-6:
        if abs(angle) > 1e-6:
            s = score(angle)
            if s > best:
                best_angle, best = angle, s
        angle += step
    # Only correct a clear tilt; tiny gains are noise.
    if abs(best_angle) < 0.5 or best < base * 1.08:
        return 0.0
    return round(best_angle, 2)


def _preprocess_single(img: Image.Image, max_dim: int = OCR_MAX_DIM, enhance: bool = True) -> Tuple[Image.Image, float]:
    """
    Per-page conditioning → (image, rotation applied):
      1. Convert to RGB, auto-rotate via EXIF orientation
      2. Deskew (OCR_DESKEW) — tilted phone photos read line by line again
      3. Downscale to max_dim (only when larger)
      4. Mild contrast (×1.2) and sharpness (×1.1) — pale ink / pencil marks
    """
    rotation = 0.0
    try:
        img = ImageOps.exif_transpose(img.convert("RGB"))
        if OCR_DESKEW:
            rotation = estimate_skew(img)
            if rotation:
                img = img.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))

        w, h = img.size
        if max(w, h) > max_dim:
            scale  = max_dim / float(max(w, h))
            new_w  = int(w * scale)
            new_h  = int(h * scale)
            img    = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"  ⚡ Auto-scaled {w}×{h} → {new_w}×{new_h}")

        if enhance:
            img = ImageEnhance.Contrast(img).enhance(1.2)
            img = ImageEnhance.Sharpness(img).enhance(1.1)
        return img, rotation
    except Exception as e:
        logger.warning(f"Image preprocessing warning: {e}")
        return img.convert("RGB"), rotation


def preprocess_images_parallel(images: List[Image.Image], enhance: bool = True) -> List[Tuple[Image.Image, float]]:
    """
    Preprocess all pages in parallel using the I/O thread pool.
    Exploits the second NUMA socket's cores while PyTorch GEMM runs on the first.
    """
    futures = [_io_executor.submit(_preprocess_single, img, OCR_MAX_DIM, enhance) for img in images]
    return [f.result() for f in futures]


# ─────────────────────────────────────────────────────────────────────────────
# PDF → Images (parallel page rasterization)
# ─────────────────────────────────────────────────────────────────────────────
def _render_page(doc_bytes: bytes, page_idx: int, dpi: int) -> Image.Image:
    """Render a single PDF page in a worker thread (PyMuPDF is thread-safe per-doc-open)."""
    zoom   = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    doc    = fitz.open(stream=doc_bytes, filetype="pdf")
    try:
        pix = doc.load_page(page_idx).get_pixmap(matrix=matrix, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


def pdf_page_count(file_bytes: bytes) -> int:
    with fitz.open(stream=file_bytes, filetype="pdf") as probe:
        return len(probe)


def pdf_to_images(file_bytes: bytes, dpi: int = OCR_PDF_DPI) -> List[Image.Image]:
    """
    Rasterize all PDF pages in parallel via the I/O thread pool.
    On a 28-worker pool a 10-page PDF renders ~8× faster than sequential.
    """
    n_pages = pdf_page_count(file_bytes)
    logger.info(f"  📄 Rasterizing {n_pages}-page PDF at {dpi} DPI in parallel…")
    futures = [
        _io_executor.submit(_render_page, file_bytes, i, dpi)
        for i in range(n_pages)
    ]
    images = [f.result() for f in futures]
    for i, img in enumerate(images):
        logger.info(f"  PDF page {i+1}/{n_pages} → {img.size}")
    return images


# ─────────────────────────────────────────────────────────────────────────────
# Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────
def compute_page_confidence(raw_md: str) -> float:
    """Character quality of one page's text, for any script.

    Signal = share of letters, vowel signs (Devanagari matras) and digits among the
    non-space characters: OCR noise ("|||| ###") has almost none. Latin text is also
    checked for vowels (noise like "xqzt" has none). The old English-only check scored
    every Hindi page as garbage."""
    text = (raw_md or "").strip()
    if not text:
        return 0.0
    cats = [unicodedata.category(c) for c in text if not c.isspace()]
    if not cats:
        return 0.0
    signal = sum(1 for k in cats if k[0] in "LMN") / len(cats)
    letters = [c for c in text if unicodedata.category(c)[0] == "L"]
    latin = [c for c in letters if "a" <= c.lower() <= "z"]
    if letters and len(latin) / len(letters) >= 0.7:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
        vowel_ratio = (sum(1 for w in words if any(v in w.lower() for v in "aeiouy")) / len(words)) if words else 0.5
        confidence = 0.35 * signal + 0.65 * vowel_ratio
    else:
        confidence = signal
    return round(min(max(confidence, 0.0), 1.0), 2)


def combine_pages(pages: List["PageOCR"]) -> str:
    """One text for the whole document, each page under a "--- Page N ---" line."""
    return "\n\n".join(f"--- Page {p.page} ---\n\n{p.markdown}".rstrip() for p in pages)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model() -> None:
    global _model, _processor, _model_error

    if _model is not None:
        return

    path = MODEL_PATH
    has_weights = os.path.isdir(path) and any(
        os.path.exists(os.path.join(path, f))
        for f in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
    )
    if not has_weights:
        logger.warning(f"Local weights not found at '{path}', using HuggingFace Hub: {FALLBACK_HF}")
        path = FALLBACK_HF

    if IS_GPU:
        device     = "cuda"
        dtype      = torch.bfloat16
        device_map = "auto"
    else:
        device     = "cpu"
        dtype      = torch.float32
        device_map = None

    logger.info(f"⏳ Loading Chandra OCR-2 from: {path}  (device={device}, dtype={dtype})")
    t0 = time.time()

    def _load(model_path: str) -> None:
        global _model, _processor
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,           # not torch_dtype — avoids deprecation warning
            device_map=device_map,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path)
        processor.tokenizer.padding_side = "left"
        # Attach processor to model so chandra generate_hf can find it
        model.processor = processor
        _instrument_model(model)      # input tokens + first-token time of every call (Chandra reports neither)
        _model, _processor = model, processor

    try:
        _load(path)
    except Exception as err:
        if path != FALLBACK_HF:
            logger.warning(f"⚠️  Local load failed ({err}). Falling back to HuggingFace Hub.")
            _load(FALLBACK_HF)
        else:
            _model_error = f"{type(err).__name__}: {str(err)[:300]}"
            raise

    elapsed = round(time.time() - t0, 2)
    logger.info(
        f"✅ Chandra OCR-2 loaded in {elapsed}s  "
        f"| intra-op threads={torch.get_num_threads()} | max side {OCR_MAX_DIM}px | "
        f"{OCR_MAX_OUTPUT_TOKENS} tokens/page"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input tokens and first-token time of each Chandra call
#
# Chandra's generate_hf reports output tokens only. It calls model.generate(**inputs) itself, so the
# loaded model's generate is wrapped once: each page's input tokens come from its attention mask
# (left padding left out), its image tokens from the image token id, the OCR instruction from
# tokenizing Chandra's prompt; a streamer notes when the first token came. Same output.
# ─────────────────────────────────────────────────────────────────────────────
_gen_stats = threading.local()


class _TokenClock:
    """A generate() streamer that only notes times (the first put() is the prompt)."""

    def __init__(self) -> None:
        self.first: Optional[float] = None
        self._prompt_seen = False

    def put(self, value: Any) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
        elif self.first is None:
            self.first = time.time()

    def end(self) -> None:
        pass


def _instrument_model(model: Any) -> None:
    original = getattr(model, "generate", None)
    if original is None or getattr(original, "_viaexam_measured", False):
        return
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    prompt_tokens: Optional[int] = None
    try:
        from chandra.prompts import PROMPT_MAPPING
        prompt_tokens = len(model.processor.tokenizer(PROMPT_MAPPING["ocr_layout"], add_special_tokens=False)["input_ids"])
    except Exception:
        pass

    def generate(*args: Any, **kwargs: Any) -> Any:
        rows: List[Dict[str, Optional[int]]] = []
        try:
            ids, mask = kwargs.get("input_ids"), kwargs.get("attention_mask")
            for i in range(int(ids.shape[0])):
                total = int(mask[i].sum()) if mask is not None else int(ids.shape[-1])
                image = int((ids[i] == image_token_id).sum()) if image_token_id is not None else None
                rows.append({"input": total, "image": image, "prompt": prompt_tokens})
        except Exception:
            rows = []
        clock = _TokenClock()
        t0 = time.time()
        if kwargs.get("streamer") is None:
            kwargs["streamer"] = clock
        try:
            out = original(*args, **kwargs)
        except ValueError as e:
            if "streamer" not in str(e) or kwargs.get("streamer") is not clock:
                raise
            kwargs.pop("streamer")                 # a transformers build that refuses a streamer here
            out = original(*args, **kwargs)
        _gen_stats.last = {"rows": rows, "started": t0, "first": clock.first}
        return out

    generate._viaexam_measured = True  # type: ignore[attr-defined]
    model.generate = generate


# ─────────────────────────────────────────────────────────────────────────────
# Batch OCR inference
# ─────────────────────────────────────────────────────────────────────────────
def _generate(images: List[Image.Image], max_tokens: int, pages: Optional[List[int]] = None,
              reason: str = "read") -> List[Tuple[str, int]]:
    """(raw output, tokens used) per image, in mini-batches, one model call at a time.
    `pages` (page numbers) and `reason` only label the batches in LangSmith."""
    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images]
    pages = pages or list(range(1, len(images) + 1))
    out: List[Tuple[str, int]] = []
    total_batches = (len(batch) + OCR_BATCH_SIZE - 1) // OCR_BATCH_SIZE
    for i in range(0, len(batch), OCR_BATCH_SIZE):
        sub = batch[i : i + OCR_BATCH_SIZE]
        b_idx = i // OCR_BATCH_SIZE + 1
        logger.info(f"  🧠 OCR batch {b_idx}/{total_batches}  ({len(sub)} page(s))…")
        t_start = time.perf_counter()
        t_wait = t_gen = time.time()
        sub_results: List[Any] = []
        error = ""
        peak = None
        stats: Optional[Dict[str, Any]] = None
        try:
            # torch.inference_mode() = no_grad + no autograd version tracking → fastest CPU path
            with torch.inference_mode():
                with _ocr_model_lock:
                    t_gen = time.time()
                    ot.vram_mark()            # inside the lock: the peak is this batch's
                    _gen_stats.last = None
                    sub_results = generate_hf(sub, _model, max_output_tokens=max_tokens)
                    peak = ot.vram_peak()
                    stats = getattr(_gen_stats, "last", None)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            _trace_batch(sub, sub_results, pages[i : i + OCR_BATCH_SIZE], b_idx, total_batches, max_tokens,
                         reason, t_wait, t_gen, time.time(), error, peak, stats)
        logger.info(f"     ✓ batch {b_idx} done in {time.perf_counter() - t_start:.2f}s")
        out.extend((r.raw or "", int(getattr(r, "token_count", 0) or 0)) for r in sub_results)
    return out


def _trace_batch(sub: List[Any], results: List[Any], pages: List[int], b_idx: int, total: int, max_tokens: int,
                 reason: str, t_wait: float, t_gen: float, t_end: float, error: str,
                 vram_peak_mb: Optional[float] = None, stats: Optional[Dict[str, Any]] = None) -> None:
    """One Chandra batch: kept for the dashboard (ocr_telemetry), and sent to LangSmith as an llm run
    when tracing is on — the pages (their sizes, never the pictures), the text read from each,
    output tokens, and the wait for the model."""
    tokens = [int(getattr(r, "token_count", 0) or 0) for r in results]
    rows = (stats or {}).get("rows") or []
    first = (stats or {}).get("first")
    started = (stats or {}).get("started")
    ot.batch(pages, tokens, reason, max_tokens, t_gen - t_wait, t_end - t_gen, vram_peak_mb, error, started_at=t_wait,
             inputs=rows if len(rows) == len(pages) else None,
             ttft_s=(first - started) if first and started else None,
             prep_s=(started - t_gen) if started else None)
    if not lst.enabled():
        return
    texts = [str(getattr(r, "raw", "") or "") for r in results]
    images = []
    for p, item in zip(pages, sub):
        size = getattr(getattr(item, "image", None), "size", None)
        images.append(f"page {p} — {size[0]}x{size[1]} px" if size else f"page {p}")
    reply = "\n\n".join(f"--- Page {p} ---\n{t}" for p, t in zip(pages, texts))
    lst.log_llm(
        f"chandra-ocr-2 · {reason} · batch {b_idx}/{total}",
        start=t_gen, end=t_end, images=images, reply=reply, error=error,
        prompt=f"OCR (layout) of {len(sub)} page(s), up to {max_tokens} tokens each",
        usage=({"input_tokens": sum(r["input"] for r in rows), "output_tokens": sum(tokens),
                "total_tokens": sum(r["input"] for r in rows) + sum(tokens)} if rows else
               {"output_tokens": sum(tokens), "total_tokens": sum(tokens)}) if tokens else None,
        first_token=first,
        model="chandra-ocr-2", max_tokens=max_tokens, tags=[f"ocr:{reason}"],
        metadata={"pages": pages, "tokens_per_page": tokens, "lock_wait_s": round(t_gen - t_wait, 3),
                  "cut_off_pages": [p for p, t in zip(pages, tokens) if t >= max_tokens - 2],
                  "batch_size": len(sub), "device": "cuda" if IS_GPU else "cpu",
                  "tokens_per_s": round(sum(tokens) / (t_end - t_gen), 1) if tokens and t_end > t_gen else None,
                  "vram_peak_mb": vram_peak_mb,
                  "input_tokens_per_page": [r["input"] for r in rows] or None,
                  "image_tokens_per_page": [r["image"] for r in rows] or None,
                  "ocr_instruction_tokens": rows[0]["prompt"] if rows else None,
                  "ttft_s": round(first - started, 3) if first and started else None},
    )


def run_batch_ocr(images: List[Image.Image]) -> List[Dict[str, Any]]:
    """
    OCR every page → [{markdown, blank, rotation, retried, truncated}] in page order.

    Pipeline:
      1. Parallel conditioning (deskew, resize, contrast) on the I/O executor
      2. Blank pages skipped
      3. Chandra OCR-2 forward pass under torch.inference_mode()
      4. Pages cut off at the token limit are read again with a bigger budget; pages that
         came back empty are read again without the contrast boost
      5. Markdown parse of raw outputs
    """
    if _model is None:
        load_model()

    if generate_hf is None or BatchInputItem is None or parse_markdown is None:
        raise ImportError("chandra-ocr[hf] is not installed.  Run: pip install chandra-ocr[hf]")

    logger.info(f"  🖼️  Conditioning {len(images)} image(s) in parallel…")
    with lst.span("condition pages", inputs={"pages": len(images)}) as span:
        t_step = time.time()
        conditioned = preprocess_images_parallel(images)
        ot.step("condition", time.time() - t_step, t_step)
        span.end(outputs={"rotated": {str(i + 1): rot for i, (_, rot) in enumerate(conditioned) if rot}})
    pages: List[Dict[str, Any]] = [
        {"markdown": "", "blank": False, "rotation": rot, "retried": False, "truncated": False}
        for _, rot in conditioned
    ]
    todo = []
    for idx, (img, _) in enumerate(conditioned):
        if is_blank_page(img):
            pages[idx]["blank"] = True
            logger.info(f"  ⬜ Page {idx + 1} is blank — skipped")
        else:
            todo.append(idx)
    if not todo:
        return pages

    outputs = _generate([conditioned[i][0] for i in todo], OCR_MAX_OUTPUT_TOKENS, pages=[i + 1 for i in todo])
    for idx, (raw, tokens) in zip(todo, outputs):
        pages[idx]["markdown"] = parse_markdown(raw) if raw else ""
        pages[idx]["truncated"] = tokens >= OCR_MAX_OUTPUT_TOKENS - 2

    # Second chances, one page at a time (rare, so no batching needed).
    for idx in todo:
        page = pages[idx]
        if page["truncated"] and OCR_MAX_OUTPUT_TOKENS_RETRY > OCR_MAX_OUTPUT_TOKENS:
            logger.warning(f"  ✂️  Page {idx + 1} hit the {OCR_MAX_OUTPUT_TOKENS}-token limit — reading it again")
            raw, tokens = _generate([conditioned[idx][0]], OCR_MAX_OUTPUT_TOKENS_RETRY, pages=[idx + 1],
                                    reason="re-read cut-off page")[0]
            if raw:
                page["markdown"] = parse_markdown(raw)
                page["truncated"] = tokens >= OCR_MAX_OUTPUT_TOKENS_RETRY - 2
            page["retried"] = True
        elif len(page["markdown"].strip()) < 5:
            logger.warning(f"  🔁 Page {idx + 1} came back empty — reading it again without contrast boost")
            plain, _ = _preprocess_single(images[idx], OCR_MAX_DIM, enhance=False)
            raw, _ = _generate([plain], OCR_MAX_OUTPUT_TOKENS, pages=[idx + 1], reason="re-read empty page")[0]
            if raw:
                page["markdown"] = parse_markdown(raw)
            page["retried"] = True
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Back-pressure
# ─────────────────────────────────────────────────────────────────────────────
_doc_slots = threading.BoundedSemaphore(OCR_MAX_CONCURRENT)
_inflight_lock = threading.Lock()
_inflight = {"running": 0, "waiting": 0, "served": 0, "rejected": 0}


class ServiceBusy(Exception):
    pass


def _in_slot(fn):
    with _doc_slots:
        with _inflight_lock:
            _inflight["waiting"] -= 1
            _inflight["running"] += 1
        try:
            return fn()
        finally:
            with _inflight_lock:
                _inflight["running"] -= 1
                _inflight["served"] += 1


async def run_document(fn):
    """Runs one document's OCR in a worker thread when a slot frees; ServiceBusy when full."""
    with _inflight_lock:
        if _inflight["running"] + _inflight["waiting"] >= OCR_MAX_CONCURRENT + OCR_MAX_WAITING:
            _inflight["rejected"] += 1
            raise ServiceBusy()
        _inflight["waiting"] += 1
    return await run_in_threadpool(_in_slot, fn)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_error
    logger.info("🚀 Pre-loading Chandra OCR-2 at server startup…")
    # Run blocking model load in thread pool so the event loop stays responsive
    try:
        await asyncio.get_running_loop().run_in_executor(None, load_model)
        logger.info("✅ Enterprise OCR Server ready — accepting requests on :8000")
    except Exception as e:
        _model_error = _model_error or f"{type(e).__name__}: {str(e)[:300]}"
        logger.error(f"❌ Chandra OCR-2 failed to load: {_model_error} — /ready reports 503 until fixed")
    ls_state = lst.status()
    if ls_state["enabled"]:
        logger.info(f"🔭 LangSmith tracing ON → {ls_state['destination']} ({ls_state['endpoint']})")
    for problem in ls_state["problems"]:
        logger.warning(f"[LangSmith] {problem}")
    yield
    lst.flush()
    logger.info("👋 Shutting down gracefully.")
    _io_executor.shutdown(wait=False)


app = FastAPI(
    title="Enterprise OCR Output API",
    description=(
        "Upload an Image (PNG/JPG/JPEG/WEBP) or PDF → "
        "Returns structured OCR Markdown + Page Quality + Tables + Formulas.\n\n"
        "Optimized for: Dual Intel Xeon Platinum 8173M | 56 Cores / 112 Threads | 256 GB RAM"
    ),
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_internal_token(request: Request, call_next):
    """With INTERNAL_API_TOKEN set, only callers that send it can use the OCR endpoints."""
    if INTERNAL_API_TOKEN and request.url.path not in OPEN_PATHS and request.method != "OPTIONS":
        sent = request.headers.get("x-internal-token", "")
        auth = request.headers.get("authorization", "")
        if not sent and auth.lower().startswith("bearer "):
            sent = auth[7:].strip()
        if not sent or not hmac.compare_digest(sent.encode(), INTERNAL_API_TOKEN.encode()):
            return JSONResponse(status_code=401, content={
                "status": "error", "error": "unauthorized", "message": "A valid X-Internal-Token header is required.",
            })
    return await call_next(request)


@app.exception_handler(ServiceBusy)
async def busy_handler(_request: Request, _exc: ServiceBusy):
    return JSONResponse(status_code=503, headers={"Retry-After": "20"}, content={
        "status": "error", "error": "busy",
        "message": f"{_inflight['running']} document(s) being read and {_inflight['waiting']} waiting; try again shortly.",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────────────────────────────────────
class PageOCR(BaseModel):
    page:       int
    markdown:   str
    confidence: float = 1.0
    blank:      bool = False
    rotation:   float = 0.0     # degrees the page was turned to straighten it
    retried:    bool = False    # read a second time (cut off, or empty the first time)
    truncated:  bool = False    # still longer than the token budget: the end may be missing


class OCROutputResponse(BaseModel):
    status:                    str
    filename:                  str
    total_pages:               int
    processing_time_seconds:   float
    average_confidence:        float
    pages:                     List[PageOCR]
    combined_markdown:         str
    tables:                    List[str]
    formulas:                  List[str]
    warnings:                  List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_structures(text: str) -> Dict[str, List[str]]:
    tables   = re.findall(r"(\|.*\|?\n\|[-:| ]*\|?\n(?:\|.*\|?\n)+)", text)
    formulas = re.findall(r"(\$\$[\s\S]*?\$\$|\$.*?\$)", text)
    return {
        "tables":   [t.strip() for t in tables],
        "formulas": [f.strip() for f in formulas],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Live hardware & model health telemetry.
    Reports: CPU threads, NUMA info, RAM, model device, PyTorch version.
    """
    ram_info: Dict[str, Any] = {}
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        ram_info = {
            "total_ram_gb":     round(vm.total     / (1024 ** 3), 1),
            "available_ram_gb": round(vm.available / (1024 ** 3), 1),
            "used_ram_percent": vm.percent,
        }

    return {
        "status":          "healthy",
        "model":           "Chandra OCR-2",
        "version":         "3.1.0",
        "device":          "CUDA" if IS_GPU else "CPU",
        "pytorch_version": torch.__version__,
        "cpu": {
            "logical_threads":       CPU_LOGICAL,
            "physical_cores":        CPU_PHYSICAL,
            "numa_sockets":          NUMA_SOCKETS,
            "cores_per_socket":      CORES_PER_SOCKET,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "io_pool_workers":       IO_WORKERS,
        },
        **ram_info,
        "ocr_batch_size":  OCR_BATCH_SIZE,
        "page": {"max_side_px": OCR_MAX_DIM, "max_tokens": OCR_MAX_OUTPUT_TOKENS, "pdf_dpi": OCR_PDF_DPI, "deskew": OCR_DESKEW},
        "documents": dict(_inflight),
        "model_loaded":    _model is not None,
        "langsmith":       lst.status(),
    }


@app.get("/telemetry/documents")
async def telemetry_documents(limit: int = 50):
    """The last documents read: pages, tokens per page, time per step and batch, GPU peak, thread id,
    LangSmith trace link (read by the AI server's /dashboard; behind INTERNAL_API_TOKEN like /v1/ocr)."""
    await run_in_threadpool(lst.project_url)     # the LangSmith project link: two API calls once, then cached
    docs = ot.documents(max(1, min(limit, ot.KEEP)))
    for d in docs:
        d["trace_url"] = lst.trace_url(d.get("trace_id") or "")
    return {
        "server": {"service": "ocr-server", "version": "3.1.0", "model": "Chandra OCR-2",
                   "device": "cuda" if IS_GPU else "cpu", "model_loaded": _model is not None,
                   "batch_size": OCR_BATCH_SIZE, "max_side_px": OCR_MAX_DIM, "max_tokens_per_page": OCR_MAX_OUTPUT_TOKENS,
                   "retry_max_tokens": OCR_MAX_OUTPUT_TOKENS_RETRY, "pdf_dpi": OCR_PDF_DPI, "max_pages": OCR_MAX_PAGES,
                   "documents": dict(_inflight), "gpu_cost_per_hour": ot.GPU_COST_PER_HOUR,
                   "langsmith": lst.status()},
        "memory": ot.memory(),
        "documents": docs,
    }


@app.get("/telemetry/stream")
async def telemetry_stream(request: Request, max_seconds: float = 0):
    """Server-sent events for the AI server's dashboard: `data: {"version": n}` once on connect and
    again only when a document starts or finishes (plus a keep-alive comment every 20 s). One long
    request instead of polling; the version is checked in memory, nothing is logged meanwhile."""
    async def events():
        last, started, beat = None, time.time(), time.time()
        while not await request.is_disconnected():
            v = ot.version()
            if v != last:
                last, beat = v, time.time()
                yield f"data: {json.dumps({'version': v, 'running': _inflight['running']})}\n\n"
            elif time.time() - beat > 20:
                beat = time.time()
                yield ": keep-alive\n\n"
            if max_seconds and time.time() - started > max_seconds:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/ready")
async def ready():
    """200 once the model is loaded, else 503 (with the load error)."""
    ok = _model is not None
    return JSONResponse(status_code=200 if ok else 503, content={
        "ready": ok, "error": _model_error, "documents": dict(_inflight),
        "limits": {"concurrent": OCR_MAX_CONCURRENT, "waiting": OCR_MAX_WAITING,
                   "max_file_mb": round(OCR_MAX_FILE_BYTES / 1024 / 1024, 1), "max_pages": OCR_MAX_PAGES},
    })


async def _read_document(file: UploadFile, request: Optional[Request] = None) -> OCROutputResponse:
    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "File is empty.")
    if len(file_bytes) > OCR_MAX_FILE_BYTES:
        raise HTTPException(413, f"File is {len(file_bytes) / 1024 / 1024:.1f} MB; the limit is {OCR_MAX_FILE_BYTES / 1024 / 1024:.0f} MB.")

    # Detect file type (magic bytes > extension > MIME)
    is_pdf = file_bytes[:5] == b"%PDF-"
    ext    = os.path.splitext(file.filename)[1].lower() if file.filename else ""

    if not is_pdf and ext not in IMAGE_EXTENSIONS and ext != PDF_EXTENSION:
        mime = (file.content_type or "").lower()
        if "pdf" in mime:
            is_pdf = True
        elif not any(t in mime for t in ["image/png", "image/jpeg", "image/jpg", "image/webp"]):
            raise HTTPException(
                400,
                f"Unsupported file type '{ext}' (mime={mime}). "
                f"Accepted: {', '.join(sorted(IMAGE_EXTENSIONS | {PDF_EXTENSION}))}",
            )
    if is_pdf or ext == PDF_EXTENSION:
        try:
            n_pages = pdf_page_count(file_bytes)
        except Exception as e:
            raise HTTPException(400, f"The PDF could not be opened: {e}")
        if n_pages > OCR_MAX_PAGES:
            raise HTTPException(413, f"The PDF has {n_pages} pages; the limit is {OCR_MAX_PAGES}.")

    logger.info(f"📥 Received: {file.filename!r}  ({len(file_bytes):,} bytes  pdf={is_pdf})")
    start = time.time()

    def _sync_process() -> OCROutputResponse:
        """All CPU-heavy work runs here in a thread, keeping asyncio free."""
        with lst.span("render pages", inputs={"pdf": bool(is_pdf or ext == PDF_EXTENSION), "dpi": OCR_PDF_DPI}) as span:
            t_step = time.time()
            if is_pdf or ext == PDF_EXTENSION:
                images = pdf_to_images(file_bytes)
            else:
                images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
            ot.step("render", time.time() - t_step, t_step)
            ot.page_sizes([f"{im.size[0]}x{im.size[1]}" for im in images])
            span.end(outputs={"pages": len(images), "sizes": [f"{im.size[0]}x{im.size[1]}" for im in images[:40]]})

        logger.info(f"  🔍 Running batch OCR across {len(images)} page(s)…")
        results = run_batch_ocr(images)

        pages: List[PageOCR] = []
        confidences = []
        warnings_out: List[str] = []
        for idx, res in enumerate(results):
            md = res["markdown"]
            conf = 1.0 if res["blank"] else compute_page_confidence(md)
            if not res["blank"]:
                confidences.append(conf)
            if res["truncated"]:
                warnings_out.append(f"Page {idx + 1} is very long; its end may be missing.")
            if not res["blank"] and not md.strip():
                warnings_out.append(f"Page {idx + 1}: no text could be read.")
            pages.append(PageOCR(page=idx + 1, markdown=md, confidence=conf, blank=res["blank"],
                                 rotation=res["rotation"], retried=res["retried"], truncated=res["truncated"]))

        combined = combine_pages(pages)
        structs  = extract_structures(combined)
        elapsed  = round(time.time() - start, 2)
        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 1.0
        ot.finish([pg.model_dump() for pg in pages], len(combined), avg_conf, warnings_out)

        logger.info(
            f"✅ OCR Done — {len(pages)} page(s)  avg-conf={avg_conf}  "
            f"{len(combined):,} chars  {elapsed}s"
            + (f"  ⚠ {'; '.join(warnings_out)}" if warnings_out else "")
        )

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
            warnings=warnings_out,
        )

    try:
        # One worker thread per document, at most OCR_MAX_CONCURRENT at once. One LangSmith trace per
        # document — inside the caller's trace when it sent trace headers (the AI evaluator does),
        # in the answer sheet's thread when it sent X-Trace-Thread-Id (the backend does).
        with lst.inbound(request.headers if request is not None else {}), lst.root(
            f"ocr · {file.filename}", tags=["ocr"],
            inputs={"file": file.filename, "bytes": len(file_bytes), "pdf": bool(is_pdf or ext == PDF_EXTENSION)},
            metadata={"model": "chandra-ocr-2", "device": "cuda" if IS_GPU else "cpu"},
        ) as trace, ot.document(file.filename, len(file_bytes), bool(is_pdf or ext == PDF_EXTENSION),
                                thread_id=lst.thread_id(), backend_run_id=lst.inbound_info().get("backend_run_id", ""),
                                trace_id=trace.trace_id):
            result = await run_document(_sync_process)
            trace.end(outputs={
                "pages": result.total_pages, "average_confidence": result.average_confidence,
                "characters": len(result.combined_markdown), "warnings": result.warnings,
                "seconds": result.processing_time_seconds, "combined_markdown": result.combined_markdown,
            })
            return result
    except (HTTPException, ServiceBusy):
        raise
    except Exception as e:
        logger.error(f"❌ OCR failed: {e}", exc_info=True)
        raise HTTPException(500, f"OCR failed: {str(e)}")


@app.post("/v1/ocr", response_model=OCROutputResponse)
async def ocr_v1(request: Request, file: UploadFile = File(...)):
    """Read an image or PDF. Pages are separated by "--- Page N ---" in combined_markdown."""
    return await _read_document(file, request)


@app.post("/ocrOutput", response_model=OCROutputResponse)
async def ocr_output(request: Request, file: UploadFile = File(...)):
    """
    Upload an image or PDF → returns batch OCR as structured JSON.
    (Original path, kept for existing callers; same as POST /v1/ocr.)
    """
    return await _read_document(file, request)
