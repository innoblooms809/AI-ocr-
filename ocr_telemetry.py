"""
ocr_telemetry.py — what each document cost the OCR server, for the AI server's dashboard (OCR tab).

Per document: pages (blank, re-read, cut off), each step with its start and length (render,
straighten/resize, every Chandra batch with its wait for the model), output tokens per page,
tokens per second, characters read, average confidence, the GPU peak of each batch, memory at the
end, the LangSmith trace and the sheet's thread id (X-Trace-Thread-Id from the backend), so the
dashboard can put a sheet's OCR next to its evaluation.

The last OCR_TELEMETRY_KEEP documents stay in memory and are served by GET /telemetry/documents
(behind the server's token, like /v1/ocr). Each finished document is also appended to
logs/telemetry/ocr-YYYY-MM-DD.jsonl (OCR_TELEMETRY_SAVE=false turns that off; files older than
OCR_TELEMETRY_KEEP_DAYS are removed), so the history survives a restart. No page text is kept.

Input tokens per page (image / OCR instruction / chat template) and the time to the first token
are measured around Chandra's model call (ocr_server._instrument_model).
Pure Python (torch / psutil when present). Nothing here may break a request.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional

logger = logging.getLogger("ocr_server.telemetry")

KEEP = max(1, int(os.getenv("OCR_TELEMETRY_KEEP", "200")))
SAVE = os.getenv("OCR_TELEMETRY_SAVE", "true").strip().lower() != "false"
KEEP_DAYS = max(1, int(os.getenv("OCR_TELEMETRY_KEEP_DAYS", "14")))
DIR = Path(os.getenv("OCR_TELEMETRY_DIR", str(Path(__file__).resolve().parent / "logs" / "telemetry")))
GPU_COST_PER_HOUR = float(os.getenv("GPU_COST_PER_HOUR", "0") or 0)

_lock = threading.Lock()
_docs: Deque[Dict[str, Any]] = deque(maxlen=KEEP)
_history_loaded = False
_version = 0          # +1 when a document starts or finishes (GET /telemetry/stream pushes it)


def version() -> int:
    return _version


def _bump() -> None:
    global _version
    with _lock:
        _version += 1


_current: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar("ocr_doc", default=None)


def _mb(n: float) -> float:
    return round(float(n) / 2 ** 20, 1)


@contextmanager
def document(filename: str, size: int, pdf: bool, thread_id: str = "", backend_run_id: str = "",
             trace_id: str = "") -> Iterator[Dict[str, Any]]:
    """Collect one document's figures (the pages' work in worker threads joins it through the context)."""
    _ensure_history()
    doc: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:12], "filename": filename, "bytes": size, "pdf": pdf,
        "thread_id": thread_id or None, "backend_run_id": backend_run_id or None, "trace_id": trace_id or None,
        "started_at": time.time(), "ended_at": None, "seconds": None, "status": "running", "error": "",
        "pages": 0, "blank_pages": [], "retried_pages": [], "cut_off_pages": [], "characters": 0,
        "average_confidence": None, "warnings": [], "render_s": 0.0, "condition_s": 0.0,
        "steps": [], "batches": [], "page_tokens": {}, "page_sizes": [],
    }
    with _lock:
        _docs.append(doc)
    _bump()
    token = _current.set(doc)
    try:
        yield doc
        doc["status"] = "ok"
    except BaseException as e:
        doc["status"] = "failed"
        doc["error"] = f"{type(e).__name__}: {e}"[:300]
        raise
    finally:
        _current.reset(token)
        doc["ended_at"] = time.time()
        doc["seconds"] = round(doc["ended_at"] - doc["started_at"], 3)
        doc["memory"] = memory()
        _save(doc)
        _bump()


def step(name: str, seconds: float, started_at: Optional[float] = None) -> None:
    """A step of the current document: "render" (PDF pages → images), "condition" (straighten, resize)."""
    doc = _current.get()
    if doc is None:
        return
    doc[f"{name}_s"] = round(doc.get(f"{name}_s", 0.0) + seconds, 3)
    start = (started_at if started_at is not None else time.time() - seconds) - doc["started_at"]
    doc["steps"].append({"name": name, "start_s": round(max(start, 0.0), 3), "seconds": round(seconds, 3)})


def page_sizes(sizes: List[str]) -> None:
    doc = _current.get()
    if doc is not None:
        doc["page_sizes"] = sizes[:60]


def batch(pages: List[int], tokens: List[int], reason: str, max_tokens: int, wait_s: float, gen_s: float,
          vram_peak_mb: Optional[float], error: str = "", started_at: Optional[float] = None,
          inputs: Optional[List[Dict[str, Any]]] = None, ttft_s: Optional[float] = None,
          prep_s: Optional[float] = None) -> None:
    """One Chandra batch of the current document (`started_at`: when it began waiting for the model;
    `inputs`: per page {"input", "image", "prompt"} tokens; `ttft_s`: model start → first token;
    `prep_s`: Chandra turning the pages into model inputs)."""
    doc = _current.get()
    if doc is None:
        return
    out = sum(tokens)
    start = (started_at if started_at is not None else time.time() - wait_s - gen_s) - doc["started_at"]
    doc["batches"].append({
        "reason": reason, "pages": pages, "tokens_per_page": tokens, "output_tokens": out, "max_tokens": max_tokens,
        "start_s": round(max(start, 0.0), 3), "wait_s": round(wait_s, 3), "gen_s": round(gen_s, 3),
        "tokens_per_s": round(out / gen_s, 1) if gen_s > 0 and out else None,
        "cut_off_pages": [p for p, t in zip(pages, tokens) if t >= max_tokens - 2],
        "vram_peak_mb": vram_peak_mb, "error": error,
        "input_tokens_per_page": [r.get("input") for r in inputs] if inputs else None,
        "image_tokens_per_page": [r.get("image") for r in inputs] if inputs else None,
        "instruction_tokens": inputs[0].get("prompt") if inputs else None,
        "input_tokens": sum(r.get("input") or 0 for r in inputs) if inputs else None,
        "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
        "prep_s": round(prep_s, 3) if prep_s is not None else None,
    })
    for p, t in zip(pages, tokens):
        doc["page_tokens"][str(p)] = doc["page_tokens"].get(str(p), 0) + t
    if inputs:
        for p, r in zip(pages, inputs):
            doc.setdefault("page_input_tokens", {})[str(p)] = r.get("input")
            doc.setdefault("page_image_tokens", {})[str(p)] = r.get("image")


def finish(pages: List[Dict[str, Any]], characters: int, average_confidence: float,
           warnings: Optional[List[str]] = None) -> None:
    """The pages' outcome (from the response) for the current document."""
    doc = _current.get()
    if doc is None:
        return
    doc["pages"] = len(pages)
    doc["blank_pages"] = [p["page"] for p in pages if p.get("blank")]
    doc["retried_pages"] = [p["page"] for p in pages if p.get("retried")]
    doc["cut_off_pages"] = [p["page"] for p in pages if p.get("truncated")]
    doc["page_confidence"] = {str(p["page"]): p.get("confidence") for p in pages}
    doc["characters"] = characters
    doc["average_confidence"] = average_confidence
    doc["warnings"] = list(warnings or [])[:20]


def summary(doc: Dict[str, Any]) -> Dict[str, Any]:
    batches = doc.get("batches") or []
    gen = sum(b["gen_s"] for b in batches)
    out = sum(b["output_tokens"] for b in batches)
    pages_read = max(1, (doc.get("pages") or 0) - len(doc.get("blank_pages") or []))
    vram = [b["vram_peak_mb"] for b in batches if b.get("vram_peak_mb") is not None]
    rereads = [b for b in batches if b.get("reason") != "read"]
    measured = [b for b in batches if b.get("input_tokens") is not None]
    ttfts = [b["ttft_s"] for b in batches if b.get("ttft_s") is not None]
    inp = sum(b["input_tokens"] for b in measured)
    image = sum(sum(t or 0 for t in b.get("image_tokens_per_page") or []) for b in measured)
    instruction = sum((b.get("instruction_tokens") or 0) * len(b["pages"]) for b in measured)
    return {
        "input_tokens": inp if measured else None,
        "image_tokens": image if measured else None,
        "instruction_tokens": instruction if measured else None,
        "template_tokens": max(inp - image - instruction, 0) if measured else None,
        "total_tokens": (inp + out) if measured else None,
        "ttft_avg_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "prefill_tps": round(inp / sum(b["ttft_s"] for b in measured if b.get("ttft_s")), 1)
        if measured and any(b.get("ttft_s") for b in measured) else None,
        "batches": len(batches), "rereads": len(rereads), "output_tokens": out, "gen_s": round(gen, 3),
        "wait_s": round(sum(b["wait_s"] for b in batches), 3),
        "tokens_per_s": round(out / gen, 1) if gen > 0 and out else None,
        "tokens_per_page": round(out / pages_read, 1) if out else 0,
        "seconds_per_page": round((doc.get("seconds") or 0) / max(1, doc.get("pages") or 1), 2),
        "largest_page_tokens": max((doc.get("page_tokens") or {"": 0}).values() or [0]),
        "vram_peak_mb": max(vram) if vram else None,
        "cost": round(gen * GPU_COST_PER_HOUR / 3600.0, 6) if GPU_COST_PER_HOUR > 0 else None,
    }


def documents(limit: int = 50) -> List[Dict[str, Any]]:
    """Newest first, each with its summary."""
    _ensure_history()
    with _lock:
        docs = list(_docs)[-limit:][::-1]
    return [dict(d, summary=summary(d)) for d in docs]


def memory() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import psutil
        proc, vm = psutil.Process(), psutil.virtual_memory()
        out.update(rss_mb=_mb(proc.memory_info().rss), threads=proc.num_threads(),
                   ram_available_mb=_mb(vm.available), ram_total_mb=_mb(vm.total))
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info(0)
            out.update(vram_allocated_mb=_mb(torch.cuda.memory_allocated(0)),
                       vram_reserved_mb=_mb(torch.cuda.memory_reserved(0)),
                       gpu_used_all_processes_mb=_mb(total - free), gpu_total_mb=_mb(total))
    except Exception:
        pass
    return out


def vram_mark() -> None:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass


def vram_peak() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return _mb(torch.cuda.max_memory_allocated(0))
    except Exception:
        pass
    return None


# ── History on disk ─────────────────────────────────────────────────────────────

def _save(doc: Dict[str, Any]) -> None:
    if not SAVE:
        return
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        day = datetime.fromtimestamp(doc["started_at"]).strftime("%Y-%m-%d")
        line = json.dumps(doc, ensure_ascii=False, default=str)
        with _lock, open(DIR / f"ocr-{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning(f"[Telemetry] Could not save document {doc.get('id')}: {e}")


def _ensure_history() -> None:
    """Once per process: load the newest saved documents and remove files older than KEEP_DAYS."""
    global _history_loaded
    if _history_loaded:
        return
    with _lock:
        if _history_loaded:
            return
        _history_loaded = True
        if not SAVE or not DIR.is_dir():
            return
        try:
            cutoff = (datetime.now() - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
            files = sorted(DIR.glob("ocr-*.jsonl"))
            for f in files:
                if f.stem[4:] < cutoff:
                    f.unlink(missing_ok=True)
            loaded: List[Dict[str, Any]] = []
            for f in reversed([f for f in files if f.exists()]):
                for line in reversed(f.read_text(encoding="utf-8").splitlines()):
                    if len(loaded) >= KEEP:
                        break
                    try:
                        doc = json.loads(line)
                    except Exception:
                        continue
                    if doc.get("status") == "running":
                        doc["status"] = "interrupted"
                    loaded.append(doc)
                if len(loaded) >= KEEP:
                    break
            known = {d["id"] for d in _docs}
            fresh = list(_docs)
            _docs.clear()
            _docs.extend([d for d in reversed(loaded) if d.get("id") not in known] + fresh)
        except Exception as e:
            logger.warning(f"[Telemetry] Could not load saved documents from {DIR}: {e}")
