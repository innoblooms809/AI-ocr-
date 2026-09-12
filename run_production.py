"""
run_production.py — Production launcher for Enterprise OCR Server v3.0

No --reload file-watcher overhead. Uvicorn tuned for high-throughput on:
  Dual Intel Xeon Platinum 8173M | 56 Physical Cores / 112 Threads | 256 GB RAM
  Windows Server 2022 Standard x64

Usage:
    .\\venv\\Scripts\\python.exe run_production.py

Or use the PowerShell script (recommended — sets Intel env vars):
    .\\run_production.ps1
"""

import os
import sys

# ─── NUMA / Intel MKL / KMP tuning ───────────────────────────────────────────
# Must be set before PyTorch / MKL initialize.
os.environ["MKL_NUM_THREADS"]      = "28"
os.environ["OMP_NUM_THREADS"]      = "28"
os.environ["OPENBLAS_NUM_THREADS"] = "28"
os.environ["KMP_AFFINITY"]         = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"]        = "1"
os.environ["KMP_SETTINGS"]         = "0"

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "ocr_server:app",

        # Network
        host="0.0.0.0",
        port=8000,

        # Workers: 1 process — model loaded once in memory (avoids 10GB × N-worker copies)
        # Use an external process manager (PM2 / Windows Service) if you need multi-process.
        workers=1,

        # No file-watch overhead in production
        reload=False,

        # Async event-loop settings
        loop="asyncio",

        # Access log format
        access_log=True,
        log_level="info",

        # Keep-alive (seconds) — long-lived connections for batched uploads
        timeout_keep_alive=120,

        # Header size limits (large PDF metadata)
        h11_max_incomplete_event_size=65536,
    )
