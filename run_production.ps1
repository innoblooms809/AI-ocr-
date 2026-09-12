<#
.SYNOPSIS
    Production launcher for Enterprise OCR Server v3.0
    Optimized for: Dual Intel Xeon Platinum 8173M | 56 Cores / 112 Threads | 256 GB RAM

.DESCRIPTION
    Sets Intel MKL, OpenMP, and KMP environment variables BEFORE launching Python
    so that all C/C++ math libraries respect the thread budget from process start.

    Uses 28 intra-op threads (single NUMA socket, avoids cross-QPI contention).

.USAGE
    From the AI-OCR directory:
        .\run_production.ps1

    Dev mode (with reload):
        .\venv\Scripts\python.exe -m uvicorn ocr_server:app --host 0.0.0.0 --port 8000 --reload
#>

# ── Intel MKL / KMP / OpenMP ──────────────────────────────────────────────────
# 28 = single NUMA socket physical cores (optimal for Xeon 8173M GEMM on CPU)
$env:MKL_NUM_THREADS      = "28"
$env:OMP_NUM_THREADS      = "28"
$env:OPENBLAS_NUM_THREADS = "28"
$env:KMP_AFFINITY         = "granularity=fine,compact,1,0"
$env:KMP_BLOCKTIME        = "1"
$env:KMP_SETTINGS         = "0"

# ── Optional: NUMA-aware process affinity (requires numactl or Process Lasso)
# To pin to socket 0 (cores 0-27) use Windows CPUSET before calling python:
#   Start-Process ... -ArgumentList ... -Affinity 0x0FFFFFFF

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Enterprise OCR Server v3.0 — Production Mode" -ForegroundColor Cyan
Write-Host "  Hardware: Dual Xeon Platinum 8173M | 56 Cores / 112 Threads" -ForegroundColor Cyan
Write-Host "  MKL / OMP threads : $($env:MKL_NUM_THREADS) (1 NUMA socket)"  -ForegroundColor Green
Write-Host "  KMP_AFFINITY      : $($env:KMP_AFFINITY)"                     -ForegroundColor Green
Write-Host "  KMP_BLOCKTIME     : $($env:KMP_BLOCKTIME)ms"                   -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# ── Launch ─────────────────────────────────────────────────────────────────────
$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python not found at: $python"
    Write-Host "Make sure your virtual environment is set up: python -m venv venv"
    exit 1
}

Write-Host "Starting OCR server at http://0.0.0.0:8000 ..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

& $python run_production.py
