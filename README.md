# Chandra OCR-2 — Enterprise OCR Server

A local OCR server built on **Chandra OCR-2** (`datalab-to/chandra-ocr-2`). It accepts PNG/JPG/WEBP images and PDFs and returns structured Markdown.

Includes:
1. **FastAPI OCR server** (`ocr_server.py`): `POST /v1/ocr`, plus `GET /health` and `GET /ready`.
2. **Production launcher** (`run_production.py` / `run_production.ps1`).

---

## 📂 Project Structure

```
AI-OCR/
│
├── ocr_server.py                  # Main OCR server (Chandra OCR-2)
├── run_production.py              # Production launcher (uvicorn, 1 worker)
├── run_production.ps1             # Windows launcher (sets Intel MKL/OMP env vars)
│
├── app/
│   ├── config/
│   │   └── settings.py
│   ├── models/
│   │   ├── base_model.py          # Model adapter interface
│   │   └── chandra/
│   │       └── model.py           # Chandra OCR-2 adapter
│   ├── ocr/
│   │   ├── loader.py              # Model registry (chandra only)
│   │   ├── predictor.py
│   │   ├── parser.py
│   │   └── postprocess.py
│   ├── preprocessing/             # Image loading / enhancement helpers
│   ├── extraction/                # Question / answer extraction
│   ├── utils/
│   └── api/
│       └── main.py                # Alternative FastAPI app
│
├── models/
│   └── chandra-ocr-2/             # Chandra OCR-2 checkpoint
│
├── tests/
│   ├── test_ocr_server.py
│   └── test_model_inference.py
│
├── requirements.txt
├── .env
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Activate the virtual environment** (PowerShell):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Check `.env`**:
   - `DEFAULT_MODEL_NAME=chandra`
   - `DEFAULT_MODEL_PATH=models/chandra-ocr-2`
   - `DEVICE=cuda` (or `cpu` without a GPU)

---

## 🚀 Running the Server

Production mode (recommended):
```powershell
.\run_production.ps1
```

Or run the launcher directly:
```powershell
.\venv\Scripts\python.exe run_production.py
```

Dev mode with auto-reload:
```powershell
.\venv\Scripts\python.exe -m uvicorn ocr_server:app --host 0.0.0.0 --port 8000 --reload
```

Check that it is up:
- `GET http://localhost:8000/health` — the process is alive
- `GET http://localhost:8000/ready` — 200 once the model is loaded, 503 otherwise

---

## 🔗 Example Request

```bash
curl -X POST "http://localhost:8000/v1/ocr" \
  -F "file=@your_document.pdf"
```

If `INTERNAL_API_TOKEN` is set, also send `-H "X-Internal-Token: <token>"`.

---

## 🧪 Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_ocr_server.py
```
