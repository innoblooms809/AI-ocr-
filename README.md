# Modular OCR and VLM Testing Benchmark

A modular, local testing bench for OCR and Vision-Language models. It is configured to run the `PaddleOCR-VL` model from your local weight files, and is built to easily test any other Vision-Language Models (such as Qwen2-VL, InternVL, etc.) that you download in the future.

Includes:
1. **Streamlit Playbench UI**: Upload document images, apply pre-processing, run OCR/table/formula/chart extractions, and inspect parsed QA lists.
2. **FastAPI Web Endpoint**: Run predictions programmatically via POST requests.
3. **Flexible Adapter Layout**: Swap or drop in standard HuggingFace models with zero code changes.

---

## 📂 Project Structure

```
ocr-project/
│
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── models/
│   │   ├── base_model.py          # Shared model adapter interface
│   │   ├── paddleocr_vl/
│   │   │   └── model.py           # Specialized PaddleOCR-VL model adapter
│   │   └── qwen/
│   │       └── model.py           # Standard HuggingFace VLM model adapter (e.g., Qwen2-VL)
│   ├── preprocessing/
│   │   ├── image_loader.py
│   │   ├── image_enhancer.py
│   │   ├── deskew.py
│   │   ├── denoise.py
│   │   └── resize.py
│   ├── ocr/
│   │   ├── loader.py              # Dynamic model path and config scanner
│   │   ├── predictor.py           # Runs full loading, preprocessing & prediction
│   │   ├── parser.py
│   │   ├── postprocess.py
│   │   └── layout.py
│   ├── extraction/
│   │   ├── question_extractor.py  # Regex question extractor
│   │   ├── answer_extractor.py    # Aligns answer text with questions
│   │   └── json_formatter.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── helpers.py
│   │   ├── file_utils.py
│   │   └── constants.py
│   └── api/
│       └── main.py                # FastAPI endpoints
│
├── models/
│   └── PaddleOCR-VL/              # Place model checkpoints here
│
├── data/
│   ├── input/
│   ├── processed/
│   └── output/
│
├── tests/
├── scripts/
├── logs/
│
├── requirements.txt
├── .env
├── .gitignore
├── streamlit_app.py                # Streamlit Web App Interface
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Activate Virtual Environment**:
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Check/Configure `.env`**:
   Adjust configurations in `.env`:
   - `DEFAULT_MODEL_NAME=paddleocr_vl`
   - `DEFAULT_MODEL_PATH=models/PaddleOCR-VL`
   - `DEVICE=cuda` (or `cpu` if running without a GPU)

---

## 🚀 Running the Streamlit Frontend

Start the playbench UI:
```powershell
.\venv\Scripts\streamlit run streamlit_app.py
```
Open the local URL in your browser (usually `http://localhost:8501`).
1. Upload an image.
2. Select the vision task (OCR, Table, Formula, Chart).
3. Click **Run OCR Analysis**.
4. View the parsed results inside the rendered markdown tab or structured Q&A tab!

---

## 🔗 Running the FastAPI Backend

Start the API server:
```powershell
.\venv\Scripts\python app/api/main.py
```
This runs the API on `http://localhost:8000`. You can query predictions programmatically:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.png" \
  -F "task=ocr" \
  -F "adapter_type=paddleocr_vl" \
  -F "model_path=models/PaddleOCR-VL"
```

---

## 📥 How to Download and Test Other Models

The system is designed to scan your `models/` directory dynamically. When you want to test a new model:

1. **Download the model checkpoint**:
   Download any standard HuggingFace Vision-Language Model (e.g. Qwen/Qwen2-VL-7B-Instruct or others) to a subdirectory inside `models/`.
   For example:
   ```
   models/
   ├── PaddleOCR-VL/
   └── Qwen2-VL-7B-Instruct/
       ├── config.json
       ├── model.safetensors
       └── ...
   ```
2. **Run Streamlit**:
   Open the Streamlit App. It will automatically detect `Qwen2-VL-7B-Instruct` in the dropdown select list!
3. **Select Class/Adapter Type**:
   - For Qwen or other standard VLM models, select **`qwen_vlm`** as the adapter.
   - Click **Run OCR Analysis**.
   *(No Python code changes are required for standard models!)*
4. **Custom Model Architectures**:
   If the model has custom inference logic, simply:
   1. Create an adapter in `app/models/new_model/model.py` subclassing `BaseOCRModel`.
   2. Register it in `MODEL_REGISTRY` inside `app/ocr/loader.py`.
