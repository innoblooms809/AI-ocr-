import streamlit as st
import os
import torch
from PIL import Image
import importlib

# Force reload models and predictor to prevent caching errors in Streamlit
import app.models.paddleocr_vl.model
import app.models.chandra.model
import app.ocr.loader
import app.ocr.predictor
importlib.reload(app.models.paddleocr_vl.model)
importlib.reload(app.models.chandra.model)
importlib.reload(app.ocr.loader)
importlib.reload(app.ocr.predictor)

from app.ocr.loader import list_local_models, MODEL_REGISTRY
from app.ocr.predictor import run_ocr_pipeline, clear_model_cache
from app.ocr.parser import parse_ocr_text
from app.ocr.postprocess import clean_markdown
from app.utils.constants import TASKS

# Set styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .qa-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🔍 Modular OCR & VLM Playbench</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload documents, process images, and test PaddleOCR-VL or other custom vision-language models locally.</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.header("🛠️ Model Configuration")

# 1. Scan and select local models
local_models = list_local_models("models")
model_options = [f"{m['name']} ({m['path']})" for m in local_models]

# Option to manually type a path (for minimal effort additions)
use_custom_path = st.sidebar.checkbox("Use custom model folder path", value=len(local_models) == 0)

if use_custom_path:
    model_path = st.sidebar.text_input(
        "Model Folder Path (local)", 
        value="models/PaddleOCR-VL"
    )
    model_name = os.path.basename(model_path)
else:
    selected_model_idx = st.sidebar.selectbox(
        "Select Downloaded Model",
        options=range(len(model_options)),
        format_func=lambda x: model_options[x]
    )
    selected_model = local_models[selected_model_idx]
    model_path = selected_model["path"]
    model_name = selected_model["name"]

# 2. Select Model Adapter Type
adapter_type = st.sidebar.selectbox(
    "Model Adapter / Class",
    options=list(MODEL_REGISTRY.keys()),
    help="Determines the tokenizer and generation format code mapping."
)

# 3. Select Inference Engine Configs
st.sidebar.subheader("⚙️ Inference Parameters")
task = st.sidebar.selectbox(
    "Vision Task",
    options=TASKS,
    format_func=lambda x: x.upper()
)

max_new_tokens = st.sidebar.slider(
    "Max Output Tokens", 
    min_value=128, 
    max_value=2048, 
    value=1024, 
    step=128
)

device = st.sidebar.selectbox(
    "Device Target",
    options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
    help="Force CPU if running low on VRAM."
)

# 4. Preprocessing Settings
st.sidebar.subheader("🖼️ Image Preprocessing")
do_resize = st.sidebar.checkbox("Resize Image", value=True, help="Prevents CPU slow-down and OOM on large images")
max_dim = st.sidebar.slider(
    "Max Resize Dimension",
    min_value=256,
    max_value=1536,
    value=768,
    step=128,
    help="Smaller sizes process MUCH faster on CPU."
)
do_deskew = st.sidebar.checkbox("Deskew (Stubbed)", value=False)
do_denoise = st.sidebar.checkbox("Denoise (Stubbed)", value=False)
contrast_factor = st.sidebar.slider("Contrast Adjustment", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# Actions
if st.sidebar.button("🧹 Clear Models Cache", help="Frees GPU memory by unloading model models"):
    clear_model_cache()
    st.sidebar.success("Cache cleared!")

# ----------------- MAIN AREA -----------------
uploaded_file = st.file_uploader(
    "Choose a document image...", 
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    # Reset cached OCR results if a new file is uploaded
    file_name = uploaded_file.name
    if "current_file" not in st.session_state or st.session_state.current_file != file_name:
        st.session_state.current_file = file_name
        if "ocr_results" in st.session_state:
            del st.session_state.ocr_results

    # Load and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Pipeline Pipeline Config")
        st.write(f"**Model Name:** `{model_name}`")
        st.write(f"**Model Path:** `{model_path}`")
        st.write(f"**Adapter Type:** `{adapter_type}`")
        st.write(f"**Engine Mode:** `{task.upper()}`")
        
        run_btn = st.button("🚀 Run OCR Analysis", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Processing image and running inference..."):
            try:
                # Run OCR
                file_bytes = uploaded_file.getvalue()
                raw_output = run_ocr_pipeline(
                    image_source=file_bytes,
                    adapter_type=adapter_type,
                    model_path=model_path,
                    task=task,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    do_resize=do_resize,
                    max_dim=max_dim,
                    do_deskew=do_deskew,
                    do_denoise=do_denoise,
                    contrast_factor=contrast_factor
                )
                
                # Postprocess
                cleaned_text = clean_markdown(raw_output)
                parsed_data = parse_ocr_text(cleaned_text)
                
                st.session_state.ocr_results = {
                    "raw_output": raw_output,
                    "cleaned_text": cleaned_text,
                    "parsed_data": parsed_data
                }
            except Exception as e:
                st.error(f"Error executing OCR pipeline: {str(e)}")
                st.exception(e)

    # Persist and display results in the downside page area
    if "ocr_results" in st.session_state:
        results = st.session_state.ocr_results
        raw_output = results["raw_output"]
        cleaned_text = results["cleaned_text"]
        parsed_data = results["parsed_data"]

        # Show OCR results directly at the bottom
        st.subheader("📝 OCR Results")
        
        st.markdown("### Rendered Markdown")
        st.markdown(cleaned_text)
        
        with st.expander("💾 Raw Output & Metadata"):
            st.markdown("### Raw Output")
            st.text(raw_output)
            
            st.markdown("### JSON Structured Result")
            structured_output = {
                "model": model_name,
                "task": task,
                "adapter": adapter_type,
                "cleaned_text": cleaned_text,
                "parsed_structure": parsed_data
            }
            st.json(structured_output)
else:
    st.info("Please upload an image in the file uploader above to begin testing.")
