import os
from dotenv import load_dotenv

# Load environments
load_dotenv()

# Settings
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "chandra")
DEFAULT_MODEL_PATH = os.getenv("DEFAULT_MODEL_PATH", "models/chandra-ocr-2")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEVICE = os.getenv("DEVICE", "cuda")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
