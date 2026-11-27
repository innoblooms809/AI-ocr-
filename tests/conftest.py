"""Test settings: test runs send no LangSmith traces.

.env.local turns tracing on for the real server. ocr_server never lets .env.local override a
variable that is already set, so the values set here (before any test imports it) win.
"""
import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OCR_TELEMETRY_SAVE"] = "false"
