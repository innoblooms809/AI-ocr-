"""LangSmith tracing of the OCR server (langsmith_tracing.py), with Chandra and the LangSmith
client both faked: nothing leaves the machine.

Run:  python -m pytest tests/test_tracing.py -q
"""
import io
import os
import sys
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ocr_server as o  # noqa: E402
import langsmith_tracing as lst  # noqa: E402


class FakeResult:
    def __init__(self, raw, tokens):
        self.raw, self.token_count = raw, tokens


class FakeItem:
    def __init__(self, image, prompt_type):
        self.image, self.prompt_type = image, prompt_type


class FakeLangSmithClient:
    def __init__(self):
        self.created, self.lock = [], threading.Lock()

    def create_run(self, **kwargs):
        with self.lock:
            self.created.append(kwargs)

    def update_run(self, run_id=None, **kwargs):
        pass


@pytest.fixture
def traced(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_not_a_key")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:9")
    fake = FakeLangSmithClient()
    monkeypatch.setattr(lst, "_client_obj", fake)
    monkeypatch.setattr(o, "_model", object())
    monkeypatch.setattr(o, "BatchInputItem", FakeItem)
    monkeypatch.setattr(o, "parse_markdown", lambda raw: raw)
    monkeypatch.setattr(o, "generate_hf", lambda batch, model, max_output_tokens: [FakeResult("1. The answer", 42) for _ in batch])
    monkeypatch.setattr(o, "INTERNAL_API_TOKEN", "")
    return fake


def page_png() -> bytes:
    img = Image.new("RGB", (800, 1000), "white")
    d = ImageDraw.Draw(img)
    for y in range(100, 900, 50):
        d.rectangle([80, y, 720, y + 12], fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ocr_request_continues_the_callers_trace(traced):
    caller = lst._RunTree(name="evaluate (AI server)", run_type="chain", client=traced)
    headers = {**caller.to_headers(), "X-Trace-Thread-Id": "sheet-42"}
    # No `with`: the app's shutdown would stop the page-conditioning thread pool for later tests.
    r = TestClient(o.app).post("/v1/ocr", files={"file": ("page.png", page_png(), "image/png")}, headers=headers)
    assert r.status_code == 200, r.text

    runs = {c["name"]: c for c in traced.created}
    root = runs["ocr · page.png"]
    assert str(root["parent_run_id"]) == str(caller.id) and str(root["trace_id"]) == str(caller.trace_id)
    assert root["extra"]["metadata"]["thread_id"] == "sheet-42"
    assert root["extra"]["metadata"]["service"] == "ocr-server"
    for step in ("render pages", "condition pages"):
        assert str(runs[step]["parent_run_id"]) == str(root["id"])
    batch = runs["chandra-ocr-2 · read · batch 1/1"]
    assert batch["run_type"] == "llm" and str(batch["parent_run_id"]) == str(root["id"])
    meta = batch["extra"]["metadata"]
    assert meta["usage_metadata"]["output_tokens"] == 42 and meta["pages"] == [1] and meta["thread_id"] == "sheet-42"
    assert "1. The answer" in batch["outputs"]["choices"][0]["message"]["content"]

    # The same document in the figures the AI server's dashboard reads.
    doc = TestClient(o.app).get("/telemetry/documents").json()["documents"][0]
    assert doc["filename"] == "page.png" and doc["status"] == "ok" and doc["thread_id"] == "sheet-42"
    assert doc["pages"] == 1 and doc["page_tokens"] == {"1": 42} and doc["summary"]["output_tokens"] == 42
    assert doc["batches"][0]["reason"] == "read" and doc["render_s"] >= 0 and doc["trace_id"] == str(caller.trace_id)


def test_ocr_request_without_headers_starts_its_own_trace(traced):
    r = TestClient(o.app).post("/ocrOutput", files={"file": ("sheet.png", page_png(), "image/png")})
    assert r.status_code == 200, r.text
    root = [c for c in traced.created if c["name"] == "ocr · sheet.png"][0]
    assert root.get("parent_run_id") is None and "thread_id" not in root["extra"]["metadata"]


def test_document_figures_push_a_new_version(traced):
    import json
    import ocr_telemetry as ot
    v0 = ot.version()
    r = TestClient(o.app).post("/v1/ocr", files={"file": ("v.png", page_png(), "image/png")})
    assert r.status_code == 200 and ot.version() == v0 + 2       # started, finished
    with TestClient(o.app).stream("GET", "/telemetry/stream?max_seconds=1") as s:
        events = [line for line in "".join(s.iter_text()).splitlines() if line.startswith("data:")]
    assert len(events) == 1 and json.loads(events[0][5:])["version"] == ot.version()


def test_document_history_survives_a_restart(tmp_path, monkeypatch):
    import ocr_telemetry as ot
    monkeypatch.setattr(ot, "SAVE", True)
    monkeypatch.setattr(ot, "DIR", tmp_path)
    monkeypatch.setattr(ot, "_history_loaded", True)
    with ot.document("kept.pdf", 10, True, thread_id="sheet-9"):
        ot.step("render", 0.2)
        ot.batch([1, 2], [100, 50], "read", 3072, 0.1, 1.5, None)
    assert list(tmp_path.glob("ocr-*.jsonl"))
    monkeypatch.setattr(ot, "_docs", type(ot._docs)(maxlen=ot.KEEP))       # a new process
    monkeypatch.setattr(ot, "_history_loaded", False)
    doc = ot.documents()[0]
    assert doc["filename"] == "kept.pdf" and doc["thread_id"] == "sheet-9" and doc["status"] == "ok"
    assert doc["summary"]["output_tokens"] == 150 and doc["steps"][0]["name"] == "render"
