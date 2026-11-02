"""Tests for ocr_server.py page conditioning and HTTP layer (the model is stubbed).

Run:  python -m pytest tests/test_ocr_server.py -q
"""

import io
import os
import sys

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ocr_server as o  # noqa: E402


def lined_page(angle: float = 0.0, size=(1200, 1600)) -> Image.Image:
    """A page of dark text-like bars, tilted by `angle` degrees."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for y in range(150, size[1] - 150, 60):
        d.rectangle([120, y, size[0] - 120, y + 14], fill="black")
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")


def test_blank_page():
    assert o.is_blank_page(Image.new("RGB", (800, 1100), "white"))
    assert not o.is_blank_page(lined_page())


def test_deskew_finds_the_tilt():
    assert o.estimate_skew(lined_page(0)) == 0.0
    angle = o.estimate_skew(lined_page(3.0))
    assert -3.6 <= angle <= -2.4      # the correction turns the page back
    angle = o.estimate_skew(lined_page(-2.0))
    assert 1.4 <= angle <= 2.6


def test_preprocess_resizes_to_max_dim_and_straightens():
    img, rotation = o._preprocess_single(lined_page(3.0, size=(2400, 3200)), max_dim=2048)
    assert max(img.size) <= 2048 + 2
    assert rotation != 0.0


def test_confidence_any_script():
    assert o.compute_page_confidence("The mitochondria is the powerhouse of the cell.") > 0.8
    assert o.compute_page_confidence("प्रकाश संश्लेषण वह प्रक्रिया है जिससे पौधे भोजन बनाते हैं।") > 0.8
    assert o.compute_page_confidence("") == 0.0
    assert o.compute_page_confidence("|||| #### ---- ^^^^") < 0.2      # OCR noise
    assert o.compute_page_confidence("2 + 2 = 4") >= 0.5               # a numerical answer


def test_combine_pages_has_page_markers():
    pages = [o.PageOCR(page=1, markdown="1. Answer"), o.PageOCR(page=2, markdown="continues")]
    text = o.combine_pages(pages)
    assert text.startswith("--- Page 1 ---") and "--- Page 2 ---\n\ncontinues" in text


@pytest.fixture
def client(monkeypatch):
    def fake_batch(images):
        return [{"markdown": f"page {i + 1} text", "blank": False, "rotation": 0.0, "retried": False, "truncated": i == 1}
                for i in range(len(images))]
    monkeypatch.setattr(o, "run_batch_ocr", fake_batch)
    monkeypatch.setattr(o, "INTERNAL_API_TOKEN", "")
    return TestClient(o.app)


def _png() -> bytes:
    buf = io.BytesIO()
    lined_page().save(buf, format="PNG")
    return buf.getvalue()


def _pdf(pages: int) -> bytes:
    import fitz
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    return doc.tobytes()


def test_ocr_endpoints(client):
    for path in ("/ocrOutput", "/v1/ocr"):
        r = client.post(path, files={"file": ("sheet.png", _png(), "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["combined_markdown"] == "--- Page 1 ---\n\npage 1 text"
    r = client.post("/v1/ocr", files={"file": ("sheet.pdf", _pdf(2), "application/pdf")})
    body = r.json()
    assert body["total_pages"] == 2 and body["pages"][1]["truncated"] is True
    assert any("Page 2" in w for w in body["warnings"])


def test_limits(client, monkeypatch):
    monkeypatch.setattr(o, "OCR_MAX_PAGES", 3)
    assert client.post("/v1/ocr", files={"file": ("big.pdf", _pdf(4), "application/pdf")}).status_code == 413
    monkeypatch.setattr(o, "OCR_MAX_FILE_BYTES", 10)
    assert client.post("/v1/ocr", files={"file": ("sheet.png", _png(), "image/png")}).status_code == 413
    assert client.post("/v1/ocr", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code in (400, 413)


def test_token_and_ready(client, monkeypatch):
    monkeypatch.setattr(o, "INTERNAL_API_TOKEN", "tok")
    assert client.post("/v1/ocr", files={"file": ("sheet.png", _png(), "image/png")}).status_code == 401
    ok = client.post("/v1/ocr", files={"file": ("sheet.png", _png(), "image/png")}, headers={"X-Internal-Token": "tok"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503   # no model in tests
