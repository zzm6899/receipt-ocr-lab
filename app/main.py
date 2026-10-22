from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import MODELS, catalog, run_model

ROOT = Path(__file__).resolve().parent
MAX_BYTES = int(os.getenv("OCR_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", 3))
MAX_PIXELS = int(os.getenv("OCR_MAX_PIXELS", 20_000_000))
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
MAX_MODELS = int(os.getenv("OCR_MAX_MODELS_PER_REQUEST", 2))
QUEUE_WAIT = float(os.getenv("OCR_QUEUE_WAIT_SECONDS", 5))
INFERENCE_TIMEOUT = float(os.getenv("OCR_INFERENCE_TIMEOUT_SECONDS", 60))
RATE_LIMIT = int(os.getenv("OCR_REQUESTS_PER_MINUTE", 5))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
SEMAPHORE = asyncio.Semaphore(int(os.getenv("OCR_CONCURRENCY", 1)))
REQUESTS: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(title="OCR Comparator", docs_url=None, redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def enforce_rate_limit(request: Request) -> None:
    now = time.monotonic()
    key = request.client.host if request.client else "unknown"
    history = REQUESTS[key]
    while history and history[0] < now - 60:
        history.popleft()
    if len(history) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests; try again in a minute.")
    history.append(now)


def validate_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    if width < 16 or height < 16 or width * height > MAX_PIXELS:
        raise HTTPException(400, "Image dimensions are outside the allowed range.")
    return image.convert("RGB")


def rasterize(data: bytes, media_type: str) -> list[Image.Image]:
    if media_type == "application/pdf":
        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise HTTPException(400, "Invalid PDF.") from exc
        try:
            if document.page_count < 1 or document.page_count > MAX_PAGES:
                raise HTTPException(400, f"PDFs must contain 1-{MAX_PAGES} pages.")
            pages = []
            for page in document:
                if page.rect.width * 1.5 * page.rect.height * 1.5 > MAX_PIXELS:
                    raise HTTPException(400, "Rendered PDF page dimensions exceed the allowed range.")
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                pages.append(validate_image(Image.open(io.BytesIO(pixmap.tobytes("png")))))
            return pages
        finally:
            document.close()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return [validate_image(image)]
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Invalid or unsupported image.") from exc


MONEY = re.compile(r"(?<!\w)(?:[$�?]\s*)?-?\d{1,7}(?:[.,]\d{2})(?!\w)")


def derived_fields(text: str) -> dict:
    fields: dict[str, str] = {}
    rules = {"total": r"\b(?:grand\s+)?total\b", "subtotal": r"\bsub\s*total\b", "tax": r"\b(?:tax|gst|vat)\b", "date": r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b"}
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        for key, pattern in rules.items():
            if key in fields:
                continue
            if key == "date":
                match = re.search(pattern, line)
                if match:
                    fields[key] = match.group(0)
            elif re.search(pattern, line, re.IGNORECASE):
                amounts = MONEY.findall(line)
                if amounts:
                    fields[key] = amounts[-1]
    return fields


def preview(image: Image.Image) -> dict:
    copy = image.copy()
    copy.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    copy.save(stream, "JPEG", quality=82, optimize=True)
    return {"data_url": "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii"), "width": copy.width, "height": copy.height}


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/models")
async def models():
    return {"models": catalog(), "limits": {"max_upload_bytes": MAX_BYTES, "max_pages": MAX_PAGES, "max_models_per_request": MAX_MODELS}}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/compare")
async def compare(request: Request, file: UploadFile = File(...), models: list[str] = Form(...)):
    enforce_rate_limit(request)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Only JPEG, PNG, WebP and PDF uploads are accepted.")
    if not models or len(models) > MAX_MODELS or len(set(models)) != len(models):
        raise HTTPException(400, f"Select 1-{MAX_MODELS} distinct models.")
    unknown = [key for key in models if key not in MODELS]
    unavailable = [key for key in models if key in MODELS and not MODELS[key].available()]
    if unknown or unavailable:
        raise HTTPException(400, {"unknown": unknown, "unavailable": unavailable})
    data = await file.read(MAX_BYTES + 1)
    await file.close()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413, f"Upload must be between 1 and {MAX_BYTES} bytes.")
    pages = rasterize(data, file.content_type)
    page_previews = [preview(page) for page in pages]
    try:
        await asyncio.wait_for(SEMAPHORE.acquire(), timeout=QUEUE_WAIT)
    except TimeoutError as exc:
        raise HTTPException(503, "OCR queue is busy; try again shortly.") from exc
    try:
        output = []
        with tempfile.TemporaryDirectory(prefix="ocr-request-") as folder:
            paths = []
            for index, page in enumerate(pages):
                path = Path(folder) / f"page-{index + 1}.png"
                page.save(path, "PNG", optimize=True)
                paths.append(path)
            for key in models:
                combined_text, combined_fields, regions, latency = [], {}, [], 0.0
                checkpoint = MODELS[key].checkpoint
                for path in paths:
                    try:
                        result = await asyncio.wait_for(asyncio.to_thread(run_model, key, path, INFERENCE_TIMEOUT), timeout=INFERENCE_TIMEOUT + 1)
                    except (TimeoutError, subprocess.TimeoutExpired) as exc:
                        raise HTTPException(504, f"{MODELS[key].name} exceeded the inference timeout.") from exc
                    combined_text.append(result["text"])
                    combined_fields.update(result["fields"])
                    regions.extend({**region, "page": len(combined_text) - 1} for region in result.get("regions", []))
                    latency += result["latency_ms"]
                    checkpoint = result["checkpoint"]
                text = "\n\n".join(combined_text)
                output.append({"key": key, "name": MODELS[key].name, "text": text, "fields": combined_fields or derived_fields(text), "fields_source": "model" if combined_fields else "regex derived from OCR text", "regions": regions, "latency_ms": round(latency, 1), "checkpoint": checkpoint})
        return {"page_count": len(pages), "pages": page_previews, "results": output, "retention": "Upload deleted after this response; previews exist only in this response."}
    finally:
        SEMAPHORE.release()

