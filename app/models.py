from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    description: str
    checkpoint: str
    env_command: str | None = None
    note: str = ""

    def command(self) -> list[str] | None:
        raw = os.getenv(self.env_command, "") if self.env_command else ""
        return shlex.split(raw) if raw else None

    def available(self) -> bool:
        if self.key == "tesseract":
            return tesseract_path() is not None
        return bool(self.command())


MODELS = {
    "tesseract": ModelSpec("tesseract", "Tesseract 5", "Open-source OCR text baseline bundled in this image.", "Runtime package version"),
    "glm_ocr": ModelSpec("glm_ocr", "GLM-OCR", "Direct-image receipt/document extraction route.", "zai-org/GLM-OCR; adapter reports exact asset identity", "OCR_GLM_COMMAND", "Requires a local Linux adapter and model mount on TrueNAS."),
    "qwen_3_5_2b": ModelSpec("qwen_3_5_2b", "Qwen3.5-2B", "Direct-image vision-language comparison route.", "Qwen/Qwen3.5-2B; adapter reports exact asset identity", "OCR_QWEN_COMMAND", "Requires a local Linux adapter and model mount on TrueNAS."),
    "layoutlmv3": ModelSpec("layoutlmv3", "LayoutLMv3 (legacy CORD)", "OCR-token and geometry field classifier.", "Local CORD checkpoint; upstream revision not recorded", note="Asset found, but no verified Linux serving adapter is packaged."),
    "trocr_layoutlmv3": ModelSpec("trocr_layoutlmv3", "TrOCR + LayoutLMv3", "TrOCR rereads candidate regions; it is not page-level OCR.", "microsoft/trocr-base-printed plus legacy LayoutLMv3", note="Experimental auxiliary route; not enabled as a public whole-document model."),
    "donut": ModelSpec("donut", "Donut (legacy CORD v2)", "OCR-free receipt JSON candidate route.", "naver-clova-ix/donut-base-finetuned-cord-v2; revision not recorded", note="Asset found, but no verified Linux serving adapter is packaged."),
}


def tesseract_path() -> str | None:
    configured = os.getenv("OCR_TESSERACT_COMMAND")
    return configured if configured and Path(configured).is_file() else shutil.which("tesseract")


def tesseract_version():
    pytesseract.pytesseract.tesseract_cmd = tesseract_path() or "tesseract"
    return pytesseract.get_tesseract_version()


def catalog() -> list[dict]:
    result = []
    for spec in MODELS.values():
        checkpoint = spec.checkpoint
        if spec.key == "tesseract" and tesseract_path():
            checkpoint = f"Tesseract {tesseract_version()}"
        result.append({"key": spec.key, "name": spec.name, "description": spec.description, "checkpoint": checkpoint, "available": spec.available(), "note": spec.note})
    return result


def run_model(key: str, image_path: Path, timeout: float) -> dict:
    spec = MODELS[key]
    started = time.perf_counter()
    if key == "tesseract":
        executable = tesseract_path()
        if not executable:
            raise RuntimeError("Tesseract is not installed")
        pytesseract.pytesseract.tesseract_cmd = executable
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image, lang="eng", config="--psm 6", timeout=timeout)
        return {"text": text.strip(), "fields": {}, "checkpoint": f"Tesseract {tesseract_version()}", "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    command = spec.command()
    if not command:
        raise RuntimeError("model is not configured on this host")
    completed = subprocess.run([*command, str(image_path)], capture_output=True, text=True, timeout=timeout, check=False, shell=False)
    if completed.returncode:
        raise RuntimeError(f"adapter exited with status {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("adapter did not return one JSON object") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("text", ""), str) or not isinstance(payload.get("fields", {}), dict):
        raise RuntimeError("adapter JSON must contain string 'text' and object 'fields'")
    return {"text": payload.get("text", ""), "fields": payload.get("fields", {}), "checkpoint": str(payload.get("checkpoint") or spec.checkpoint), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

