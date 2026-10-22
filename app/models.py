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
from pytesseract import Output


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
            data = pytesseract.image_to_data(image, lang="eng", config="--psm 6", timeout=timeout, output_type=Output.DICT)
            width, height = image.size
        words = []
        lines: dict[tuple[int, int, int], list[str]] = {}
        for index, text in enumerate(data["text"]):
            text = text.strip()
            confidence = float(data["conf"][index])
            if not text or confidence < 0:
                continue
            line_key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            lines.setdefault(line_key, []).append(text)
            words.append({
                "text": text,
                "field": None,
                "confidence": round(min(1.0, confidence / 100), 4),
                "bbox": [
                    data["left"][index] / width,
                    data["top"][index] / height,
                    data["width"][index] / width,
                    data["height"][index] / height,
                ],
            })
        return {"text": "\n".join(" ".join(line) for line in lines.values()), "fields": {}, "regions": words, "checkpoint": f"Tesseract {tesseract_version()}", "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

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
    return {"text": payload.get("text", ""), "fields": payload.get("fields", {}), "regions": normalize_regions(payload.get("regions", [])), "checkpoint": str(payload.get("checkpoint") or spec.checkpoint), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def normalize_regions(regions) -> list[dict]:
    """Validate normalized adapter boxes without inventing missing confidence."""
    if not isinstance(regions, list):
        raise RuntimeError("adapter 'regions' must be an array")
    output = []
    for region in regions[:5000]:
        if not isinstance(region, dict) or not isinstance(region.get("bbox"), list) or len(region["bbox"]) != 4:
            raise RuntimeError("each adapter region must contain bbox [x, y, width, height]")
        bbox = [float(value) for value in region["bbox"]]
        if any(value < 0 or value > 1 for value in bbox) or bbox[2] == 0 or bbox[3] == 0 or bbox[0] + bbox[2] > 1.0001 or bbox[1] + bbox[3] > 1.0001:
            raise RuntimeError("adapter region coordinates must be normalized to 0..1")
        confidence = region.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if confidence < 0 or confidence > 1:
                raise RuntimeError("adapter region confidence must be 0..1 or null")
        output.append({"bbox": bbox, "text": str(region.get("text", ""))[:1000], "field": None if region.get("field") is None else str(region["field"])[:100], "confidence": confidence})
    return output

