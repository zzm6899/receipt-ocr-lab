import io

import pytest
from fastapi import HTTPException
from PIL import Image

from app.main import derived_fields, preview, rasterize
from app.models import catalog, normalize_regions


def png(width=200, height=100):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, "PNG")
    return output.getvalue()


def test_catalog_is_honest_about_optional_models():
    by_key = {row["key"]: row for row in catalog()}
    assert by_key["glm_ocr"]["available"] is False
    assert by_key["qwen_3_5_2b"]["available"] is False
    assert "no verified" in by_key["layoutlmv3"]["note"].lower()


def test_rejects_fake_image():
    with pytest.raises(HTTPException) as caught:
        rasterize(b"not an image", "image/png")
    assert caught.value.status_code == 400


def test_rasterizes_valid_image():
    pages = rasterize(png(), "image/png")
    assert len(pages) == 1
    assert pages[0].size == (200, 100)


def test_pixel_limit(monkeypatch):
    monkeypatch.setattr("app.main.MAX_PIXELS", 100)
    with pytest.raises(HTTPException) as caught:
        rasterize(png(), "image/png")
    assert caught.value.status_code == 400


def test_derived_fields_are_conservative_and_label_driven():
    assert derived_fields("Shop\nDate 22/10/2026\nSubtotal $10.00\nGST $1.00\nTOTAL $11.00") == {"date": "22/10/2026", "subtotal": "$10.00", "tax": "$1.00", "total": "$11.00"}


def test_preview_is_bounded_and_contains_only_rendered_pixels():
    result = preview(Image.new("RGB", (3200, 800), "white"))
    assert result["data_url"].startswith("data:image/jpeg;base64,")
    assert result["width"] == 1600
    assert result["height"] == 400


def test_normalized_regions_preserve_missing_confidence():
    region = normalize_regions([{"bbox": [0.1, 0.2, 0.3, 0.4], "text": "TOTAL", "field": "total", "confidence": None}])[0]
    assert region == {"bbox": [0.1, 0.2, 0.3, 0.4], "text": "TOTAL", "field": "total", "confidence": None}


@pytest.mark.parametrize("bbox", [[-0.1, 0, 0.2, 0.2], [0.9, 0, 0.2, 0.2], [0, 0, 0, 0.2]])
def test_normalized_regions_reject_invalid_coordinates(bbox):
    with pytest.raises(RuntimeError):
        normalize_regions([{"bbox": bbox}])

