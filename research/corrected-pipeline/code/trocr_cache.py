"""Stage 2: TrOCR text for each Tesseract line crop (same crops, order and batching as benchmark_cord.recognize_lines)."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import json, sys, os, time
from collections import defaultdict
from pathlib import Path
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
W = ROOT; M = Path(os.environ.get('MODELS_DIR', str(ROOT/'models')))/'trocr-base-printed'
torch.set_num_threads(int(os.environ.get('NT', '2')))
USE_CACHE = os.environ.get('USE_CACHE', '1') == '1'
tp = TrOCRProcessor.from_pretrained(str(M)); tm = VisionEncoderDecoderModel.from_pretrained(str(M)).eval()
import gc, ctypes; gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
def line_crops(im, words):
    lines = defaultdict(list)
    for w in words: lines[tuple(w["line"])].append(w)
    ordered = sorted(lines.values(), key=lambda ws: (min(w["top"] for w in ws), min(w["left"] for w in ws)))
    crops, bounds = [], []
    for ws in ordered:
        x0 = min(w["left"] for w in ws); y0 = min(w["top"] for w in ws)
        x1 = max(w["left"] + w["width"] for w in ws); y1 = max(w["top"] + w["height"] for w in ws)
        bounds.append([x0, y0, x1, y1, list(ws[0]["line"])])
        crops.append(im.crop((max(0, x0-4), max(0, y0-4), min(im.width, x1+4), min(im.height, y1+4))))
    return crops, bounds
def recognise(crops):
    out = []
    for s in range(0, len(crops), 12):
        batch = crops[s:s+12]
        pix = tp(images=batch, return_tensors="pt").pixel_values
        with torch.inference_mode(): ids = tm.generate(pix, max_new_tokens=64, use_cache=USE_CACHE)
        out.extend(tp.batch_decode(ids, skip_special_tokens=True))
    return out
def image_path(key):
    kind, a, *b = key.split('/')
    if kind == 'cord': return W/'cord'/'cord_data'/'images'/a/f'{b[0]}.jpg'
    return W/'photos'/a/'input.png'
if __name__ == "__main__":
    keys = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    outroot = Path(sys.argv[2]) if len(sys.argv) > 2 else W/'cache'/'trocr'
    for key in keys:
        dest = outroot/(key + '.json'); dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): continue
        ocr = json.loads((W/'cache'/'ocr'/(key + '.json')).read_text())
        im = Image.open(image_path(key)).convert("RGB")
        t = time.perf_counter(); crops, bounds = line_crops(im, ocr["words"]); texts = recognise(crops) if crops else []
        rec = {"key": key, "use_cache": USE_CACHE, "seconds": round(time.perf_counter()-t, 2),
               "lines": [{"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "line": b[4], "text": tx} for b, tx in zip(bounds, texts)]}
        tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(rec, ensure_ascii=False)); os.replace(tmp, dest)
        print(key, len(crops), rec["seconds"], flush=True)
