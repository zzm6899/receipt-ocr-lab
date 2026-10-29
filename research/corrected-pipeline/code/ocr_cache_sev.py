import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import sys, json, os, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from PIL import Image
import pytesseract
from ocr_cache import detect
W = ROOT
keys = []
for sub in ('severity_test', 'severity_extension_test'):
    for p in sorted((W/'cord'/'cord_data'/'images'/sub).glob('*.jpg')):
        keys.append((f"cord/{sub}/{p.stem}", p))
for key, path in keys:
    dest = W/'cache'/'ocr'/(key + '.json'); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): continue
    t = time.perf_counter(); im = Image.open(path).convert("RGB"); words = detect(im)
    rec = {"key": key, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": list(im.size), "tesseract": str(pytesseract.get_tesseract_version()), "config": "--psm 6", "seconds": round(time.perf_counter()-t, 2), "words": words}
    tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(rec)); os.replace(tmp, dest)
    print(key, len(words), rec["seconds"], flush=True)
