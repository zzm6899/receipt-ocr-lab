"""Stage 1: Tesseract --psm 6 words for every image (same call as benchmark_cord.detect)."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import json, sys, os, hashlib, time
from pathlib import Path
from PIL import Image
import pytesseract
from pytesseract import Output
W = ROOT
def detect(im):  # verbatim logic of benchmark_cord.detect
    d = pytesseract.image_to_data(im, config="--psm 6", output_type=Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t: continue
        try: conf = max(0., min(1., float(d["conf"][i]) / 100))
        except: conf = 0.
        words.append({"text": t, "left": int(d["left"][i]), "top": int(d["top"][i]), "width": int(d["width"][i]), "height": int(d["height"][i]),
                      "line": [d["page_num"][i], d["block_num"][i], d["par_num"][i], d["line_num"][i]], "conf": conf})
    return words
def jobs():
    out = []
    for split, conds in (("validation", ["clean", "color_cast", "fade", "occlusion", "rotate180"]),
                         ("test", ["clean", "color_cast", "fade", "occlusion", "rotate180", "synthetic_crumple"])):
        for c in conds:
            for i in range(100):
                rid = f"{split}-{i:03d}"
                out.append((f"cord/{c}/{rid}", W/'cord'/'cord_data'/'images'/c/f'{rid}.jpg'))
    for p in sorted((W/'photos').glob('*/input.png')):
        out.append((f"photo/{p.parent.name}", p))
    return out
if __name__ == "__main__":
    k, n = int(sys.argv[1]), int(sys.argv[2])
    for j, (key, path) in enumerate(jobs()):
        if j % n != k: continue
        dest = W/'cache'/'ocr'/(key + '.json'); dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): continue
        t = time.perf_counter(); im = Image.open(path).convert("RGB"); words = detect(im)
        rec = {"key": key, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
               "size": list(im.size), "tesseract": str(pytesseract.get_tesseract_version()), "config": "--psm 6", "seconds": round(time.perf_counter()-t, 2), "words": words}
        tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(rec)); os.replace(tmp, dest)
        print(key, len(words), rec["seconds"], flush=True)
