"""Export official CORD splits and run repeatable local receipt-model inference.

Outputs are isolated under this task folder. Source parquet files are not changed.
"""
from __future__ import annotations

import argparse, json, time, io, random, hashlib
from pathlib import Path
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cord_data"
VARIANTS = ("clean", "color_cast", "fade", "occlusion", "rotate180", "synthetic_crumple")

def synthetic_crumple(im: Image.Image, seed: int):
    """Deterministic geometric crease proxy; it is not a physical paper simulation."""
    import numpy as np
    import random
    rng=random.Random(seed); src=im.convert("RGB"); w,h=src.size
    # Mild projective skew, sampled from an inset quadrilateral into the full frame.
    dst=[(0,0),(w-1,0),(w-1,h-1),(0,h-1)]
    source=[(int(.015*w),int(.02*h)),(int(.985*w),int(.01*h)),(int(.975*w),int(.985*h)),(int(.02*w),int(.97*h))]
    A=[]; B=[]
    for (x,y),(u,v) in zip(dst,source):
        A.append([x,y,1,0,0,0,-u*x,-u*y]); B.append(u)
        A.append([0,0,0,x,y,1,-v*x,-v*y]); B.append(v)
    coeff=np.linalg.solve(np.asarray(A,dtype=float),np.asarray(B,dtype=float)).tolist()
    warped=src.transform((w,h),Image.Transform.PERSPECTIVE,coeff,resample=Image.Resampling.BICUBIC)
    # Two seeded curved fold-shadow/highlight pairs; intentionally described as synthetic.
    layer=Image.new("RGBA",(w,h),(0,0,0,0)); shadow=Image.new("RGBA",(w,h),(0,0,0,0)); ds=ImageDraw.Draw(shadow)
    highlight=Image.new("RGBA",(w,h),(0,0,0,0)); dh=ImageDraw.Draw(highlight)
    for j in range(2):
        if j==0: start=(int(w*rng.uniform(.22,.38)),0); end=(int(w*rng.uniform(.58,.78)),h-1)
        else: start=(0,int(h*rng.uniform(.30,.48))); end=(w-1,int(h*rng.uniform(.62,.80)))
        control=(int((start[0]+end[0])/2+rng.uniform(-.10,.10)*w),int((start[1]+end[1])/2+rng.uniform(-.10,.10)*h))
        pts=[]
        for k in range(41):
            t=k/40; omt=1-t
            pts.append((round(omt*omt*start[0]+2*omt*t*control[0]+t*t*end[0]),round(omt*omt*start[1]+2*omt*t*control[1]+t*t*end[1])))
        ds.line(pts,fill=(40,32,24,58),width=max(7,round(min(w,h)*.018)))
        dh.line([(x+2,y) for x,y in pts],fill=(255,255,255,42),width=max(2,round(min(w,h)*.004)))
    from PIL import ImageFilter
    shadow=shadow.filter(ImageFilter.GaussianBlur(max(2,round(min(w,h)*.009))))
    layer=Image.alpha_composite(shadow,highlight)
    return Image.alpha_composite(warped.convert("RGBA"),layer).convert("RGB")

def variants(im: Image.Image, seed: int):
    im = im.convert("RGB")
    yield "clean", im.copy()
    # Fixed, legibility-preserving variants; no label or OCR data is used.
    color = Image.new("RGB", im.size, (218, 228, 255))
    yield "color_cast", Image.blend(im, color, 0.18)
    yield "fade", ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.12)).enhance(0.58)
    occ = im.copy()
    draw = ImageDraw.Draw(occ)
    w, h = occ.size
    # Opaque elliptical stain/obstruction near one edge, fixed seed by document ID.
    rng = random.Random(seed)
    x = int(w * rng.uniform(.78, .84)); y = int(h * rng.uniform(.35, .62))
    draw.ellipse((x, y, min(w, x + int(w*.12)), min(h, y + int(h*.11))), fill=(112, 91, 67, 100))
    yield "occlusion", occ
    yield "rotate180", im.rotate(180, expand=False)
    yield "synthetic_crumple", synthetic_crumple(im, seed)

def export(split: str, limit: int | None):
    source = ROOT / f"cord-v2-{split}.parquet"
    table = pq.read_table(source)
    rows = table.to_pylist()
    out = DATA / split
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, row in enumerate(rows[:limit]):
        b = row["image"]["bytes"]
        if not b:
            raise RuntimeError(f"CORD row {i} has no image bytes")
        gt = json.loads(row["ground_truth"])
        split_in_gt = gt.get("meta", {}).get("split")
        if split_in_gt and not (split == "validation" and split_in_gt == "valid") and split_in_gt != split:
            raise RuntimeError(f"split mismatch row={i}, label={split_in_gt}")
        image_id = f"{split}-{i:03d}"
        img = Image.open(io.BytesIO(b)).convert("RGB")
        entry = {"receipt_id": image_id, "row_index": i, "source_split": split,
                 "source_sha256": hashlib.sha256(b).hexdigest(), "width": img.width,
                 "height": img.height, "ground_truth": gt}
        entry["n_menu_rows"] = len(gt.get("gt_parse", {}).get("menu", [])) if isinstance(gt.get("gt_parse", {}).get("menu", []), list) else int(bool(gt.get("gt_parse", {}).get("menu")))
        with (out / f"{image_id}.json").open("w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        for name, v in variants(img, i):
            d = DATA / "images" / name
            d.mkdir(parents=True, exist_ok=True)
            v.save(d / f"{image_id}.jpg", quality=94, subsampling=0)
        manifest.append({k: entry[k] for k in ("receipt_id", "row_index", "source_split", "source_sha256", "width", "height", "n_menu_rows")})
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"split": split, "exported": len(manifest), "images_per_receipt": len(VARIANTS), "manifest": str(out / "manifest.json")}))

def infer(split: str, limit: int | None, variants_arg: list[str], endpoint: str):
    manifests = json.loads((DATA / split / "manifest.json").read_text(encoding="utf-8"))
    if limit is not None: manifests = manifests[:limit]
    out = DATA / "predictions" / split
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in manifests:
        for variant in variants_arg:
            p = DATA / "images" / variant / f"{m['receipt_id']}.jpg"
            start = time.perf_counter()
            try:
                with p.open("rb") as f:
                    response = requests.post(endpoint + "/receipts", files={"file": (p.name, f, "image/jpeg")}, timeout=600)
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - start
                record = {"receipt_id": m["receipt_id"], "variant": variant, "source_split": split,
                          "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "wall_seconds": elapsed,
                          "result": data.get("result"), "candidates": data.get("candidates"),
                          "model_runs": data.get("model_runs"), "warnings": data.get("processing_warnings")}
                rows.append(record)
                with (out / f"{m['receipt_id']}__{variant}.json").open("w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                print(json.dumps({"receipt": m["receipt_id"], "variant": variant, "seconds": round(elapsed,2), "runs": data.get("model_runs")}), flush=True)
            except Exception as e:
                record = {"receipt_id": m["receipt_id"], "variant": variant, "source_split": split,
                          "error": repr(e), "wall_seconds": time.perf_counter() - start}
                rows.append(record)
                with (out / f"{m['receipt_id']}__{variant}.json").open("w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)
                print(json.dumps(record), flush=True)
    with (out / "index.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["export", "infer"])
    p.add_argument("split", choices=["validation", "test"])
    p.add_argument("--limit", type=int)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--endpoint", default="http://127.0.0.1:8123")
    args = p.parse_args()
    if args.command == "export": export(args.split, args.limit)
    else: infer(args.split, args.limit, args.variants, args.endpoint)

if __name__ == "__main__": main()
