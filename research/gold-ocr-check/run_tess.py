"""LayoutLMv3 CORD checkpoint on CORD's annotated words and boxes (gold OCR), clean test split.
Reuses the word-label and row-grouping logic of benchmark_cord.py so only the OCR input changes."""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification
torch.set_num_threads(2)
H = Path.home()
TASK = H/'mnt'/'Codex'/'2026-09-30'/'task'
MODEL = H/'mnt'/'layoutlmv3-finetuned-cord'
BOXES = sys.argv[3]
GAP = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
OUT = H/'gold_ocr'/(f'preds_tess_{BOXES}' + (f'_{GAP}' if BOXES == 'gapseg' else ''))
OUT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
start, end = int(sys.argv[1]), int(sys.argv[2])
lp = AutoProcessor.from_pretrained(str(MODEL), apply_ocr=False)
lm = AutoModelForTokenClassification.from_pretrained(str(MODEL)).eval()
id2label = lm.config.id2label

def predict_layout(im, words):  # identical logic to benchmark_cord.predict_layout
    width, height = im.size
    boxes = [[max(0,min(1000,int(w["left"]/width*1000))), max(0,min(1000,int(w["top"]/height*1000))),
              max(0,min(1000,int((w["left"]+w["width"])/width*1000))), max(0,min(1000,int((w["top"]+w["height"])/height*1000)))] for w in words]
    enc = lp(im, [w["text"] for w in words], boxes=boxes, return_tensors="pt", truncation=True)
    word_ids = enc.word_ids(0)
    with torch.inference_mode(): logits = lm(**enc).logits[0].softmax(-1)
    labels = logits.argmax(-1).tolist(); conf = logits.max(-1).values.tolist()
    best = {}
    for ti, wi in enumerate(word_ids):
        if wi is None: continue
        if wi not in best or conf[ti] > best[wi][1]: best[wi] = (id2label[labels[ti]], conf[ti])
    covered = len(best)
    return [{**w, "tag": best.get(i, ("O", 0))[0]} for i, w in enumerate(words)], covered

def tagged_rows(tagged, key):  # benchmark_cord.tagged_rows, grouping key selectable
    lines = defaultdict(list)
    for w in tagged: lines[w[key]].append(w)
    rows = []
    for _, ws in lines.items():
        r = {"nm": [], "cnt": [], "price": [], "unitprice": [], "itemsubtotal": []}
        for w in ws:
            tag = w["tag"].upper().replace("B-", "").replace("I-", "")
            if tag.startswith("MENU."):
                f = tag.split(".")[-1].lower()
                if f in r: r[f].append(w["text"])
        row = {k: " ".join(v).strip() for k, v in r.items() if v}
        if row: rows.append(row)
    return rows

def fields_from_tags(tagged):  # benchmark_cord.fields_from_tags
    groups = defaultdict(list)
    for w in tagged:
        t = w["tag"].upper().replace("B-", "").replace("I-", "")
        if t.startswith("TOTAL.") or t.startswith("SUB_TOTAL."):
            groups[t.split(".", 1)[1].lower()].append(w["text"])
    return {"total": " ".join(v)} if (v := groups.get("total_price")) else {}

TW = json.loads((TASK/'gold_ocr_check'/'tess_words_cord_test.json').read_text())["words"]
manifest = json.loads((TASK/'cord_data'/'test'/'manifest.json').read_text())
for m in manifest[start:end]:
    rid = m["receipt_id"]
    im = Image.open(TASK/'cord_data'/'images'/'clean'/f'{rid}.jpg').convert("RGB")
    words = [dict(w, line=tuple(w["line"])) for w in TW[rid]]
    if BOXES in ('line', 'gapseg'):
        seg = defaultdict(list)
        if BOXES == 'line':
            for w in words: seg[w['line']].append(w)
        else:  # split each Tesseract line into segments at wide horizontal gaps (proxy for CORD entity segments)
            byline = defaultdict(list)
            for w in words: byline[w['line']].append(w)
            for ln, ws in byline.items():
                ws = sorted(ws, key=lambda w: w['left']); hgt = sorted(w['height'] for w in ws)[len(ws)//2]; k = 0
                for i, w in enumerate(ws):
                    if i and w['left'] - (ws[i-1]['left'] + ws[i-1]['width']) > GAP * hgt: k += 1
                    seg[(ln, k)].append(w)
        for ws in seg.values():
            x0 = min(w['left'] for w in ws); y0 = min(w['top'] for w in ws)
            x1 = max(w['left'] + w['width'] for w in ws); y1 = max(w['top'] + w['height'] for w in ws)
            for w in ws: w.update(left=x0, top=y0, width=x1 - x0, height=y1 - y0)
    tagged, covered = predict_layout(im, words) if words else ([], 0)
    rec = {"receipt_id": rid, "boxes": BOXES, "n_words": len(words), "covered": covered,
           "layout_fields": fields_from_tags(tagged), "layout_rows": tagged_rows(tagged, "line")}
    (OUT/f'{rid}.json').write_text(json.dumps(rec, ensure_ascii=False))
    print(rid, len(words), covered, flush=True)
