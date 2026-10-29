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
ORDER = sys.argv[3] if len(sys.argv) > 3 else 'annotation'
BOXES = sys.argv[4] if len(sys.argv) > 4 else 'word'
OUT = H/'gold_ocr'/(('preds' if ORDER == 'annotation' else 'preds_reading') + ('' if BOXES == 'word' else '_segment'))
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

manifest = json.loads((TASK/'cord_data'/'test'/'manifest.json').read_text())
for m in manifest[start:end]:
    rid = m["receipt_id"]
    entry = json.loads((TASK/'cord_data'/'test'/f'{rid}.json').read_text())
    gt = entry["ground_truth"]
    im = Image.open(TASK/'cord_data'/'images'/'clean'/f'{rid}.jpg').convert("RGB")
    words = []
    for li, line in enumerate(gt["valid_line"]):
        cat = str(line.get("category", "")).upper()
        for wi, w in enumerate(line["words"]):
            q = w["quad"]; xs = [q["x1"], q["x2"], q["x3"], q["x4"]]; ys = [q["y1"], q["y2"], q["y3"], q["y4"]]
            words.append({"text": w["text"], "left": min(xs), "top": min(ys), "width": max(xs)-min(xs), "height": max(ys)-min(ys),
                          "row": w.get("row_id"), "seg": li, "group": (cat.split(".")[0], line.get("group_id")), "gold": ("B-" if wi == 0 else "I-") + cat})
    if BOXES == 'segment':  # LayoutLMv3 CORD/FUNSD convention: every word carries its segment (line) box
        seg = defaultdict(list)
        for w in words: seg[w['seg']].append(w)
        for ws in seg.values():
            x0 = min(w['left'] for w in ws); y0 = min(w['top'] for w in ws)
            x1 = max(w['left'] + w['width'] for w in ws); y1 = max(w['top'] + w['height'] for w in ws)
            for w in ws: w.update(left=x0, top=y0, width=x1 - x0, height=y1 - y0)
    if ORDER == 'reading':
        rows_y = defaultdict(list)
        for w in words: rows_y[w['row']].append(w['top'] + w['height']/2)
        ry = {k: sum(v)/len(v) for k, v in rows_y.items()}
        words = sorted(words, key=lambda w: (ry[w['row']], w['left']))
    tic = time.perf_counter()
    tagged, covered = predict_layout(im, words)
    rec = {"receipt_id": rid, "variant": "clean", "input": "cord_gold_words_and_boxes", "order": ORDER, "boxes": BOXES, "n_words": len(words), "words_scored_by_model": covered,
           "image_size_matches_meta": [im.width, im.height] == [gt["meta"]["image_size"]["width"], gt["meta"]["image_size"]["height"]],
           "seconds": round(time.perf_counter()-tic, 2), "layout_fields": fields_from_tags(tagged),
           "layout_rows_physical_row": tagged_rows(tagged, "row"), "layout_rows_gold_group": tagged_rows(tagged, "group"),
           "word_tags": [[w["text"], w["gold"], w["tag"]] for w in tagged]}
    (OUT/f'{rid}.json').write_text(json.dumps(rec, ensure_ascii=False))
    print(rid, rec["n_words"], covered, rec["seconds"], rec["layout_fields"].get("total"), flush=True)
