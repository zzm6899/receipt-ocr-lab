"""Stage 3: LayoutLMv3 (CORD checkpoint) on cached Tesseract words or TrOCR line text.

Input presentation modes
  word        one box per word (the thesis pipeline)
  line        every word shares its Tesseract line box
  gapK        Tesseract lines split into segments at horizontal gaps wider than K x median word height
  num         lines split where tokens change between numeric and text
  gapKnum     both splits
Long inputs: 'trunc' keeps the thesis behaviour (processor truncation at 512 tokens, the rest labelled O);
'window' labels every word by running consecutive whole-line windows that each fit in 512 tokens.
The image, checkpoint, label set and word-label rule (best-confidence sub-token) are unchanged.
"""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import json, os, re, sys, time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification

W = ROOT; M = Path(os.environ.get('MODELS_DIR', str(ROOT/'models')))/'layoutlmv3-finetuned-cord'
torch.set_num_threads(int(os.environ.get('NT', '2')))
lp = AutoProcessor.from_pretrained(str(M), apply_ocr=False)
lm = AutoModelForTokenClassification.from_pretrained(str(M)).eval()
id2label = lm.config.id2label
MAXTOK = 512

NUMERIC = re.compile(r"^[^A-Za-z]*\d[^A-Za-z]*$|^[xX]\d+$|^\d+[xX]$")

def image_path(key):
    kind, a, *b = key.split('/')
    if kind == 'cord': return W/'cord'/'cord_data'/'images'/a/f'{b[0]}.jpg'
    return W/'photos'/a/'input.png'

def projected_trocr_words(lines):
    """benchmark_cord.recognize_lines token projection (character-length boxes inside each line crop)."""
    result = []
    for ln in lines:
        x0, y0, x1, y1, line = ln["x0"], ln["y0"], ln["x1"], ln["y1"], tuple(ln["line"]); toks = ln["text"].strip().split()
        if not toks: continue
        widths = [max(1, len(t)) for t in toks]; denom = sum(widths); cursor = x0
        for token, weight in zip(toks, widths):
            nxt = cursor + (x1 - x0) * weight / denom
            result.append({"text": token, "left": round(cursor), "top": y0, "width": max(1, round(nxt - cursor)), "height": max(1, y1 - y0), "line": line, "conf": 0.0})
            cursor = nxt
    return result

def tess_segments(words, mode):
    """Return a segment id per Tesseract word (index aligned) and the union box of each segment."""
    m = re.fullmatch(r"(word|line|num|gap([0-9.]+)(num)?)", mode)
    if not m: raise ValueError(mode)
    byline = defaultdict(list)
    for i, w in enumerate(words): byline[tuple(w["line"])].append(i)
    seg_of = [None] * len(words); sid = 0
    for ln, idx in byline.items():
        idx = sorted(idx, key=lambda i: words[i]["left"])
        hs = sorted(words[i]["height"] for i in idx); hmed = hs[len(hs)//2]
        gap = float(m.group(2)) if m.group(2) else None
        usenum = mode == 'num' or bool(m.group(3))
        for j, i in enumerate(idx):
            if j:
                p = idx[j-1]; new = mode == 'word'
                if gap is not None and words[i]["left"] - (words[p]["left"] + words[p]["width"]) > gap * hmed: new = True
                if usenum and bool(NUMERIC.match(words[i]["text"])) != bool(NUMERIC.match(words[p]["text"])): new = True
                if new: sid += 1
            seg_of[i] = sid
        sid += 1
    boxes = {}
    for i, s in enumerate(seg_of):
        w = words[i]; b = boxes.get(s)
        x0, y0, x1, y1 = w["left"], w["top"], w["left"] + w["width"], w["top"] + w["height"]
        boxes[s] = [x0, y0, x1, y1] if b is None else [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
    return seg_of, boxes

def norm(t): return re.sub(r"[^0-9a-z]", "", t.lower())

def assign_trocr_to_segments(tw_line, tess_line_words, tess_seg):
    """Align one line's TrOCR tokens to the Tesseract words of that line; matched tokens take the word's segment,
    unmatched tokens take the segment of the nearest preceding matched token (or the first segment)."""
    a = [norm(t) for t in tw_line]; b = [norm(w["text"]) for w in tess_line_words]
    sm = SequenceMatcher(None, a, b, autojunk=False); seg = [None] * len(a)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1): seg[i1 + k] = tess_seg[j1 + k]
        elif tag == 'replace':
            for k in range(i2 - i1): seg[i1 + k] = tess_seg[min(j2 - 1, j1 + round(k * (j2 - j1) / max(1, i2 - i1)))]
    last = tess_seg[0] if tess_seg else 0
    for k in range(len(seg)):
        if seg[k] is None: seg[k] = last
        last = seg[k]
    return seg

def build_input(key, source, mode):
    ocr = json.loads((W/'cache'/'ocr'/(key + '.json')).read_text())["words"]
    for w in ocr: w["line"] = tuple(w["line"])
    if source == 'tess':
        words = [dict(w) for w in ocr]
        if mode != 'word':
            seg_of, boxes = tess_segments(ocr, mode)
            for w, s in zip(words, seg_of):
                x0, y0, x1, y1 = boxes[s]; w.update(left=x0, top=y0, width=x1 - x0, height=y1 - y0, seg=s)
        return words
    lines = json.loads((W/'cache'/'trocr'/(key + '.json')).read_text())["lines"]
    words = projected_trocr_words(lines)
    if mode == 'word': return words
    seg_of, boxes = tess_segments(ocr, mode)
    tess_by_line = defaultdict(list)
    for i, w in enumerate(ocr): tess_by_line[w["line"]].append(i)
    out = []
    for ln in lines:
        toks = ln["text"].strip().split()
        if not toks: continue
        idx = sorted(tess_by_line[tuple(ln["line"])], key=lambda i: ocr[i]["left"])
        segs = assign_trocr_to_segments(toks, [ocr[i] for i in idx], [seg_of[i] for i in idx])
        for t, s in zip(toks, segs):
            x0, y0, x1, y1 = boxes[s]
            out.append({"text": t, "left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0, "line": tuple(ln["line"]), "seg": s, "conf": 0.0})
    return out

def norm_boxes(words, width, height):
    return [[max(0, min(1000, int(w["left"]/width*1000))), max(0, min(1000, int(w["top"]/height*1000))),
             max(0, min(1000, int((w["left"]+w["width"])/width*1000))), max(0, min(1000, int((w["top"]+w["height"])/height*1000)))] for w in words]

def label_words(im, words):
    """benchmark_cord.predict_layout labelling rule; returns (tag, conf) per word or None if not covered."""
    boxes = norm_boxes(words, *im.size)
    enc = lp(im, [w["text"] for w in words], boxes=boxes, return_tensors="pt", truncation=True)
    word_ids = enc.word_ids(0)
    with torch.inference_mode(): logits = lm(**enc).logits[0].softmax(-1)
    labels = logits.argmax(-1).tolist(); conf = logits.max(-1).values.tolist(); best = {}
    for ti, wi in enumerate(word_ids):
        if wi is None: continue
        if wi not in best or conf[ti] > best[wi][1]: best[wi] = (id2label[labels[ti]], conf[ti])
    return [best.get(i) for i in range(len(words))]

def token_counts(words):
    enc = lp.tokenizer([w["text"] for w in words], boxes=[[0, 0, 0, 0]] * len(words), add_special_tokens=False, truncation=False)
    c = [0] * len(words)
    for wi in enc.word_ids():
        if wi is not None: c[wi] += 1
    return c

def windows(words):
    """Consecutive whole-line windows (in input order) that each fit in MAXTOK-2 tokens."""
    counts = token_counts(words); out, cur, n = [], [], 0
    groups = []
    for i, w in enumerate(words):
        if groups and words[groups[-1][-1]]["line"] == w["line"]: groups[-1].append(i)
        else: groups.append([i])
    for g in groups:
        k = sum(counts[i] for i in g)
        if cur and n + k > MAXTOK - 2: out.append(cur); cur, n = [], 0
        if k > MAXTOK - 2:  # a single over-long line: split it by words
            for i in g:
                if cur and n + counts[i] > MAXTOK - 2: out.append(cur); cur, n = [], 0
                cur.append(i); n += counts[i]
        else: cur.extend(g); n += k
    if cur: out.append(cur)
    return out, sum(counts) + 2

def run(key, source, mode, longmode):
    im = Image.open(image_path(key)).convert("RGB"); words = build_input(key, source, mode)
    if not words: return {"key": key, "n_words": 0, "n_tokens": 0, "windows": 0, "words": []}
    if longmode == 'trunc':
        lab = label_words(im, words); nwin = 1; ntok = None
    else:
        wins, ntok = windows(words); lab = [None] * len(words)
        for win in wins:
            sub = label_words(im, [words[i] for i in win])
            for i, l in zip(win, sub): lab[i] = l
        nwin = len(wins)
    out = [[w["text"], list(w["line"]), w.get("seg"), (l or ("O", 0.0))[0], round((l or ("O", 0.0))[1], 4), int(l is not None)] for w, l in zip(words, lab)]
    return {"key": key, "source": source, "mode": mode, "long": longmode, "n_words": len(words), "n_tokens": ntok, "windows": nwin,
            "uncovered": sum(1 for l in lab if l is None), "words": out}

if __name__ == "__main__":
    keys = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    source = sys.argv[2]; modes = sys.argv[3].split(','); longmode = sys.argv[4]
    for key in keys:
        for mode in modes:
            dest = W/'cache'/'layout'/f'{source}_{mode}_{longmode}'/(key + '.json'); dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists(): continue
            t = time.perf_counter(); rec = run(key, source, mode, longmode); rec["seconds"] = round(time.perf_counter() - t, 2)
            tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(rec, ensure_ascii=False)); os.replace(tmp, dest)
            print(key, mode, rec["n_words"], rec.get("windows"), rec.get("uncovered"), rec["seconds"], flush=True)
