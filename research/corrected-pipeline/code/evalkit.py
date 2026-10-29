"""Decoders for LayoutLMv3 word labels and the thesis scorers (imported verbatim from the project scripts)."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import csv, json, re, sys, importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np

W = ROOT; STAGE = W/'stage'; UP = Path(os.environ.get('PROJECT_ROOT', str(ROOT/'project')))
sys.path.insert(0, str(UP/'2026-09-30'/'task'))
from row_metrics import score_rows as cord_score_rows   # noqa: E402

# --- CORD scorer helpers, extracted verbatim from score_cord.py (functions only) ---
_src = (UP/'2026-09-30'/'task'/'score_cord.py').read_text()
_ns = {}
for name in ("parse_amount", "gt_total", "menu_list", "gt_menu_rows", "wilson"):
    m = re.search(rf"^def {name}\(.*?(?=^def |\Z|^manifest=)", _src, flags=re.S | re.M)
    exec("import re,unicodedata\nfrom decimal import Decimal,InvalidOperation\n" + m.group(0), _ns)
cord_parse_amount, gt_total, gt_menu_rows, wilson = _ns["parse_amount"], _ns["gt_total"], _ns["gt_menu_rows"], _ns["wilson"]

# --- photo scorers: total parser used for the thesis photo totals, row scorer used for Table 5.2 ---
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
_sre = _load('shared_row_evaluator', UP/'2026-10-04'/'task-6'/'evidence'/'evaluator_sources'/'shared_row_evaluator.py')
strict_aud_amount = _sre.strict_aud_amount
_srv2 = _load('score_receipy_v2', STAGE/'2026-09-30'/'task'/'photo_pilot'/'receipy_extension'/'score_receipy_v2.py')

# ---------------- decoders ----------------
def ent(tag): return tag[2:] if tag[:2] in ("B-", "I-") else tag

def totals_join(words):
    """benchmark_cord.fields_from_tags: every word labelled TOTAL.TOTAL_PRICE, joined in input order."""
    vs = [w[0] for w in words if ent(w[3]).upper() == "TOTAL.TOTAL_PRICE"]
    return " ".join(vs) if vs else None

NUMTOK = re.compile(r"\d+(?:[.,]\d+)*")
def single_amount(text): return text is not None and len(NUMTOK.findall(text)) == 1

def totals_span(words):
    """Split TOTAL.TOTAL_PRICE words into spans (a B- tag or a break in the label starts a new span);
    return the most confident span that contains exactly one number; otherwise fall back to the join."""
    spans, cur, prev_total = [], None, False
    for w in words:
        is_t = ent(w[3]).upper() == "TOTAL.TOTAL_PRICE"
        if is_t and prev_total and w[3].startswith("I-"): cur.append(w)
        elif is_t: cur = [w]; spans.append(cur)
        prev_total = is_t
    best = None
    for i, s in enumerate(spans):
        text = " ".join(x[0] for x in s)
        if not single_amount(text): continue
        score = (sum(x[4] for x in s) / len(s), i)
        if best is None or score > best[0]: best = (score, text)
    return best[1] if best else totals_join(words)

FIELDS = ("nm", "cnt", "price", "unitprice", "itemsubtotal")
def _field(tag):
    t = ent(tag).upper()
    if not t.startswith("MENU."): return None
    f = t.split(".")[-1].lower()
    return f if f in FIELDS else None

def rows_line(words):
    """benchmark_cord.tagged_rows: menu labels grouped by physical OCR line."""
    lines = defaultdict(list)
    for w in words: lines[tuple(w[1])].append(w)
    rows = []
    for _, ws in lines.items():
        r = {k: [] for k in FIELDS}
        for w in ws:
            f = _field(w[3])
            if f: r[f].append(w[0])
        row = {k: " ".join(v).strip() for k, v in r.items() if v}
        if row: rows.append(row)
    return rows

def rows_reset(words):
    """Menu labels in reading order; a new row starts when a field that the current row already holds
    begins again (a new B- span or a break in the label). Rows may cross physical lines."""
    rows, cur, prev_f = [], {}, None
    for w in words:
        f = _field(w[3])
        if f is None: prev_f = None; continue
        cont = (f == prev_f and w[3].startswith("I-"))
        if not cont and f in cur:
            rows.append(cur); cur = {}
        cur.setdefault(f, []).append(w[0]); prev_f = f
    if cur: rows.append(cur)
    return [{k: " ".join(v).strip() for k, v in r.items()} for r in rows]

def segvote(words):
    """Give every word of a segment the segment's label (the entity with the largest summed word confidence),
    as in CORD's training data where all words of a segment share one label; B- on the first word, I- after."""
    if not words or words[0][2] is None: return words
    score = defaultdict(lambda: defaultdict(float))
    for w in words: score[w[2]][ent(w[3])] += w[4]
    lab = {s: max(d.items(), key=lambda kv: kv[1])[0] for s, d in score.items()}
    out, prev = [], None
    for w in words:
        e = lab[w[2]]
        tag = "O" if e == "O" else (("I-" if prev == w[2] else "B-") + e)
        out.append([w[0], w[1], w[2], tag, w[4], w[5]]); prev = w[2]
    return out

TOTAL_DECODERS = {"join": totals_join, "span": totals_span}
ROW_DECODERS = {"line": rows_line, "reset": rows_reset}

# ---------------- references ----------------
def cord_refs(split):
    man = json.loads((W/'cord'/'cord_data'/split/'manifest.json').read_text())
    out = {}
    for m in man:
        rid = m["receipt_id"]; gt = json.loads((W/'cord'/'cord_data'/split/f'{rid}.json').read_text())["ground_truth"]
        out[rid] = {"total": gt_total(gt), "rows": gt_menu_rows(gt)}
    return out

def photo_refs():
    """Frozen total references per view (as scored in the thesis) and frozen row references for the 15 ledgers."""
    pv = list(csv.DictReader(open(UP/'2026-10-04'/'task-7'/'evidence'/'six_model_comparison_v1'/'attempts'/'fullcohort_five-route-122-02'/'scores_total_only_v1'/'per_view_total_predictions.csv', encoding='utf-8-sig')))
    totals = {}
    for r in pv:
        if r["route"] != "LayoutLMv3": continue
        totals[(r["receipt_id"], r["condition"])] = {"known": r["primary_reference_known"] == "1", "total": r["primary_reference_total"], "image_id": r["image_id"],
                                                    "pixel": r["prediction_pixel_sha256"]}
    refs = json.loads((STAGE/'2026-09-30'/'task'/'photo_pilot'/'receipy_extension'/'frozen_v2'/'references.json').read_text())
    pairing = json.loads((STAGE/'2026-09-30'/'task'/'photo_pilot'/'receipy_extension'/'frozen_v2'/'pairing_manifest.json').read_text())
    pair_ids = [x["receipt_id"] for x in pairing["primary_pairs"]]
    rows = {}
    for x in refs["records"]:
        if x["receipt_id"] in pair_ids and x.get("complete_visible_inventory_candidate"):
            rows[x["receipt_id"]] = _srv2.target_rows(x)
    return totals, rows

def photo_total_ok(pred, ref):
    if not ref["known"]: return None
    p, t = strict_aud_amount(pred), strict_aud_amount(ref["total"])
    return int(p is not None and t is not None and p == t)

def cord_total_ok(pred, truth):
    if truth is None: return None
    p = cord_parse_amount(pred); return int(p is not None and p == cord_parse_amount(truth))

def f1(tp, fp, fn): return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
