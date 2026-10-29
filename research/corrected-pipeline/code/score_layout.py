"""Score cached LayoutLMv3 runs under every decoder pair, for CORD splits or the photo cohort."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import evalkit as E

W = ROOT

def load(run, key):
    p = W/'cache'/'layout'/run/(key + '.json')
    return json.loads(p.read_text()) if p.exists() else None

def score_cord(run, split, conds, tdec, rdec, vote=False):
    refs = E.cord_refs(split); res = {}
    for c in conds:
        k = n = 0; tp = fp = fn = 0; missing = 0; per = {}
        for rid, r in refs.items():
            rec = load(run, f'cord/{c}/{rid}')
            if rec is None: missing += 1; continue
            words = E.segvote(rec["words"]) if vote else rec["words"]
            ok = E.cord_total_ok(E.TOTAL_DECODERS[tdec](words), r["total"])
            if ok is not None: k += ok; n += 1
            a, b, d, _, _ = E.cord_score_rows(r["rows"], E.ROW_DECODERS[rdec](words), ["nm", "cnt", "price"])
            tp += a; fp += b; fn += d; per[rid] = (ok, a, b, d)
        res[c] = dict(k=k, n=n, tp=tp, fp=fp, fn=fn, f1=E.f1(tp, fp, fn), missing=missing, per=per)
    return res

def score_photo(run, tdec, rdec, vote=False):
    tot, rows = E.photo_refs(); out = {}
    for cond in ("clear", "scrunched"):
        k = n = 0; tp = fp = fn = 0; missing = 0; per = {}
        for (rid, c), ref in tot.items():
            if c != cond: continue
            rec = load(run, f'photo/{rid}__{c}')
            if rec is None: missing += 1; continue
            words = E.segvote(rec["words"]) if vote else rec["words"]; pred = E.TOTAL_DECODERS[tdec](words)
            ok = E.photo_total_ok(pred, ref)
            if ok is not None: k += ok; n += 1
            rs = None
            if rid in rows:
                s = E._srv2.score_rows(rows[rid], E.ROW_DECODERS[rdec](words)); tp += s["tp"]; fp += s["fp"]; fn += s["fn"]; rs = (s["tp"], s["fp"], s["fn"])
            per[(rid, c)] = dict(ok=ok, pred=pred, rows=rs)
        out[cond] = dict(k=k, n=n, tp=tp, fp=fp, fn=fn, f1=E.f1(tp, fp, fn), missing=missing, per=per)
    return out
