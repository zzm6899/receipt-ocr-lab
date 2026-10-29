"""Build every corrected-pipeline number used in the thesis from cached outputs (CORD test and photos)."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import evalkit as E
import stats as S
from row_metrics import donut_rows

W = ROOT; OUT = W/'results'; OUT.mkdir(exist_ok=True)
FROZEN = json.loads((OUT/'frozen_config.json').read_text())
TC, RC = FROZEN["totals_config"], FROZEN["rows_config"]
CONDS = ["clean", "color_cast", "fade", "occlusion", "rotate180", "synthetic_crumple"]

def cached(source, mode, longmode, key):
    p = W/'cache'/'layout'/f'{source}_{mode}_{longmode}'/(key + '.json')
    return json.loads(p.read_text())["words"] if p.exists() else None

def corrected(source, key):
    """Frozen corrected decoder: total from the totals pass, rows from the rows pass (None if not run)."""
    wt = cached(source, TC["mode"], "window", key); wr = cached(source, RC["mode"], "window", key)
    if wt is None or wr is None: return None
    if TC["vote"]: wt = E.segvote(wt)
    if RC["vote"]: wr = E.segvote(wr)
    return E.TOTAL_DECODERS[TC["totals"]](wt), E.ROW_DECODERS[RC["rows"]](wr)

def original_rerun(source, key):
    w = cached(source, "word", "trunc", key)
    return None if w is None else (E.totals_join(w), E.rows_line(w))

def variant(source, mode, longmode, vote, tdec, key):
    w = cached(source, mode, longmode, key)
    if w is None: return None
    if vote: w = E.segvote(w)
    return E.TOTAL_DECODERS[tdec](w), E.rows_line(w)

# ---------------- CORD test ----------------
def cord_saved(rid, v):
    d = json.loads((W/'cord'/'cord_data'/'direct_predictions'/'test'/f'{rid}__{v}.json').read_text())
    t = d.get('donut_json', {}).get('total') if isinstance(d.get('donut_json'), dict) else None
    if isinstance(t, list): t = t[0] if t else None
    dt = t.get('total_price') if isinstance(t, dict) else None
    if not isinstance(dt, (str, int, float)): dt = None
    return {"Tesseract + rules": (d.get('ocr_rules_total'), d.get('ocr_rows', [])),
            "LayoutLMv3": ((d.get('layout_fields') or {}).get('total'), d.get('layout_rows', [])),
            "TrOCR → LayoutLMv3": ((d.get('cascade_fields') or {}).get('total'), d.get('cascade_rows', [])),
            "Donut": (dt, donut_rows(d.get('donut_json', {})))}

def cord_tables():
    refs = E.cord_refs('test'); ids = list(refs)
    routes = ["Tesseract + rules", "LayoutLMv3", "TrOCR → LayoutLMv3", "Donut", "LayoutLMv3 (original, rerun)",
              "LayoutLMv3 (corrected)", "TrOCR → LayoutLMv3 (corrected)", "TrOCR → LayoutLMv3 (original, rerun)"]
    per = defaultdict(dict)   # (route, cond) -> rid -> (ok, tp, fp, fn)
    for v in CONDS:
        for rid in ids:
            r = refs[rid]; saved = cord_saved(rid, v); key = f'cord/{v}/{rid}'
            outs = dict(saved)
            outs["LayoutLMv3 (original, rerun)"] = original_rerun('tess', key)
            outs["LayoutLMv3 (corrected)"] = corrected('tess', key)
            outs["TrOCR → LayoutLMv3 (corrected)"] = corrected('trocr', key)
            outs["TrOCR → LayoutLMv3 (original, rerun)"] = original_rerun('trocr', key)
            for route, o in outs.items():
                if o is None: continue
                ok = E.cord_total_ok(o[0], r["total"]); tp, fp, fn, _, _ = E.cord_score_rows(r["rows"], o[1], ["nm", "cnt", "price"])
                per[(route, v)][rid] = (ok, tp, fp, fn)
    summary = []
    for route in routes:
        for v in CONDS:
            d = per.get((route, v), {})
            if len(d) < len(ids): summary.append(dict(route=route, cond=v, status=f"incomplete {len(d)}/{len(ids)}")); continue
            k = sum(x[0] or 0 for x in d.values()); n = sum(x[0] is not None for x in d.values())
            tp, fp, fn = (sum(x[i] for x in d.values()) for i in (1, 2, 3))
            lo, hi = S.wilson(k, n)
            row = dict(route=route, cond=v, status="ok", k=k, n=n, lo=lo, hi=hi, tp=tp, fp=fp, fn=fn, f1=S.f1_counts((tp, fp, fn)))
            base = {"LayoutLMv3 (corrected)": "LayoutLMv3", "TrOCR → LayoutLMv3 (corrected)": "TrOCR → LayoutLMv3",
                    "LayoutLMv3 (original, rerun)": "LayoutLMv3", "TrOCR → LayoutLMv3 (original, rerun)": "TrOCR → LayoutLMv3"}.get(route)
            if base:
                b = per[(base, v)]
                a_ok = [d[r][0] or 0 for r in ids]; b_ok = [b[r][0] or 0 for r in ids]; el = [int(d[r][0] is not None) for r in ids]
                row["dtot"] = S.paired_diff_boot(a_ok, b_ok, el, ids)
                row["df1"] = S.paired_f1_boot([d[r][1:] for r in ids], [b[r][1:] for r in ids], ids)
            summary.append(row)
    return summary, per

# ---------------- photos ----------------
def photo_tables():
    tot, rowrefs = E.photo_refs()
    C = W/'stage'/'2026-10-04'/'task-7'/'evidence'/'six_model_comparison_v1'/'attempts'/'fullcohort_five-route-122-02'/'classic_four'
    classic = {}
    for f in C.glob('*.json'):
        d = json.loads(f.read_text())
        if d.get('condition') in ('clear', 'scrunched'): classic[(d['receipt_id'], d['condition'])] = d
    Q = W/'stage'/'2026-10-05'/'task'/'deliverables'/'qwen3_5_2b_comparison_v1_recovery'/'receipt_photo_model_results_qwen35_2b_2026-10-06'
    cmp_rows = list(csv.DictReader(open(Q/'comparison_all_views.csv', encoding='utf-8-sig')))
    direct = {(r['receipt_id'], r['condition'], r['route']): r for r in cmp_rows}
    outs = defaultdict(dict)  # route -> view -> dict(total_ok, rows)
    for view, ref in tot.items():
        d = classic[view]; key = f'photo/{view[0]}__{view[1]}'
        cands = {"Tesseract + rules": (d.get('ocr_total'), d.get('ocr_rows') or []),
                 "LayoutLMv3": (d.get('layout_total'), d.get('layout_rows') or []),
                 "TrOCR + LayoutLMv3": (d.get('cascade_total'), d.get('cascade_rows') or []),
                 "Donut": (d.get('donut_total'), d.get('donut_rows') or []),
                 "LayoutLMv3 (original, rerun)": original_rerun('tess', key),
                 "LayoutLMv3 (corrected)": corrected('tess', key),
                 "TrOCR + LayoutLMv3 (corrected)": corrected('trocr', key),
                 "TrOCR + LayoutLMv3 (original, rerun)": original_rerun('trocr', key),
                 "Ablation: word boxes + windows": variant('tess', 'word', 'window', False, 'join', key),
                 "Ablation: line boxes, truncated": variant('tess', 'line', 'trunc', False, 'join', key),
                 "Ablation: line boxes + windows": variant('tess', 'line', 'window', False, 'join', key),
                 "Ablation: line boxes + windows + vote": variant('tess', 'line', 'window', True, 'join', key)}
        for route, o in cands.items():
            if o is None: continue
            rec = {"ok": E.photo_total_ok(o[0], ref), "pred": o[0]}
            if view[0] in rowrefs:
                rec["rows"] = E._srv2.score_rows(rowrefs[view[0]], o[1])
            outs[route][view] = rec
        for route in ("GLM-OCR", "Qwen3.5-2B"):
            r = direct[(view[0], view[1], route)]
            outs[route][view] = {"ok": (int(r['exact_match']) if r['reference_known'] == '1' else None), "pred": r['prediction_total']}
    return tot, rowrefs, outs

def photo_summary(tot, rowrefs, outs):
    res = []
    for route, d in outs.items():
        if len(d) < 100: res.append(dict(route=route, status=f"incomplete {len(d)}/100")); continue
        row = dict(route=route, status="ok")
        for cond in ("clear", "scrunched", None):
            xs = [v for (rid, c), v in d.items() if (cond is None or c == cond) and v["ok"] is not None]
            k = sum(x["ok"] for x in xs); n = len(xs); row[cond or "all"] = (k, n, *S.wilson(k, n))
        rv = [v["rows"] for (rid, c), v in d.items() if "rows" in v]
        if rv:
            tp = sum(x["tp"] for x in rv); fp = sum(x["fp"] for x in rv); fn = sum(x["fn"] for x in rv)
            exact = sum(1 for x in rv if x["fp"] == 0 and x["fn"] == 0)
            row["rows"] = dict(tp=tp, fp=fp, fn=fn, P=tp/(tp+fp) if tp+fp else 0.0, R=tp/(tp+fn), F1=S.f1_counts((tp, fp, fn)), exact=exact, views=len(rv))
        # paired clear -> scrunched on receipts with both totals known
        both = [rid for rid in {v[0] for v in d} if d.get((rid, 'clear'), {}).get("ok") is not None and d.get((rid, 'scrunched'), {}).get("ok") is not None]
        b = sum(1 for r in both if d[(r, 'clear')]["ok"] == 1 and d[(r, 'scrunched')]["ok"] == 0)
        c = sum(1 for r in both if d[(r, 'clear')]["ok"] == 0 and d[(r, 'scrunched')]["ok"] == 1)
        sc = [d[(r, 'scrunched')]["ok"] for r in both]; cl = [d[(r, 'clear')]["ok"] for r in both]
        row["paired"] = dict(n=len(both), change=S.paired_diff_boot(sc, cl, [1]*len(both), both), p=S.mcnemar_exact(b, c))
        res.append(row)
    return res

def corrected_vs_original(outs, a, b):
    """Paired comparison on the 99 known views: a minus b, receipt-cluster bootstrap, exact McNemar."""
    views = [v for v in outs[a] if outs[a][v]["ok"] is not None]
    ka = [outs[a][v]["ok"] for v in views]; kb = [outs[b][v]["ok"] for v in views]; cl = [v[0] for v in views]
    bb = sum(1 for x, y in zip(ka, kb) if x == 1 and y == 0); cc = sum(1 for x, y in zip(ka, kb) if x == 0 and y == 1)
    return dict(n=len(views), diff=S.paired_diff_boot(ka, kb, [1]*len(views), cl), a_only=bb, b_only=cc, p=S.mcnemar_exact(bb, cc))

if __name__ == "__main__":
    summary, per = cord_tables()
    tot, rowrefs, outs = photo_tables(); ps = photo_summary(tot, rowrefs, outs)
    comps = {}
    for a, b in [("LayoutLMv3 (corrected)", "LayoutLMv3"), ("LayoutLMv3 (original, rerun)", "LayoutLMv3"),
                 ("TrOCR + LayoutLMv3 (corrected)", "TrOCR + LayoutLMv3"), ("LayoutLMv3 (corrected)", "LayoutLMv3 (original, rerun)"),
                 ("TrOCR + LayoutLMv3 (corrected)", "TrOCR + LayoutLMv3 (original, rerun)"), ("TrOCR + LayoutLMv3 (original, rerun)", "TrOCR + LayoutLMv3")]:
        if len(outs.get(a, {})) == 100 and len(outs.get(b, {})) == 100: comps[f"{a} vs {b}"] = corrected_vs_original(outs, a, b)
    json.dump({"cord": summary, "photo": ps, "photo_comparisons": comps, "frozen": FROZEN}, open(OUT/'final_results.json', 'w'), indent=1, default=str)
    for r in summary:
        if r["status"] != "ok": print(r["route"], r["cond"], r["status"]); continue
        extra = f' Δtot {r["dtot"][0]:+.1f} ({r["dtot"][1]:+.1f},{r["dtot"][2]:+.1f}) ΔF1 {r["df1"][0]:+.3f} ({r["df1"][1]:+.3f},{r["df1"][2]:+.3f})' if "dtot" in r else ""
        print(f'{r["route"]:32s} {r["cond"]:18s} {r["k"]:3d}/{r["n"]} F1 {r["f1"]:.3f} ({r["tp"]}/{r["fp"]}/{r["fn"]}){extra}')
    for r in ps:
        if r["status"] != "ok": print(r["route"], r["status"]); continue
        rw = r.get("rows"); rs = f' rows {rw["tp"]}/{rw["tp"]+rw["fp"]} P {rw["P"]:.3f} R {rw["R"]:.3f} F1 {rw["F1"]:.3f} exact {rw["exact"]}/{rw["views"]}' if rw else ""
        p = r["paired"]
        print(f'{r["route"]:40s} clear {r["clear"][0]}/{r["clear"][1]} scr {r["scrunched"][0]}/{r["scrunched"][1]} all {r["all"][0]}/{r["all"][1]} ({r["all"][2]*100:.1f}-{r["all"][3]*100:.1f}){rs} | paired {p["change"][0]:+.1f} ({p["change"][1]:+.1f},{p["change"][2]:+.1f}) p={p["p"]:.3f}')
    for k, v in comps.items(): print(k, v)

# ---- clear->scrunched paired change, replicating analyze_receipt_photo_metrics.py exactly ----
import math, random
def _perc(a, q):
    a = sorted(a); p = (len(a)-1)*q; lo = int(p); hi = math.ceil(p); return a[lo]+(a[hi]-a[lo])*(p-lo) if a else None
def _mcnemar(k, n):
    if not n: return 1.0
    masses = [math.comb(n, i)/2**n for i in range(n+1)]; return min(1, sum(x for x in masses if x <= masses[k]+1e-15))
def thesis_paired(outs, route, seed):
    Q = W/'stage'/'2026-10-05'/'task'/'deliverables'/'qwen3_5_2b_comparison_v1_recovery'/'receipt_photo_model_results_qwen35_2b_2026-10-06'
    pid = {(r['receipt_id'], r['condition']): r['pair_id'] for r in csv.DictReader(open(Q/'comparison_all_views.csv', encoding='utf-8-sig'))}
    by = defaultdict(dict)
    for (rid, c), v in outs[route].items(): by[pid[(rid, c)]][c] = v
    pairs = [(p, v['clear']['ok'], v['scrunched']['ok']) for p, v in sorted(by.items()) if v['clear']['ok'] is not None and v['scrunched']['ok'] is not None]
    co = sum(c == 1 and s == 0 for _, c, s in pairs); so = sum(c == 0 and s == 1 for _, c, s in pairs); diff = [s-c for _, c, s in pairs]
    rng = random.Random(seed); boot = [sum(rng.choice(diff) for _ in diff)/len(diff)*100 for _ in range(10000)]
    return dict(n=len(pairs), change=sum(diff)/len(diff)*100, lo=_perc(boot, .025), hi=_perc(boot, .975), p=_mcnemar(so, co+so), clear_only=co, scr_only=so)

def thesis_clean_deltas(per, route, ids):
    """Table A.1 style clean-to-condition change, replicating score_cord.py (same seed and resample matrix)."""
    import numpy as np
    N = len(ids); rng = np.random.default_rng(20260930); boot = rng.integers(0, N, size=(10000, N))
    clean = np.array([per[(route, 'clean')][r][0] or 0 for r in ids], float)
    elig = np.array([int(per[(route, 'clean')][r][0] is not None) for r in ids], float)
    out = {}
    for v in CONDS[1:]:
        x = np.array([per[(route, v)][r][0] or 0 for r in ids], float)
        d = (x - clean)*elig; denom = elig[boot].sum(1); db = np.divide(d[boot].sum(1), denom, out=np.zeros(len(denom)), where=denom > 0)
        lo, hi = np.quantile(db, [.025, .975]); out[v] = (d.sum()/elig.sum()*100, lo*100, hi*100)
    return out
