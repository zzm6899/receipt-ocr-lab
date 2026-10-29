"""Corrected LayoutLMv3 (Tesseract route) on the severity suite and the stronger extension."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import json, sys, ast
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import evalkit as E
import final_tables as F

W = ROOT
SEV = ['clean'] + [f'{f}_{s}' for f in ('fade', 'blur', 'occlusion', 'lighting', 'crumple') for s in ('mild', 'moderate', 'severe')]
EXT = sorted({p.stem.split('__')[1] for p in (W/'cord'/'cord_data'/'images'/'severity_extension_test').glob('*.jpg')})

def lit(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return x
    return x

def run():
    refs = E.cord_refs('test'); ids = list(refs); out = {}
    for sub, conds in (('severity_test', SEV), ('severity_extension_test', EXT)):
        for c in conds:
            orig = []; corr = []
            for rid in ids:
                r = refs[rid]; key = f'cord/{sub}/{rid}__{c}'
                o = F.corrected('tess', key)
                if o is None: corr.append(None)
                else:
                    tp, fp, fn, _, _ = E.cord_score_rows(r["rows"], o[1], ["nm", "cnt", "price"]); corr.append((E.cord_total_ok(o[0], r["total"]), tp, fp, fn))
                if sub == 'severity_test':
                    d = json.loads((W/'cord'/'cord_data'/'results'/'severity_test'/f'{rid}__{c}.json').read_text())
                    lf = lit(d['layout_fields']) or {}
                    tp, fp, fn, _, _ = E.cord_score_rows(r["rows"], lit(d['layout_rows']), ["nm", "cnt", "price"]); orig.append((E.cord_total_ok(lf.get('total'), r["total"]), tp, fp, fn))
            out[(sub, c)] = (orig, corr)
    return ids, out

if __name__ == "__main__":
    ids, out = run(); res = {}
    clean_c = out[('severity_test', 'clean')][1]
    for (sub, c), (orig, corr) in out.items():
        if any(x is None for x in corr): print(sub, c, 'incomplete', sum(x is None for x in corr)); continue
        k = sum(x[0] or 0 for x in corr); n = sum(x[0] is not None for x in corr)
        tp, fp, fn = (sum(x[i] for x in corr) for i in (1, 2, 3)); f1 = E.f1(tp, fp, fn)
        ok = [x[0] or 0 for x in corr]; base = [x[0] or 0 for x in clean_c]; el = [int(x[0] is not None) for x in corr]
        # paired change against the severity suite's clean control (same convention as the thesis extension scorer)
        rng = np.random.default_rng(20261001); boot = rng.integers(0, len(ids), size=(10000, len(ids)))
        d = (np.array(ok) - np.array(base))*np.array(el); den = np.array(el)[boot].sum(1); db = d[boot].sum(1)/den
        lo, hi = np.quantile(db, [.025, .975])
        line = dict(k=k, n=n, f1=f1, tp=tp, fp=fp, fn=fn, delta=(d.sum()/sum(el)*100, lo*100, hi*100))
        if orig: line["orig_k"] = sum(x[0] or 0 for x in orig); line["orig_f1"] = E.f1(*(sum(x[i] for x in orig) for i in (1, 2, 3)))
        res[f'{sub}/{c}'] = line
        print(f'{sub:24s} {c:22s} corrected {k}/{n} F1 {f1:.3f} Δclean {line["delta"][0]:+.1f} ({line["delta"][1]:+.1f},{line["delta"][2]:+.1f})' + (f'  original {line["orig_k"]}/{n} F1 {line["orig_f1"]:.3f}' if orig else ''))
    json.dump(res, open(W/'results'/'severity_corrected.json', 'w'), indent=1)
