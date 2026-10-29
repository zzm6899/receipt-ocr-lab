import json, re, sys, unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
H = Path.home(); TASK = H/'mnt'/'Codex'/'2026-09-30'/'task'
sys.path.insert(0, str(TASK)); from row_metrics import score_rows
src = (TASK/'score_cord.py').read_text()
# reuse score_cord's own amount parser and ground-truth helpers verbatim (functions only, no module code)
ns = {}
for name in ("parse_amount", "gt_total", "menu_list", "gt_menu_rows", "wilson"):
    m = re.search(rf"^def {name}\(.*?(?=^def |\Z|^manifest=)", src, flags=re.S | re.M)
    exec("import re,unicodedata\nfrom decimal import Decimal,InvalidOperation\n" + m.group(0), ns)
parse_amount, gt_total, gt_menu_rows, wilson = ns["parse_amount"], ns["gt_total"], ns["gt_menu_rows"], ns["wilson"]
manifest = json.loads((TASK/'cord_data'/'test'/'manifest.json').read_text()); ids = [m["receipt_id"] for m in manifest]
SUP = ["nm", "cnt", "price"]
PRED = sys.argv[1] if len(sys.argv) > 1 else 'preds'
def spans(tags):
    out, cur = set(), None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and (cur is None or cur[2] != t[2:])):
            if cur: out.add((cur[0], i, cur[2])); cur = None
            if t.startswith("B-") or t.startswith("I-"): cur = (i, None, t[2:])
    return out
res = {"gold": [], "tess": []}; ent_tp = ent_p = ent_g = 0; wacc = wn = 0
for rid in ids:
    gt = json.loads((TASK/'cord_data'/'test'/f'{rid}.json').read_text())["ground_truth"]
    truth = gt_total(gt); rows = gt_menu_rows(gt)
    g = json.loads((H/'gold_ocr'/PRED/f'{rid}.json').read_text())
    t = json.loads((TASK/'cord_data'/'direct_predictions'/'test'/f'{rid}__clean.json').read_text())
    for key, pred_total, prow, grow in (("gold", (g["layout_fields"] or {}).get("total"), g["layout_rows_physical_row"], g["layout_rows_gold_group"]),
                                        ("tess", (t.get("layout_fields") or {}).get("total"), t.get("layout_rows", []), None)):
        ok = int(truth is not None and parse_amount(pred_total) is not None and parse_amount(pred_total) == parse_amount(truth))
        res[key].append({"elig": int(truth is not None), "ok": ok, "rows": score_rows(rows, prow, SUP), "rows_group": score_rows(rows, grow, SUP) if grow is not None else None})
    fix = lambda tag: tag.replace('MENU.SUB.', 'MENU.SUB_')
    gold_tags = [fix(x[1]) for x in g["word_tags"]]; pred_tags = [x[2] for x in g["word_tags"]]
    gs, ps = spans(gold_tags), spans(pred_tags); ent_tp += len(gs & ps); ent_p += len(ps); ent_g += len(gs)
    wacc += sum(a == b for a, b in zip(gold_tags, pred_tags)); wn += len(gold_tags)
def summ(key, rk="rows"):
    r = res[key]; k = sum(x["ok"] for x in r); n = sum(x["elig"] for x in r)
    c = np.array([x[rk][:3] for x in r]); tp, fp, fn = c.sum(0)
    return k, n, wilson(k, n), int(tp), int(fp), int(fn), 2*tp/(2*tp+fp+fn)
out = {}
for key in ("tess", "gold"):
    k, n, (lo, hi), tp, fp, fn, f1 = summ(key); out[key] = dict(total=f"{k}/{n}", total_acc=round(k/n, 4), wilson=[round(lo, 3), round(hi, 3)], row_tp=tp, row_fp=fp, row_fn=fn, row_f1=round(f1, 3))
k, n, _, tp, fp, fn, f1 = summ("gold", "rows_group"); out["gold_oracle_grouping"] = dict(row_tp=tp, row_fp=fp, row_fn=fn, row_f1=round(f1, 3))
P, R = ent_tp/ent_p, ent_tp/ent_g; out["gold_entity"] = dict(precision=round(P, 4), recall=round(R, 4), f1=round(2*P*R/(P+R), 4), word_accuracy=round(wacc/wn, 4), entities_gold=ent_g)
# paired receipt bootstrap of the total-accuracy difference (gold minus Tesseract input)
rng = np.random.default_rng(20261029); N = len(ids); boot = rng.integers(0, N, size=(10000, N))
e = np.array([x["elig"] for x in res["gold"]]); d = np.array([x["ok"] for x in res["gold"]]) - np.array([x["ok"] for x in res["tess"]])
bd = (d*e)[boot].sum(1)/e[boot].sum(1)*100; out["total_diff_pp"] = dict(est=round((d*e).sum()/e.sum()*100, 1), ci95=[round(v, 1) for v in np.quantile(bd, [.025, .975])])
cg = np.array([x["rows"][:3] for x in res["gold"]]); ct = np.array([x["rows"][:3] for x in res["tess"]])
f = lambda c: np.divide(2*c[:, 0], 2*c[:, 0]+c[:, 1]+c[:, 2]); bf = f(cg[boot].sum(1)) - f(ct[boot].sum(1))
out["row_f1_diff"] = dict(est=round(out["gold"]["row_f1"]-out["tess"]["row_f1"], 3), ci95=[round(v, 3) for v in np.quantile(bf, [.025, .975])])
print(json.dumps(out, indent=1)); (H/'gold_ocr'/f'summary_{PRED}.json').write_text(json.dumps(out, indent=1))
