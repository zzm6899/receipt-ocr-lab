"""Choose the corrected LayoutLMv3 configuration on the CORD validation split only, then freeze it.
Rule (fixed before any corrected test or photo run):
  totals: maximise exact totals pooled over the five validation conditions; ties -> fewer changes from the thesis pipeline
  rows:   maximise grouped-row F1 pooled over the same conditions; same tie rule
"""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import json, sys, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from score_layout import score_cord

CONDS = ["clean", "color_cast", "fade", "occlusion", "rotate180"]
MODES = ["word", "line", "num", "gap1.5"]          # gap3 and gap1.5+num were dropped after the clean validation pass (dominated)
COMPLEXITY = {"word": 0, "line": 1, "num": 2, "gap1.5": 2}
grid = []
for mode in MODES:
    for vote in ([False] if mode == "word" else [False, True]):
        for td in ["join", "span"]:
            for rd in ["line", "reset"]:
                res = score_cord(f"tess_{mode}_window", "validation", CONDS, td, rd, vote)
                k = sum(r["k"] for r in res.values()); n = sum(r["n"] for r in res.values())
                tp = sum(r["tp"] for r in res.values()); fp = sum(r["fp"] for r in res.values()); fn = sum(r["fn"] for r in res.values())
                miss = sum(r["missing"] for r in res.values())
                grid.append(dict(mode=mode, vote=vote, totals=td, rows=rd, k=k, n=n, acc=k/n, tp=tp, fp=fp, fn=fn,
                                 f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0, missing=miss,
                                 per_cond={c: (res[c]["k"], res[c]["n"], round(res[c]["f1"], 4)) for c in CONDS},
                                 changes=COMPLEXITY[mode] + int(vote) + int(td != "join") + int(rd != "line")))
assert all(g["missing"] == 0 for g in grid), "validation runs incomplete"
best_t = max(grid, key=lambda g: (g["k"], -g["changes"]))
best_r = max(grid, key=lambda g: (round(g["f1"], 6), -g["changes"]))
frozen = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "split": "validation", "conditions": CONDS,
          "totals_config": {k: best_t[k] for k in ("mode", "vote", "totals")}, "rows_config": {k: best_r[k] for k in ("mode", "vote", "rows")},
          "long_inputs": "window", "totals_validation": f'{best_t["k"]}/{best_t["n"]}', "rows_validation_f1": round(best_r["f1"], 4)}
out = Path(str(ROOT/'results')); out.mkdir(exist_ok=True)
(out/'validation_grid.json').write_text(json.dumps(grid, indent=1))
s = json.dumps(frozen, indent=1); (out/'frozen_config.json').write_text(s)
print(s); print('sha256', hashlib.sha256(s.encode()).hexdigest())
for g in sorted(grid, key=lambda g: -g["k"])[:12]:
    print(g["mode"], int(g["vote"]), g["totals"], g["rows"], f'{g["k"]}/{g["n"]}', round(g["f1"], 3), g["per_cond"])
