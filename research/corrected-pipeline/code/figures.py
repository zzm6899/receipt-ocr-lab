"""Regenerate Figure 5.1 (photo totals) and Figure 5.3 (CORD primary suite totals) with the corrected routes."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import stats as S

R = json.load(open(str(ROOT/'results/final_results.json')))
OUT = Path(str(ROOT/'results/figs')); OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "hatch.linewidth": 0.9})

def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---------- Figure 5.1: photo totals by route and condition ----------
order = [("Tesseract + rules", "Tesseract"), ("LayoutLMv3", "LayoutLMv3"), ("LayoutLMv3 (corrected)", "LayoutLMv3\n(corrected)"),
         ("TrOCR + LayoutLMv3", "TrOCR +\nLayoutLMv3"), ("TrOCR + LayoutLMv3 (corrected)", "TrOCR +\nLayoutLMv3\n(corrected)"),
         ("Donut", "Donut"), ("GLM-OCR", "GLM-OCR"), ("Qwen3.5-2B", "Qwen3.5-2B")]
ph = {r["route"]: r for r in R["photo"] if r["status"] == "ok"}
order = [o for o in order if o[0] in ph]
fig, ax = plt.subplots(figsize=(12, 5.4), dpi=150); style(ax)
w = 0.38
for i, (route, label) in enumerate(order):
    corr = "corrected" in route
    for j, (cond, col) in enumerate((("clear", BLUE), ("scrunched", ORANGE))):
        k, n, lo, hi = ph[route][cond]; x = i + (j - 0.5) * w
        ax.bar(x, k / n * 100, w - 0.03, color=col, edgecolor="white" if not corr else col, hatch="////" if corr else None,
               facecolor=col if not corr else "white", linewidth=1.2 if corr else 0, zorder=2)
        ax.text(x, k / n * 100 + 1.5, f"{k}/{n}", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=9.5)
ax.set_ylim(0, 105); ax.set_ylabel("Exact designated-total match (%)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=BLUE, label="Clear"), Patch(facecolor=ORANGE, label="Scrunched"),
                   Patch(facecolor="white", edgecolor=INK2, hatch="////", label="Corrected pipeline (Section 4.8)")],
          frameon=False, loc="upper left", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT/"fig5_1_photo_totals.png"); plt.close(fig)

# ---------- Figure 5.3: CORD primary suite totals ----------
conds = [("clean", "Clean"), ("color_cast", "Blue cast"), ("fade", "Fade"), ("occlusion", "Occlusion"), ("rotate180", "180°"), ("synthetic_crumple", "Synthetic\ncrumple")]
routes = [("Tesseract + rules", BLUE, False), ("LayoutLMv3", ORANGE, False), ("LayoutLMv3 (corrected)", ORANGE, True),
          ("TrOCR → LayoutLMv3", AQUA, False), ("TrOCR → LayoutLMv3 (corrected)", AQUA, True), ("Donut", YELLOW, False)]
cd = {(r["route"], r["cond"]): r for r in R["cord"] if r["status"] == "ok"}
fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150); style(ax)
w = 0.13
for ci, (c, cl) in enumerate(conds):
    present = [rt for rt in routes if (rt[0], c) in cd]
    for ri, (route, col, corr) in enumerate(present):
        r = cd[(route, c)]; x = ci + (ri - (len(present) - 1) / 2) * w; p = r["k"] / r["n"] * 100
        ax.bar(x, p, w - 0.02, facecolor="white" if corr else col, edgecolor=col, hatch="////" if corr else None, linewidth=1.2 if corr else 0, zorder=2)
        ax.errorbar(x, p, yerr=[[p - r["lo"] * 100], [r["hi"] * 100 - p]], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
ax.set_xticks(range(len(conds)), [c[1] for c in conds]); ax.set_ylim(0, 105); ax.set_ylabel("Exact total accuracy (%)")
ax.legend(handles=[Patch(facecolor=col if not corr else "white", edgecolor=col, hatch="////" if corr else None, label=route) for route, col, corr in routes],
          frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT/"fig5_3_cord_totals.png"); plt.close(fig)
print("figures written")

# ---------- Figure 5.4: severity, fade and blur ----------
import csv
sevp = Path(str(ROOT/'results/severity_corrected.json'))
if sevp.exists():
    sev = json.load(open(sevp))
    rows = list(csv.DictReader(open(str(ROOT/'cord/cord_data/results/severity_test/severity_metrics.csv'), encoding='utf-8')))
    by = {(r['model'], r['condition']): r for r in rows}
    models = [("Tesseract+rules", "Tesseract + rules", BLUE, False), ("LayoutLMv3", "LayoutLMv3", ORANGE, False), (None, "LayoutLMv3, corrected", ORANGE, True),
              ("TrOCR->LayoutLMv3", "TrOCR → LayoutLMv3", AQUA, False), ("Donut", "Donut", YELLOW, False)]
    names = {r['model'] for r in rows}
    def mname(m):
        for n in names:
            if n.replace('→', '->').replace(' ', '') == m.replace(' ', ''): return n
        return m
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, sharey=True)
    for ax, fam, title in ((axes[0], 'fade', 'Fade (simulated)'), (axes[1], 'blur', 'Blur (simulated)')):
        style(ax); levels = ['clean', f'{fam}_mild', f'{fam}_moderate', f'{fam}_severe']; xs = range(4)
        for m, label, col, corr in models:
            ks = []; los = []; his = []
            for lv in levels:
                if corr: k = sev[f'severity_test/{lv}']['k']; n = sev[f'severity_test/{lv}']['n']
                else: r = by[(mname(m), lv)]; k = int(r['total_exact_correct']); n = int(r['total_gt_n'])
                lo, hi = S.wilson(k, n); ks.append(k/n*100); los.append(k/n*100 - lo*100); his.append(hi*100 - k/n*100)
            ax.errorbar(xs, ks, yerr=[los, his], color=col, lw=2, ls='--' if corr else '-', marker='o', ms=6, capsize=3, label=label,
                        mfc='white' if corr else col)
        ax.set_xticks(list(xs), ['Clean', 'Mild', 'Moderate', 'Severe']); ax.set_title(title, color=INK, fontsize=12, loc='left'); ax.set_ylim(0, 105)
    axes[0].set_ylabel('Exact total accuracy (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc='lower center', fontsize=9.5)
    fig.tight_layout(rect=[0, 0.09, 1, 1]); fig.savefig(OUT/"fig5_4_severity.png"); plt.close(fig)
    print("figure 5.4 written")
