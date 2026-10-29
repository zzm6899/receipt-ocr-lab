# Photo-study analysis

CPU-only analysis code and aggregate results for the paired clear/scrunched receipt-photo study in *Structured Information Extraction from Receipt Images* (Z. Morgan, UTS 41030 Engineering Capstone B, 2026). No model is run here: the scripts score saved model outputs against the frozen reference labels and write the summary tables and chart.

## Contents

- `analyze_receipt_photo_metrics.py`: totals, Wilson intervals, paired clear/scrunched changes (receipt-level bootstrap, exact McNemar with Holm adjustment), item-row scores, runtime summaries and the chart
- `stratify_totals_by_item_count.py`: total accuracy by annotated item count (`item_count_complexity_addendum/`)
- `analysis_reproducibility.ipynb`: executed notebook that reruns the scorer
- `analysis_note.md`: the analysis note written with these outputs on 7 October 2026
- `*.csv`, `analysis_evidence.json`, `total_accuracy_clear_vs_scrunched.png`: the outputs

## Not included

The receipt photographs, the reference labels and the raw model outputs are not published. The receipts contain personal details of the author and of the friends who contributed some of them.

## Reproducing

The scripts read the saved outputs and labels from the original project folders, which are private. With those folders available, set `RECEIPT_PROJECT_ROOT` to the folder that holds the dated project directories and run:

```
python analyze_receipt_photo_metrics.py
python stratify_totals_by_item_count.py
```

Without the private inputs, the CSV files here are the recorded outputs.

## Changes from the original handoff

Absolute workstation paths were replaced with `<PROJECT_ROOT>` / `<USER_DOCUMENTS>` placeholders, or with the `RECEIPT_PROJECT_ROOT` variable in the two scripts, in: `analysis_evidence.json`, `analysis_reproducibility.ipynb`, `analyze_receipt_photo_metrics.py`, `stratify_totals_by_item_count.py`, `checkpoint_provenance.csv` and `item_count_complexity_addendum/verified_input_hashes.csv`. SHA-256 values in `package_manifest.json` and `item_count_complexity_addendum/output_manifest.csv` describe the original files, so they will not match those six edited files. In `stratify_totals_by_item_count.py`, the private inputs are now read from `RECEIPT_PROJECT_ROOT` rather than from beside the script, and input hashes are looked up under the same `<PROJECT_ROOT>` keys recorded in `analysis_evidence.json`; the checks themselves are unchanged. `item_count_complexity_addendum/per_receipt_total_item_counts.csv` is withheld because it lists each private receipt's reference total. Every other file is byte-identical to the original. Status fields such as `assistant-audited/unverified` are reproduced exactly as the scripts wrote them on 7 October 2026.
