# Receipt total accuracy by annotated product-row count

This is a frozen-input, CPU-only descriptive join. The script verifies SHA-256 hashes recorded in the parent analysis manifest, maps `legacy_receipt_###` to `receipt_###` and `new_###` directly, and confirms that each known saved total matches the joined reference value. The 50 selected receipt IDs map one-to-one; 600 completed score rows cover six routes by two views.

Item-count proxy is available for 49/50 selected receipts. One partial/fragment reference is kept in the per-receipt file and shown in a separate unbinned row. One of 100 total-view targets is unknown (six route decisions); unknown labels stay out of accuracy denominators and remain counted in coverage fields.

Product-count bin coverage is 1–3: 35, 4–5: 2, and 6+: 12 receipts. Under raw reference-entry counts it is 1–3: 33, 4–5: 3, and 6+: 13; two Receipy ledgers move bins (`new_005`, `new_013`) because raw entries include promotions/discounts. The 4–5 product bin has only two receipts, so its percentages are especially unstable. Legacy rows lack a `row_type`, so every listed `line_items` row is treated as a product for this proxy; full inventory completeness is not independently asserted.

Exact total matches / known reference labels by route and condition (Wilson intervals and coverage counts are in the CSV):

| Route | Clear 1–3 | Clear 4–5 | Clear 6+ | Scrunched 1–3 | Scrunched 4–5 | Scrunched 6+ |
|---|---:|---:|---:|---:|---:|---:|
| Tesseract+rules | 9/35 (25.7%) | 1/2 (50.0%) | 4/12 (33.3%) | 6/34 (17.6%) | 0/2 (0.0%) | 0/12 (0.0%) |
| LayoutLMv3 | 1/35 (2.9%) | 0/2 (0.0%) | 5/12 (41.7%) | 0/34 (0.0%) | 0/2 (0.0%) | 0/12 (0.0%) |
| TrOCR+LayoutLMv3 | 8/35 (22.9%) | 0/2 (0.0%) | 5/12 (41.7%) | 4/34 (11.8%) | 0/2 (0.0%) | 0/12 (0.0%) |
| Donut | 10/35 (28.6%) | 0/2 (0.0%) | 4/12 (33.3%) | 10/34 (29.4%) | 1/2 (50.0%) | 4/12 (33.3%) |
| GLM-OCR | 28/35 (80.0%) | 2/2 (100.0%) | 12/12 (100.0%) | 31/34 (91.2%) | 1/2 (50.0%) | 12/12 (100.0%) |
| Qwen3.5-2B | 28/35 (80.0%) | 1/2 (50.0%) | 11/12 (91.7%) | 27/34 (79.4%) | 1/2 (50.0%) | 12/12 (100.0%) |

Across 161 readable/score-eligible product-name strings, length median is 23 characters (IQR 15–35; range 4–86). The sum of available product-name characters per receipt has median 49 (IQR 29–122; range 0–323) across 49 binned receipts. This is an item-list text-volume proxy, not full receipt text length or OCR token count.

`per_receipt_total_item_counts.csv` retains item counts, coverage status, unknown item fields, unknown total-label status, and each route's saved match flag. `product_name_length_summary.csv` describes individual name lengths; `receipt_item_name_text_volume_proxy.csv` sums those characters per receipt. These are item-list text proxies only; no complete receipt transcription or OCR-token count exists in these references. `item_count_distribution_summary.csv` records raw-versus-product bin sensitivity. These descriptive strata do not establish that more items cause worse accuracy. The labels are assistant-audited and unverified.

Reproduce from the analysis handoff directory with:

```powershell
..\.venv-pilot\Scripts\python.exe .\stratify_totals_by_item_count.py
```
