# Research code

This folder contains the code and recorded results for *Structured Information Extraction from Receipt Images* (Zachary Morgan, UTS 41030 Engineering Capstone B, 2026). Use it to follow the evaluation behind the public web demo.

| Folder | What it holds | Thesis |
|---|---|---|
| `cord/` | The CORD v2 benchmark pipeline (Tesseract + rules, LayoutLMv3, TrOCR → LayoutLMv3, Donut), image-damage recipes, scorers, the 600 saved primary-suite predictions and the metric tables | Sections 4.3, 5.3, 5.4; Appendix A.1–A.4, A.7, B.1 |
| `photo-analysis/` | Scoring and statistics for the paired clear/scrunched photo study, with aggregate outputs only | Sections 5.1, 5.2; Appendix A.5–A.6, E |
| `photo-scoring/` | The frozen photo scorers: `shared_row_evaluator.py` (total amount parser) and `score_receipy_v2.py` (one-to-one item rows) | Section 4.4; Appendix E.1 |
| `gold-ocr-check/` | The LayoutLMv3 input check: the same checkpoint on CORD's annotated words and boxes against Tesseract words | Section 7.2; Appendix A.8 |
| `corrected-pipeline/` | The corrected LayoutLMv3 routes: segment boxes, windowed labelling of long receipts and per-segment label decoding, chosen on CORD validation and then frozen | Section 4.8; Appendix A.9 |

## Privacy

The photos and their reference labels stay private because the receipts contain personal details about me and the friends who shared them. I also withheld raw model outputs and any OCR text from the photos. This folder includes code, outputs from CORD and aggregate results from the photo study.

## Reference label status

I used an AI assistant to transcribe and audit the photo reference labels. On 1 October 2026, I checked every total and item row by hand against all 100 photos. That check isn't in the pipeline's registers, so its generated files still carry the earlier labels (`assistant-audited/unverified`, human decision `pending`). The frozen reference files and their hashes did not change.

## Running

Everything here scores saved outputs on a CPU, except the inference scripts (`cord/benchmark_cord.py`, `cord/benchmark_photo_exif_transposed.py` and the `corrected-pipeline` stages), which need the model checkpoints listed in thesis Appendix B.5. The CORD benchmark expects CORD v2 (`naver-clova-ix/cord-v2`) exported with `cord/cord_pilot.py`. See each folder's README for paths and environment variables. `cord/score_orientation.py` is the shared scorer imported by the severity scorers; its SHA-256 matches the one recorded in `cord/results/run_manifest.json`. Apart from path settings and the two input-path fixes noted in `photo-analysis/README.md`, the scripts are published as they were run, so recorded fields keep their original meaning (for example, `elapsed_seconds` in `cord/benchmark_photo_exif_transposed.py` is time since the start of the run).
