# Corrected LayoutLMv3 pipeline (post-diagnosis revision, 28–29 October 2026)

Code, frozen settings and results for the corrected LayoutLMv3 routes in thesis Section 4.8 and Appendix A.9.

## What changed and what did not

Unchanged from the thesis pipeline: input images (pixel hashes checked against the original runs), the Tesseract call (`--psm 6`), TrOCR line crops, order and batching, the LayoutLMv3 checkpoint (`nielsr/layoutlmv3-finetuned-cord`, `model.safetensors` SHA-256 8b209f25…), the TrOCR checkpoint (`microsoft/trocr-base-printed`, 1cf4a6ee…), the label set and the best-sub-token word-labelling rule.

Changed:

1. **Segment boxes.** The words of a segment share its box: whole Tesseract lines for totals, lines split at number/text changes for rows.
2. **Long inputs.** Words are labelled in windows of whole lines that each fit in 512 tokens. Truncation had dropped the end of 19 of the 100 photographed views and 3 of the 100 clean CORD test receipts.
3. **Label decoding.** A per-segment label vote, the single most confident total span containing one number, and reading-order row grouping.

## Selection protocol

All options were scored on the CORD validation split only (100 receipts × clean, colour cast, fade, occlusion and 180° rotation). `results/frozen_config.json` was written before any corrected test or photo run (SHA-256 5c4f24b266d85985a326203e5eaa741b36f6f5191ad81a7709870aa1c44e7dbe). Line boxes and two gap thresholds had been tried on the clean test split during the earlier diagnosis (`../gold-ocr-check`), so the test split is not untouched for those settings. The photographs were not used for any choice.

## Environment

CPU only: Python 3.13, PyTorch 2.5.1+cpu, Transformers 4.46.3, Tesseract 5.3.4 (the original runs used Tesseract 5.2.0 on a Windows GPU workstation). The original LayoutLMv3 routes were rerun unchanged in this environment as a control (thesis Table A.12). TrOCR decoding used the key-value cache; a 45-line check gave identical text without it.

## Running

Set `PIPELINE_ROOT` to a working folder holding `cord/cord_data/` (CORD export with images) and, for the photo study, the private photo inputs; `MODELS_DIR` to a folder holding `layoutlmv3-finetuned-cord/` and `trocr-base-printed/`; and `PROJECT_ROOT` to the original project folder that holds the thesis scorers (`2026-09-30/task/score_cord.py`, `row_metrics.py`, and the photo scorers in `../photo-scoring/`). Stages, in order:

1. `code/ocr_cache.py K N` (shard K of N) and `code/ocr_cache_sev.py` — Tesseract words per image.
2. `code/trocr_cache.py KEYS` — TrOCR text per Tesseract line crop.
3. `code/layout_run.py KEYS tess|trocr MODES word|line|num|gapK[num] trunc|window` — LayoutLMv3 labels.
4. `code/select_val.py` — validation grid and freezing rule (writes `results/frozen_config.json`).
5. `code/final_tables.py`, `code/sev_tables.py`, `code/figures.py` — every reported number, interval and figure.

`code/evalkit.py` holds the decoders and imports the thesis scorers verbatim. Before any corrected result was computed, the scorers were checked by reproducing every original thesis figure from the saved original outputs. `code/ocr_probe.py` is an exploratory check of other Tesseract modes on validation receipts (none beat `--psm 6`).

## Results

`results/validation_grid.json`, `results/frozen_config.json`, `results/final_results.json` (aggregate CORD and photo results), `results/severity_corrected.json` and `results/figs/`. Key lists (`keys_*.txt`) name the inputs in run order. OCR and label caches are not included: CORD caches are regenerable with stages 1–3, and the photo caches contain private receipt text.
