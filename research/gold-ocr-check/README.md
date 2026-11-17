# LayoutLMv3 input check (29 October 2026)

What changes when LayoutLMv3 gets annotated words and boxes instead of words from Tesseract? This check runs the same local LayoutLMv3 CORD checkpoint on all 100 clean CORD v2 test receipts with different inputs. It scores totals and menu rows with the thesis scorer (`score_cord.py` helpers and `row_metrics.score_rows`).

- `run.py`: CORD's annotated words and boxes. Args: start end [annotation|reading] [word|segment]. `segment` gives every word its annotated segment box, the convention LayoutLMv3 was fine-tuned with.
- `run_tess.py`: Tesseract 5.3.4 (--psm 6) words with word boxes, whole-line boxes or gap-split segments.
- `score.py`: totals (exact amount), row F1 (nm, cnt, price) and entity F1 against CORD's labels, with paired receipt bootstrap against the original Tesseract-input predictions.
- `summary.json`: all results. `preds*/`: per-receipt predictions.
- `cord_test_clean.zip`, `tess_words_cord_test.json`: inputs used.

Not included in this repository: the `preds*/` folders, `cord_test_clean.zip` and `tess_words_cord_test.json` (all regenerable from CORD v2 with these scripts). The scripts assume the original folder layout (`$HOME/mnt/Codex/2026-09-30/task` for the project, `$HOME/mnt/layoutlmv3-finetuned-cord` for the checkpoint); edit `TASK` and `MODEL` at the top of each script to run elsewhere.
