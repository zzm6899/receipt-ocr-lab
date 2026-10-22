# OCR model comparator

Stateless upload-and-compare website prepared for `ocr.z2hs.au`. Docker Compose maps TrueNAS host port **5099** to app port 8000.

## Verified support

| Route | Default status | Meaning |
|---|---|---|
| Tesseract 5 | Available | Installed in the image; returns measured text and latency. |
| GLM-OCR | Unavailable until configured | Direct-image asset exists; deployment needs a Linux adapter and separately mounted model files. |
| Qwen3.5-2B | Unavailable until configured | Direct-image asset exists; deployment needs a Linux adapter and separately mounted model files. |
| LayoutLMv3 | Unavailable | Legacy CORD checkpoint exists, but its upstream revision was not recorded and no Linux serving adapter is packaged. The failed/unfinished training candidate is excluded. |
| TrOCR + LayoutLMv3 | Unavailable | Existing interface uses TrOCR only to reread regions seeded by OCR candidates; it is not whole-page OCR. |
| Donut | Unavailable | Legacy CORD v2 asset exists, but no verified Linux serving adapter is packaged. |

Unavailable routes remain visible and cannot be selected. Nothing is simulated, and the UI makes no accuracy claims.

## Per-model hardware notes

These are deployment facts and planning budgets, not universal minimum requirements. Only the rows labelled **measured** have been run on this project's hardware; system-RAM minima for the research routes have not been measured.

| Route | Weights / precision | CPU support | GPU / VRAM evidence | System RAM |
|---|---|---|---|---|
| Tesseract 5 | CPU package; no neural checkpoint | Yes; current public route | No GPU needed. **Verified** on this TrueNAS CPU in a container limited to 2 GiB RAM. | 2 GiB is the current safe container limit, not a measured minimum. |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 1B parameters, BF16 model card; local asset revision `2e85a62840ccac27daa451df36c736c4636b8628` | Not benchmarked here | **Measured:** 2.22 GiB peak reserved VRAM on an RTX 4070 for one image. This is observed use, not a minimum; allow additional runtime headroom. | Minimum not measured. Provisional isolated-worker budget: 8 GiB, validate before public enablement. |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B-class multimodal model; local asset revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; repository weight file about 4.55 GB | Frameworks may support CPU, but this route is not benchmarked on CPU here | **Measured:** 4.81 GiB peak reserved VRAM on an RTX 4070 for one image. Allow headroom above that measurement. | Minimum not measured. Provisional isolated-worker budget: 12 GiB, validate before public enablement. |
| [LayoutLMv3 CORD](https://huggingface.co/nielsr/layoutlmv3-finetuned-cord) | Local safetensors file about 504 MB; precision/revision unrecorded | Expected through PyTorch, but no packaged Linux adapter or benchmark | Not measured. Provisional GPU budget cannot be stated honestly until the adapter is verified. | Minimum not measured; provisional worker budget 4 GiB. |
| [TrOCR base printed](https://huggingface.co/microsoft/trocr-base-printed) + LayoutLMv3 | Local TrOCR assets about 1.33 GB plus the 504 MB LayoutLMv3 classifier | Expected through PyTorch, but not benchmarked | Not measured. Existing route rereads OCR-seeded regions; it is not whole-page OCR. | Minimum not measured; provisional worker budget 8 GiB. |
| [Donut CORD v2](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2) | Local assets about 806 MB; MIT model card | Expected through PyTorch, but no packaged Linux adapter or benchmark | Not measured. | Minimum not measured; provisional worker budget 6 GiB. |

GLM-OCR's model card is MIT and notes the optional PP-DocLayoutV3 pipeline is Apache-2.0. Qwen3.5-2B is Apache-2.0. Donut CORD v2 is MIT. The legacy LayoutLMv3 and TrOCR local asset cards do not record a license in this handoff, so redistribute code only until their upstream licenses are audited. The current TrueNAS host has no verified container GPU and recently had only about 1.1 GiB memory available; therefore it must remain Tesseract-only until host memory and GPU capability are remediated and rechecked.

## Detection overlays

Tesseract returns real word boxes and its own TSV confidence. The comparison viewer maps normalized boxes onto the EXIF-corrected or rendered page and provides page, overlay, opacity, and zoom controls. Colours are confidence bands from the producing model only; they are not attention, saliency, accuracy, or calibrated across models. Adapters may optionally return `regions` with normalized `bbox`, text, field, and confidence values. Models that provide no spatial output show "No spatial overlay is available" instead of invented geometry.

## Run

```sh
docker compose up --build
```

Open `http://HOST:5099`. Confirm port 5099 is free on the **TrueNAS host** before deployment; the desktop check is not sufficient.

For TrueNAS, use `compose.truenas.yaml` after the GitHub Actions run has successfully published `ghcr.io/zzm6899/receipt-ocr-lab:main`. The workflow tests every change and publishes an OCI image on pushes to `main` and `v*` tags. It uses only the repository-scoped `GITHUB_TOKEN` with `contents: read` and `packages: write`; no PAT is required.

Publishing a new image does **not** restart TrueNAS. Apply updates deliberately from the TrueNAS app/Compose interface by pulling the current image and redeploying the stack. For reproducible production installs, replace `:main` with a verified release tag or image digest after the first successful publish.

## Optional local model adapter contract

Set `OCR_GLM_COMMAND` or `OCR_QWEN_COMMAND` to a fixed executable command inside the container. The service appends one temporary PNG path. The command writes exactly one JSON object to stdout:

```json
{"text":"raw model output","fields":{"total":"12.34"},"regions":[{"bbox":[0.1,0.2,0.3,0.1],"text":"TOTAL","field":"total","confidence":0.91}],"checkpoint":"model/revision-or-local-digest"}
```

Coordinates are fractions of the processed page in `[x, y, width, height]` order; confidence is `0..1` or `null`. The adapter and model live on TrueNAS in a GPU image or read-only mount. Do not point these variables at a desktop or remote inference endpoint. Commands run without a shell, have a timeout, and receive only the temporary image path. Adapters must not log images or extracted text.

## Public deployment requirements

- Recheck port 5099, GPU/VRAM, NVIDIA Container Toolkit support, and model redistribution licenses on TrueNAS.
- Put a reverse proxy in front for TLS and `ocr.z2hs.au`; set its request-body limit to 10 MB.
- Keep one app worker. The semaphore serializes inference; Compose also bounds CPU, memory, PIDs and temporary storage.
- Defaults: 5 requests/minute/client, 10 MB/file, 3 PDF pages, 20 megapixels/page, 2 models/request, 5-second queue wait and 60 seconds/model/page.
- Do not mount NAS document libraries. Mount only model assets/adapters read-only. Uploads use an isolated temporary directory and are deleted before the response completes.
- Add a GPU reservation and raise memory only after the TrueNAS hardware and adapter are verified.

## Verify

```sh
python -m pip install -r requirements-dev.txt
pytest -q
docker compose config
```

This source tree contains no checkpoints, receipts, research evidence, credentials, hostnames or NAS paths. Model weights stay outside Git and require separate license review before redistribution.

The structured fields shown for Tesseract are conservative regex-derived display aids and are labelled as such. They are not accuracy measurements. Prior experimental scores are deliberately omitted because they do not establish public-upload accuracy.


