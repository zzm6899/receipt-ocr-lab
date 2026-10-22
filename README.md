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
{"text":"raw model output","fields":{"total":"12.34"},"checkpoint":"model/revision-or-local-digest"}
```

The adapter and model live on TrueNAS in a GPU image or read-only mount. Do not point these variables at a desktop or remote inference endpoint. Commands run without a shell, have a timeout, and receive only the temporary image path. Adapters must not log images or extracted text.

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


