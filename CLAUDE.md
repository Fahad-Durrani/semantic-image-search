# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local semantic image-search app over ~7,810 images. Images are embedded once with **Apple's official MobileCLIP-S2** (the `mobileclip` package from github.com/apple/ml-mobileclip, using the local `mobileclip_s2.pt` checkpoint). A Flask server then encodes text queries — or an uploaded/selected image — with the same model and ranks images by cosine similarity, with MMR re-ranking for diversity. The frontend (search, autocomplete, similar-image, drag-and-drop upload, lightbox) is a single HTML string embedded in `app.py`.

> Note: an earlier version used open-clip-torch's `MobileCLIP2-S2` (`dfndr2b`) weights from HuggingFace. That was abandoned because it applies a different image normalization than Apple's. Do **not** reintroduce open_clip for model loading — use the `mobileclip` package so the normalization matches.

## Commands

```powershell
# One-time setup
pip install -e ./ml-mobileclip --no-deps   # Apple package; deps (torch/open_clip/timm) already satisfied
pip install flask Pillow numpy tqdm
# checkpoint: checkpoints/mobileclip_s2.pt  (download from
#   https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt )

# Build the embedding index (~10 min on CPU). Shows a tqdm progress bar.
python build_index.py

# Run the app
python app.py            # -> http://127.0.0.1:5000

# Quick smoke test
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3"
```

No tests/linter/build step. Re-run `build_index.py` only when the contents of `images_repo/` change.

## Directory layout (paths are hardcoded in both scripts — keep in sync)

- `images_repo/` — the flat image library, files named `1.jpg … 7811.png`. Served by `GET /images/<filename>`.
- `checkpoints/mobileclip_s2.pt` — Apple checkpoint (~380 MB, not in git).
- `cache/embeddings.npz` — built index. **Two aligned arrays: `embeddings` (N×512 float32, L2-normalized) and `filenames` (bare names).** Note the key is `filenames`, not `paths`.
- `cache/suggestions.json` — autocomplete counts, auto-rebuilt when `VOCAB` changes (keyed by an md5 hash of the vocab).
- `ml-mobileclip/` — cloned Apple repo, installed editable. Must persist (editable install references it).
- `build_index.py`, `app.py` — the two entry points.

## Architecture

Two entry points share the same model + L2-normalization, so cosine similarity is a plain dot product at query time:

- **`build_index.py`** — offline indexer. Loads `mobileclip_s2` from the checkpoint, batches images (`BATCH_SIZE=32`) through `encode_image`, L2-normalizes, writes `cache/embeddings.npz`. Per-image try/except skips corrupt files (e.g. `4784.jpg`), so `len(filenames)` can be one or two less than the file count on disk.
- **`app.py`** — Flask server. On startup loads the npz into globals, loads the model for query-time text/image encoding, kicks off a background thread to build the **colour palette index**, then builds the suggestion + prefix index (encodes the ~500-term `VOCAB` once). Endpoints: `/search` (text, supports `-negative` terms and `&color=`), `/similar` (by indexed filename), `/search_by_image` (POST upload), `/suggest` (autocomplete), `/colors` (available colour names), `/rebuild` (POST, starts incremental re-index) + `/rebuild_stream` (SSE progress), `/images/<f>`, `/`.
- The single-page frontend (in `FRONTEND_HTML`) has: text/image search, autocomplete, **recent searches** (localStorage), **colour filter bar**, **dark mode** toggle (CSS variables + `[data-theme="dark"]`, persisted), lightbox, find-similar, and a **Rebuild Index** modal driven by the SSE stream.

### Extra cache files (besides embeddings.npz / suggestions.json)
- `cache/colors.json` — `{filename: [dominant colour names]}`, built once by an HSV scan (`_extract_dominant_colors`, 25×25 downscale). Powers the colour filter.
- `cache/skip_list.json` — filenames that failed to open during a rebuild, so they aren't retried (e.g. corrupt `4784.jpg`).

### Things that will bite you

- **`model.eval()` is mandatory** — MobileCLIP has batchnorm; skipping eval mode silently corrupts embeddings.
- **Index ↔ filename alignment**: row `i` of `embeddings` ↔ `filenames[i]`. `build_index.py` (full rebuild) and `/rebuild` (incremental, appends new files only) both keep them aligned and re-save the npz.
- **Search is pure relevance order** (`run_search`) — the old MMR/diversity, expand, and re-rank features were removed per user request. Dedup is kept.
- **Score range**: MobileCLIP text-image cosine sims run ~0.2–0.31 for good matches, not near 1.0. The UI normalizes the bar width to the top hit's score.
- **Dedup**: near-identical images (cosine > `DEDUP_THRESHOLD` 0.999) are collapsed into one result with a `+N identical` badge rather than shown separately.
- **No FAISS** — search is plain numpy matmul over the (N×512) matrix; fast enough at this scale. (faiss-cpu was deliberately not installed.)
- **Colour filter** is applied by zeroing scores of images whose `colors.json` entry lacks the chosen colour, before ranking.

### Helper scripts (one-off, kept for reference)

`copy_images.py` (flatten nested folders → one dir, collision-safe), `rename_folder.py` (rename a folder's images to 1..N). These produced the current `images_repo/`.
