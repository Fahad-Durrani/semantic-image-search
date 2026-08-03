# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local semantic image-search app over ~7,810 images. Images are embedded once with **Apple's official MobileCLIP** (the `mobileclip` package from github.com/apple/ml-mobileclip), using two switchable checkpoint variants — **S2** (`mobileclip_s2.pt`, the original/default) and **S0** (`mobileclip_s0.pt`, added later). A Flask server then encodes text queries — or an uploaded/selected image — with whichever model is selected and ranks images by cosine similarity. The frontend (search, autocomplete, similar-image, drag-and-drop upload, lightbox, model dropdown) is a single HTML string embedded in `app.py`.

> Note: an earlier version used open-clip-torch's `MobileCLIP2-S2` (`dfndr2b`) weights from HuggingFace. That was abandoned because it applies a different image normalization than Apple's. Do **not** reintroduce open_clip for model loading — use the `mobileclip` package so the normalization matches.

## Commands

```powershell
# One-time setup
pip install -e ./ml-mobileclip --no-deps   # Apple package; deps (torch/open_clip/timm) already satisfied
pip install flask Pillow numpy tqdm
# checkpoints (same CDN, filename varies by variant):
#   checkpoints/mobileclip_s2.pt  <- https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt
#   checkpoints/mobileclip_s0.pt  <- https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt

# Build the embedding index for each variant you want available (~10 min on CPU each).
python build_index.py --model s2   # -> cache/embeddings.npz
python build_index.py --model s0   # -> cache/embeddings_s0.npz

# Run the app
python app.py            # -> http://127.0.0.1:5000  (loads BOTH models at startup)

# Quick smoke test
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=s2"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=s0"
```

No tests/linter/build step. Re-run `build_index.py --model <s2|s0>` only when the contents of `images_repo/` change (or use the in-app **Rebuild Index** button, which re-embeds new images for both models together).

## Directory layout (paths are hardcoded in both scripts — keep in sync)

- `images_repo/` — the flat image library, files named `1.jpg … 7811.png`. Served by `GET /images/<filename>`.
- `checkpoints/mobileclip_s2.pt`, `checkpoints/mobileclip_s0.pt` — Apple checkpoints (S2 ~380 MB, S0 ~215 MB, not in git).
- `cache/embeddings.npz` (S2) / `cache/embeddings_s0.npz` (S0) — built indexes. **Each has two aligned arrays: `embeddings` (N×512 float32, L2-normalized) and `filenames` (bare names).** Note the key is `filenames`, not `paths`.
- `cache/suggestions.json` (S2) / `cache/suggestions_s0.json` (S0) — autocomplete counts, auto-rebuilt when `VOCAB` changes (keyed by an md5 hash of the vocab). Counts differ per model since they're derived from that model's embeddings.
- `ml-mobileclip/` — cloned Apple repo, installed editable. Must persist (editable install references it).
- `build_index.py`, `app.py` — the two entry points. Both define a `MODEL_CONFIGS = {"s2": {...}, "s0": {...}}` dict mapping a short key to `{model_name, checkpoint, cache_file}` (`app.py`'s entries also carry `label`/`suggestions_file`) — **keep these two dicts in sync** when adding another variant.

## Architecture

Two entry points share the same `mobileclip` loading path + L2-normalization, so cosine similarity is a plain dot product at query time. Both S2 and S0 use the same embedding dimensionality (512), so nothing downstream needs to special-case shape.

- **`build_index.py`** — offline indexer, takes `--model {s2,s0}` (default `s2`). Loads that variant from its checkpoint, batches images (`BATCH_SIZE=32`) through `encode_image`, L2-normalizes, writes that variant's `cache_file`. Per-image try/except skips corrupt files (e.g. `4784.jpg`), so `len(filenames)` can be one or two less than the file count on disk. Run it once per model you want available.
- **`app.py`** — Flask server. On startup loops over `MODEL_CONFIGS` and loads **both** models simultaneously into a `MODELS` dict (keyed `"s2"`/`"s0"`, each holding its own `embeddings`/`filenames`/`model`/`tokenizer`/`image_transform`/`device`), so requests can switch models with no restart. It also kicks off a background thread to build the **colour palette index** (shared across models — built from raw pixels, not embeddings) and builds a suggestion + prefix index **per model** (encodes the ~500-term `VOCAB` once against each model's embeddings). Endpoints: `/search` (text, supports `-negative` terms, `&color=`, `&model=s2|s0`), `/similar` (by indexed filename, `&model=`), `/search_by_image` (POST upload, `&model=`), `/suggest` (autocomplete, `&model=`), `/models` (list of `{key, label, default}` from `MODEL_CONFIGS` — what the frontend populates every model dropdown/panel from, so adding a model server-side needs no frontend change), `/colors` (available colour names, model-agnostic), `/rebuild` (POST, starts incremental re-index for **both** models together) + `/rebuild_stream` (SSE progress), `/images/<f>`, `/`.
- The single-page frontend (in `FRONTEND_HTML`) has two top-level views toggled by the **Compare Models** header button (`compareMode`, persisted): **single view** (`#singleView`, the original layout) with a **model dropdown** (populated from `/models`, persisted via `localStorage`), autocomplete, **recent searches** (localStorage), **colour filter bar**, lightbox, find-similar; and **compare view** (`#compareView`) with a shared query/Top K/colour-filter bar and a row of independent model panels (`comparePanels`, persisted) — each panel has its own model `<select>` and remove button, queries `/search` independently with the shared query but its own `model=`, and panel count is capped at `MODEL_LIST.length` so "+ Add model" naturally scales as more models are added to `MODEL_CONFIGS`. Card rendering (`buildCardsHTML`) and click wiring (`wireCardClicks` → lightbox / "Similar") are shared between both views. Also: **dark mode** toggle (CSS variables + `[data-theme="dark"]`, persisted) and a **Rebuild Index** modal driven by the SSE stream.

### Extra cache files (besides the per-model embeddings*.npz / suggestions*.json)
- `cache/colors.json` — `{filename: [dominant colour names]}`, built once by an HSV scan (`_extract_dominant_colors`, 25×25 downscale). Powers the colour filter. Shared across models.
- `cache/skip_list.json` — filenames that failed to open during a rebuild, so they aren't retried (e.g. corrupt `4784.jpg`). Shared across models.

### Things that will bite you

- **`model.eval()` is mandatory** — MobileCLIP has batchnorm; skipping eval mode silently corrupts embeddings. True for every variant.
- **Index ↔ filename alignment**: row `i` of a model's `embeddings` ↔ that model's `filenames[i]`. `build_index.py` (full rebuild) and `/rebuild` (incremental, appends new files only, run for both models together) both keep them aligned and re-save each model's npz. If you ever build one model's index separately from the other's, their `filenames` arrays can end up covering different file sets — the app doesn't detect or guard against this.
- **Search is pure relevance order** (`run_search`) — the old MMR/diversity, expand, and re-rank features were removed per user request. Dedup is kept. `run_search` now takes a `model_key` and reads that model's embeddings/filenames from the `MODELS` dict.
- **Score range**: MobileCLIP text-image cosine sims run ~0.2–0.31 for good matches for S2, not near 1.0; S0 (a smaller model) may have a somewhat different typical range. The UI normalizes the bar width to the top hit's score, so this isn't hardcoded anywhere in app logic — just worth knowing when eyeballing raw scores.
- **Dedup**: near-identical images (cosine > `DEDUP_THRESHOLD` 0.999) are collapsed into one result with a `+N identical` badge rather than shown separately.
- **No FAISS** — search is plain numpy matmul over the (N×512) matrix; fast enough at this scale. (faiss-cpu was deliberately not installed.)
- **Colour filter** is applied by zeroing scores of images whose `colors.json` entry lacks the chosen colour, before ranking. Unaffected by which model is selected.

### Helper scripts (one-off, kept for reference)

`copy_images.py` (flatten nested folders → one dir, collision-safe), `rename_folder.py` (rename a folder's images to 1..N). These produced the current `images_repo/`.
