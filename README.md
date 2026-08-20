# 🔎 Semantic Image Search

A fast, fully-local **semantic image search** engine over a personal library of
~7,800 images. Search your photos with natural language ("a lion in the wild",
"sunset over the ocean"), find visually similar images, or drop in a photo to
search *by image* — all running on your own machine, no cloud, no API keys.

Images are embedded once — six switchable models, all loaded at once:
**Apple's official [MobileCLIP](https://github.com/apple/ml-mobileclip)**
in four variants (**S2**, **S0**, **S0 (fp16)**, **S0 (int8)**), plus two
models from wkcn's unrelated **TinyCLIP** project included purely for
comparison — **[TinyCLIP-ViT-8M/16](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)**
and **TinyCLIP-ResNet-19M/Text-19M** (a CLIP-style ResNet-50 vision tower
instead of a transformer, with its own 1024-dimensional embedding space).
A Flask server then encodes your text query (or an uploaded image) with
whichever model is selected and ranks the library by cosine similarity; a
**Compare Models** view lets you run the same query against every model
side by side.

![Demo: top result for eight text queries](docs/demo.jpg)

> *Top-1 result for eight different text queries, straight from the running app.
> See **[docs/RESULTS.md](docs/RESULTS.md)** for the full test report with scores.*

---

## ✨ Features

- **Natural-language search** — rank images by meaning, not filenames or tags.
- **Negative terms** — `beach -people` down-weights unwanted concepts.
- **Search by image** — drag-and-drop or upload a photo to find similar ones.
- **Find similar** — one click from any result to its nearest neighbours.
- **Colour filter** — narrow results to images containing a dominant colour.
- **Autocomplete** — prefix suggestions from a ~500-term vocabulary.
- **Recent searches**, **dark mode**, and a **lightbox** viewer (all in the UI).
- **Near-duplicate dedup** — pixel-identical images collapse into one hit with a
  `+N identical` badge.
- **Rebuild index** from the UI (incremental, streamed progress) when you add images.
- **Compare Models view** — toggle to a side-by-side layout that runs one query
  against every configured model at once (independent panels, add/remove as
  models are added), instead of switching the single dropdown back and forth.
- **100% local & offline** after the one-time model + image setup. No FAISS, no GPU
  required (runs on CPU); search is a plain NumPy matrix multiply.

---

## 🖥️ The web interface

The whole UI is a single page served at `http://127.0.0.1:5000` — a search bar
with autocomplete, recent-search chips, a colour-filter bar, dark-mode toggle,
and a responsive results grid where every card shows its rank, filename, and a
normalised relevance bar. Click any image for a lightbox with a **Find Similar**
button; drop an image on the bar to search by image.

**Text search — `"Dog walking in park"`:**

![Web UI: searching "Dog walking in park"](docs/ui-search-dog.png)

**Mixed content — `"Chat screenshot"`** (note the `+1 identical` dedup badge on the
last card, where a near-duplicate was collapsed):

![Web UI: searching "Chat screenshot"](docs/ui-search-chats.png)

---

## 🧠 How it works

```
                 build_index.py --model {s2,s0,s0_fp16,s0_int8,tinyclip,tinyclip_resnet19m} (offline, once per model)
   images_repo/  ─────────────────────────────►  cache/embeddings.npz (s2)
   (7,800 imgs)      encode_image                    cache/embeddings_s0.npz (s0)
                     + L2-normalise                  cache/embeddings_s0_fp16.npz (s0_fp16)
                                                      cache/embeddings_s0_int8.npz (s0_int8)
                                                      cache/embeddings_tinyclip.npz (tinyclip)
                                                      cache/embeddings_tinyclip_resnet19m.npz
                                                      (N×D, L2-normalised; D=512 for every
                                                       variant except tinyclip_resnet19m, D=1024)

                 app.py (serving, loads ALL models at startup)
   "a red car"  ──► encode_text ──► D-d vec ──►  dot product vs. every row
   + &model=...     (selected model)               of the selected model's own
                                                    matrix ──► top-k ──► JSON + web UI
```

Because both images and text are embedded by the **same model** and
**L2-normalised**, cosine similarity is just a dot product. At ~7,800 images the
whole index is a small matrix, so a brute-force NumPy `matmul` is instant — no
vector database needed. Six model variants — **S2**, **S0**, **S0 (fp16)**,
**S0 (int8)**, **TinyCLIP-ViT-8M/16**, and **TinyCLIP-ResNet-19M/Text-19M** —
are each embedded into their own cache file and loaded simultaneously; a
dropdown in the UI switches which one answers a given request. S0 (fp16) is
the same S0 architecture and weights, just exported at half precision — a
smaller checkpoint (~108 MB vs. ~215 MB) to compare against the full-precision
S0. S0 (int8) goes further still: a *pair* of separately int8-quantized
image/text encoders (~13 MB + ~44 MB) — much lossier than fp16 (cosine
agreement with fp32 S0 drops to ~0.84 on images, ~0.70 on text). The two
TinyCLIP models are a different case entirely: not MobileCLIP variants at all,
but a separate, much smaller model family (wkcn's TinyCLIP project) — included
to compare against genuinely different architectures/training runs, not just
compressed versions of the same one. TinyCLIP-ViT-8M/16 (~23M params, loaded
via HuggingFace `transformers`) happens to share the 512-d embedding space, so
it drops in unchanged; TinyCLIP-ResNet-19M/Text-19M (~63M raw params, ~38M
excluding its token-embedding table, loaded via `open_clip` with a registered
custom config and a remapped checkpoint — see [CLAUDE.md](CLAUDE.md) for why
that remapping was necessary) uses a CLIP-style **ResNet-50** vision tower
instead of a transformer and outputs **1024-d** embeddings — the one variant
with a different embedding dimensionality, which the pipeline handles fine
since nothing hardcodes a fixed size anywhere.

**Score range.** MobileCLIP text→image cosine similarities are *not* confidences
near 1.0 — good matches sit around **0.24–0.31**. Ranking order is what matters.
The TinyCLIP models' cosine similarities run on their own different scales
(TinyCLIP-ViT-8M/16 higher across the board — even unrelated text/image pairs
can score ~0.35; TinyCLIP-ResNet-19M/Text-19M sits closer to MobileCLIP's range
but in its own 1024-d space), so none of these scores are directly comparable
across models — ranking order within a single model is what matters.
See [docs/RESULTS.md](docs/RESULTS.md).

---

## 📦 Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Developed on 3.12 (Windows 11). |
| **~2 GB RAM free** | For the model + index in memory. |
| **The image library** | Not in this repo (too large). Put your own images in `images_repo/`, named however you like. |
| **MobileCLIP checkpoints** | `mobileclip_s2.pt` (~380 MB) and, optionally, `mobileclip_s0.pt` (~215 MB) / `mobileclip_s0_fp16.pt` (~108 MB) / `mobileclip_s0_image_int8.pt` + `mobileclip_s0_text_int8.pt` (~13 MB + ~44 MB), downloaded/provided once each (see below). Not in this repo. |
| **TinyCLIP snapshot** (optional) | A local copy of the `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` HF hub repo (~90 MB), fetched once (see below). Not in this repo. |
| **TinyCLIP-ResNet-19M checkpoint** (optional) | `tinyclip_resnet19m_text19m_laion400m.pt` (~121 MB) + its `open_clip` config JSON, fetched once (see below). Not in this repo. |
| GPU | Optional. CPU works fine at this scale. |

> **What's *not* in this repo (by design):** the image library (`images_repo/`,
> multi-GB), the model checkpoint (`checkpoints/`, exceeds GitHub's 100 MB limit),
> and generated caches (`cache/`). These are listed in `.gitignore`. The steps
> below regenerate everything.

---

## 🚀 Getting started

### 1. Clone

```powershell
git clone https://github.com/Fahad-Durrani/semantic-image-search.git
cd semantic-image-search
```

To pull later updates:

```powershell
git pull
```

### 2. Get Apple's MobileCLIP package

The model is loaded via Apple's `mobileclip` package (this is important — an
earlier version used `open_clip`'s weights, which apply a *different* image
normalisation and were abandoned). Clone it next to the project and install it
editable:

```powershell
git clone https://github.com/apple/ml-mobileclip.git
pip install -e ./ml-mobileclip --no-deps
```

> `--no-deps` because torch / open_clip / timm are installed in the next step.
> The `ml-mobileclip/` folder must stay in place (editable install points at it).

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download the checkpoints

The app loads **all configured** MobileCLIP variants at startup (so the model
dropdown can switch instantly, with no restart) — download **`mobileclip_s2.pt`**
and **`mobileclip_s0.pt`** into a `checkpoints/` folder; both come from the same
Apple CDN:

```powershell
mkdir checkpoints
# S2 -- from: https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt
Invoke-WebRequest `
  -Uri "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt" `
  -OutFile "checkpoints/mobileclip_s2.pt"

# S0 -- from: https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt
Invoke-WebRequest `
  -Uri "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt" `
  -OutFile "checkpoints/mobileclip_s0.pt"
```

`mobileclip_s0_fp16.pt` (a half-precision export of the S0 architecture) isn't
on Apple's CDN — place it at `checkpoints/mobileclip_s0_fp16.pt` if you have a
copy. Its weights are already "reparameterized" (MobileOne branches folded), so
loading it takes a different code path than the other two checkpoints; see
`load_model()` in `build_index.py` / `_load_model()` in `app.py`.

Similarly, `mobileclip_s0_image_int8.pt` + `mobileclip_s0_text_int8.pt` (a pair
of int8-quantized TorchScript encoders for the S0 architecture) aren't on
Apple's CDN either — place both at `checkpoints/mobileclip_s0_image_int8.pt`
and `checkpoints/mobileclip_s0_text_int8.pt` if you have copies. These are
self-contained scripted modules, not state dicts, so they're loaded with
`torch.jit.load()` and run on CPU only (int8 kernels have no CUDA backend).

**TinyCLIP** (`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`) is a different model
family entirely, loaded via HuggingFace `transformers` rather than the
`mobileclip` package. Fetch a local snapshot once (requires `pip install
huggingface_hub`, already pulled in by `transformers`):

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M', local_dir='checkpoints/tinyclip-vit-8m-16-text-3m-yfcc15m')"
```

This downloads config/tokenizer/weight files into
`checkpoints/tinyclip-vit-8m-16-text-3m-yfcc15m/`; after that, `app.py` loads
it with `local_files_only=True` and never hits the network again.

**TinyCLIP-ResNet-19M/Text-19M** is a different case again — a raw `open_clip`-
format checkpoint (not on HuggingFace) plus a model config that has to be
registered at runtime. Fetch both once:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ResNet-19M-Text-19M-LAION400M.pt" `
  -OutFile "checkpoints/tinyclip_resnet19m_text19m_laion400m.pt"
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/wkcn/TinyCLIP/main/src/open_clip/model_configs/TinyCLIP-ResNet-19M-Text-19M.json" `
  -OutFile "checkpoints/TinyCLIP-ResNet-19M-Text-19M.json"
```

Only wkcn's "Manual" (uniformly-scaled) TinyCLIP checkpoints load this way —
their "auto inheritance" variants use an irregular per-layer-pruned
architecture that plain `open_clip` can't represent at all. See
[CLAUDE.md](CLAUDE.md) if you're tempted to add another TinyCLIP release.

> `app.py` will refuse to start if any configured checkpoint or its embedding
> cache is missing (see step 6). To make a variant truly optional, remove its
> entry from the `MODEL_CONFIGS` dict in both `app.py` and `build_index.py`.

### 5. Add your images

Put your image files in `images_repo/` (a flat folder). Supported extensions:
`.jpg .jpeg .png .webp .avif .jfif`. Any filenames work.

### 6. Build the embedding index (one time per model, ~10-15 min on CPU each)

```powershell
python build_index.py --model s2
python build_index.py --model s0
python build_index.py --model s0_fp16
python build_index.py --model s0_int8
python build_index.py --model tinyclip
python build_index.py --model tinyclip_resnet19m
```

This writes `cache/embeddings.npz` (s2) / `cache/embeddings_s0.npz` (s0) /
`cache/embeddings_s0_fp16.npz` (s0_fp16) / `cache/embeddings_s0_int8.npz`
(s0_int8) / `cache/embeddings_tinyclip.npz` (tinyclip) /
`cache/embeddings_tinyclip_resnet19m.npz` (tinyclip_resnet19m) and shows a
`tqdm` progress bar for each. Corrupt files are skipped automatically. Re-run
only when the contents of `images_repo/` change (or use the in-app **Rebuild
Index** button, which incrementally re-indexes new images for all models).

### 7. Run the app

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. All models load at startup;
use the **Model** dropdown next to Top K to switch between S2, S0, S0 (fp16),
S0 (int8), TinyCLIP-ViT-8M/16, and TinyCLIP-ResNet-19M/Text-19M.

Quick smoke test from the terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=s2"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=s0"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=s0_fp16"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=s0_int8"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=tinyclip"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=lion&k=3&model=tinyclip_resnet19m"
```

---

## 🌐 API reference

The server exposes a small JSON/HTTP API (default host `127.0.0.1:5000`):

| Method & path | Purpose | Key params |
|---------------|---------|------------|
| `GET /` | The single-page web UI | — |
| `GET /search` | Text search | `q` (query, supports `-negative` terms), `k` (1–50, default 10), `color`, `model` (`s2`\|`s0`\|`s0_fp16`\|`s0_int8`\|`tinyclip`\|`tinyclip_resnet19m`, default `s2`) |
| `GET /similar` | Nearest neighbours of an indexed image | `img` (filename), `k`, `model` |
| `POST /search_by_image` | Search by an uploaded image | multipart `image` file, `k`, `model` |
| `GET /suggest` | Autocomplete | `q` (prefix ≥2 chars), `limit`, `model` |
| `GET /models` | List available models (for the dropdown / compare panels) | — |
| `GET /colors` | List available colour-filter names | — (model-agnostic) |
| `POST /rebuild` | Start an incremental re-index (all models) | — |
| `GET /rebuild_stream` | Server-sent-events progress for a rebuild | — |
| `GET /images/<filename>` | Serve an image from `images_repo/` | — |

**Example — text search with a negative term, S0 model:**

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search?q=beach -people&k=5&model=s0"
```

Response shape:

```json
{
  "query": "lion", "positive": "lion", "negatives": [], "k": 3, "model": "s2",
  "results": [
    { "rank": 1, "filename": "4582.jpg", "score": 0.2951,
      "url": "/images/4582.jpg", "duplicate_count": 0 }
  ]
}
```

---

## 🗂️ Project layout

```
semantic-image-search/
├── app.py              # Flask server + embedded single-page frontend; loads all models
├── build_index.py      # Offline indexer → cache/embeddings*.npz (--model s2|s0|s0_fp16|s0_int8|tinyclip|tinyclip_resnet19m)
├── copy_images.py      # Helper: flatten nested folders into one dir
├── rename_folder.py    # Helper: rename a folder's images to 1..N
├── requirements.txt
├── CLAUDE.md           # Detailed architecture notes / gotchas
├── docs/
│   ├── demo.jpg        # README hero montage
│   ├── RESULTS.md      # Model test report (real outputs)
│   └── examples/       # Thumbnails of the test results
│
│   # ---- generated / provided locally, not in git ----
├── images_repo/        # Your image library
├── checkpoints/        # mobileclip_s2.pt, mobileclip_s0.pt, mobileclip_s0_fp16.pt,
│                       # mobileclip_s0_image_int8.pt, mobileclip_s0_text_int8.pt,
│                       # tinyclip-vit-8m-16-text-3m-yfcc15m/ (HF snapshot dir),
│                       # tinyclip_resnet19m_text19m_laion400m.pt + TinyCLIP-ResNet-19M-Text-19M.json
├── cache/              # embeddings.npz + suggestions.json (s2), embeddings_s0.npz + suggestions_s0.json (s0),
│                       # embeddings_s0_fp16.npz + suggestions_s0_fp16.json (s0_fp16),
│                       # embeddings_s0_int8.npz + suggestions_s0_int8.json (s0_int8),
│                       # embeddings_tinyclip.npz + suggestions_tinyclip.json (tinyclip),
│                       # embeddings_tinyclip_resnet19m.npz + suggestions_tinyclip_resnet19m.json (tinyclip_resnet19m),
│                       # colors.json, skip_list.json (shared)
└── ml-mobileclip/      # Apple's package (cloned, installed editable)
```

Paths are hard-coded in `app.py` and `build_index.py` — both define a
`MODEL_CONFIGS` dict mapping `"s2"`/`"s0"`/`"s0_fp16"`/`"s0_int8"`/`"tinyclip"`/
`"tinyclip_resnet19m"` to their checkpoint/cache paths; keep these two dicts in
sync if you move things or add another variant. See **[CLAUDE.md](CLAUDE.md)**
for architecture details and the "things that will bite you" list (e.g.
`model.eval()` is mandatory — MobileCLIP has batchnorm).

---

## 🧪 Test results

A set of eight diverse text queries was run against the live index; the actual
top-4 results (with cosine scores) are documented in **[docs/RESULTS.md](docs/RESULTS.md)**.
Summary of top-1 scores:

| Query | Top-1 score |
|-------|:-----------:|
| a lion in the wild | 0.298 |
| city skyline at night | 0.308 |
| red sports car | 0.287 |
| a cup of coffee on a table | 0.283 |
| snow covered mountains | 0.288 |
| a dog playing outdoors | 0.272 |
| sunset over the ocean | 0.257 |
| a plate of healthy food | 0.257 |

---

## 🛠️ Troubleshooting

- **`Checkpoint not found for MobileCLIP-S2/S0/S0 (fp16)/S0 (int8)/TinyCLIP-*`** — you skipped step 4 for that variant.
- **`No module named 'mobileclip'`** — run step 2 (`pip install -e ./ml-mobileclip --no-deps`).
- **`No module named 'transformers'`** — only needed for the `tinyclip` variant; run step 3 (`pip install -r requirements.txt`) or remove its `MODEL_CONFIGS` entry.
- **`Cache not found for MobileCLIP-S2/S0/S0 (fp16)/S0 (int8)/TinyCLIP-*`** — run `python build_index.py --model s2` / `s0` / `s0_fp16` / `s0_int8` / `tinyclip` / `tinyclip_resnet19m` (step 6).
- **TinyCLIP-ViT-8M/16 autocomplete counts look too high** — expected; its baseline cosine similarities run higher than MobileCLIP's, so the shared `SUGGEST_THRESHOLD` is miscalibrated for it. Ranking within a search is still meaningful.
- **`RuntimeError`/missing-key errors loading a different TinyCLIP release** — only wkcn's "Manual" (uniformly-scaled) checkpoints load via the `open_clip_config` path; "auto inheritance" releases use an irregular per-layer-pruned architecture that plain `open_clip` can't build at all. See CLAUDE.md before adding another one.
- **Weird / low-quality results** — make sure you're using Apple's `mobileclip`
  package, *not* `open_clip`; they normalise images differently.
- **Port 5000 in use** — change the port in the last line of `app.py`
  (`app.run(host="127.0.0.1", port=5000)`).

---

## 📄 Notes & credits

- Models: **[Apple MobileCLIP](https://github.com/apple/ml-mobileclip)** (MobileCLIP-S2, MobileCLIP-S0, a half-precision export of S0, and an int8-quantized export of S0) and **[wkcn/TinyCLIP](https://github.com/wkcn/TinyCLIP)** (TinyCLIP-ViT-8M-16-Text-3M-YFCC15M via HuggingFace `transformers`, and TinyCLIP-ResNet-19M-Text-19M-LAION400M via `open_clip`). Checkpoints © their respective authors, under their respective licences.
- This project is a personal / educational local search tool. The image library and
  the model checkpoint are **not** distributed here.
