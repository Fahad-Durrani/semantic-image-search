# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
import threading
import queue
import colorsys
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from PIL import Image
from collections import Counter

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(BASE_DIR, "images_repo")
COLORS_FILE = os.path.join(BASE_DIR, "cache", "colors.json")
SKIP_FILE   = os.path.join(BASE_DIR, "cache", "skip_list.json")

# Keep in sync with build_index.py's MODEL_CONFIGS.
MODEL_CONFIGS = {
    "s2": {
        "model_name": "mobileclip_s2",
        "label":      "MobileCLIP-S2",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s2.pt"),
        "cache_file": os.path.join(BASE_DIR, "cache", "embeddings.npz"),
        "suggestions_file": os.path.join(BASE_DIR, "cache", "suggestions.json"),
    },
    "s0": {
        "model_name": "mobileclip_s0",
        "label":      "MobileCLIP-S0",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0.pt"),
        "cache_file": os.path.join(BASE_DIR, "cache", "embeddings_s0.npz"),
        "suggestions_file": os.path.join(BASE_DIR, "cache", "suggestions_s0.json"),
    },
    "s0_fp16": {
        "model_name": "mobileclip_s0",
        "label":      "MobileCLIP-S0 (fp16)",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0_fp16.pt"),
        "cache_file": os.path.join(BASE_DIR, "cache", "embeddings_s0_fp16.npz"),
        "suggestions_file": os.path.join(BASE_DIR, "cache", "suggestions_s0_fp16.json"),
        # This checkpoint stores already-reparameterized (folded) MobileOne
        # branches in fp16, not the raw multi-branch state dict create_model_
        # and_transforms()'s default load path expects -- reparameterize the
        # freshly-built model first so its keys match, then load into it.
        "reparam_checkpoint": True,
    },
    "s0_int8": {
        "model_name": "mobileclip_s0",
        "label":      "MobileCLIP-S0 (int8)",
        # Two independently int8-quantized TorchScript modules (image/text),
        # not a single state-dict checkpoint like the other variants.
        "checkpoint": {
            "image": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0_image_int8.pt"),
            "text":  os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0_text_int8.pt"),
        },
        "cache_file": os.path.join(BASE_DIR, "cache", "embeddings_s0_int8.npz"),
        "suggestions_file": os.path.join(BASE_DIR, "cache", "suggestions_s0_int8.json"),
        "quantized": True,
    },
    "tinyclip": {
        "model_name": "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "label":      "TinyCLIP-ViT-8M/16",
        # A local snapshot of the HF hub repo (config.json, tokenizer.json,
        # model.safetensors, preprocessor_config.json), not a mobileclip
        # checkpoint file -- loaded via transformers, not the mobileclip package.
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "tinyclip-vit-8m-16-text-3m-yfcc15m"),
        "cache_file": os.path.join(BASE_DIR, "cache", "embeddings_tinyclip.npz"),
        "suggestions_file": os.path.join(BASE_DIR, "cache", "suggestions_tinyclip.json"),
        "hf_clip": True,
    },
    "tinyclip_resnet19m": {
        # Registered with open_clip.add_model_config() at load time using this
        # name (must match the config JSON's filename stem).
        "model_name": "TinyCLIP-ResNet-19M-Text-19M",
        "label":      "TinyCLIP-ResNet-19M/Text-19M",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "tinyclip_resnet19m_text19m_laion400m.pt"),
        "open_clip_config": os.path.join(BASE_DIR, "checkpoints", "TinyCLIP-ResNet-19M-Text-19M.json"),
        "cache_file": os.path.join(BASE_DIR, "cache", "embeddings_tinyclip_resnet19m.npz"),
        "suggestions_file": os.path.join(BASE_DIR, "cache", "suggestions_tinyclip_resnet19m.json"),
        # 1024-d embeddings, unlike every other variant (512-d). Nothing in
        # this file assumes a fixed dimension, so this is safe -- just
        # documented since it's the one place that differs.
        "embed_dim": 1024,
    },
}
DEFAULT_MODEL_KEY = "s2"

DEFAULT_K         = 10
DEDUP_THRESHOLD   = 0.999
SUGGEST_THRESHOLD = 0.20
NEGATIVE_WEIGHT   = 0.5
SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".jfif"}

COLOR_HEX = {
    "red":    "#e53935", "orange": "#fb8c00", "yellow": "#fdd835",
    "green":  "#43a047", "cyan":   "#00acc1", "blue":   "#1e88e5",
    "purple": "#8e24aa", "pink":   "#e91e63", "brown":  "#6d4c41",
    "white":  "#f5f5f5", "gray":   "#78909c", "black":  "#212121",
}

VOCAB = list(dict.fromkeys([
    # --- Domestic pets ---
    "dog", "dogs", "puppy", "puppies", "pup",
    "cat", "cats", "kitten", "kittens",
    "rabbit", "rabbits", "bunny", "hamster", "gerbil",
    "parrot", "budgie", "cockatiel",
    "goldfish", "turtle", "gecko", "iguana",

    # --- Farm animals ---
    "horse", "horses", "pony", "foal", "colt", "mare", "stallion",
    "cow", "cows", "bull", "calf", "cattle",
    "sheep", "lamb", "goat", "goats",
    "pig", "piglet", "pigs",
    "chicken", "rooster", "hen", "chick",
    "turkey", "duck", "ducks", "goose", "geese",
    "donkey", "mule", "llama", "alpaca",

    # --- Big cats ---
    "tiger", "tigers", "lion", "lions", "lioness", "cub",
    "leopard", "cheetah", "panther", "jaguar", "lynx", "bobcat", "cougar", "puma",

    # --- Wild canines ---
    "wolf", "wolves", "fox", "foxes", "coyote", "jackal", "dingo",

    # --- Bears ---
    "bear", "bears", "panda", "grizzly", "polar",

    # --- Deer family ---
    "deer", "fawn", "elk", "moose", "reindeer", "caribou",

    # --- African & savanna wildlife ---
    "elephant", "giraffe", "zebra", "rhinoceros", "hippo",
    "antelope", "gazelle", "impala", "wildebeest", "bison", "buffalo",
    "hyena", "meerkat", "warthog",

    # --- Primates ---
    "gorilla", "chimpanzee", "orangutan", "baboon", "monkey", "lemur", "gibbon",

    # --- Small & misc mammals ---
    "raccoon", "squirrel", "chipmunk", "otter", "badger",
    "hedgehog", "opossum", "skunk", "mongoose",
    "kangaroo", "koala", "wallaby",
    "seal", "walrus", "dolphin", "whale", "orca",
    "bat", "mole", "weasel",

    # --- Raptors ---
    "eagle", "hawk", "falcon", "osprey", "vulture", "kite",

    # --- Owls ---
    "owl",

    # --- Wading & water birds ---
    "heron", "crane", "stork", "flamingo", "pelican", "spoonbill",

    # --- Exotic & tropical birds ---
    "toucan", "macaw", "cockatoo", "peacock", "kingfisher",
    "hummingbird", "woodpecker", "hornbill",

    # --- Common birds ---
    "bird", "birds", "sparrow", "robin", "pigeon", "dove", "swift",
    "swan", "puffin", "penguin", "albatross",
    "pheasant", "quail", "jay",

    # --- Reptiles ---
    "snake", "cobra", "python",
    "lizard", "chameleon",
    "crocodile", "alligator",
    "frog", "toad", "salamander",

    # --- Insects ---
    "butterfly", "bee", "dragonfly", "ladybug", "moth", "beetle",

    # --- Marine life ---
    "shark", "jellyfish", "octopus", "crab", "lobster", "seahorse", "ray",

    # --- Dog breeds ---
    "husky", "labrador", "shepherd", "rottweiler", "bulldog",
    "poodle", "beagle", "dalmatian", "boxer", "terrier",
    "collie", "pug", "chihuahua", "dachshund",
    "greyhound", "samoyed", "corgi", "doberman",
    "spaniel", "retriever", "setter", "pointer",
    "vizsla", "weimaraner", "schnauzer", "mastiff",
    "maltese", "bichon", "malamute", "akita",
    "basenji", "ridgeback", "whippet", "newfoundland",

    # --- Cat breeds ---
    "siamese", "persian", "ragdoll", "bengal",
    "sphynx", "abyssinian", "burmese",

    # --- Horse breeds ---
    "thoroughbred", "arabian", "appaloosa", "friesian", "mustang", "clydesdale",

    # --- People ---
    "woman", "women", "girl", "girls",
    "man", "men", "boy", "boys",
    "child", "children", "baby", "toddler", "teenager",
    "couple", "family", "group", "crowd",
    "person", "people",
    "portrait", "selfie", "face", "profile",
    "model", "athlete", "soldier", "rider", "jockey",

    # --- Human actions ---
    "running", "jogging", "sprinting",
    "walking", "hiking", "strolling",
    "jumping", "leaping", "diving",
    "swimming", "surfing", "kayaking",
    "cycling", "riding", "galloping", "trotting",
    "climbing", "rappelling",
    "playing", "dancing", "singing",
    "sitting", "standing", "lying", "crouching",
    "sleeping", "resting",
    "smiling", "laughing", "crying",
    "eating", "drinking", "cooking",
    "reading", "working", "fishing", "hunting",
    "skiing", "snowboarding", "skating",
    "stretching", "meditating", "training",
    "herding", "grazing", "prowling", "stalking",

    # --- Expressions / mood ---
    "happy", "playful", "fierce", "calm", "curious", "alert",
    "aggressive", "relaxed", "proud", "majestic",

    # --- Landscapes ---
    "forest", "jungle", "rainforest", "woodland", "grove",
    "meadow", "field", "prairie", "savanna", "steppe", "pasture",
    "mountain", "mountains", "peak", "summit", "cliff", "ridge",
    "hill", "hills", "valley", "canyon", "gorge",
    "desert", "dunes",
    "tundra", "glacier", "iceberg",
    "lake", "pond", "river", "stream", "creek",
    "ocean", "sea", "bay", "coast", "shoreline",
    "beach", "island",
    "waterfall", "rapids", "marsh", "swamp",
    "cave", "rock", "cliff",

    # --- Sky & weather ---
    "sky", "clouds", "cloudy",
    "sunset", "sunrise", "twilight", "dusk", "dawn",
    "night", "stars", "moon",
    "fog", "mist", "haze",
    "rain", "storm", "lightning",
    "snow", "snowfall", "blizzard",
    "rainbow", "sunshine", "overcast",

    # --- Flora ---
    "flower", "flowers", "wildflower", "blossom", "bloom",
    "rose", "tulip", "sunflower", "daisy", "lily", "orchid", "lavender",
    "tree", "trees", "oak", "pine", "palm", "birch", "willow", "bamboo",
    "leaf", "leaves", "foliage",
    "grass", "moss", "fern", "cactus",
    "garden", "bush", "mushroom",

    # --- Urban & architecture ---
    "city", "skyline", "skyscraper", "building",
    "street", "road", "bridge", "alley",
    "park", "fountain", "statue",
    "house", "cabin", "cottage", "barn", "farmhouse",
    "church", "castle", "ruins", "lighthouse",
    "market", "village",

    # --- Sports ---
    "soccer", "football", "basketball", "tennis", "golf",
    "volleyball", "baseball", "rugby", "cricket",
    "boxing", "wrestling", "gymnastics", "archery",
    "equestrian", "marathon",

    # --- Visual descriptors ---
    "wild", "domestic", "free",
    "young", "adult", "elderly", "juvenile",
    "tiny", "large", "giant",
    "fluffy", "furry", "feathered", "scaled",
    "spotted", "striped", "patchy",
    "white", "black", "brown", "golden", "grey", "orange",
    "colorful", "dark", "bright",
    "beautiful", "elegant", "rare", "exotic", "endangered",
    "cute", "adorable",
    "closeup", "macro", "silhouette", "aerial", "underwater",
    "wildlife", "nature", "outdoor", "indoor",
    "winter", "summer", "spring", "autumn",
    "tropical", "arctic", "alpine",
    "misty", "snowy", "foggy", "rainy", "sunny",
    "nocturnal", "camouflage",
]))

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load once at startup
# ---------------------------------------------------------------------------
import re
import mobileclip
from mobileclip.modules.common.mobileone import reparameterize_model

MODELS = {}   # model_key -> {embeddings, filenames, n_images, model, tokenizer, image_transform, device}


def _remap_tinyclip_open_clip_key(key):
    """TinyCLIP's raw training checkpoints wrap real open_clip parameter names
    under a training-only prefix (observed forms: "image_encoder_without_ddp.",
    "_image_encoder.module.", "text_encoder_without_ddp.", "_text_encoder.module.",
    "_logit_scale.module.(module.)?logit_scale") that vanilla open_clip's CLIP
    class doesn't expect. Strip it so the remaining key matches the model's
    real state dict (e.g. "_image_encoder.module.visual.conv1.weight" ->
    "visual.conv1.weight"). Verified exact match (no missing/unexpected keys)
    against TinyCLIP-ResNet-19M-Text-19M's checkpoint."""
    if "logit_scale" in key:
        return "logit_scale"
    return re.sub(r"^_?(image_encoder|text_encoder)(_without_ddp)?(\.module)*\.", "", key)


class _QuantizedDualEncoder:
    """Wraps separately-quantized TorchScript image/text encoders (each a
    self-contained scripted module callable as `model(tensor)`) behind the
    same .encode_image()/.encode_text() interface used for the regular
    mobileclip CLIP model."""

    def __init__(self, image_model, text_model):
        self.image_model = image_model
        self.text_model  = text_model

    def eval(self):
        self.image_model.eval()
        self.text_model.eval()
        return self

    def encode_image(self, tensor):
        return self.image_model(tensor)

    def encode_text(self, tokens):
        return self.text_model(tokens)


class _HFClipEncoder:
    """Wraps a HuggingFace `transformers` CLIPModel -- a different loading path
    entirely from Apple's mobileclip package -- behind the same
    .encode_image()/.encode_text() interface used for the mobileclip models."""

    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    @staticmethod
    def _pooled(output):
        # Some transformers versions return a plain tensor from get_image_features/
        # get_text_features, others wrap it in an output object with .pooler_output.
        return output.pooler_output if hasattr(output, "pooler_output") else output

    def encode_image(self, pixel_values):
        return self._pooled(self.model.get_image_features(pixel_values=pixel_values))

    def encode_text(self, tokens):
        return self._pooled(self.model.get_text_features(**tokens))


class _HFImageTransform:
    """Per-image preprocessing callable matching mobileclip's `preprocess(img)`
    signature, backed by a HF CLIPImageProcessor."""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, img):
        return self.image_processor(images=img, return_tensors="pt")["pixel_values"][0]


class _HFTextTokenizer:
    """Batch-of-strings tokenizer callable matching mobileclip's tokenizer
    signature (returns a `.to(device)`-able object), backed by a HF tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        return self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)


def _load_model(model_name, checkpoint, device, reparam_checkpoint=False, quantized=False,
                 hf_clip=False, open_clip_config=None):
    if open_clip_config:
        import open_clip
        open_clip.add_model_config(open_clip_config)
        model, _, image_transform = open_clip.create_model_and_transforms(
            model_name, pretrained=None, device=device)
        raw = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        remapped = {_remap_tinyclip_open_clip_key(k): v for k, v in state_dict.items()}
        model.load_state_dict(remapped)
        tokenizer = open_clip.get_tokenizer(model_name)
    elif hf_clip:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(checkpoint, local_files_only=True).to(device)
        processor = CLIPProcessor.from_pretrained(checkpoint, local_files_only=True)
        model = _HFClipEncoder(clip_model)
        image_transform = _HFImageTransform(processor.image_processor)
        tokenizer = _HFTextTokenizer(processor.tokenizer)
    elif quantized:
        # Quantized int8 ops only run on CPU (fbgemm/qnnpack backends), regardless
        # of CUDA availability -- the caller is expected to pass device="cpu".
        image_model = torch.jit.load(checkpoint["image"], map_location=device)
        text_model  = torch.jit.load(checkpoint["text"], map_location=device)
        model = _QuantizedDualEncoder(image_model, text_model)
        # Preprocessing transform is architecture-derived, not weight-derived,
        # so a throwaway float32 model is enough to fetch it.
        _, _, image_transform = mobileclip.create_model_and_transforms(
            model_name, pretrained=None, reparameterize=False, device=device)
        tokenizer = mobileclip.get_tokenizer(model_name)
    elif reparam_checkpoint:
        model, _, image_transform = mobileclip.create_model_and_transforms(
            model_name, pretrained=None, reparameterize=False, device=device)
        model = reparameterize_model(model)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        tokenizer = mobileclip.get_tokenizer(model_name)
    else:
        model, _, image_transform = mobileclip.create_model_and_transforms(
            model_name, pretrained=checkpoint, device=device)
        tokenizer = mobileclip.get_tokenizer(model_name)
    model.eval()
    return model, image_transform, tokenizer


for _key, _cfg in MODEL_CONFIGS.items():
    if not os.path.isfile(_cfg["cache_file"]):
        sys.exit(f"Cache not found for {_cfg['label']}: {_cfg['cache_file']} "
                  f"-- run `python build_index.py --model {_key}` first.")
    if _cfg.get("hf_clip"):
        _missing_checkpoints = [] if os.path.isdir(_cfg["checkpoint"]) else [_cfg["checkpoint"]]
    else:
        _checkpoint_paths = list(_cfg["checkpoint"].values()) if isinstance(_cfg["checkpoint"], dict) else [_cfg["checkpoint"]]
        if _cfg.get("open_clip_config"):
            _checkpoint_paths.append(_cfg["open_clip_config"])
        _missing_checkpoints = [p for p in _checkpoint_paths if not os.path.isfile(p)]
    if _missing_checkpoints:
        sys.exit(f"Checkpoint not found for {_cfg['label']}: {_missing_checkpoints[0]}")

    print(f"Loading embedding cache for {_cfg['label']}...", flush=True)
    _data       = np.load(_cfg["cache_file"])
    _embeddings = _data["embeddings"]   # (N, D) float32, L2-normalised
    _filenames  = _data["filenames"]    # (N,)
    print(f"  {len(_filenames)} images indexed.", flush=True)

    print(f"Loading {_cfg['label']}...", flush=True)
    # Quantized int8 ops only run on CPU, regardless of CUDA availability.
    _device = "cpu" if _cfg.get("quantized") else ("cuda" if torch.cuda.is_available() else "cpu")
    _model, _image_transform, _tokenizer = _load_model(
        _cfg["model_name"], _cfg["checkpoint"], _device,
        reparam_checkpoint=_cfg.get("reparam_checkpoint", False),
        quantized=_cfg.get("quantized", False),
        hf_clip=_cfg.get("hf_clip", False),
        open_clip_config=_cfg.get("open_clip_config"))
    print(f"  Model ready on {_device}.", flush=True)

    MODELS[_key] = {
        "embeddings":      _embeddings,
        "filenames":       _filenames,
        "n_images":        len(_filenames),
        "model":           _model,
        "tokenizer":       _tokenizer,
        "image_transform": _image_transform,
        "device":          _device,
    }

N_IMAGES = MODELS[DEFAULT_MODEL_KEY]["n_images"]

# ---------------------------------------------------------------------------
# Color palette index (built once in background, cached to disk)
# ---------------------------------------------------------------------------
COLOR_INDEX = {}
SKIP_SET    = set()   # filenames that failed to encode; persisted to cache/skip_list.json


def _load_skip_set():
    global SKIP_SET
    if os.path.isfile(SKIP_FILE):
        try:
            with open(SKIP_FILE, encoding="utf-8") as f:
                SKIP_SET = set(json.load(f))
        except Exception:
            pass


_load_skip_set()


def _rgb_to_color_name(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if v < 0.12:
        return "black"
    if v > 0.88 and s < 0.12:
        return "white"
    if s < 0.12:
        return "gray"
    h360 = h * 360
    if h360 < 15 or h360 >= 345:
        return "red"
    if h360 < 45:
        return "orange"
    if h360 < 65:
        return "yellow"
    if h360 < 155:
        return "green"
    if h360 < 195:
        return "cyan"
    if h360 < 255:
        return "blue"
    if h360 < 285:
        return "purple"
    return "pink"


def _extract_dominant_colors(img_path, n=3):
    try:
        img = Image.open(img_path).convert("RGB").resize((25, 25), Image.LANCZOS)
        counts = Counter(_rgb_to_color_name(r, g, b) for r, g, b in img.getdata())
        return [c for c, _ in counts.most_common(n)]
    except Exception:
        return []


def _build_color_index_bg():
    global COLOR_INDEX
    if os.path.isfile(COLORS_FILE):
        try:
            with open(COLORS_FILE, encoding="utf-8") as f:
                COLOR_INDEX = json.load(f)
            print(f"  Color index loaded ({len(COLOR_INDEX)} images).", flush=True)
            return
        except Exception:
            pass
    print(f"Building color index for {N_IMAGES} images (background)...", flush=True)
    idx = {}
    for i, fname in enumerate(MODELS[DEFAULT_MODEL_KEY]["filenames"]):
        path = os.path.join(IMAGES_DIR, str(fname))
        idx[str(fname)] = _extract_dominant_colors(path)
        if (i + 1) % 1000 == 0:
            print(f"  color index: {i + 1}/{N_IMAGES}", flush=True)
    try:
        with open(COLORS_FILE, "w", encoding="utf-8") as f:
            json.dump(idx, f)
    except Exception:
        pass
    COLOR_INDEX = idx
    print(f"  Color index ready ({len(idx)} images).", flush=True)


threading.Thread(target=_build_color_index_bg, daemon=True).start()

# ---------------------------------------------------------------------------
# Suggestion / autocomplete index
# ---------------------------------------------------------------------------
_VOCAB_HASH = hashlib.md5("|".join(sorted(VOCAB)).encode()).hexdigest()[:12]


def build_suggestion_index(model_key):
    m                = MODELS[model_key]
    suggestions_file = MODEL_CONFIGS[model_key]["suggestions_file"]

    if os.path.isfile(suggestions_file):
        with open(suggestions_file, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("_vocab_hash") == _VOCAB_HASH:
            cached.pop("_vocab_hash")
            return cached
        print(f"Vocab changed -- rebuilding suggestion index for {model_key}...", flush=True)
    else:
        print(f"Building suggestion index for {model_key} ({len(VOCAB)} terms)...", flush=True)

    result     = {}
    batch_size = 64
    for i in range(0, len(VOCAB), batch_size):
        batch  = VOCAB[i : i + batch_size]
        tokens = m["tokenizer"](batch).to(m["device"])
        with torch.no_grad():
            feats = m["model"].encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs = feats.cpu().numpy().astype("float32")
        sims = m["embeddings"] @ vecs.T
        for j, term in enumerate(batch):
            result[term] = int((sims[:, j] > SUGGEST_THRESHOLD).sum())

    payload = dict(result)
    payload["_vocab_hash"] = _VOCAB_HASH
    with open(suggestions_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"  Done -- {len(result)} terms indexed.", flush=True)
    return result


def build_prefix_index(suggestions):
    index = {}
    for term, count in sorted(suggestions.items(), key=lambda x: -x[1]):
        if count == 0:
            continue
        for length in range(2, len(term) + 1):
            prefix = term[:length]
            bucket = index.setdefault(prefix, [])
            if len(bucket) < 20:
                bucket.append({"term": term, "count": count})
    return index


for _key in MODEL_CONFIGS:
    MODELS[_key]["suggestions"]  = build_suggestion_index(_key)
    MODELS[_key]["prefix_index"] = build_prefix_index(MODELS[_key]["suggestions"])
print(f"Server ready -- http://127.0.0.1:5000\n", flush=True)

# ---------------------------------------------------------------------------
# Index rebuild (background thread + SSE queue)
# Incrementally encodes any images in images_repo that aren't indexed yet.
# ---------------------------------------------------------------------------
_rebuild_queue   = queue.Queue()
_rebuild_lock    = threading.Lock()
_rebuild_running = False


def _do_rebuild():
    global N_IMAGES, _rebuild_running, COLOR_INDEX, SKIP_SET

    _rebuild_queue.put({"type": "start"})
    all_disk_files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )

    ref_filenames = MODELS[DEFAULT_MODEL_KEY]["filenames"]
    indexed_set   = set(ref_filenames.tolist())
    new_files     = [f for f in all_disk_files
                     if f not in indexed_set and f not in SKIP_SET]

    if not new_files:
        _rebuild_queue.put({"type": "progress", "done": 0, "total": 0,
                            "msg": "Index already up to date -- nothing to do."})
        _rebuild_queue.put({"type": "done", "total": N_IMAGES})
        with _rebuild_lock:
            _rebuild_running = False
        return

    total     = len(new_files)
    n_models  = len(MODEL_CONFIGS)
    _rebuild_queue.put({"type": "progress", "done": 0, "total": total,
                        "msg": f"Found {total} new image(s) to index across {n_models} model(s)"})

    failed_fnames = set()
    for model_key, m in MODELS.items():
        new_embs, new_fnames = [], []
        batch_size = 16
        for i in range(0, total, batch_size):
            batch_files = new_files[i : i + batch_size]
            imgs, valid = [], []
            for fname in batch_files:
                try:
                    img = Image.open(os.path.join(IMAGES_DIR, fname)).convert("RGB")
                    imgs.append(m["image_transform"](img))
                    valid.append(fname)
                except Exception:
                    failed_fnames.add(fname)
            if imgs:
                tensor = torch.stack(imgs).to(m["device"])
                with torch.no_grad():
                    feats = m["model"].encode_image(tensor)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                new_embs.append(feats.cpu().numpy().astype("float32"))
                new_fnames.extend(valid)
            _rebuild_queue.put({"type": "progress",
                                "done": min(i + batch_size, total), "total": total,
                                "msg": f"[{MODEL_CONFIGS[model_key]['label']}] Encoded {len(new_fnames)} image(s)"})

        if new_embs:
            added_embs   = np.vstack(new_embs)
            added_fnames = np.array(new_fnames)
            m["embeddings"] = np.vstack([m["embeddings"], added_embs])
            m["filenames"]  = np.append(m["filenames"], added_fnames)
            m["n_images"]   = len(m["filenames"])
            np.savez(MODEL_CONFIGS[model_key]["cache_file"],
                     embeddings=m["embeddings"], filenames=m["filenames"])

    N_IMAGES = MODELS[DEFAULT_MODEL_KEY]["n_images"]

    # Persist files that couldn't be opened (by any model) so they're not retried next rebuild
    if failed_fnames:
        SKIP_SET.update(failed_fnames)
        try:
            with open(SKIP_FILE, "w", encoding="utf-8") as f:
                json.dump(sorted(SKIP_SET), f)
        except Exception:
            pass

    added_fnames_final = [f for f in new_files if f not in failed_fnames]
    if added_fnames_final:
        _rebuild_queue.put({"type": "progress", "done": total, "total": total,
                            "msg": "Extracting colors for new images..."})
        for fname in added_fnames_final:
            COLOR_INDEX[fname] = _extract_dominant_colors(os.path.join(IMAGES_DIR, fname))
        try:
            with open(COLORS_FILE, "w", encoding="utf-8") as f:
                json.dump(COLOR_INDEX, f)
        except Exception:
            pass

    _rebuild_queue.put({"type": "done", "total": N_IMAGES})
    with _rebuild_lock:
        _rebuild_running = False


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def encode_text(text, model_key):
    m      = MODELS[model_key]
    tokens = m["tokenizer"]([text]).to(m["device"])
    with torch.no_grad():
        feat = m["model"].encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")[0]


def encode_image_pil(pil_img, model_key):
    m      = MODELS[model_key]
    tensor = m["image_transform"](pil_img).unsqueeze(0).to(m["device"])
    with torch.no_grad():
        feat = m["model"].encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")[0]


def parse_query(query):
    """Split 'tiger -cage -cub' into ('tiger', ['cage', 'cub'])."""
    words     = query.split()
    positives = [w for w in words if not w.startswith("-")]
    negatives = [w.lstrip("-") for w in words if w.startswith("-") and len(w) > 1]
    return " ".join(positives).strip(), negatives


def _apply_color_filter(scores, filenames, color):
    if not color or not COLOR_INDEX:
        return scores
    scores = scores.copy()
    mask   = np.array([color not in COLOR_INDEX.get(str(f), []) for f in filenames])
    scores[mask] = -1.0
    return scores


def run_search(vec, k, model_key, exclude_idx=None, color_filter=None):
    """Relevance-ranked search. Pixel-identical duplicates (cosine > DEDUP_THRESHOLD)
    are collapsed into the result they duplicate and reported as duplicate_count."""
    m          = MODELS[model_key]
    embeddings = m["embeddings"]
    filenames  = m["filenames"]
    n_images   = m["n_images"]

    scores = embeddings @ vec
    if exclude_idx is not None:
        scores = scores.copy()
        scores[exclude_idx] = -1.0
    scores = _apply_color_filter(scores, filenames, color_filter)

    oversample = min(k * 10, n_images)
    if oversample >= n_images:
        candidates = np.argsort(scores)[::-1].tolist()
    else:
        top_pool   = np.argpartition(scores, -oversample)[-oversample:]
        candidates = top_pool[np.argsort(scores[top_pool])[::-1]].tolist()
    candidates = [c for c in candidates if scores[c] > -0.5]

    selected, dup_counts = [], []
    remaining = list(candidates)
    while len(selected) < k and remaining:
        best_idx = remaining.pop(0)
        best_emb = embeddings[best_idx]
        dup_count = 0
        if remaining:
            dup_sims  = embeddings[np.array(remaining, dtype=np.int64)] @ best_emb
            dup_mask  = dup_sims > DEDUP_THRESHOLD
            dup_count = int(dup_mask.sum())
            remaining = [i for i, d in zip(remaining, dup_mask) if not d]
        selected.append(best_idx)
        dup_counts.append(dup_count)

    return [
        {
            "rank":            i + 1,
            "filename":        str(filenames[idx]),
            "score":           round(float(scores[idx]), 4),
            "url":             f"/images/{filenames[idx]}",
            "duplicate_count": dup_counts[i],
        }
        for i, idx in enumerate(selected)
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _resolve_model_key():
    key = request.args.get("model", DEFAULT_MODEL_KEY).strip().lower()
    if key not in MODEL_CONFIGS:
        return None
    return key


@app.route("/")
def index():
    return (FRONTEND_HTML
             .replace("{{N_IMAGES}}", f"{N_IMAGES:,}")
             .replace("{{N_IMAGES_RAW}}", str(N_IMAGES)))


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/search")
def search():
    model_key = _resolve_model_key()
    if model_key is None:
        return jsonify({"error": "Unknown model"}), 400

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        k = min(int(request.args.get("k", DEFAULT_K)), MODELS[model_key]["n_images"])
    except ValueError:
        k = DEFAULT_K

    color_filter = request.args.get("color", "").strip().lower() or None

    positive, negatives = parse_query(query)
    if not positive:
        return jsonify({"error": "Query has no positive terms"}), 400

    vec = encode_text(positive, model_key)
    if negatives:
        neg_vecs = np.stack([encode_text(t, model_key) for t in negatives])
        vec      = vec - NEGATIVE_WEIGHT * neg_vecs.mean(axis=0)
        norm     = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm

    results = run_search(vec, k, model_key, color_filter=color_filter)
    return jsonify({"query": query, "positive": positive,
                    "negatives": negatives, "k": k, "model": model_key, "results": results})


@app.route("/similar")
def similar():
    model_key = _resolve_model_key()
    if model_key is None:
        return jsonify({"error": "Unknown model"}), 400

    filename = request.args.get("img", "").strip()
    if not filename:
        return jsonify({"error": "No image specified"}), 400

    filenames = MODELS[model_key]["filenames"]
    matches   = np.where(filenames == filename)[0]
    if len(matches) == 0:
        return jsonify({"error": f"Image not found: {filename}"}), 404

    query_idx = int(matches[0])
    try:
        k = min(int(request.args.get("k", DEFAULT_K)), MODELS[model_key]["n_images"])
    except ValueError:
        k = DEFAULT_K

    vec     = MODELS[model_key]["embeddings"][query_idx].copy()
    results = run_search(vec, k, model_key, exclude_idx=query_idx)
    return jsonify({"query_image": filename, "k": k, "model": model_key, "results": results})


@app.route("/search_by_image", methods=["POST"])
def search_by_image():
    model_key = _resolve_model_key()
    if model_key is None:
        return jsonify({"error": "Unknown model"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    f = request.files["image"]
    try:
        pil_img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    try:
        k = min(int(request.args.get("k", DEFAULT_K)), MODELS[model_key]["n_images"])
    except ValueError:
        k = DEFAULT_K

    vec     = encode_image_pil(pil_img, model_key)
    results = run_search(vec, k, model_key)
    return jsonify({"query_image": f.filename or "uploaded image", "k": k, "model": model_key, "results": results})


@app.route("/suggest")
def suggest():
    model_key = _resolve_model_key()
    if model_key is None:
        return jsonify([])

    prefix = request.args.get("q", "").strip().lower()
    if len(prefix) < 2:
        return jsonify([])
    limit = min(int(request.args.get("limit", "8")), 20)
    return jsonify(MODELS[model_key]["prefix_index"].get(prefix, [])[:limit])


@app.route("/models")
def models():
    return jsonify([
        {"key": key, "label": cfg["label"], "default": key == DEFAULT_MODEL_KEY}
        for key, cfg in MODEL_CONFIGS.items()
    ])


@app.route("/colors")
def colors():
    available = sorted({c for cols in COLOR_INDEX.values() for c in cols})
    return jsonify(available)


@app.route("/rebuild", methods=["POST"])
def rebuild():
    global _rebuild_running
    with _rebuild_lock:
        if _rebuild_running:
            return jsonify({"error": "Rebuild already in progress"}), 409
        _rebuild_running = True
    threading.Thread(target=_do_rebuild, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/rebuild_stream")
def rebuild_stream():
    def generate():
        while True:
            try:
                item = _rebuild_queue.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MobileCLIP Image Search</title>
<style>
:root {
  --bg: #f0f2f5; --bg2: #fff; --bg3: #f5f5f5;
  --hdr-bg: #1a1a2e; --hdr-text: #fff;
  --text: #1a1a2e; --text2: #666; --text3: #999;
  --border: #e0e0e0; --input-border: #ccc;
  --accent: #1a73e8; --accent-h: #1558c0; --accent-lite: #f0f6ff;
  --neg-bg: #fce8e6; --neg-text: #c62828;
  --dup-bg: #fff3e0; --dup-text: #e65100;
  --card-img-bg: #eee;
  --sh-sm: 0 1px 5px rgba(0,0,0,.10);
  --sh-md: 0 4px 14px rgba(0,0,0,.15);
}
[data-theme="dark"] {
  --bg: #111827; --bg2: #1f2937; --bg3: #374151;
  --hdr-bg: #030712; --hdr-text: #f9fafb;
  --text: #f9fafb; --text2: #9ca3af; --text3: #6b7280;
  --border: #374151; --input-border: #4b5563;
  --accent: #60a5fa; --accent-h: #3b82f6; --accent-lite: #1e3a5f;
  --neg-bg: #450a0a; --neg-text: #fca5a5;
  --dup-bg: #431407; --dup-text: #fdba74;
  --card-img-bg: #374151;
  --sh-sm: 0 1px 5px rgba(0,0,0,.35);
  --sh-md: 0 4px 14px rgba(0,0,0,.50);
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); min-height: 100vh;
       transition: background .2s, color .2s; }

header { background: var(--hdr-bg); color: var(--hdr-text); padding: 16px 32px;
         display: flex; align-items: center; gap: 12px;
         box-shadow: 0 2px 8px rgba(0,0,0,.25); }
header h1 { font-size: 1.18rem; font-weight: 700; }
header .sub { font-size: .74rem; opacity: .5; margin-top: 2px; }
#compareToggleBtn, #rebuildBtn {
  background: rgba(255,255,255,.12); color: var(--hdr-text);
  border: 1px solid rgba(255,255,255,.25); border-radius: 7px;
  padding: 7px 14px; font-size: .82rem; cursor: pointer;
  transition: background .15s; white-space: nowrap; flex-shrink: 0;
}
#compareToggleBtn { margin-left: auto; }
#compareToggleBtn:hover, #rebuildBtn:hover { background: rgba(255,255,255,.24); }
#compareToggleBtn.active { background: var(--accent); border-color: var(--accent); }
#themeToggle {
  background: rgba(255,255,255,.12); color: var(--hdr-text);
  border: 1px solid rgba(255,255,255,.25); border-radius: 7px;
  padding: 7px 11px; font-size: .9rem; cursor: pointer; line-height: 1;
  transition: background .15s; white-space: nowrap; flex-shrink: 0;
}
#themeToggle:hover { background: rgba(255,255,255,.24); }

.search-bar {
  background: var(--bg2); border-bottom: 1px solid var(--border);
  padding: 12px 32px; display: flex; gap: 10px; align-items: center;
  position: sticky; top: 0; z-index: 20;
  box-shadow: 0 1px 4px rgba(0,0,0,.08); transition: box-shadow .2s, background .2s;
}
.search-bar.drag-over { box-shadow: 0 0 0 3px var(--accent) inset; background: var(--accent-lite); }
.autocomplete-wrap { flex: 1; position: relative; }
.search-bar input[type=text] {
  width: 100%; padding: 10px 16px; border: 1.5px solid var(--input-border);
  border-radius: 8px; font-size: 1rem; outline: none; transition: border-color .15s;
  background: var(--bg2); color: var(--text);
}
.search-bar input[type=text]:focus { border-color: var(--accent); }
.ac-dropdown {
  display: none; position: absolute; top: calc(100% + 5px); left: 0; right: 0;
  background: var(--bg2); border: 1.5px solid var(--border); border-radius: 8px;
  box-shadow: var(--sh-md); z-index: 50; overflow: hidden;
}
.ac-dropdown.open { display: block; }
.ac-item { display: flex; align-items: center; justify-content: space-between;
           padding: 9px 14px; cursor: pointer; transition: background .1s; }
.ac-item:hover, .ac-item.active { background: var(--accent-lite); }
.ac-term { font-size: .93rem; color: var(--text); font-weight: 500; }
.ac-term em { font-style: normal; color: var(--accent); }
.ac-count { font-size: .76rem; color: var(--text3); white-space: nowrap; margin-left: 12px; }

#uploadBtn {
  padding: 10px 12px; background: var(--bg3); color: var(--text2);
  border: 1.5px solid var(--input-border); border-radius: 8px; font-size: 1rem;
  cursor: pointer; transition: background .15s; white-space: nowrap; flex-shrink: 0;
}
#uploadBtn:hover { background: var(--accent-lite); border-color: var(--accent); color: var(--accent); }
#algoBtn {
  padding: 10px 14px; background: var(--bg3); color: var(--text2);
  border: 1.5px solid var(--input-border); border-radius: 8px; font-size: .85rem; font-weight: 600;
  cursor: pointer; transition: background .15s; white-space: nowrap; flex-shrink: 0;
}
#algoBtn:hover { background: var(--accent-lite); border-color: var(--accent); color: var(--accent); }
#algoBtn.active { background: var(--accent); border-color: var(--accent); color: #fff; }

/* algo threshold panel */
.algo-panel {
  background: var(--bg2); border-bottom: 1px solid var(--border);
  padding: 10px 32px; display: flex; align-items: flex-end; gap: 20px; flex-wrap: wrap;
}
.algo-field { display: flex; flex-direction: column; gap: 4px; }
.algo-field label { font-size: .72rem; color: var(--text2); font-weight: 600; }
.algo-field input {
  padding: 7px 9px; border: 1.5px solid var(--input-border); border-radius: 7px;
  font-size: .88rem; width: 100px; background: var(--bg2); color: var(--text); outline: none;
}
.algo-field input:focus { border-color: var(--accent); }
.algo-divider { width: 1px; align-self: stretch; background: var(--border); margin: 0 4px; }
.algo-legend { display: flex; gap: 16px; align-items: center; margin-left: auto; flex-wrap: wrap; }
.algo-legend-item { display: flex; align-items: center; gap: 6px; font-size: .78rem; color: var(--text2); white-space: nowrap; }
.algo-legend-dot, .card-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.dot-top { background: var(--accent); }
.dot-probable { background: #f59e0b; }
.dot-cutoff { background: var(--text3); }
.card-dot {
  position: absolute; top: 8px; left: 8px; width: 10px; height: 10px;
  border: 2px solid #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.15); z-index: 1;
}

/* tiered results */
.tier-section { padding: 8px 0; }
.tier-section + .tier-section { border-top: 1px solid var(--border); margin-top: 8px; padding-top: 16px; }
.tier-head { display: flex; align-items: center; gap: 8px; padding: 0 0 6px; }
.tier-head.clickable { cursor: pointer; }
.tier-title { font-size: .92rem; font-weight: 700; color: var(--text); }
.tier-count { color: var(--text3); font-size: .8rem; }
.tier-sub { font-size: .78rem; color: var(--text2); font-style: italic; padding: 0 0 8px; }
.tier-chevron { margin-left: auto; color: var(--text3); font-size: .8rem; transition: transform .15s; }
.tier-section.collapsed .tier-chevron { transform: rotate(-90deg); }
.tier-section.collapsed .tier-sub, .tier-section.collapsed .tier-grid { display: none; }

.no-match-banner { text-align: center; padding: 40px 32px 24px; border-bottom: 1px solid var(--border); }
.nm-icon { font-size: 2.2rem; line-height: 1; margin-bottom: 10px; opacity: .7; }
.nm-icon-empty { opacity: .45; }
.nm-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.nm-sub { font-size: .84rem; color: var(--text2); max-width: 420px; margin: 0 auto; }
.ctrl-wrap { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.ctrl-wrap label { font-size: .82rem; color: var(--text2); }
.ctrl-wrap input[type=number] {
  width: 58px; padding: 10px 6px; border: 1.5px solid var(--input-border);
  border-radius: 8px; font-size: .95rem; text-align: center; outline: none;
  background: var(--bg2); color: var(--text);
}
.ctrl-wrap input[type=number]:focus { border-color: var(--accent); }
.ctrl-wrap select {
  padding: 10px 8px; border: 1.5px solid var(--input-border);
  border-radius: 8px; font-size: .85rem; outline: none;
  background: var(--bg2); color: var(--text); cursor: pointer;
}
.ctrl-wrap select:focus { border-color: var(--accent); }
#searchBtn, #compareSearchBtn {
  padding: 10px 24px; background: var(--accent); color: #fff;
  border: none; border-radius: 8px; font-size: .95rem; font-weight: 600;
  cursor: pointer; transition: background .15s; white-space: nowrap;
}
#searchBtn:hover, #compareSearchBtn:hover { background: var(--accent-h); }
#searchBtn:disabled, #compareSearchBtn:disabled { opacity: .55; cursor: default; }

/* recent searches bar */
.history-bar {
  background: var(--bg2); border-bottom: 1px solid var(--border);
  padding: 7px 32px; display: flex; align-items: center; gap: 8px;
  font-size: .78rem; overflow-x: auto;
}
.hist-label { color: var(--text3); flex-shrink: 0; font-weight: 600; }
.hist-pills { display: flex; gap: 6px; flex: 1; overflow-x: auto; }
.hist-pills::-webkit-scrollbar { display: none; }
.hist-pill {
  background: var(--bg3); color: var(--text2); border-radius: 20px;
  padding: 3px 11px; cursor: pointer; white-space: nowrap; flex-shrink: 0;
  border: 1px solid var(--border); font-size: .74rem; transition: background .12s;
}
.hist-pill:hover { background: var(--accent-lite); color: var(--accent); border-color: var(--accent); }
.hist-clear { background: none; border: none; color: var(--text3); cursor: pointer;
              font-size: .74rem; flex-shrink: 0; padding: 2px 6px; }
.hist-clear:hover { color: var(--neg-text); }

/* color filter bar */
.color-bar {
  background: var(--bg2); border-bottom: 1px solid var(--border);
  padding: 7px 32px; display: flex; align-items: center; gap: 8px;
  font-size: .78rem; flex-wrap: wrap;
}
.color-label { color: var(--text3); flex-shrink: 0; font-weight: 600; }
.color-chip {
  border-radius: 20px; padding: 3px 11px; cursor: pointer; font-size: .73rem;
  font-weight: 600; color: #fff; flex-shrink: 0; user-select: none;
  border: 2px solid transparent; transition: transform .1s, border-color .1s;
  text-shadow: 0 1px 2px rgba(0,0,0,.4);
}
.color-chip:hover { transform: scale(1.08); }
.color-chip.active { border-color: var(--text); box-shadow: 0 0 0 2px var(--text); }
.color-chip-white { color: #333 !important; text-shadow: none !important; }
#colorClear { background: none; border: none; color: var(--text3); cursor: pointer;
              font-size: .74rem; padding: 2px 6px; }
#colorClear:hover { color: var(--neg-text); }

#status { padding: 8px 32px; font-size: .82rem; color: var(--text2);
          min-height: 34px; display: flex; align-items: center; gap: 8px; }
.spinner { width: 14px; height: 14px; border: 2px solid var(--border);
           border-top-color: var(--accent); border-radius: 50%;
           animation: spin .7s linear infinite; display: none; flex-shrink: 0; }
@keyframes spin { to { transform: rotate(360deg); } }

#results {
  padding: 16px 32px 48px;
}
.tier-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  align-items: start; gap: 16px;
}
.card {
  background: var(--bg2); border-radius: 10px; box-shadow: var(--sh-sm);
  cursor: pointer; transition: transform .12s, box-shadow .12s;
  position: relative; overflow: hidden;
}
.card:hover { transform: translateY(-3px); box-shadow: var(--sh-md); }
.card-img { width: 100%; height: auto; display: block; background: var(--card-img-bg); }
.card-body { padding: 10px 12px; }
.card-rank { font-size: .67rem; font-weight: 700; color: var(--text3);
             text-transform: uppercase; letter-spacing: .06em; margin-bottom: 3px; }
.card-fname { font-family: 'Courier New', monospace; font-size: .7rem;
              color: var(--text2); word-break: break-all; margin-bottom: 8px; }
.score-row { display: flex; align-items: center; gap: 7px; }
.score-bg { flex: 1; height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 3px; background: var(--accent); }
.score-val { font-size: .78rem; font-weight: 700; color: var(--accent); white-space: nowrap; }
.dup-badge { margin-top: 7px; padding: 3px 7px; border-radius: 4px;
             background: var(--dup-bg); color: var(--dup-text); font-size: .68rem; font-weight: 600; }
.similar-btn {
  position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,.55); color: #fff;
  border: none; border-radius: 5px; padding: 4px 10px; font-size: .7rem;
  font-weight: 600; cursor: pointer; opacity: 0; transition: opacity .15s; pointer-events: none;
}
.card:hover .similar-btn { opacity: 1; pointer-events: auto; }

#lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.86);
            z-index: 100; align-items: center; justify-content: center; flex-direction: column; gap: 12px; }
#lightbox.open { display: flex; }
#lightbox img { max-width: 90vw; max-height: 78vh; object-fit: contain;
                border-radius: 6px; box-shadow: 0 8px 32px rgba(0,0,0,.5); }
.lb-caption { color: #fff; font-size: .84rem; opacity: .7; text-align: center; }
.lb-actions { display: flex; gap: 10px; }
.lb-btn { padding: 8px 20px; border-radius: 7px; border: none; cursor: pointer;
          font-size: .85rem; font-weight: 600; }
.lb-similar-btn { background: var(--accent); color: #fff; }
.lb-similar-btn:hover { background: var(--accent-h); }
.lb-close-btn { background: rgba(255,255,255,.15); color: #fff; }
.lb-close-btn:hover { background: rgba(255,255,255,.25); }
.lb-esc { position: absolute; top: 18px; right: 24px; color: #fff;
          font-size: 2rem; cursor: pointer; opacity: .55; line-height: 1; }
.lb-esc:hover { opacity: 1; }

/* rebuild modal */
.rebuild-modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.75);
                 z-index: 200; align-items: center; justify-content: center; }
.rebuild-modal.open { display: flex; }
.rebuild-inner { background: var(--bg2); border-radius: 12px; padding: 28px 32px;
                 min-width: 340px; max-width: 480px; box-shadow: var(--sh-md); }
.rebuild-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 16px; color: var(--text); }
.rebuild-progress-bar { height: 8px; background: var(--bg3); border-radius: 4px;
                        overflow: hidden; margin-bottom: 10px; }
.rebuild-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width .3s; width: 0%; }
.rebuild-msg { font-size: .82rem; color: var(--text2); margin-bottom: 16px; }
.rebuild-close { padding: 7px 18px; background: var(--accent); color: #fff; border: none;
                 border-radius: 7px; font-size: .85rem; font-weight: 600; cursor: pointer; display: none; }

.empty { grid-column: 1/-1; text-align: center; padding: 60px 0; color: var(--text3); font-size: 1rem; }
.neg-tag { display: inline-block; background: var(--neg-bg); color: var(--neg-text);
           border-radius: 4px; padding: 1px 6px; font-size: .75rem; font-weight: 600; margin-left: 4px; }

/* ---- Shared "card" shell for both single view and compare view ----
   No overflow:hidden here -- singleView's search bar relies on position:sticky,
   which stops working under any ancestor with overflow != visible. Rounded
   corners are applied directly to the first/last child instead. */
.view-shell {
  width: calc(100% - 48px); max-width: 1400px; margin: 24px auto 40px;
  border: 1px solid var(--border); border-radius: 16px;
  background: var(--bg2); box-shadow: var(--sh-md);
}
.view-shell > *:first-child { border-top-left-radius: 15px; border-top-right-radius: 15px; }
.view-shell > *:last-child  { border-bottom-left-radius: 15px; border-bottom-right-radius: 15px; }

/* ---- Compare view ---- */
.cmp-header { padding: 20px 24px 4px; }
.cmp-header h2 { font-size: 1.15rem; font-weight: 700; color: var(--text); }
.cmp-sub { font-size: .82rem; color: var(--text2); margin-top: 2px; }
#compareSearchBar { position: static; box-shadow: none; padding-left: 24px; padding-right: 24px; }
#compareColorBar { padding-left: 24px; padding-right: 24px; }
#comparePanels {
  display: flex; flex-wrap: wrap; gap: 16px; padding: 16px 24px; align-items: flex-start;
}
.cmp-panel {
  flex: 1 1 320px; min-width: 280px; background: var(--bg2); border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden; box-shadow: var(--sh-sm);
}
.cmp-panel-head {
  display: flex; align-items: center; gap: 8px; padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}
.cmp-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.cmp-model-select {
  flex: 1; padding: 7px 8px; border: 1.5px solid var(--input-border); border-radius: 7px;
  font-size: .88rem; font-weight: 600; background: var(--bg2); color: var(--text); cursor: pointer;
}
.cmp-remove-btn {
  background: none; border: none; color: var(--text3); font-size: 1.1rem; cursor: pointer;
  line-height: 1; padding: 2px 4px; border-radius: 5px;
}
.cmp-remove-btn:hover { background: var(--bg3); color: var(--text); }
.cmp-remove-btn:disabled { opacity: .3; cursor: not-allowed; }
.cmp-status { padding: 8px 12px 0; font-size: .78rem; color: var(--text2); }
.cmp-results {
  padding: 10px 12px 16px; display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px;
}
.cmp-results .card-body { padding: 7px 8px; }
.cmp-results .card-fname { font-size: .63rem; }
.cmp-results .score-val { font-size: .7rem; }
.cmp-empty {
  grid-column: 1/-1; text-align: center; padding: 40px 0; color: var(--text3); font-size: .88rem;
}
.cmp-add-row { padding: 0 24px 24px; }
#cmpAddBtn {
  padding: 9px 18px; background: var(--bg2); color: var(--text); border: 1.5px dashed var(--input-border);
  border-radius: 8px; font-size: .85rem; font-weight: 600; cursor: pointer;
}
#cmpAddBtn:hover { border-color: var(--accent); color: var(--accent); }
#cmpAddBtn:disabled { opacity: .4; cursor: not-allowed; }
</style>
</head>
<body>

<header>
  <div>
    <h1>MobileCLIP Image Search</h1>
    <div class="sub"><span id="imgCount">{{N_IMAGES}}</span> images indexed</div>
  </div>
  <button id="compareToggleBtn" title="Compare models side by side">&#8646; Compare Models</button>
  <button id="rebuildBtn" title="Index any newly added images">&#8635; Rebuild Index</button>
  <button id="themeToggle" title="Toggle dark mode">&#127769;</button>
</header>

<div id="singleView" class="view-shell">

<div class="search-bar" id="searchBar">
  <div class="autocomplete-wrap">
    <input type="text" id="queryInput"
           placeholder='Try "eagle -cage", "two horses", "waterfall"&hellip;'
           autocomplete="off" autofocus>
    <div class="ac-dropdown" id="acDropdown"></div>
  </div>
  <button id="uploadBtn" title="Search by image (or drag an image onto this bar)">&#128247;</button>
  <input type="file" id="fileInput" accept="image/*" style="display:none">
  <div class="ctrl-wrap">
    <label for="modelSelect">Model</label>
    <select id="modelSelect"></select>
  </div>
  <button id="algoBtn" title="Configure result tiers">&#9881; Algo</button>
  <div class="ctrl-wrap">
    <label for="kInput">Top&nbsp;K</label>
    <input type="number" id="kInput" value="10" min="1" max="{{N_IMAGES_RAW}}">
  </div>
  <button id="searchBtn" onclick="doSearch()">Search</button>
</div>

<div class="algo-panel" id="algoPanel" style="display:none">
  <div class="algo-field">
    <label for="topThresholdInput">Top threshold</label>
    <input type="number" id="topThresholdInput" step="0.01" min="0" max="1" value="0.24">
  </div>
  <div class="algo-field">
    <label for="midThresholdInput">Mid threshold</label>
    <input type="number" id="midThresholdInput" step="0.01" min="0" max="1" value="0.20">
  </div>
  <div class="algo-field">
    <label for="cutoffInput">Cutoff</label>
    <input type="number" id="cutoffInput" step="0.01" min="0" max="1" value="0.10">
  </div>
  <div class="algo-divider"></div>
  <div class="algo-field">
    <label for="t1Input">T1 &mdash; blue bar</label>
    <input type="number" id="t1Input" step="0.01" min="0" max="1" value="0.24">
  </div>
  <div class="algo-field">
    <label for="t2Input">T2 &mdash; yellow bar</label>
    <input type="number" id="t2Input" step="0.01" min="0" max="1" value="0.20">
  </div>
  <div class="algo-field">
    <label for="t3Input">T3 &mdash; red bar</label>
    <input type="number" id="t3Input" step="0.01" min="0" max="1" value="0.10">
  </div>
  <div class="algo-legend">
    <span class="algo-legend-item"><span class="algo-legend-dot dot-top"></span>Top match</span>
    <span class="algo-legend-item"><span class="algo-legend-dot dot-probable"></span>Partial</span>
    <span class="algo-legend-item"><span class="algo-legend-dot dot-cutoff"></span>Below cutoff</span>
  </div>
</div>

<div class="history-bar" id="historyBar" style="display:none">
  <span class="hist-label">Recent:</span>
  <div class="hist-pills" id="historyPills"></div>
  <button class="hist-clear" onclick="clearHistory()">Clear</button>
</div>

<div class="color-bar" id="colorBar" style="display:none">
  <span class="color-label">Filter by color:</span>
  <div id="colorPills"></div>
  <button id="colorClear" onclick="clearColorFilter()" style="display:none">&times; Clear</button>
</div>

<div id="status">
  <div class="spinner" id="spinner"></div>
  <span id="statusText">Enter a query above and press Search or Enter.</span>
</div>

<div id="results">
  <div class="empty">Your results will appear here.</div>
</div>

</div>

<div id="compareView" class="view-shell" style="display:none">
  <div class="cmp-header">
    <h2>Image search &mdash; model comparison</h2>
    <div class="cmp-sub">Query all selected models simultaneously and compare results side by side</div>
  </div>

  <div class="search-bar" id="compareSearchBar">
    <div class="autocomplete-wrap">
      <input type="text" id="compareQueryInput"
             placeholder='Try "eagle -cage", "two horses", "waterfall"&hellip;'
             autocomplete="off">
    </div>
    <div class="ctrl-wrap">
      <label for="compareKInput">Top&nbsp;K</label>
      <input type="number" id="compareKInput" value="10" min="1" max="{{N_IMAGES_RAW}}">
    </div>
    <button id="compareSearchBtn" onclick="doCompareSearch()">Search</button>
  </div>

  <div class="color-bar" id="compareColorBar" style="display:none">
    <span class="color-label">Filter by color:</span>
    <div id="compareColorPills"></div>
    <button id="compareColorClear" onclick="clearCompareColorFilter()" style="display:none">&times; Clear</button>
  </div>

  <div id="comparePanels"></div>
  <div class="cmp-add-row">
    <button id="cmpAddBtn" onclick="addComparePanel()">+ Add model</button>
  </div>
</div>

<div id="lightbox">
  <span class="lb-esc" id="lbEsc">&times;</span>
  <img id="lbImg" src="" alt="">
  <div class="lb-caption" id="lbCaption"></div>
  <div class="lb-actions">
    <button class="lb-btn lb-similar-btn" id="lbSimilarBtn">Find Similar</button>
    <button class="lb-btn lb-close-btn" id="lbCloseBtn">Close</button>
  </div>
</div>

<div class="rebuild-modal" id="rebuildModal">
  <div class="rebuild-inner">
    <div class="rebuild-title">Rebuilding Image Index</div>
    <div class="rebuild-progress-bar"><div class="rebuild-fill" id="rebuildFill"></div></div>
    <div class="rebuild-msg" id="rebuildMsg">Starting&hellip;</div>
    <button class="rebuild-close" id="rebuildClose" onclick="closeRebuild()">Done</button>
  </div>
</div>

<script>
const queryInput   = document.getElementById('queryInput');
const kInput       = document.getElementById('kInput');
const modelSelect  = document.getElementById('modelSelect');
const searchBtn  = document.getElementById('searchBtn');
const uploadBtn  = document.getElementById('uploadBtn');
const fileInput  = document.getElementById('fileInput');
const spinner    = document.getElementById('spinner');
const statusText = document.getElementById('statusText');
const resultsEl  = document.getElementById('results');
const lightbox   = document.getElementById('lightbox');
const lbImg      = document.getElementById('lbImg');
const lbCaption  = document.getElementById('lbCaption');
const searchBar  = document.getElementById('searchBar');
const algoBtn           = document.getElementById('algoBtn');
const algoPanel         = document.getElementById('algoPanel');
const topThresholdInput = document.getElementById('topThresholdInput');
const midThresholdInput = document.getElementById('midThresholdInput');
const cutoffInput       = document.getElementById('cutoffInput');
const t1Input           = document.getElementById('t1Input');
const t2Input           = document.getElementById('t2Input');
const t3Input           = document.getElementById('t3Input');

let currentLbFilename = '';
let activeColor       = '';

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---------------------------------------------------------------------------
// Model selector (persisted across reloads). MODEL_LIST comes from /models,
// so new variants added to MODEL_CONFIGS on the server just show up here.
// ---------------------------------------------------------------------------
let activeModel = localStorage.getItem('searchModel') || 's2';
let MODEL_LIST   = [];

async function loadModels() {
  try {
    const res = await fetch('/models');
    MODEL_LIST = await res.json();
  } catch(_) {}
  if (!MODEL_LIST.length) MODEL_LIST = [{ key: 's2', label: 'MobileCLIP-S2', default: true }];
  if (!MODEL_LIST.some(function(m) { return m.key === activeModel; })) {
    const def = MODEL_LIST.find(function(m) { return m.default; });
    activeModel = def ? def.key : MODEL_LIST[0].key;
  }
  modelSelect.innerHTML = MODEL_LIST.map(function(m) {
    return '<option value="' + m.key + '">' + escHtml(m.label) + '</option>';
  }).join('');
  modelSelect.value = activeModel;
  initCompareMode();
}
modelSelect.addEventListener('change', function() {
  activeModel = modelSelect.value;
  localStorage.setItem('searchModel', activeModel);
  if (queryInput.value.trim()) doSearch();
});

// ---------------------------------------------------------------------------
// Similar / model switching helper -- shared by single view, compare panels,
// and the lightbox, so "Find Similar" always searches with the right model.
// ---------------------------------------------------------------------------
function goSimilar(fname, modelKey) {
  setCompareMode(false);
  if (modelKey && modelKey !== activeModel) {
    activeModel = modelKey;
    modelSelect.value = modelKey;
    localStorage.setItem('searchModel', activeModel);
  }
  doSimilarSearch(fname);
}

// ---------------------------------------------------------------------------
// Compare mode -- query N models at once, side by side. Panel count is
// bounded by MODEL_LIST, so it scales automatically as models are added.
// ---------------------------------------------------------------------------
const compareToggleBtn  = document.getElementById('compareToggleBtn');
const singleView        = document.getElementById('singleView');
const compareView       = document.getElementById('compareView');
const compareQueryInput = document.getElementById('compareQueryInput');
const compareKInput     = document.getElementById('compareKInput');
const comparePanelsEl   = document.getElementById('comparePanels');
const cmpAddBtn         = document.getElementById('cmpAddBtn');

let compareMode         = localStorage.getItem('compareMode') === '1';
let comparePanels       = [];   // [{ id, model }]
let comparePanelSeq     = 0;
let compareActiveColor  = '';
const PANEL_COLORS = ['#3b82f6', '#22c55e', '#a855f7', '#f97316', '#ef4444', '#14b8a6'];

function setCompareMode(on) {
  compareMode = on;
  localStorage.setItem('compareMode', on ? '1' : '0');
  singleView.style.display  = on ? 'none' : '';
  compareView.style.display = on ? '' : 'none';
  compareToggleBtn.classList.toggle('active', on);
  if (on && !compareQueryInput.value.trim() && queryInput.value.trim()) {
    compareQueryInput.value = queryInput.value;
  }
}

function initCompareMode() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem('comparePanels') || 'null'); } catch(_) {}
  const valid = saved && Array.isArray(saved) && saved.length &&
                saved.every(function(m) { return MODEL_LIST.some(function(x) { return x.key === m; }); });
  const models = valid ? saved : MODEL_LIST.slice(0, 2).map(function(m) { return m.key; });
  comparePanels = models.map(function(m) { return { id: comparePanelSeq++, model: m }; });
  renderComparePanels();
  setCompareMode(compareMode);
}

function saveComparePanels() {
  localStorage.setItem('comparePanels', JSON.stringify(comparePanels.map(function(p) { return p.model; })));
}

function updateAddButtonState() {
  cmpAddBtn.disabled = comparePanels.length >= MODEL_LIST.length;
}

function renderComparePanels() {
  comparePanelsEl.innerHTML = comparePanels.map(function(p, i) {
    const opts = MODEL_LIST.map(function(m) {
      return '<option value="' + m.key + '"' + (m.key === p.model ? ' selected' : '') + '>' +
             escHtml(m.label) + '</option>';
    }).join('');
    return '<div class="cmp-panel" data-id="' + p.id + '">' +
      '<div class="cmp-panel-head">' +
        '<span class="cmp-dot" style="background:' + PANEL_COLORS[i % PANEL_COLORS.length] + '"></span>' +
        '<select class="cmp-model-select">' + opts + '</select>' +
        '<button class="cmp-remove-btn" title="Remove"' +
          (comparePanels.length <= 1 ? ' disabled' : '') + '>&times;</button>' +
      '</div>' +
      '<div class="cmp-status"></div>' +
      '<div class="cmp-results"><div class="cmp-empty">Enter a query to search</div></div>' +
    '</div>';
  }).join('');

  comparePanelsEl.querySelectorAll('.cmp-panel').forEach(function(panelEl) {
    const id = parseInt(panelEl.dataset.id);
    panelEl.querySelector('.cmp-model-select').addEventListener('change', function(e) {
      const panel = comparePanels.find(function(p) { return p.id === id; });
      panel.model = e.target.value;
      saveComparePanels();
      if (compareQueryInput.value.trim()) searchPanel(panel, panelEl);
    });
    panelEl.querySelector('.cmp-remove-btn').addEventListener('click', function() {
      comparePanels = comparePanels.filter(function(p) { return p.id !== id; });
      saveComparePanels();
      renderComparePanels();
      if (compareQueryInput.value.trim()) doCompareSearch();
    });
    wireCardClicks(panelEl.querySelector('.cmp-results'),
      comparePanels.find(function(p) { return p.id === id; }).model);
  });
  updateAddButtonState();
}

function addComparePanel() {
  const used = comparePanels.map(function(p) { return p.model; });
  const next = MODEL_LIST.find(function(m) { return used.indexOf(m.key) === -1; });
  if (!next) return;
  comparePanels.push({ id: comparePanelSeq++, model: next.key });
  saveComparePanels();
  renderComparePanels();
  if (compareQueryInput.value.trim()) doCompareSearch();
}

function getCompareK() { return parseInt(compareKInput.value) || 10; }

async function searchPanel(panel, panelEl) {
  const statusEl = panelEl.querySelector('.cmp-status');
  const gridEl   = panelEl.querySelector('.cmp-results');
  const q = compareQueryInput.value.trim();
  if (!q) return;
  statusEl.textContent = 'Searching\\u2026';
  let url = '/search?q=' + encodeURIComponent(q) + '&k=' + getCompareK() + '&model=' + panel.model;
  if (compareActiveColor) url += '&color=' + encodeURIComponent(compareActiveColor);
  try {
    const res  = await fetch(url);
    const data = await res.json();
    if (data.error) { statusEl.textContent = 'Error: ' + data.error; return; }
    statusEl.textContent = data.results.length + ' results';
    gridEl.innerHTML = buildCardsHTML(data.results, 'cmp-empty');
    wireCardClicks(gridEl, panel.model);
  } catch(err) { statusEl.textContent = 'Request failed: ' + err.message; }
}

function doCompareSearch() {
  if (!compareQueryInput.value.trim()) return;
  comparePanelsEl.querySelectorAll('.cmp-panel').forEach(function(panelEl) {
    const id    = parseInt(panelEl.dataset.id);
    const panel = comparePanels.find(function(p) { return p.id === id; });
    searchPanel(panel, panelEl);
  });
}
compareQueryInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') doCompareSearch();
});
compareToggleBtn.addEventListener('click', function() { setCompareMode(!compareMode); });

// ---------------------------------------------------------------------------
// Compare view color filter (separate state from the single-view filter)
// ---------------------------------------------------------------------------
async function loadCompareColors() {
  try {
    const res    = await fetch('/colors');
    const colors = await res.json();
    if (!colors.length) return;
    const bar  = document.getElementById('compareColorBar');
    const pils = document.getElementById('compareColorPills');
    bar.style.display = 'flex';
    pils.innerHTML = colors.map(function(c) {
      const extra = (c === 'white') ? ' color-chip-white' : '';
      return '<span class="color-chip' + extra + '" data-color="' + c +
             '" style="background:' + (COLOR_HEX[c] || '#888') + '">' + c + '</span>';
    }).join('');
    pils.querySelectorAll('.color-chip').forEach(function(el) {
      el.addEventListener('click', function() { toggleCompareColor(el.dataset.color); });
    });
  } catch(_) {}
}
function toggleCompareColor(c) {
  compareActiveColor = (compareActiveColor === c) ? '' : c;
  document.querySelectorAll('#compareColorPills .color-chip').forEach(function(el) {
    el.classList.toggle('active', el.dataset.color === compareActiveColor);
  });
  const clearBtn = document.getElementById('compareColorClear');
  if (clearBtn) clearBtn.style.display = compareActiveColor ? 'inline-block' : 'none';
  if (compareQueryInput.value.trim()) doCompareSearch();
}
function clearCompareColorFilter() { if (compareActiveColor) toggleCompareColor(compareActiveColor); }

// ---------------------------------------------------------------------------
// Dark mode
// ---------------------------------------------------------------------------
const themeToggle = document.getElementById('themeToggle');
function applyTheme(dark) {
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : '');
  themeToggle.textContent = dark ? '\\u2600\\uFE0F' : '\\uD83C\\uDF19';
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}
(function initTheme() {
  const saved = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved === 'dark' || (!saved && prefersDark));
})();
themeToggle.addEventListener('click', function() {
  applyTheme(document.documentElement.getAttribute('data-theme') !== 'dark');
});

// ---------------------------------------------------------------------------
// Recent searches (localStorage)
// ---------------------------------------------------------------------------
function addToHistory(q, mode) {
  let h = JSON.parse(localStorage.getItem('searchHistory') || '[]');
  h = h.filter(function(x) { return !(x.q === q && x.mode === mode); });
  h.unshift({ q: q, mode: mode || 'text' });
  h = h.slice(0, 20);
  localStorage.setItem('searchHistory', JSON.stringify(h));
  renderHistory();
}
function renderHistory() {
  const h    = JSON.parse(localStorage.getItem('searchHistory') || '[]');
  const bar  = document.getElementById('historyBar');
  const pils = document.getElementById('historyPills');
  if (!h.length) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';
  pils.innerHTML = h.map(function(x, i) {
    const icon = x.mode === 'similar' ? '\\uD83D\\uDCF7 ' : '';
    return '<span class="hist-pill" data-i="' + i + '">' + icon + escHtml(x.q) + '</span>';
  }).join('');
  pils.querySelectorAll('.hist-pill').forEach(function(el) {
    el.addEventListener('click', function() {
      const item = h[parseInt(el.dataset.i)];
      if (item.mode === 'similar') { doSimilarSearch(item.q); }
      else { queryInput.value = item.q; doSearch(); }
    });
  });
}
function clearHistory() { localStorage.removeItem('searchHistory'); renderHistory(); }

// ---------------------------------------------------------------------------
// Filter by color
// ---------------------------------------------------------------------------
const COLOR_HEX = {
  red:'#e53935', orange:'#fb8c00', yellow:'#fdd835', green:'#43a047',
  cyan:'#00acc1', blue:'#1e88e5', purple:'#8e24aa', pink:'#e91e63',
  brown:'#6d4c41', white:'#f5f5f5', gray:'#78909c', black:'#212121'
};
async function loadColors() {
  try {
    const res    = await fetch('/colors');
    const colors = await res.json();
    if (!colors.length) return;
    const bar  = document.getElementById('colorBar');
    const pils = document.getElementById('colorPills');
    bar.style.display = 'flex';
    pils.innerHTML = colors.map(function(c) {
      const extra = (c === 'white') ? ' color-chip-white' : '';
      return '<span class="color-chip' + extra + '" data-color="' + c +
             '" style="background:' + (COLOR_HEX[c] || '#888') + '">' + c + '</span>';
    }).join('');
    pils.querySelectorAll('.color-chip').forEach(function(el) {
      el.addEventListener('click', function() { toggleColor(el.dataset.color); });
    });
  } catch(_) {}
}
function toggleColor(c) {
  activeColor = (activeColor === c) ? '' : c;
  document.querySelectorAll('.color-chip').forEach(function(el) {
    el.classList.toggle('active', el.dataset.color === activeColor);
  });
  const clearBtn = document.getElementById('colorClear');
  if (clearBtn) clearBtn.style.display = activeColor ? 'inline-block' : 'none';
  if (queryInput.value.trim()) doSearch();
}
function clearColorFilter() { if (activeColor) toggleColor(activeColor); }

// ---------------------------------------------------------------------------
// Algo threshold panel -- purely re-buckets whatever's already on screen,
// so every change re-renders instantly with no new /search request.
// ---------------------------------------------------------------------------
(function initAlgoPanel() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem('algoThresholds') || 'null'); } catch(_) {}
  if (saved) {
    if (saved.top    !== undefined) topThresholdInput.value = saved.top;
    if (saved.mid    !== undefined) midThresholdInput.value = saved.mid;
    if (saved.cutoff !== undefined) cutoffInput.value       = saved.cutoff;
    if (saved.t1     !== undefined) t1Input.value           = saved.t1;
    if (saved.t2     !== undefined) t2Input.value           = saved.t2;
    if (saved.t3     !== undefined) t3Input.value           = saved.t3;
  }
  if (localStorage.getItem('algoPanelOpen') === '1') {
    algoPanel.style.display = 'flex';
    algoBtn.classList.add('active');
  }
})();
algoBtn.addEventListener('click', function() {
  const opening = algoPanel.style.display === 'none';
  algoPanel.style.display = opening ? 'flex' : 'none';
  algoBtn.classList.toggle('active', opening);
  localStorage.setItem('algoPanelOpen', opening ? '1' : '0');
});
[topThresholdInput, midThresholdInput, cutoffInput, t1Input, t2Input, t3Input].forEach(function(el) {
  el.addEventListener('input', function() {
    localStorage.setItem('algoThresholds', JSON.stringify({
      top: topThresholdInput.value, mid: midThresholdInput.value, cutoff: cutoffInput.value,
      t1: t1Input.value, t2: t2Input.value, t3: t3Input.value
    }));
    if (lastResults.length) renderResults(lastResults);
  });
});

// ---------------------------------------------------------------------------
// Autocomplete
// ---------------------------------------------------------------------------
const acDropdown = document.getElementById('acDropdown');
let acItems = [], acIdx = -1, acTimer = null;

queryInput.addEventListener('input', function() {
  clearTimeout(acTimer);
  acTimer = setTimeout(fetchSuggestions, 180);
});
queryInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') {
    if (acIdx >= 0 && acDropdown.classList.contains('open')) { e.preventDefault(); selectSuggestion(acIdx); }
    else { closeDropdown(); doSearch(); }
    return;
  }
  if (!acDropdown.classList.contains('open')) return;
  if (e.key === 'ArrowDown') { e.preventDefault(); acIdx = Math.min(acIdx + 1, acItems.length - 1); renderActive(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); acIdx = Math.max(acIdx - 1, -1); renderActive(); }
  else if (e.key === 'Escape') closeDropdown();
});
document.addEventListener('click', function(e) {
  if (!e.target.closest('.autocomplete-wrap')) closeDropdown();
});
function getLastWord() {
  const parts = queryInput.value.split(' ');
  return parts[parts.length - 1].replace(/^-/, '').toLowerCase();
}
async function fetchSuggestions() {
  const prefix = getLastWord();
  if (prefix.length < 2) { closeDropdown(); return; }
  try {
    const res  = await fetch('/suggest?q=' + encodeURIComponent(prefix) + '&limit=8&model=' + activeModel);
    const data = await res.json();
    acItems = data; acIdx = -1;
    if (!data.length) { closeDropdown(); return; }
    acDropdown.innerHTML = data.map(function(it, i) {
      const hi = '<em>' + it.term.slice(0, prefix.length) + '</em>' + it.term.slice(prefix.length);
      return '<div class="ac-item" data-i="' + i + '">' +
        '<span class="ac-term">' + hi + '</span>' +
        '<span class="ac-count">' + it.count.toLocaleString() + ' images</span></div>';
    }).join('');
    acDropdown.querySelectorAll('.ac-item').forEach(function(el) {
      el.addEventListener('mousedown', function(e) { e.preventDefault(); selectSuggestion(parseInt(el.dataset.i)); });
    });
    acDropdown.classList.add('open');
  } catch(_) {}
}
function selectSuggestion(i) {
  const parts = queryInput.value.split(' ');
  parts[parts.length - 1] = acItems[i].term;
  queryInput.value = parts.join(' ');
  closeDropdown(); doSearch();
}
function closeDropdown() { acDropdown.classList.remove('open'); acIdx = -1; }
function renderActive() {
  acDropdown.querySelectorAll('.ac-item').forEach(function(el, i) {
    el.classList.toggle('active', i === acIdx);
    if (i === acIdx) el.scrollIntoView({ block: 'nearest' });
  });
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
function getK() { return parseInt(kInput.value) || 10; }

async function doSearch() {
  const q = queryInput.value.trim();
  if (!q) return;
  addToHistory(q, 'text');
  startLoading('Searching\\u2026');
  let url = '/search?q=' + encodeURIComponent(q) + '&k=' + getK() + '&model=' + activeModel;
  if (activeColor) url += '&color=' + encodeURIComponent(activeColor);
  try {
    const res  = await fetch(url);
    const data = await res.json();
    if (data.error) { statusText.textContent = 'Error: ' + data.error; return; }
    renderResults(data.results, buildTextStatus(data));
  } catch(err) { statusText.textContent = 'Request failed: ' + err.message; }
  finally { stopLoading(); }
}
function buildTextStatus(data) {
  let s = data.results.length + ' results for "' + escHtml(data.positive) + '"';
  s += ' &mdash; model: <b>' + escHtml(data.model) + '</b>';
  if (activeColor) s += ' &mdash; color: <b>' + activeColor + '</b>';
  if (data.negatives && data.negatives.length) {
    s += ' &mdash; excluding: ' + data.negatives.map(function(n) {
      return '<span class="neg-tag">-' + escHtml(n) + '</span>';
    }).join(' ');
  }
  return s;
}
async function doSimilarSearch(filename) {
  addToHistory(filename, 'similar');
  queryInput.value = '';
  startLoading('Finding images similar to ' + filename + '\\u2026');
  try {
    const res  = await fetch('/similar?img=' + encodeURIComponent(filename) + '&k=' + getK() + '&model=' + activeModel);
    const data = await res.json();
    if (data.error) { statusText.textContent = 'Error: ' + data.error; return; }
    renderResults(data.results, data.results.length + ' images similar to <b>' + escHtml(data.query_image) + '</b>' +
      ' &mdash; model: <b>' + escHtml(data.model) + '</b>');
  } catch(err) { statusText.textContent = 'Request failed: ' + err.message; }
  finally { stopLoading(); }
}
async function doUploadSearch(file) {
  queryInput.value = '';
  startLoading('Encoding uploaded image\\u2026');
  const fd = new FormData();
  fd.append('image', file);
  try {
    const res  = await fetch('/search_by_image?k=' + getK() + '&model=' + activeModel, { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { statusText.textContent = 'Error: ' + data.error; return; }
    renderResults(data.results, data.results.length + ' images similar to uploaded image' +
      ' &mdash; model: <b>' + escHtml(data.model) + '</b>');
  } catch(err) { statusText.textContent = 'Request failed: ' + err.message; }
  finally { stopLoading(); }
}
function startLoading(msg) {
  searchBtn.disabled = true; spinner.style.display = 'block';
  statusText.innerHTML = msg; resultsEl.innerHTML = '';
}
function stopLoading() { searchBtn.disabled = false; spinner.style.display = 'none'; }

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
function barColor(score, th) {
  if (score >= th.t1) return 'var(--accent)';
  if (score >= th.t2) return '#f59e0b';
  return '#ef4444';
}

function buildCardsHTML(results, emptyClass, dotClass, th, globalMax) {
  if (!results.length) return '<div class="' + (emptyClass || 'empty') + '">No results.</div>';
  const max = globalMax || results[0].score || 1;
  const dot = dotClass ? '<span class="card-dot ' + dotClass + '"></span>' : '';
  return results.map(function(r) {
    const color = th ? barColor(r.score, th) : '';
    const pct = Math.round((r.score / max) * 100);
    const fillStyle = 'width:' + pct + '%' + (color ? ';background:' + color : '');
    const valStyle = color ? ' style="color:' + color + '"' : '';
    const dup = r.duplicate_count > 0
      ? '<div class="dup-badge">+' + r.duplicate_count + ' identical</div>' : '';
    return '<div class="card" data-url="' + r.url + '" data-fname="' + r.filename +
           '" data-score="' + r.score + '" data-rank="' + r.rank + '">' +
      dot +
      '<button class="similar-btn" data-fname="' + r.filename + '">Similar</button>' +
      '<img class="card-img" src="' + r.url + '" loading="lazy" alt="">' +
      '<div class="card-body">' +
        '<div class="card-rank">#' + r.rank + '</div>' +
        '<div class="card-fname">' + r.filename + '</div>' +
        '<div class="score-row">' +
          '<div class="score-bg"><div class="score-fill" style="' + fillStyle + '"></div></div>' +
          '<span class="score-val"' + valStyle + '>' + r.score.toFixed(4) + '</span>' +
        '</div>' + dup +
      '</div></div>';
  }).join('');
}

function wireCardClicks(container, modelKey) {
  container.onclick = function(e) {
    const btn = e.target.closest('.similar-btn');
    if (btn) { e.stopPropagation(); goSimilar(btn.dataset.fname, modelKey); return; }
    const card = e.target.closest('.card');
    if (card) openLightbox(card.dataset.url, card.dataset.fname,
                           parseFloat(card.dataset.score), parseInt(card.dataset.rank), modelKey);
  };
}

// ---------------------------------------------------------------------------
// Tiered results (Top results / Partial matches / Below cutoff) -- purely a
// client-side re-bucketing of the already-fetched `results` array by score,
// so adjusting thresholds re-renders instantly with no new request.
// ---------------------------------------------------------------------------
let lastResults = [];

function getAlgoThresholds() {
  const top = parseFloat(topThresholdInput.value);
  const mid = parseFloat(midThresholdInput.value);
  const cut = parseFloat(cutoffInput.value);
  const t1  = parseFloat(t1Input.value);
  const t2  = parseFloat(t2Input.value);
  const t3  = parseFloat(t3Input.value);
  return {
    top: isNaN(top) ? 0.24 : top,
    mid: isNaN(mid) ? 0.20 : mid,
    cutoff: isNaN(cut) ? 0.10 : cut,
    t1: isNaN(t1) ? 0.24 : t1,
    t2: isNaN(t2) ? 0.20 : t2,
    t3: isNaN(t3) ? 0.10 : t3,
  };
}

function renderTierSection(title, subtitle, dotClass, items, th, globalMax, collapsedDefault) {
  if (!items.length) return '';
  const collapsible = collapsedDefault !== null;
  const countLabel = items.length + (items.length === 1 ? ' photo' : ' photos');
  const chevron = collapsible ? '<span class="tier-chevron">&#9662;</span>' : '';
  const head = '<div class="tier-head' + (collapsible ? ' clickable" onclick="toggleTierSection(this)' : '') + '">' +
    '<span class="algo-legend-dot ' + dotClass + '"></span>' +
    '<span class="tier-title">' + escHtml(title) + '</span>' +
    '<span class="tier-count">' + countLabel + '</span>' +
    chevron +
  '</div>';
  const sub = subtitle ? '<div class="tier-sub">' + escHtml(subtitle) + '</div>' : '';
  const grid = '<div class="tier-grid">' + buildCardsHTML(items, 'tier-empty', dotClass, th, globalMax) + '</div>';
  return '<div class="tier-section' + (collapsedDefault ? ' collapsed' : '') + '">' + head + sub + grid + '</div>';
}

function toggleTierSection(headEl) {
  headEl.closest('.tier-section').classList.toggle('collapsed');
}

function renderResults(results, statusHTML) {
  lastResults = results;
  if (statusHTML !== undefined) statusText.innerHTML = statusHTML;
  if (!results.length) { resultsEl.innerHTML = '<div class="empty">No results.</div>'; return; }

  const th = getAlgoThresholds();
  const globalMax = results[0].score || 1;
  const topTier = [], partialTier = [], cutoffTier = [];
  results.forEach(function(r) {
    if (r.score >= th.top) topTier.push(r);
    else if (r.score >= th.mid) partialTier.push(r);
    else if (r.score >= th.cutoff) cutoffTier.push(r);
    // else: below the cutoff floor -- excluded entirely, not shown, not counted
  });

  const noStrongMatches = !topTier.length && partialTier.length > 0;
  const nothingFound    = !topTier.length && !partialTier.length;

  let html = '';
  if (noStrongMatches) {
    html += '<div class="no-match-banner">' +
      '<div class="nm-icon">&#128269;</div>' +
      '<div class="nm-title">No strong matches</div>' +
      '<div class="nm-sub">Found ' + partialTier.length +
        ' partial match' + (partialTier.length === 1 ? '' : 'es') + ' below</div>' +
    '</div>';
  } else if (nothingFound) {
    html += '<div class="no-match-banner">' +
      '<div class="nm-icon nm-icon-empty">&#128444;</div>' +
      '<div class="nm-title">Nothing found</div>' +
      '<div class="nm-sub">Try a shorter phrase or add more detail to narrow down your results</div>' +
    '</div>';
  }
  html += renderTierSection('Top results', null, 'dot-top', topTier, th, globalMax, false);
  html += renderTierSection('Partial matches', 'May share some details with your search',
                             'dot-probable', partialTier, th, globalMax, true);
  html += renderTierSection('Below cutoff', 'Lower relevance than your Mid threshold',
                             'dot-cutoff', cutoffTier, th, globalMax, true);

  resultsEl.innerHTML = html;
  wireCardClicks(resultsEl, activeModel);
}

// ---------------------------------------------------------------------------
// Lightbox
// ---------------------------------------------------------------------------
let currentLbModel = '';
function openLightbox(url, fname, score, rank, modelKey) {
  currentLbFilename = fname;
  currentLbModel    = modelKey || activeModel;
  lbImg.src = url;
  lbCaption.textContent = '#' + rank + ' \\xb7 ' + fname + ' \\xb7 score ' + score.toFixed(4);
  lightbox.classList.add('open');
}
function closeLightbox() { lightbox.classList.remove('open'); lbImg.src = ''; }
lightbox.addEventListener('click', function(e) { if (e.target === lightbox) closeLightbox(); });
document.getElementById('lbEsc').addEventListener('click', closeLightbox);
document.getElementById('lbCloseBtn').addEventListener('click', closeLightbox);
document.getElementById('lbSimilarBtn').addEventListener('click', function() {
  closeLightbox(); goSimilar(currentLbFilename, currentLbModel);
});
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeLightbox(); });

// ---------------------------------------------------------------------------
// Image upload
// ---------------------------------------------------------------------------
uploadBtn.addEventListener('click', function() { fileInput.click(); });
fileInput.addEventListener('change', function(e) {
  const f = e.target.files[0];
  if (f) doUploadSearch(f);
  e.target.value = '';
});
searchBar.addEventListener('dragover', function(e) { e.preventDefault(); searchBar.classList.add('drag-over'); });
searchBar.addEventListener('dragleave', function() { searchBar.classList.remove('drag-over'); });
searchBar.addEventListener('drop', function(e) {
  e.preventDefault(); searchBar.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) doUploadSearch(f);
});

// ---------------------------------------------------------------------------
// Rebuild index (POST /rebuild, stream progress over SSE)
// ---------------------------------------------------------------------------
const rebuildModal = document.getElementById('rebuildModal');
const rebuildFill  = document.getElementById('rebuildFill');
const rebuildMsg   = document.getElementById('rebuildMsg');
const rebuildClose = document.getElementById('rebuildClose');
let rebuildES = null;

document.getElementById('rebuildBtn').addEventListener('click', async function() {
  rebuildFill.style.width = '0%';
  rebuildMsg.textContent  = 'Starting\\u2026';
  rebuildClose.style.display = 'none';
  rebuildModal.classList.add('open');
  try {
    const res  = await fetch('/rebuild', { method: 'POST' });
    const data = await res.json();
    if (data.error) { rebuildMsg.textContent = data.error; rebuildClose.style.display = 'inline-block'; return; }
  } catch(err) { rebuildMsg.textContent = 'Failed to start: ' + err.message; rebuildClose.style.display = 'inline-block'; return; }

  if (rebuildES) rebuildES.close();
  rebuildES = new EventSource('/rebuild_stream');
  rebuildES.onmessage = function(ev) {
    const m = JSON.parse(ev.data);
    if (m.type === 'progress') {
      const pct = m.total ? Math.round((m.done / m.total) * 100) : 100;
      rebuildFill.style.width = pct + '%';
      if (m.msg) rebuildMsg.textContent = m.msg;
    } else if (m.type === 'done') {
      rebuildFill.style.width = '100%';
      rebuildMsg.textContent  = 'Done. ' + m.total.toLocaleString() + ' images indexed.';
      document.getElementById('imgCount').textContent = m.total.toLocaleString();
      rebuildClose.style.display = 'inline-block';
      rebuildES.close(); rebuildES = null;
      loadColors();
    }
  };
  rebuildES.onerror = function() {
    rebuildMsg.textContent = 'Connection lost. The rebuild may still be running in the background.';
    rebuildClose.style.display = 'inline-block';
    if (rebuildES) { rebuildES.close(); rebuildES = null; }
  };
});
function closeRebuild() { rebuildModal.classList.remove('open'); }

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
renderHistory();
loadColors();
loadCompareColors();
loadModels();
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
