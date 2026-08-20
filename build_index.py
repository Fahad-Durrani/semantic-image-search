"""Build the embedding index for the image search app.

Uses Apple's official `mobileclip` package (ml-mobileclip) with a local
mobileclip_s2.pt / mobileclip_s0.pt / mobileclip_s0_fp16.pt checkpoint (or,
for s0_int8, a pair of separately-quantized TorchScript encoders) -- this
applies Apple's own image normalization, unlike the open_clip/HuggingFace
variant used previously. Two exceptions: tinyclip, a genuinely different
model family (wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M) loaded via HuggingFace
`transformers`; and tinyclip_resnet19m (wkcn/TinyCLIP-ResNet-19M-Text-19M),
loaded via `open_clip` with a registered custom model config and a remapped
checkpoint (see below). tinyclip_resnet19m additionally has a 1024-d
embedding space, not 512 -- nothing in this file assumes a fixed dimension.

Outputs cache/embeddings.npz (s2) / embeddings_s0.npz (s0) /
embeddings_s0_fp16.npz (s0_fp16) / embeddings_s0_int8.npz (s0_int8) /
embeddings_tinyclip.npz (tinyclip) / embeddings_tinyclip_resnet19m.npz
(tinyclip_resnet19m) with two aligned arrays:
  embeddings : (N, D) float32, L2-normalised (D is 512 for every variant
               except tinyclip_resnet19m, where D is 1024)
  filenames  : (N,)      bare filenames (e.g. "1.jpg") served from images_repo

A tqdm progress bar tracks embedding creation.
"""
import os
import re
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import mobileclip
from mobileclip.modules.common.mobileone import reparameterize_model

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images_repo")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
SUPPORTED  = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.jfif'}
BATCH_SIZE = 32

# Keep in sync with app.py's MODEL_CONFIGS.
MODEL_CONFIGS = {
    "s2": {
        "model_name": "mobileclip_s2",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s2.pt"),
        "cache_file": os.path.join(CACHE_DIR, "embeddings.npz"),
    },
    "s0": {
        "model_name": "mobileclip_s0",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0.pt"),
        "cache_file": os.path.join(CACHE_DIR, "embeddings_s0.npz"),
    },
    "s0_fp16": {
        "model_name": "mobileclip_s0",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0_fp16.pt"),
        "cache_file": os.path.join(CACHE_DIR, "embeddings_s0_fp16.npz"),
        # This checkpoint stores already-reparameterized (folded) MobileOne
        # branches in fp16, not the raw multi-branch state dict create_model_
        # and_transforms()'s default load path expects -- reparameterize the
        # freshly-built model first so its keys match, then load into it.
        "reparam_checkpoint": True,
    },
    "s0_int8": {
        "model_name": "mobileclip_s0",
        # Two independently int8-quantized TorchScript modules (image/text),
        # not a single state-dict checkpoint like the other variants.
        "checkpoint": {
            "image": os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0_image_int8.pt"),
            "text":  os.path.join(BASE_DIR, "checkpoints", "mobileclip_s0_text_int8.pt"),
        },
        "cache_file": os.path.join(CACHE_DIR, "embeddings_s0_int8.npz"),
        "quantized": True,
    },
    "tinyclip": {
        "model_name": "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        # A local snapshot of the HF hub repo (config.json, tokenizer.json,
        # model.safetensors, preprocessor_config.json), not a mobileclip
        # checkpoint file -- loaded via transformers, not the mobileclip package.
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "tinyclip-vit-8m-16-text-3m-yfcc15m"),
        "cache_file": os.path.join(CACHE_DIR, "embeddings_tinyclip.npz"),
        "hf_clip": True,
    },
    "tinyclip_resnet19m": {
        # Registered with open_clip.add_model_config() at load time using this
        # name (must match the config JSON's filename stem).
        "model_name": "TinyCLIP-ResNet-19M-Text-19M",
        "checkpoint": os.path.join(BASE_DIR, "checkpoints", "tinyclip_resnet19m_text19m_laion400m.pt"),
        "open_clip_config": os.path.join(BASE_DIR, "checkpoints", "TinyCLIP-ResNet-19M-Text-19M.json"),
        "cache_file": os.path.join(CACHE_DIR, "embeddings_tinyclip_resnet19m.npz"),
        # 1024-d embeddings, unlike every other variant (512-d). Nothing here
        # assumes a fixed dimension, so this is safe -- just documented since
        # it's the one place that differs.
        "embed_dim": 1024,
    },
}


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
    same .encode_image()/.encode_text() interface the rest of this script
    uses for the regular mobileclip CLIP model."""

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
    .encode_image()/.encode_text() interface this script uses everywhere else."""

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


def load_model(model_name, checkpoint, device, reparam_checkpoint=False, quantized=False,
                hf_clip=False, open_clip_config=None):
    if open_clip_config:
        import open_clip
        open_clip.add_model_config(open_clip_config)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None, device=device)
        raw = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        remapped = {_remap_tinyclip_open_clip_key(k): v for k, v in state_dict.items()}
        model.load_state_dict(remapped)
        model.eval()
    elif hf_clip:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(checkpoint, local_files_only=True).to(device)
        processor = CLIPProcessor.from_pretrained(checkpoint, local_files_only=True)
        model = _HFClipEncoder(clip_model).eval()
        preprocess = _HFImageTransform(processor.image_processor)
    elif quantized:
        # Quantized int8 ops only run on CPU (fbgemm/qnnpack backends), regardless
        # of CUDA availability -- the caller is expected to pass device="cpu".
        image_model = torch.jit.load(checkpoint["image"], map_location=device)
        text_model  = torch.jit.load(checkpoint["text"], map_location=device)
        model = _QuantizedDualEncoder(image_model, text_model).eval()
        # Preprocessing transform is architecture-derived, not weight-derived,
        # so a throwaway float32 model is enough to fetch it.
        _, _, preprocess = mobileclip.create_model_and_transforms(
            model_name, pretrained=None, reparameterize=False, device=device)
    elif reparam_checkpoint:
        model, _, preprocess = mobileclip.create_model_and_transforms(
            model_name, pretrained=None, reparameterize=False, device=device)
        model = reparameterize_model(model)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        model, _, preprocess = mobileclip.create_model_and_transforms(
            model_name, pretrained=checkpoint, device=device)
        model = model.eval()
    return model, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="s2")
    args = parser.parse_args()

    cfg             = MODEL_CONFIGS[args.model]
    MODEL_NAME      = cfg["model_name"]
    CHECKPOINT      = cfg["checkpoint"]
    CACHE_FILE      = cfg["cache_file"]
    QUANTIZED       = cfg.get("quantized", False)
    HF_CLIP         = cfg.get("hf_clip", False)
    OPEN_CLIP_CFG   = cfg.get("open_clip_config")

    if HF_CLIP:
        missing = [] if os.path.isdir(CHECKPOINT) else [CHECKPOINT]
    else:
        checkpoint_paths = list(CHECKPOINT.values()) if isinstance(CHECKPOINT, dict) else [CHECKPOINT]
        if OPEN_CLIP_CFG:
            checkpoint_paths.append(OPEN_CLIP_CFG)
        missing = [p for p in checkpoint_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Checkpoint not found: {missing[0]}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Quantized int8 ops only run on CPU, regardless of CUDA availability.
    device = "cpu" if QUANTIZED else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} from {CHECKPOINT} (device: {device}) ...", flush=True)
    model, preprocess = load_model(MODEL_NAME, CHECKPOINT, device,
                                    reparam_checkpoint=cfg.get("reparam_checkpoint", False),
                                    quantized=QUANTIZED, hf_clip=HF_CLIP,
                                    open_clip_config=OPEN_CLIP_CFG)
    if not QUANTIZED:
        model = model.to(device)
    print("  Model loaded.", flush=True)

    image_files = sorted(
        p for p in Path(IMAGES_DIR).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    print(f"Found {len(image_files)} images in {IMAGES_DIR}\n", flush=True)

    embeddings = []
    filenames = []
    failed = []

    batches = [image_files[i:i + BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE)]

    with torch.no_grad():
        for batch in tqdm(batches, desc="Embedding images", unit="batch"):
            tensors, names = [], []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(preprocess(img))
                    names.append(p.name)
                except Exception as e:
                    failed.append((p.name, str(e)))
            if not tensors:
                continue
            feats = model.encode_image(torch.stack(tensors).to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy().astype("float32"))
            filenames.extend(names)

    all_embeddings = np.concatenate(embeddings, axis=0).astype("float32")
    np.savez(CACHE_FILE, embeddings=all_embeddings, filenames=np.array(filenames))

    print(f"\nDone: {len(filenames)} images embedded, {len(failed)} skipped.", flush=True)
    print(f"Saved to: {CACHE_FILE}  (shape: {all_embeddings.shape})", flush=True)
    if failed:
        print(f"\nSkipped files ({len(failed)}):", flush=True)
        for name, reason in failed[:20]:
            print(f"  {name}: {reason}", flush=True)


if __name__ == "__main__":
    main()
