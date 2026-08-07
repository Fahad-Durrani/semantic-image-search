"""Build the embedding index for the image search app.

Uses Apple's official `mobileclip` package (ml-mobileclip) with a local
mobileclip_s2.pt / mobileclip_s0.pt / mobileclip_s0_fp16.pt checkpoint -- this
applies Apple's own image normalization, unlike the open_clip/HuggingFace
variant used previously.

Outputs cache/embeddings.npz (s2) / embeddings_s0.npz (s0) /
embeddings_s0_fp16.npz (s0_fp16) with two aligned arrays:
  embeddings : (N, 512) float32, L2-normalised
  filenames  : (N,)      bare filenames (e.g. "1.jpg") served from images_repo

A tqdm progress bar tracks embedding creation.
"""
import os
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
}


def load_model(model_name, checkpoint, device, reparam_checkpoint=False):
    if reparam_checkpoint:
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

    cfg        = MODEL_CONFIGS[args.model]
    MODEL_NAME = cfg["model_name"]
    CHECKPOINT = cfg["checkpoint"]
    CACHE_FILE = cfg["cache_file"]

    if not os.path.isfile(CHECKPOINT):
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} from {CHECKPOINT} (device: {device}) ...", flush=True)
    model, preprocess = load_model(MODEL_NAME, CHECKPOINT, device,
                                    reparam_checkpoint=cfg.get("reparam_checkpoint", False))
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
