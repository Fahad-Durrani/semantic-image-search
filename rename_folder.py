"""Rename every image in 'ALL images' to 1..N (extensions preserved).
Two-phase rename avoids collisions with existing numeric names.
Run BEFORE generating embeddings (embeddings are built from the renamed files)."""
import os
from pathlib import Path

DST = Path(r"c:\Users\Fahad\Desktop\Image testing\ALL images")
SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.jfif'}

def main():
    files = sorted(p for p in DST.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    n = len(files)
    print(f"Renaming {n} images to 1..{n} (extensions preserved)...")

    # Phase 1: every file -> unique temp name
    temps = []
    for i, src in enumerate(files):
        ext = src.suffix.lower()
        tmp = src.with_name(f"__tmp_{i}__{ext}")
        os.rename(src, tmp)
        temps.append((tmp, ext))

    # Phase 2: temp -> final 1..N
    for i, (tmp, ext) in enumerate(temps):
        os.rename(tmp, tmp.with_name(f"{i + 1}{ext}"))

    print(f"Done. Renamed {n} files to 1..{n}.")

if __name__ == "__main__":
    main()
