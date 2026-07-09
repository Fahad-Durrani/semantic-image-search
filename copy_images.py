"""Recursively copy all embeddable images from 'Image folders' (and its
subdirectories) into a single flat 'ALL images' folder. Duplicate filenames
across subfolders are made unique with a numeric suffix so nothing is lost."""
import shutil
from pathlib import Path

SRC = Path(r"c:\Users\Fahad\Desktop\Image testing\Image folders")
DST = Path(r"c:\Users\Fahad\Desktop\Image testing\ALL images")
SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.jfif'}

def main():
    DST.mkdir(exist_ok=True)

    sources = sorted(
        p for p in SRC.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    print(f"Found {len(sources)} embeddable images under {SRC.name}")

    copied = 0
    used = set()
    for src in sources:
        stem, ext = src.stem, src.suffix.lower()
        target = DST / f"{stem}{ext}"
        # Resolve collisions (same filename from different subfolders)
        i = 1
        while target.name in used or target.exists():
            target = DST / f"{stem}_{i}{ext}"
            i += 1
        used.add(target.name)
        shutil.copy2(src, target)
        copied += 1

    total = sum(1 for p in DST.iterdir() if p.is_file())
    print(f"Copied {copied} images.")
    print(f"'ALL images' now contains {total} files.")

if __name__ == "__main__":
    main()
