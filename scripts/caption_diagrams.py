"""Generate caption sidecar files for diagram images, ONCE, offline.

Run this on your own machine after adding new diagram PNGs, with your Groq key:
    $env:GROQ_API_KEY="gsk_..."        # PowerShell
    python scripts/caption_diagrams.py data/sample/diagrams

It writes `<image>.caption.txt` next to each image. Commit those caption files so
the deployed app never needs to call the vision API at startup. Re-run only when
you add or change diagrams.
"""
import sys
from pathlib import Path
from eih.ingestion import vision


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample/diagrams")
    images = [p for p in root.rglob("*") if vision.is_image(p)]
    if not images:
        print(f"No images found under {root}"); return
    for img in images:
        cache = img.with_suffix(img.suffix + ".caption.txt")
        if cache.exists():
            print(f"skip (cached): {img.name}"); continue
        print(f"captioning: {img.name} ...")
        text = vision.caption_image(img)
        cache.write_text(text, encoding="utf-8")
        print(f"  -> wrote {cache.name}")


if __name__ == "__main__":
    main()
