from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out = Path(__file__).parent / "pages"
    out.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (240, 120), (255, 255, 255))
    d = ImageDraw.Draw(img)

    # Keep these boxes in sync with stage/ocr/page_000.cleaned.json
    boxes = [
        (10, 10, 80, 50),
        (90, 10, 170, 50),
        (180, 10, 235, 50),
        (10, 65, 90, 110),
        (100, 65, 220, 110),
    ]

    for b in boxes:
        d.rectangle(b, outline=(0, 0, 0), width=2)

    img.save(out / "page_000.png", format="PNG")


if __name__ == "__main__":
    main()
