from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from .job import JobPaths
from .types import Page
from .utils import ensure_dir


@dataclass(frozen=True)
class PageProvider:
    input_path: str
    input_type: str  # pdf|images
    dpi: int
    paths: JobPaths

    def iter_pages(self) -> Iterator[tuple[Page, Image.Image]]:
        if self.input_type == "pdf":
            yield from self._iter_pdf_pages()
        elif self.input_type == "images":
            yield from self._iter_image_folder()
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

    def _iter_pdf_pages(self) -> Iterator[tuple[Page, Image.Image]]:
        try:
            import fitz  # PyMuPDF
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyMuPDF is required for --type pdf. Install pymupdf.") from e

        pdf_path = Path(self.input_path)
        doc = fitz.open(pdf_path)

        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        ensure_dir(self.paths.pages_dir)
        for i in range(doc.page_count):
            page_num = i + 1
            page_id = f"page_{page_num:03d}"
            source_ref = f"{pdf_path.name}#page={page_num}"
            rel_path = f"pages/{page_id}.png"
            abs_path = self.paths.job_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            if abs_path.exists():
                img = Image.open(abs_path).convert("RGB")
            else:
                p = doc.load_page(i)
                pix = p.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.tobytes("png")
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img.save(abs_path, format="PNG")

            yield Page(page_index=i, page_id=page_id, source_ref=source_ref, image_path=rel_path), img

    def _iter_image_folder(self) -> Iterator[tuple[Page, Image.Image]]:
        folder = Path(self.input_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"--type images expects a folder: {folder}")

        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

        ensure_dir(self.paths.pages_dir)
        for i, img_path in enumerate(files):
            page_num = i + 1
            page_id = f"page_{page_num:03d}"
            source_ref = f"{folder.name}/{img_path.name}"
            rel_path = f"pages/{page_id}.png"
            abs_path = self.paths.job_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            if abs_path.exists():
                img = Image.open(abs_path).convert("RGB")
            else:
                img = Image.open(img_path).convert("RGB")
                img.save(abs_path, format="PNG")

            yield Page(page_index=i, page_id=page_id, source_ref=source_ref, image_path=rel_path), img
