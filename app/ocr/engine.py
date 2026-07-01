"""
EasyOCR-based text extraction with bounding boxes.

EasyOCR returns, for every detected text region:
    (bbox, text, confidence)
where bbox is a list of 4 (x, y) corner points (not axis-aligned, since
scanned documents are often slightly rotated/skewed). We normalize that
into plain dataclasses so the rest of the pipeline never touches the
EasyOCR API directly -- swapping in PaddleOCR or another engine later
only means rewriting this one file.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_reader = None


def get_reader():
    """Lazily load the EasyOCR reader (loads model weights on first use)."""
    global _reader
    if _reader is None:
        import easyocr

        logger.info(
            "Loading EasyOCR reader (languages=%s, gpu=%s)",
            settings.easyocr_language_list,
            settings.easyocr_gpu,
        )
        _reader = easyocr.Reader(settings.easyocr_language_list, gpu=settings.easyocr_gpu)
    return _reader


@dataclass
class TextBlock:
    text: str
    bbox: list  # 4 corner points: [[x,y],[x,y],[x,y],[x,y]]
    confidence: float

    def to_dict(self) -> dict:
        return {"text": self.text, "bbox": self.bbox, "confidence": self.confidence}

    @property
    def x_min(self) -> float:
        return min(p[0] for p in self.bbox)

    @property
    def x_max(self) -> float:
        return max(p[0] for p in self.bbox)

    @property
    def y_min(self) -> float:
        return min(p[1] for p in self.bbox)

    @property
    def y_max(self) -> float:
        return max(p[1] for p in self.bbox)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass
class PageResult:
    page_number: int
    width: int
    height: int
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        return " ".join(b.text for b in self.blocks)


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    """Rasterize every page of a PDF to a PIL image (EasyOCR needs pixels, not vector text)."""
    from pdf2image import convert_from_bytes

    return convert_from_bytes(pdf_bytes, dpi=dpi)


def load_document_images(file_bytes: bytes, content_type: str) -> list[Image.Image]:
    if content_type == "application/pdf":
        return pdf_to_images(file_bytes)
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def extract_page(image: Image.Image, page_number: int) -> PageResult:
    reader = get_reader()
    np_image = np.array(image)
    results = reader.readtext(np_image)  # [(bbox, text, confidence), ...]
    blocks = [
        TextBlock(
            text=text,
            bbox=[[float(x), float(y)] for x, y in bbox],
            confidence=float(conf),
        )
        for bbox, text, conf in results
    ]
    width, height = image.size
    return PageResult(page_number=page_number, width=width, height=height, blocks=blocks)


def extract_document(file_bytes: bytes, content_type: str) -> list[PageResult]:
    """End-to-end: raw file bytes in, per-page OCR results with bounding boxes out."""
    images = load_document_images(file_bytes, content_type)
    return [extract_page(img, i + 1) for i, img in enumerate(images)]
