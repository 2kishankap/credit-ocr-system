"""
Bounding-box visualization for QA review.

Credit officers reviewing an automated extraction need to see, at a
glance, *where on the page* each value came from and how much to trust
it. This module draws two layers on top of the original page image:

  1. All raw OCR blocks, in a light neutral color (shows OCR coverage)
  2. Extracted field bounding boxes, color-coded by validation status
     (green = valid, amber = needs review, red = invalid), labeled with
     the field name

The result is saved to the "annotated" storage stage (see app/paths.py)
so it can be served straight back to the review UI.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from app.ocr.engine import PageResult
from app.models.document import ValidationStatus

OCR_BOX_COLOR = (100, 149, 237)  # cornflower blue, low-emphasis OCR coverage
STATUS_COLORS = {
    ValidationStatus.VALID: (34, 139, 34),        # green
    ValidationStatus.NEEDS_REVIEW: (218, 165, 32),  # amber
    ValidationStatus.INVALID: (200, 30, 30),        # red
}


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_ocr_overlay(image: Image.Image, page: PageResult) -> Image.Image:
    """Layer 1: faint boxes around every OCR-detected text block."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for block in page.blocks:
        polygon = [tuple(point) for point in block.bbox]
        draw.polygon(polygon, outline=OCR_BOX_COLOR, width=1)
    return annotated


def draw_field_overlay(
    image: Image.Image,
    field_boxes: list[dict],
) -> Image.Image:
    """
    Layer 2: highlighted, labeled boxes for each extracted field.

    field_boxes: [{"field_name": str, "bbox": [[x,y]x4], "status": ValidationStatus}, ...]
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font()

    for item in field_boxes:
        bbox = item.get("bbox")
        if not bbox:
            continue
        status = item.get("status", ValidationStatus.NEEDS_REVIEW)
        color = STATUS_COLORS.get(status, OCR_BOX_COLOR)
        polygon = [tuple(point) for point in bbox]
        draw.polygon(polygon, outline=color, width=3)

        x_min = min(p[0] for p in bbox)
        y_min = min(p[1] for p in bbox)
        label = item.get("field_name", "")
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [x_min, max(0, y_min - text_h - 4), x_min + text_w + 6, y_min],
            fill=color,
        )
        draw.text((x_min + 3, max(0, y_min - text_h - 4)), label, fill=(255, 255, 255), font=font)

    return annotated


def annotate_page(image: Image.Image, page: PageResult, field_boxes: list[dict]) -> Image.Image:
    """Composite both overlay layers onto a copy of the original page image."""
    with_ocr = draw_ocr_overlay(image, page)
    return draw_field_overlay(with_ocr, field_boxes)


def image_to_png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
