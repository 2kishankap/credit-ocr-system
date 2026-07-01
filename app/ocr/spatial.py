"""
Spatial analysis over EasyOCR output.

Raw OCR gives us an unordered bag of text blocks with bounding boxes.
Credit documents (loan applications, pay stubs, bank statements) are
tabular/form-like, so *where* text sits relative to other text carries
as much meaning as the text itself -- "Loan Amount" next to "$45,000"
is a key-value pair; the same two strings on unrelated lines are not.

This module turns the raw block list into:
  1. Reading-order lines (words on the same visual row, left-to-right)
  2. A layout-preserving text rendering (spacing approximates columns)
  3. Candidate label -> value pairs based on proximity

These outputs are *hints*, not the final answer. They get passed to the
LLM extraction step (app/llm/extraction.py) alongside the raw text so the
model has both the words and their structural relationships to reason
over -- this is what makes extraction reliable on messy scanned forms.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ocr.engine import PageResult, TextBlock

# Fraction of the shorter block's height that two blocks must share
# vertically to be considered part of the same visual line.
LINE_Y_OVERLAP_THRESHOLD = 0.5

# A value candidate must sit within (label_height * this ratio) pixels
# of the label to be considered "nearby" rather than an unrelated block.
LABEL_VALUE_MAX_GAP_RATIO = 4.0


@dataclass
class Line:
    blocks: list[TextBlock]

    @property
    def text(self) -> str:
        return " ".join(b.text for b in sorted(self.blocks, key=lambda b: b.x_min))

    @property
    def y_center(self) -> float:
        return sum(b.center[1] for b in self.blocks) / len(self.blocks)


@dataclass
class LabelValuePair:
    label: str
    value: str
    label_bbox: list
    value_bbox: list
    confidence: float
    page_number: int


def _vertical_overlap(a: TextBlock, b: TextBlock) -> float:
    top = max(a.y_min, b.y_min)
    bottom = min(a.y_max, b.y_max)
    overlap = max(0.0, bottom - top)
    shorter = min(a.y_max - a.y_min, b.y_max - b.y_min) or 1.0
    return overlap / shorter


def group_into_lines(blocks: list[TextBlock]) -> list[Line]:
    """Cluster text blocks into visual rows using vertical bbox overlap."""
    ordered = sorted(blocks, key=lambda b: b.y_min)
    lines: list[Line] = []
    for block in ordered:
        placed = False
        for line in lines:
            if any(_vertical_overlap(block, existing) >= LINE_Y_OVERLAP_THRESHOLD for existing in line.blocks):
                line.blocks.append(block)
                placed = True
                break
        if not placed:
            lines.append(Line(blocks=[block]))
    lines.sort(key=lambda l: l.y_center)
    for line in lines:
        line.blocks.sort(key=lambda b: b.x_min)
    return lines


def render_layout_text(page: PageResult, char_width_px: float = 9.0) -> str:
    """
    Render OCR output as plain text that approximates original column
    spacing, so the LLM sees roughly the layout a human reader would.
    """
    lines = group_into_lines(page.blocks)
    rendered: list[str] = []
    for line in lines:
        pieces = []
        cursor = 0
        for block in line.blocks:
            col = max(0, round(block.x_min / char_width_px))
            pad = max(1, col - cursor)
            pieces.append(" " * pad + block.text)
            cursor = col + len(block.text)
        rendered.append("".join(pieces).rstrip())
    return "\n".join(rendered)


def _looks_like_label(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(":") or (stripped.isupper() and 1 <= len(stripped.split()) <= 5)


def find_label_value_pairs(page: PageResult) -> list[LabelValuePair]:
    """
    Heuristic candidate generator: for every block that looks like a
    field label, find the nearest block to its right (same line) or
    directly below it (next line) and treat that as the value.

    This does not replace LLM extraction -- it feeds it, giving the
    model strong structural hints alongside the raw text so it extracts
    more reliably than it would from a flat text dump.
    """
    lines = group_into_lines(page.blocks)
    pairs: list[LabelValuePair] = []

    for line_idx, line in enumerate(lines):
        for i, block in enumerate(line.blocks):
            if not _looks_like_label(block.text):
                continue
            label_text = block.text.strip().rstrip(":").strip()
            if not label_text:
                continue

            label_height = block.y_max - block.y_min
            max_gap = label_height * LABEL_VALUE_MAX_GAP_RATIO

            # Prefer a value block later on the same line.
            value_block = None
            for candidate in line.blocks[i + 1 :]:
                if candidate.x_min - block.x_max <= max(max_gap, 150):
                    value_block = candidate
                    break

            # Fall back to the nearest roughly-aligned block on the next line.
            if value_block is None and line_idx + 1 < len(lines):
                next_line = lines[line_idx + 1]
                aligned = [c for c in next_line.blocks if abs(c.x_min - block.x_min) <= max(max_gap, 100)]
                if aligned:
                    value_block = min(aligned, key=lambda c: abs(c.x_min - block.x_min))

            if value_block is not None:
                pairs.append(
                    LabelValuePair(
                        label=label_text,
                        value=value_block.text.strip(),
                        label_bbox=block.bbox,
                        value_bbox=value_block.bbox,
                        confidence=min(block.confidence, value_block.confidence),
                        page_number=page.page_number,
                    )
                )
    return pairs
