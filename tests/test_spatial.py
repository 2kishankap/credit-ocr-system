"""
Unit tests for spatial analysis (app.ocr.spatial).

These build synthetic TextBlock objects with hand-picked bounding boxes
to exercise line-grouping and label/value pairing without needing
EasyOCR, poppler, or any real document -- the goal is to pin down the
geometric logic in isolation.
"""
from app.ocr.engine import PageResult, TextBlock
from app.ocr.spatial import find_label_value_pairs, group_into_lines, render_layout_text


def make_block(text: str, x: float, y: float, w: float, h: float, confidence: float = 0.95) -> TextBlock:
    """Axis-aligned rectangle bbox in EasyOCR's 4-corner format."""
    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return TextBlock(text=text, bbox=bbox, confidence=confidence)


def test_group_into_lines_separates_rows_and_orders_left_to_right():
    blocks = [
        make_block("World", x=100, y=10, w=40, h=15),
        make_block("Hello", x=10, y=10, w=40, h=15),
        make_block("Row2", x=10, y=50, w=40, h=15),
    ]
    lines = group_into_lines(blocks)

    assert len(lines) == 2
    assert lines[0].text == "Hello World"
    assert lines[1].text == "Row2"


def test_group_into_lines_uses_vertical_overlap_not_exact_y():
    # Slightly offset y-coordinates (as real OCR output is) should still
    # be treated as the same line as long as they substantially overlap.
    blocks = [
        make_block("Loan", x=10, y=10, w=40, h=20),
        make_block("Amount:", x=60, y=14, w=60, h=18),
    ]
    lines = group_into_lines(blocks)
    assert len(lines) == 1
    assert lines[0].text == "Loan Amount:"


def test_render_layout_text_preserves_reading_order():
    blocks = [
        make_block("Name:", x=0, y=0, w=50, h=20),
        make_block("Jane Doe", x=200, y=0, w=90, h=20),
        make_block("Income:", x=0, y=40, w=60, h=20),
        make_block("$85,000", x=200, y=40, w=70, h=20),
    ]
    page = PageResult(page_number=1, width=800, height=600, blocks=blocks)
    text = render_layout_text(page)
    lines = text.splitlines()

    assert len(lines) == 2
    assert "Name:" in lines[0] and "Jane Doe" in lines[0]
    assert lines[0].index("Jane Doe") > lines[0].index("Name:")
    assert "Income:" in lines[1] and "$85,000" in lines[1]


def test_find_label_value_pairs_same_line():
    blocks = [
        make_block("Loan Amount:", x=10, y=10, w=100, h=20),
        make_block("$45,000", x=130, y=10, w=70, h=20),
    ]
    page = PageResult(page_number=1, width=800, height=600, blocks=blocks)
    pairs = find_label_value_pairs(page)

    assert len(pairs) == 1
    assert pairs[0].label == "Loan Amount"
    assert pairs[0].value == "$45,000"
    assert pairs[0].page_number == 1


def test_find_label_value_pairs_next_line_fallback():
    # Form layouts sometimes put the value directly below a label instead
    # of beside it -- the pairing should still find it via x-alignment.
    blocks = [
        make_block("APPLICANT NAME:", x=10, y=10, w=140, h=20),
        make_block("Jane Doe", x=15, y=40, w=80, h=20),
    ]
    page = PageResult(page_number=2, width=800, height=600, blocks=blocks)
    pairs = find_label_value_pairs(page)

    assert len(pairs) == 1
    assert pairs[0].label == "APPLICANT NAME"
    assert pairs[0].value == "Jane Doe"
    assert pairs[0].page_number == 2


def test_find_label_value_pairs_ignores_unrelated_text():
    blocks = [
        make_block("Notes:", x=10, y=10, w=50, h=20),
        make_block("Unrelated text far away", x=600, y=400, w=150, h=20),
    ]
    page = PageResult(page_number=1, width=800, height=600, blocks=blocks)
    pairs = find_label_value_pairs(page)
    assert pairs == []
