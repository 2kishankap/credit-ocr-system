"""Unit tests for the stage-based storage path builders (app.paths)."""
from app.paths import annotated_page_path, document_prefix, llm_extraction_path, ocr_page_path, raw_path


def test_all_paths_share_document_prefix():
    doc_id = "abc-123"
    prefix = document_prefix(doc_id)

    assert raw_path(doc_id, "loan.pdf").startswith(prefix)
    assert ocr_page_path(doc_id, 1).startswith(prefix)
    assert llm_extraction_path(doc_id).startswith(prefix)
    assert annotated_page_path(doc_id, 1).startswith(prefix)


def test_paths_are_organized_by_stage():
    doc_id = "abc-123"
    assert raw_path(doc_id, "loan.pdf") == "documents/abc-123/raw/loan.pdf"
    assert ocr_page_path(doc_id, 2) == "documents/abc-123/ocr/page-2.json"
    assert llm_extraction_path(doc_id) == "documents/abc-123/llm/extraction.json"
    assert annotated_page_path(doc_id, 3) == "documents/abc-123/annotated/page-3.png"
