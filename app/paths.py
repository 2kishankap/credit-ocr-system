"""
Stage-based blob storage layout.

Every artifact produced while processing a document is written under a
per-document prefix, split by processing stage:

    documents/{document_id}/raw/{filename}          original upload
    documents/{document_id}/ocr/page-{n}.json        EasyOCR blocks + layout text
    documents/{document_id}/llm/extraction.json      structured fields from the LLM
    documents/{document_id}/annotated/page-{n}.png   bounding-box overlay images

Keeping this in one module means every service (API, Celery worker,
tests) constructs paths the same way -- there's exactly one place that
knows the folder convention.
"""
import enum


class Stage(str, enum.Enum):
    RAW = "raw"
    OCR = "ocr"
    LLM = "llm"
    ANNOTATED = "annotated"


def document_prefix(document_id: str) -> str:
    return f"documents/{document_id}"


def raw_path(document_id: str, filename: str) -> str:
    return f"{document_prefix(document_id)}/{Stage.RAW.value}/{filename}"


def ocr_page_path(document_id: str, page_number: int) -> str:
    return f"{document_prefix(document_id)}/{Stage.OCR.value}/page-{page_number}.json"


def llm_extraction_path(document_id: str) -> str:
    return f"{document_prefix(document_id)}/{Stage.LLM.value}/extraction.json"


def annotated_page_path(document_id: str, page_number: int) -> str:
    return f"{document_prefix(document_id)}/{Stage.ANNOTATED.value}/page-{page_number}.png"
