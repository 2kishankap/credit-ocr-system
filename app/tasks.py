"""
End-to-end document processing pipeline, run as a Celery background task.

    raw storage -> OCR -> spatial analysis -> LLM extraction
        -> business-rule validation -> bounding-box annotation
        -> stage-based storage, with a Document.status update after
           every stage so the API can report real-time progress.

Each stage is wrapped so a failure anywhere marks the document FAILED
with a recorded error message instead of leaving it stuck "in progress"
forever -- credit officers need to know a document needs manual
handling, not just that nothing happened yet.
"""
from __future__ import annotations

import json
import logging
import uuid

from app.celery_app import celery_app
from app.database import get_session
from app.llm.extraction import extract_fields
from app.models.document import Document, DocumentStatus, ExtractedField, OcrPage, ValidationStatus
from app.ocr.engine import extract_document, load_document_images
from app.ocr.spatial import find_label_value_pairs
from app.ocr.visualize import annotate_page, image_to_png_bytes
from app.paths import annotated_page_path, llm_extraction_path, ocr_page_path
from app.storage import get_storage
from app.validation.rules import validate_document

logger = logging.getLogger(__name__)


def _match_bbox_for_value(value: str, pairs: list) -> list | None:
    """
    Best-effort visual grounding: the LLM extracts values from text, not
    pixels, so it never returns a bbox. We recover an approximate one by
    matching the extracted value against the spatial label/value hints
    generated for that page (see app.ocr.spatial.find_label_value_pairs).
    A normalized substring match is enough for the QA overlay use case --
    it doesn't need to be pixel-perfect, just close enough to point a
    reviewer at the right area of the page.
    """
    if not value:
        return None
    normalized_value = value.strip().lower()
    for pair in pairs:
        candidate = pair.value.strip().lower()
        if not candidate:
            continue
        if normalized_value == candidate or normalized_value in candidate or candidate in normalized_value:
            return pair.value_bbox
    return None


def _set_status(document_id: str, status: DocumentStatus, error_message: str | None = None) -> None:
    with get_session() as session:
        document = session.get(Document, uuid.UUID(document_id))
        if document is None:
            logger.warning("Document %s not found while setting status %s", document_id, status)
            return
        document.status = status
        if error_message is not None:
            document.error_message = error_message


@celery_app.task(name="app.tasks.process_document", bind=True, max_retries=1)
def process_document(self, document_id: str) -> dict:
    storage = get_storage()

    try:
        with get_session() as session:
            document = session.get(Document, uuid.UUID(document_id))
            if document is None:
                raise ValueError(f"Document {document_id} does not exist")
            blob_path = document.blob_path
            content_type = document.content_type
            document_type = document.document_type

        # --- Stage 1: OCR -----------------------------------------------
        _set_status(document_id, DocumentStatus.OCR_IN_PROGRESS)
        file_bytes = storage.download(blob_path)
        pages = extract_document(file_bytes, content_type)
        logger.info("Document %s: OCR extracted %d page(s)", document_id, len(pages))

        with get_session() as session:
            document = session.get(Document, uuid.UUID(document_id))
            document.page_count = len(pages)
            for page in pages:
                storage.upload(
                    ocr_page_path(document_id, page.page_number),
                    json.dumps(
                        {"page_number": page.page_number, "blocks": [b.to_dict() for b in page.blocks]}
                    ).encode("utf-8"),
                    content_type="application/json",
                )
                session.add(
                    OcrPage(
                        document_id=document.id,
                        page_number=page.page_number,
                        raw_text=page.raw_text,
                        blocks=[b.to_dict() for b in page.blocks],
                        width=page.width,
                        height=page.height,
                    )
                )
        _set_status(document_id, DocumentStatus.OCR_COMPLETE)

        # --- Stage 2: LLM structured extraction --------------------------
        _set_status(document_id, DocumentStatus.EXTRACTION_IN_PROGRESS)
        extracted = extract_fields(pages)
        storage.upload(
            llm_extraction_path(document_id),
            json.dumps(
                {name: {"value": fe.value, "confidence": fe.confidence, "source_page": fe.source_page}
                 for name, fe in extracted.items()}
            ).encode("utf-8"),
            content_type="application/json",
        )
        _set_status(document_id, DocumentStatus.EXTRACTION_COMPLETE)

        # --- Stage 3: business rule validation ---------------------------
        _set_status(document_id, DocumentStatus.VALIDATING)
        validation = validate_document(extracted, document_type=document_type)

        field_rows = []
        with get_session() as session:
            document = session.get(Document, uuid.UUID(document_id))
            for field_name, field_extraction in extracted.items():
                fv = validation.field_results[field_name]
                row = ExtractedField(
                    document_id=document.id,
                    field_name=field_name,
                    field_value=field_extraction.value,
                    confidence=field_extraction.confidence,
                    source_page=field_extraction.source_page,
                    validation_status=fv.status,
                    validation_notes=fv.notes,
                )
                session.add(row)
                field_rows.append(
                    {
                        "field_name": field_name,
                        "value": field_extraction.value,
                        "source_page": field_extraction.source_page,
                        "status": fv.status,
                    }
                )
        _set_status(document_id, DocumentStatus.VALIDATION_COMPLETE)

        # --- Stage 4: bounding-box annotation for QA review --------------
        # Note: extracted field values from the LLM don't carry a precise
        # bbox (the LLM reasons over text, not pixels), so annotation here
        # highlights the spatial label/value hints that best match each
        # field's page -- giving reviewers a fast visual sanity check even
        # though it's an approximation, not a pixel-exact citation.
        _set_status(document_id, DocumentStatus.ANNOTATING)
        images = load_document_images(file_bytes, content_type)
        for page, image in zip(pages, images):
            page_pairs = find_label_value_pairs(page)
            page_field_boxes = []
            for fr in field_rows:
                if fr["source_page"] != page.page_number:
                    continue
                bbox = _match_bbox_for_value(fr["value"], page_pairs)
                if bbox:
                    page_field_boxes.append({"field_name": fr["field_name"], "bbox": bbox, "status": fr["status"]})
            annotated_image = annotate_page(image, page, page_field_boxes)
            storage.upload(
                annotated_page_path(document_id, page.page_number),
                image_to_png_bytes(annotated_image),
                content_type="image/png",
            )

        _set_status(document_id, DocumentStatus.COMPLETE)
        with get_session() as session:
            document = session.get(Document, uuid.UUID(document_id))
            from datetime import datetime

            document.processed_at = datetime.utcnow()

        return {"document_id": document_id, "status": DocumentStatus.COMPLETE.value, "pages": len(pages)}

    except Exception as exc:  # noqa: BLE001 -- pipeline boundary, must not raise unhandled
        logger.exception("Processing failed for document %s", document_id)
        _set_status(document_id, DocumentStatus.FAILED, error_message=str(exc))
        raise
