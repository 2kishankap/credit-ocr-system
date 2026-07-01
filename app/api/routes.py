"""
Public HTTP API.

Deliberately thin: every route validates input, talks to the DB/storage,
and (for uploads) enqueues a Celery task. No OCR, LLM, or validation
logic lives here -- that all belongs to the pipeline modules under
app/ocr, app/llm, and app/validation, exercised end-to-end by
app/tasks.process_document.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.document import Document, DocumentStatus, ExtractedField
from app.paths import annotated_page_path, raw_path
from app.schemas import FieldResult, ReviewResponse, StatusResponse, UploadResponse
from app.storage import get_storage
from app.tasks import process_document

router = APIRouter()
settings = get_settings()

SUPPORTED_CONTENT_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/tiff"}


@router.post("/documents", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form("unknown"),
    db: Session = Depends(get_db),
) -> UploadResponse:
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. Supported: {sorted(SUPPORTED_CONTENT_TYPES)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit")

    document = Document(
        filename=file.filename,
        content_type=file.content_type,
        blob_path="",  # set below once we know the document id
        document_type=document_type,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    db.flush()  # assigns document.id without committing

    blob_path = raw_path(str(document.id), file.filename)
    get_storage().upload(blob_path, contents, content_type=file.content_type)
    document.blob_path = blob_path
    db.commit()
    db.refresh(document)

    async_result = process_document.delay(str(document.id))

    return UploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        task_id=async_result.id,
    )


@router.get("/documents/{document_id}/status", response_model=StatusResponse)
def get_status(document_id: uuid.UUID, db: Session = Depends(get_db)) -> StatusResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return StatusResponse(
        document_id=document.id,
        filename=document.filename,
        document_type=document.document_type,
        status=document.status,
        page_count=document.page_count,
        uploaded_at=document.uploaded_at,
        processed_at=document.processed_at,
        error_message=document.error_message,
    )


@router.get("/documents/{document_id}/review", response_model=ReviewResponse)
def get_review(document_id: uuid.UUID, db: Session = Depends(get_db)) -> ReviewResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    fields = (
        db.query(ExtractedField)
        .filter(ExtractedField.document_id == document_id)
        .order_by(ExtractedField.field_name)
        .all()
    )
    field_results = [
        FieldResult(
            field_name=f.field_name,
            field_value=f.field_value,
            data_type=f.data_type,
            confidence=f.confidence,
            source_page=f.source_page,
            validation_status=f.validation_status,
            validation_notes=f.validation_notes,
        )
        for f in fields
    ]

    overall_status = None
    if field_results:
        statuses = {f.validation_status for f in field_results}
        if "invalid" in statuses:
            overall_status = "invalid"
        elif "needs_review" in statuses:
            overall_status = "needs_review"
        else:
            overall_status = "valid"

    annotated_urls = [
        f"/documents/{document_id}/pages/{page_num}/annotated" for page_num in range(1, document.page_count + 1)
    ]

    return ReviewResponse(
        document_id=document.id,
        filename=document.filename,
        document_type=document.document_type,
        status=document.status,
        overall_validation_status=overall_status,
        fields=field_results,
        annotated_page_urls=annotated_urls,
    )


@router.get("/documents/{document_id}/pages/{page_number}/annotated")
def get_annotated_page(document_id: uuid.UUID, page_number: int, db: Session = Depends(get_db)) -> Response:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        image_bytes = get_storage().download(annotated_page_path(str(document_id), page_number))
    except Exception as exc:  # blob not found / not yet generated
        raise HTTPException(status_code=404, detail="Annotated page not available yet") from exc
    return Response(content=image_bytes, media_type="image/png")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
