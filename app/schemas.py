"""Pydantic request/response models for the public API."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.models.document import DocumentStatus, ValidationStatus


class UploadResponse(BaseModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    task_id: str


class StatusResponse(BaseModel):
    document_id: UUID
    filename: str
    document_type: str
    status: DocumentStatus
    page_count: int
    uploaded_at: datetime
    processed_at: datetime | None
    error_message: str | None


class FieldResult(BaseModel):
    field_name: str
    field_value: str | None
    data_type: str
    confidence: float
    source_page: int | None
    validation_status: ValidationStatus
    validation_notes: str | None


class ReviewResponse(BaseModel):
    document_id: UUID
    filename: str
    document_type: str
    status: DocumentStatus
    overall_validation_status: ValidationStatus | None
    fields: list[FieldResult]
    annotated_page_urls: list[str]
