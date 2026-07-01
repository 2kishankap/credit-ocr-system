"""
Core data model for the OCR credit-evaluation pipeline.

Document          -- one row per uploaded file (PDF or image)
OcrPage           -- one row per page, holding raw OCR text + word-level
                      bounding boxes produced by EasyOCR
ExtractedField    -- one row per structured field the LLM pulled out of a
                      document (applicant name, income, loan amount, ...),
                      including a validation verdict
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DocumentStatus(str, enum.Enum):
    """
    Ordered pipeline stages. The worker updates `Document.status` after each
    stage completes so the API can expose real-time progress without polling
    Celery directly (see app/tasks.py and the /documents/{id}/status route).
    """

    UPLOADED = "uploaded"              # raw file stored, task enqueued
    OCR_IN_PROGRESS = "ocr_in_progress"
    OCR_COMPLETE = "ocr_complete"      # text + bounding boxes persisted
    EXTRACTION_IN_PROGRESS = "extraction_in_progress"
    EXTRACTION_COMPLETE = "extraction_complete"  # LLM structured fields persisted
    VALIDATING = "validating"
    VALIDATION_COMPLETE = "validation_complete"  # business rules applied
    ANNOTATING = "annotating"
    COMPLETE = "complete"              # annotated overlays generated, ready for review
    FAILED = "failed"


class ValidationStatus(str, enum.Enum):
    VALID = "valid"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    blob_path: Mapped[str] = mapped_column(String(512), nullable=False)
    document_type: Mapped[str] = mapped_column(String(64), default="unknown")  # loan_application, pay_stub, bank_statement, ...
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"), default=DocumentStatus.UPLOADED, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pages: Mapped[list["OcrPage"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="OcrPage.page_number"
    )
    fields: Mapped[list["ExtractedField"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class OcrPage(Base):
    __tablename__ = "ocr_pages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    # Each block: {"text": str, "bbox": [[x,y],[x,y],[x,y],[x,y]], "confidence": float}
    blocks: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped["Document"] = relationship(back_populates="pages")


class ExtractedField(Base):
    __tablename__ = "extracted_fields"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    field_name: Mapped[str] = mapped_column(String(128), nullable=False)
    field_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_type: Mapped[str] = mapped_column(String(32), default="string")  # string|number|date|currency
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_bbox: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    validation_status: Mapped[ValidationStatus] = mapped_column(
        Enum(ValidationStatus, name="validation_status"), default=ValidationStatus.NEEDS_REVIEW
    )
    validation_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="fields")
