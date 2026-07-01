"""
Business-rule validation for extracted credit-document fields.

The LLM extraction step (app/llm/extraction.py) gives us a value and a
confidence score per field, but "the model is confident" is not the same
as "this is a usable number for a credit decision." This module applies
deterministic, auditable business rules on top of the LLM output:

  - format checks (SSN, phone, email, dates, currency)
  - range/sanity checks (credit score 300-850, non-negative income, ...)
  - required-field checks that vary by document type
  - a confidence floor below which a field is always flagged for review

Credit officers need to trust *why* a field was flagged, so every rule
failure produces a human-readable note rather than a bare boolean.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from app.llm.extraction import CREDIT_FIELD_SCHEMA, FieldExtraction
from app.models.document import ValidationStatus

# Fields required per document type. Documents outside this map (or typed
# "unknown") fall back to a minimal universal set.
REQUIRED_FIELDS_BY_DOCUMENT_TYPE: dict[str, list[str]] = {
    "loan_application": ["applicant_name", "loan_amount_requested", "annual_income", "loan_purpose"],
    "pay_stub": ["applicant_name", "employer_name", "annual_income"],
    "bank_statement": ["applicant_name", "document_date"],
    "credit_report": ["applicant_name", "credit_score"],
    "unknown": ["applicant_name"],
}

# Below this confidence, a field is always sent for human review even if
# it passes every format/range check.
MIN_AUTO_APPROVE_CONFIDENCE = 0.6

_SSN_LAST_FOUR_RE = re.compile(r"^\d{4}$")
_PHONE_RE = re.compile(r"^[\d\s().+-]{7,20}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class FieldValidation:
    field_name: str
    status: ValidationStatus
    notes: str | None


@dataclass
class DocumentValidation:
    field_results: dict[str, FieldValidation]
    missing_required_fields: list[str]

    @property
    def overall_status(self) -> ValidationStatus:
        if self.missing_required_fields:
            return ValidationStatus.INVALID
        if any(fv.status == ValidationStatus.INVALID for fv in self.field_results.values()):
            return ValidationStatus.INVALID
        if any(fv.status == ValidationStatus.NEEDS_REVIEW for fv in self.field_results.values()):
            return ValidationStatus.NEEDS_REVIEW
        return ValidationStatus.VALID


def _validate_currency(value: str) -> str | None:
    cleaned = value.replace(",", "").replace("$", "").strip()
    try:
        amount = float(cleaned)
    except ValueError:
        return f"'{value}' is not a valid currency amount"
    if amount < 0:
        return "amount cannot be negative"
    if amount > 100_000_000:
        return "amount exceeds a plausible range and should be reviewed"
    return None


def _validate_number(value: str) -> str | None:
    try:
        float(value)
    except ValueError:
        return f"'{value}' is not numeric"
    return None


def _validate_date(value: str) -> str | None:
    if not _DATE_RE.match(value):
        return f"'{value}' is not in YYYY-MM-DD format"
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return f"'{value}' is not a real calendar date"
    return None


def _validate_field_specific(field_name: str, value: str) -> str | None:
    if field_name == "credit_score":
        try:
            score = float(value)
        except ValueError:
            return f"'{value}' is not numeric"
        if not (300 <= score <= 850):
            return f"credit score {score:g} is outside the valid 300-850 range"
    elif field_name == "ssn_last_four" and not _SSN_LAST_FOUR_RE.match(value):
        return f"'{value}' is not exactly 4 digits"
    elif field_name == "phone_number" and not _PHONE_RE.match(value):
        return f"'{value}' does not look like a phone number"
    elif field_name == "email" and not _EMAIL_RE.match(value):
        return f"'{value}' does not look like a valid email address"
    return None


def _validate_value(field_name: str, data_type: str, value: str) -> str | None:
    """Returns an error message string, or None if the value passes all checks."""
    type_error = {
        "currency": _validate_currency,
        "number": _validate_number,
        "date": _validate_date,
    }.get(data_type, lambda _v: None)(value)
    if type_error:
        return type_error
    return _validate_field_specific(field_name, value)


def validate_document(
    extracted: dict[str, FieldExtraction],
    document_type: str = "unknown",
) -> DocumentValidation:
    """Apply business rules to every extracted field plus required-field checks."""
    required = REQUIRED_FIELDS_BY_DOCUMENT_TYPE.get(document_type, REQUIRED_FIELDS_BY_DOCUMENT_TYPE["unknown"])

    field_results: dict[str, FieldValidation] = {}
    for field_name, extraction in extracted.items():
        data_type = CREDIT_FIELD_SCHEMA.get(field_name, "string")

        if extraction.value is None:
            status = ValidationStatus.NEEDS_REVIEW if field_name in required else ValidationStatus.VALID
            notes = "required field missing" if field_name in required else None
            field_results[field_name] = FieldValidation(field_name, status, notes)
            continue

        error = _validate_value(field_name, data_type, extraction.value)
        if error:
            field_results[field_name] = FieldValidation(field_name, ValidationStatus.INVALID, error)
        elif extraction.confidence < MIN_AUTO_APPROVE_CONFIDENCE:
            field_results[field_name] = FieldValidation(
                field_name,
                ValidationStatus.NEEDS_REVIEW,
                f"model confidence {extraction.confidence:.2f} below auto-approve threshold "
                f"({MIN_AUTO_APPROVE_CONFIDENCE:.2f})",
            )
        else:
            field_results[field_name] = FieldValidation(field_name, ValidationStatus.VALID, None)

    missing_required = [
        name for name in required if extracted.get(name) is None or extracted[name].value is None
    ]

    return DocumentValidation(field_results=field_results, missing_required_fields=missing_required)
