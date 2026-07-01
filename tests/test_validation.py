"""Unit tests for the business-rule validation engine (app.validation.rules)."""
from app.llm.extraction import FieldExtraction
from app.models.document import ValidationStatus
from app.validation.rules import MIN_AUTO_APPROVE_CONFIDENCE, validate_document


def fe(value, confidence=0.9, source_page=1) -> FieldExtraction:
    return FieldExtraction(value=value, confidence=confidence, source_page=source_page)


def test_valid_loan_application_passes():
    extracted = {
        "applicant_name": fe("Jane Doe"),
        "loan_amount_requested": fe("45000"),
        "annual_income": fe("85000"),
        "loan_purpose": fe("Home renovation"),
        "credit_score": fe("720"),
    }
    result = validate_document(extracted, document_type="loan_application")

    assert result.missing_required_fields == []
    assert result.overall_status == ValidationStatus.VALID
    assert result.field_results["credit_score"].status == ValidationStatus.VALID


def test_missing_required_field_marks_invalid():
    extracted = {
        "applicant_name": fe("Jane Doe"),
        "annual_income": fe("85000"),
        "loan_purpose": fe(None),  # explicitly not found
    }
    result = validate_document(extracted, document_type="loan_application")

    assert "loan_amount_requested" in result.missing_required_fields
    assert "loan_purpose" in result.missing_required_fields
    assert result.overall_status == ValidationStatus.INVALID


def test_credit_score_out_of_range_is_invalid():
    extracted = {
        "applicant_name": fe("Jane Doe"),
        "credit_score": fe("910"),  # above the 300-850 scale
    }
    result = validate_document(extracted, document_type="credit_report")

    assert result.field_results["credit_score"].status == ValidationStatus.INVALID
    assert "300-850" in result.field_results["credit_score"].notes


def test_malformed_currency_is_invalid():
    extracted = {
        "applicant_name": fe("Jane Doe"),
        "annual_income": fe("not-a-number"),
    }
    result = validate_document(extracted, document_type="pay_stub")
    assert result.field_results["annual_income"].status == ValidationStatus.INVALID


def test_low_confidence_forces_needs_review():
    below_threshold = MIN_AUTO_APPROVE_CONFIDENCE - 0.1
    extracted = {
        "applicant_name": fe("Jane Doe", confidence=below_threshold),
    }
    result = validate_document(extracted, document_type="unknown")
    assert result.field_results["applicant_name"].status == ValidationStatus.NEEDS_REVIEW


def test_null_optional_field_is_valid_not_penalized():
    extracted = {
        "applicant_name": fe("Jane Doe"),
        "co_applicant_name": fe(None),  # optional field, not required anywhere
    }
    result = validate_document(extracted, document_type="unknown")
    assert result.field_results["co_applicant_name"].status == ValidationStatus.VALID
