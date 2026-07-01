"""
LLM-driven structured field extraction for credit documents.

The OCR + spatial-analysis stages produce, per page:
  - layout-preserving text (app.ocr.spatial.render_layout_text)
  - heuristic label/value candidate pairs (app.ocr.spatial.find_label_value_pairs)

This module combines both into a single prompt so the local Llama model
gets the raw words *and* the structural hints, then asks it to return a
fixed JSON schema of credit-evaluation fields. Forcing a fixed schema
(rather than "extract whatever you find") is what makes downstream
business-rule validation possible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.llm.client import OllamaClient, get_ollama_client
from app.ocr.engine import PageResult
from app.ocr.spatial import find_label_value_pairs, render_layout_text

logger = logging.getLogger(__name__)

# The fixed set of fields we try to pull out of every credit request
# document. `data_type` drives both the prompt instructions and the
# business-rule validators in app/validation/rules.py.
CREDIT_FIELD_SCHEMA: dict[str, str] = {
    "applicant_name": "string",
    "date_of_birth": "date",
    "ssn_last_four": "string",
    "phone_number": "string",
    "email": "string",
    "employer_name": "string",
    "annual_income": "currency",
    "monthly_debt_payments": "currency",
    "loan_amount_requested": "currency",
    "loan_purpose": "string",
    "credit_score": "number",
    "property_address": "string",
    "requested_term_months": "number",
    "co_applicant_name": "string",
    "document_date": "date",
}

SYSTEM_PROMPT = """You are a financial document analysis assistant used by a credit union's \
loan processing team. You extract structured data from OCR'd credit request documents \
(loan applications, pay stubs, bank statements, credit reports). You are precise, never \
invent values that are not present in the text, and always respond with a single JSON object."""


@dataclass
class FieldExtraction:
    value: str | None
    confidence: float
    source_page: int | None


def _build_prompt(pages: list[PageResult]) -> str:
    sections = []
    for page in pages:
        layout_text = render_layout_text(page)
        pairs = find_label_value_pairs(page)
        hint_lines = "\n".join(f'  - "{p.label}" -> "{p.value}"' for p in pairs) or "  (none detected)"
        sections.append(
            f"--- Page {page.page_number} (layout-preserving OCR text) ---\n"
            f"{layout_text}\n\n"
            f"--- Page {page.page_number} spatial label/value hints ---\n"
            f"{hint_lines}"
        )
    document_text = "\n\n".join(sections)

    schema_lines = "\n".join(f'  "{name}": "{dtype}"' for name, dtype in CREDIT_FIELD_SCHEMA.items())

    return f"""Extract the following fields from the credit document below. Respond with ONLY a
JSON object shaped exactly like this (types shown are hints about the expected format,
not literal values):

{{
{schema_lines}
}}

Rules:
- If a field is not present in the document, use null for its value.
- "currency" fields: return the numeric amount only (no "$" or commas), as a string.
- "date" fields: return in YYYY-MM-DD format if determinable, else the original text.
- "number" fields: digits only, no formatting.
- Do not invent or infer values that are not supported by the text.
- Additionally include a sibling object called "_confidence" mapping each field name to a
  number between 0.0 and 1.0 representing how certain you are the extracted value is correct
  (0.0 if the field is null).
- Additionally include a sibling object called "_source_page" mapping each non-null field
  name to the page number it was found on.

DOCUMENT:
{document_text}
"""


def extract_fields(pages: list[PageResult], client: OllamaClient | None = None) -> dict[str, FieldExtraction]:
    """Run one LLM call across all pages of a document and return typed field results."""
    client = client or get_ollama_client()
    prompt = _build_prompt(pages)
    logger.info("Requesting structured extraction from %s (%d pages)", client.model, len(pages))
    raw = client.generate_json(prompt, system=SYSTEM_PROMPT)

    confidences = raw.get("_confidence", {}) or {}
    source_pages = raw.get("_source_page", {}) or {}

    results: dict[str, FieldExtraction] = {}
    for field_name in CREDIT_FIELD_SCHEMA:
        value = raw.get(field_name)
        value = None if value in ("", "null", "None") else value
        results[field_name] = FieldExtraction(
            value=str(value) if value is not None else None,
            confidence=float(confidences.get(field_name, 0.0) or 0.0),
            source_page=source_pages.get(field_name),
        )
    return results
