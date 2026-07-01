"""
Unit tests for app.llm.extraction using a stub Ollama client, so the
extraction/parsing logic can be verified without a running Ollama
server or model weights.
"""
from app.llm.extraction import CREDIT_FIELD_SCHEMA, extract_fields
from app.ocr.engine import PageResult, TextBlock


class StubOllamaClient:
    """Mimics OllamaClient.generate_json's interface with a canned response."""

    def __init__(self, response: dict):
        self._response = response
        self.model = "stub-model"
        self.last_prompt = None

    def generate_json(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> dict:
        self.last_prompt = prompt
        return self._response


def sample_page() -> PageResult:
    block = TextBlock(text="Applicant: Jane Doe", bbox=[[0, 0], [100, 0], [100, 20], [0, 20]], confidence=0.9)
    return PageResult(page_number=1, width=800, height=600, blocks=[block])


def test_extract_fields_maps_known_schema_fields():
    stub_response = {
        "applicant_name": "Jane Doe",
        "annual_income": "85000",
        "_confidence": {"applicant_name": 0.92, "annual_income": 0.81},
        "_source_page": {"applicant_name": 1, "annual_income": 1},
    }
    client = StubOllamaClient(stub_response)

    results = extract_fields([sample_page()], client=client)

    assert set(results.keys()) == set(CREDIT_FIELD_SCHEMA.keys())
    assert results["applicant_name"].value == "Jane Doe"
    assert results["applicant_name"].confidence == 0.92
    assert results["applicant_name"].source_page == 1
    # fields absent from the stub response should come back as null/0-confidence
    assert results["credit_score"].value is None
    assert results["credit_score"].confidence == 0.0


def test_extract_fields_treats_null_like_strings_as_none():
    stub_response = {"applicant_name": "null", "loan_purpose": "", "_confidence": {}, "_source_page": {}}
    client = StubOllamaClient(stub_response)

    results = extract_fields([sample_page()], client=client)

    assert results["applicant_name"].value is None
    assert results["loan_purpose"].value is None


def test_prompt_includes_schema_and_spatial_hints():
    client = StubOllamaClient({"_confidence": {}, "_source_page": {}})
    extract_fields([sample_page()], client=client)

    assert "applicant_name" in client.last_prompt
    assert "Page 1" in client.last_prompt
