# Credit OCR System

A local-first, microservices-based pipeline that turns scanned credit request
documents (loan applications, pay stubs, bank statements, credit reports)
into structured, validated, reviewable data — without sending anything to an
external API. Everything runs on your machine: OCR via EasyOCR, field
extraction via a local Llama 3.1 model served by Ollama, and orchestration
via FastAPI + Celery + PostgreSQL + Redis + Azurite.

This README is written as a tutorial: it explains not just how to run the
system, but why each piece exists, so you can extend it confidently.

## What this system does

- Extracts key financial data from PDFs and scanned documents using OCR
- Analyzes information using local AI models (no external APIs, no data
  leaves your machine)
- Validates data across multiple document types with business rules
- Visualizes processing results with bounding-box overlays for quality
  assurance
- Stores results in organized, stage-based storage (raw → OCR → LLM →
  annotated) for easy retrieval and auditing
- Orchestrates the complete document processing workflow with error
  handling, status tracking, and async background processing

**Before:** a loan officer manually reviews a 15–20 page application,
cross-checking figures by hand — often an hour or more per case.
**After:** the system extracts and validates the same data in minutes,
flags anything uncertain for human review, and gives the officer a
bounding-box overlay to sanity-check every number against the source page.

## Architecture

```
                         ┌─────────────┐
   client / review UI ──▶│   FastAPI   │◀── GET /documents/{id}/status
                         │   (api)     │◀── GET /documents/{id}/review
                         └──────┬──────┘◀── GET .../pages/{n}/annotated
                                │ enqueues
                                ▼
                         ┌─────────────┐        ┌──────────┐
                         │    Redis    │◀──────▶│  Celery  │
                         │  (broker)   │        │  worker  │
                         └─────────────┘        └────┬─────┘
                                                      │ runs the pipeline
        ┌─────────────────────────────────────────────┴───────────────────────┐
        ▼                    ▼                    ▼                    ▼
  ┌───────────┐        ┌───────────┐        ┌───────────┐        ┌───────────┐
  │  Azurite  │        │  EasyOCR  │        │  Ollama   │        │PostgreSQL │
  │  (blob    │        │  (OCR +   │        │ (Llama    │        │(metadata +│
  │  storage) │        │  bboxes)  │        │  3.1 8B)  │        │ results)  │
  └───────────┘        └───────────┘        └───────────┘        └───────────┘
```

Every service is a container. `api` and `worker` share one Docker image and
one Python codebase (`app/`) — the API enqueues work, the worker executes it,
and neither duplicates pipeline logic.

## The processing pipeline

Each uploaded document moves through these stages, tracked in
`Document.status` (`app/models/document.py`) so the API can report
real-time progress:

1. **Document Upload** — `POST /documents` stores the raw file in Azurite
   under `documents/{id}/raw/`, creates a `Document` row with metadata, and
   enqueues a Celery task. Status: `uploaded`.
2. **OCR Processing** — `app/ocr/engine.py` rasterizes PDF pages (via
   `pdf2image`/poppler) and runs EasyOCR on each page, producing text blocks
   with 4-point bounding boxes and confidence scores. Status:
   `ocr_in_progress` → `ocr_complete`.
3. **Spatial Analysis** — `app/ocr/spatial.py` groups OCR blocks into
   reading-order lines, renders layout-preserving text, and generates
   candidate label→value pairs by proximity (e.g. `"Loan Amount:"` next to
   `"$45,000"`). This turns an unordered bag of text into something with
   structure the LLM can reason over.
4. **LLM Field Extraction** — `app/llm/extraction.py` sends the layout text
   plus spatial hints to a local Llama 3.1 8B model (via Ollama), asking for
   a fixed JSON schema of credit fields with per-field confidence scores.
   Status: `extraction_in_progress` → `extraction_complete`.
5. **Business Rule Validation** — `app/validation/rules.py` checks every
   extracted field against format rules (SSN, email, currency, dates),
   range rules (credit score 300–850), a confidence floor, and
   document-type-specific required-field lists. Status: `validating` →
   `validation_complete`.
6. **Visualization Generation** — `app/ocr/visualize.py` draws bounding-box
   overlays: all OCR coverage in blue, and each extracted field boxed and
   labeled in green/amber/red by validation status. Status: `annotating` →
   `complete`.
7. **Stage-Based Storage** — every artifact (raw file, OCR JSON per page,
   LLM extraction JSON, annotated PNGs) is written to Azurite under
   `documents/{id}/{stage}/...` (see `app/paths.py`), so you can inspect any
   intermediate stage independently.
8. **User Review** — `GET /documents/{id}/review` returns every extracted
   field with its value, confidence, validation status/notes, and links to
   the annotated page images.

If any stage raises, the task catches it, sets `status = failed` with the
error message recorded on the document, and re-raises so Celery's retry/
dead-letter handling still applies — a document never gets silently stuck.

## Project structure

```
credit-ocr-system/
├── app/
│   ├── main.py              FastAPI app entrypoint
│   ├── config.py            Environment-driven settings (pydantic-settings)
│   ├── database.py          SQLAlchemy engine/session (lazy-initialized)
│   ├── storage.py           Azurite/Blob Storage wrapper
│   ├── paths.py             Stage-based blob path builders
│   ├── celery_app.py        Celery app + Redis broker config
│   ├── tasks.py             process_document: the full pipeline as one task
│   ├── schemas.py           Pydantic request/response models
│   ├── api/routes.py        HTTP routes (upload, status, review, annotated image)
│   ├── models/document.py   SQLAlchemy models: Document, OcrPage, ExtractedField
│   ├── ocr/
│   │   ├── engine.py        EasyOCR wrapper (PDF → images → text + bboxes)
│   │   ├── spatial.py       Line grouping, layout text, label/value pairing
│   │   └── visualize.py     Bounding-box overlay rendering
│   ├── llm/
│   │   ├── client.py        Ollama HTTP client (JSON-mode generation)
│   │   └── extraction.py    Prompt construction + structured field parsing
│   └── validation/
│       └── rules.py         Business rules engine
├── tests/                   Unit tests (no Docker required)
├── docker-compose.yml       Full local stack
├── Dockerfile                Shared image for api + worker
├── requirements.txt
└── .env.example
```

## Prerequisites

- Docker and Docker Compose
- ~8GB free disk space (Llama 3.1 8B model weights are ~4.7GB)
- A machine with at least 8GB RAM free for the `ollama` container (CPU
  inference works but is slow; a GPU-enabled Docker setup will be much
  faster if you have one)

## Getting started

```bash
cd credit-ocr-system
cp .env.example .env

docker compose up --build
```

On first startup:
- `postgres`, `redis`, and `azurite` come up immediately
- `ollama` starts, then `ollama-init` pulls `llama3.1:8b` into a shared
  volume (only happens once — subsequent `docker compose up` runs skip the
  download since the model is already cached)
- `api` and `worker` wait for their dependencies to report healthy before
  starting

The API is now available at `http://localhost:8000`. Interactive API docs
(Swagger UI) are at `http://localhost:8000/docs`.

## Usage walkthrough

**1. Upload a document**

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_documents/loan_application.pdf" \
  -F "document_type=loan_application"
```

```json
{
  "document_id": "b3f1c2a4-...",
  "filename": "loan_application.pdf",
  "status": "uploaded",
  "task_id": "9e21..."
}
```

`document_type` drives which fields are required during validation
(see `REQUIRED_FIELDS_BY_DOCUMENT_TYPE` in `app/validation/rules.py`).
Supported out of the box: `loan_application`, `pay_stub`, `bank_statement`,
`credit_report`, or `unknown`.

**2. Poll processing status**

```bash
curl http://localhost:8000/documents/b3f1c2a4-.../status
```

```json
{
  "document_id": "b3f1c2a4-...",
  "status": "extraction_in_progress",
  "page_count": 4,
  "uploaded_at": "2026-07-01T14:02:11Z",
  "processed_at": null,
  "error_message": null
}
```

Watch `status` move through `ocr_in_progress` → `ocr_complete` →
`extraction_in_progress` → `extraction_complete` → `validating` →
`validation_complete` → `annotating` → `complete` (or `failed`).

**3. Review the extracted, validated fields**

```bash
curl http://localhost:8000/documents/b3f1c2a4-.../review
```

```json
{
  "document_id": "b3f1c2a4-...",
  "status": "complete",
  "overall_validation_status": "needs_review",
  "fields": [
    {
      "field_name": "annual_income",
      "field_value": "85000",
      "confidence": 0.88,
      "source_page": 1,
      "validation_status": "valid",
      "validation_notes": null
    },
    {
      "field_name": "credit_score",
      "field_value": "742",
      "confidence": 0.55,
      "validation_status": "needs_review",
      "validation_notes": "model confidence 0.55 below auto-approve threshold (0.60)"
    }
  ],
  "annotated_page_urls": ["/documents/b3f1c2a4-.../pages/1/annotated"]
}
```

**4. View the bounding-box overlay**

```bash
curl http://localhost:8000/documents/b3f1c2a4-.../pages/1/annotated -o page1.png
```

Open `page1.png`: OCR coverage is outlined in blue, and each extracted
field is boxed and labeled in green (valid), amber (needs review), or red
(invalid) — so a credit officer can visually confirm a number against the
source document in seconds.

## How the spatial analysis works

Raw OCR output is an unordered list of `(text, bounding_box, confidence)`
tuples. Two problems fall out of that immediately: words on the same visual
line arrive in no particular order, and there's no notion of "this label
goes with that value." `app/ocr/spatial.py` solves both:

- **`group_into_lines`** clusters blocks into rows using vertical bounding-box
  overlap (not exact y-equality, since real OCR output is never perfectly
  aligned), then sorts each row left-to-right.
- **`render_layout_text`** turns those rows into plain text where horizontal
  spacing approximates the original column positions — giving the LLM a
  page that reads the way a human would scan it, rather than a flat word
  soup.
- **`find_label_value_pairs`** looks for blocks that look like field labels
  (end in `:`, or short all-caps phrases) and finds the nearest value
  candidate — either later on the same line, or aligned on the next line
  (common in forms where the value sits below the label). These pairs are
  passed to the LLM as structural hints alongside the raw text, which is
  what makes extraction reliable on messy, real-world scanned forms rather
  than clean typed text.

## How validation works

`app/validation/rules.py` applies deterministic rules on top of the LLM's
output so a "confident-sounding" wrong answer doesn't slip through silently:

- **Format checks** — currency amounts must parse as non-negative numbers,
  dates must be `YYYY-MM-DD`, SSN-last-4 must be 4 digits, emails/phone
  numbers must match basic patterns.
- **Range checks** — e.g. `credit_score` must fall between 300 and 850.
- **Confidence floor** — any field below `MIN_AUTO_APPROVE_CONFIDENCE`
  (default 0.6) is flagged `needs_review` even if it's well-formatted.
- **Required fields by document type** — a `loan_application` requires
  `applicant_name`, `loan_amount_requested`, `annual_income`, and
  `loan_purpose`; a `pay_stub` requires different fields. Missing a required
  field marks the whole document `invalid`.

Every failure carries a human-readable note (`validation_notes`) so a
reviewer sees *why* a field was flagged, not just a red flag with no
explanation.

## Extending the system

- **Add a new document type**: add an entry to
  `REQUIRED_FIELDS_BY_DOCUMENT_TYPE` in `app/validation/rules.py`.
- **Add a new field to extract**: add it to `CREDIT_FIELD_SCHEMA` in
  `app/llm/extraction.py` with its data type; add any field-specific
  validation to `_validate_field_specific` in `app/validation/rules.py`.
- **Swap the OCR engine**: only `app/ocr/engine.py` talks to EasyOCR
  directly — replace `get_reader`/`extract_page` to use PaddleOCR or another
  engine and the rest of the pipeline is unaffected as long as you keep
  returning `TextBlock`/`PageResult`.
- **Swap the LLM**: `app/llm/client.py` is the only place that knows about
  Ollama's HTTP API — point `OLLAMA_MODEL` at any model you've pulled
  (`ollama pull mistral`, etc.) as long as it supports JSON-mode generation.
- **Move to real Azure Blob Storage**: change
  `AZURE_STORAGE_CONNECTION_STRING` in `.env` to a real Azure connection
  string — `app/storage.py` doesn't know or care that it was talking to an
  emulator.

## Running tests

Unit tests cover the pure logic (spatial analysis, validation rules, path
building, and LLM response parsing against a stubbed client) and don't
require Docker, EasyOCR, or a running Ollama server:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Troubleshooting

- **`worker` keeps restarting on first boot** — it's waiting on
  `ollama-init` to finish pulling the model; check
  `docker compose logs ollama-init`.
- **OCR is slow** — EasyOCR runs on CPU by default (`EASYOCR_GPU=false` in
  `.env`). If you have an NVIDIA GPU and the NVIDIA Container Toolkit
  installed, set it to `true` and add GPU reservations to the `worker`
  service in `docker-compose.yml`.
- **A document is stuck in `failed`** — check
  `error_message` from `GET /documents/{id}/status`, then
  `docker compose logs worker` for the full traceback.
- **LLM returns malformed JSON** — `OllamaClient._parse_json` trims
  anything outside the outermost `{ }` as a fallback; if extraction still
  fails repeatedly, try a larger/less quantized model or lower
  `temperature` further (already 0.0 by default).
