Drop sample credit request PDFs or scanned images (loan applications, pay
stubs, bank statements) here to test the pipeline locally, e.g.:

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_documents/loan_application.pdf" \
  -F "document_type=loan_application"
```

No sample files are bundled with this repository — use your own test
documents (redact or synthesize any real personal/financial data).
