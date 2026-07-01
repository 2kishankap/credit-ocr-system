"""FastAPI application entry point."""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router
from app.config import get_settings
from app.database import init_db

settings = get_settings()
logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Credit OCR System",
    description="OCR + local-LLM pipeline for extracting and validating credit request document data.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    """Minimal upload + live-status page. Not part of the JSON API -- see /docs for that."""
    return FileResponse(STATIC_DIR / "index.html")


app.include_router(router)
