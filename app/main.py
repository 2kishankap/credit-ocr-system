"""FastAPI application entry point."""
import logging

from fastapi import FastAPI

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


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(router)
