"""
Centralized application configuration.

All settings are loaded from environment variables (see .env.example).
Using pydantic-settings means every value is validated and type-checked
at process startup, so a misconfigured container fails fast instead of
raising a confusing error deep inside a Celery task.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    database_url: str = "postgresql+psycopg2://credit_ocr:credit_ocr_pw@postgres:5432/credit_ocr"

    # Celery / Redis
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # Azurite / Blob storage
    azure_storage_connection_string: str = (
        "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://azurite:10000/devstoreaccount1;"
    )
    azure_container_name: str = "credit-documents"

    # Ollama
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_request_timeout: int = 180

    # OCR
    easyocr_languages: str = "en"
    easyocr_gpu: bool = False

    # App
    log_level: str = "INFO"
    max_upload_mb: int = 25

    @property
    def easyocr_language_list(self) -> list[str]:
        return [lang.strip() for lang in self.easyocr_languages.split(",") if lang.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process and cached (env vars don't change at runtime)."""
    return Settings()
