"""
Blob storage wrapper around Azurite (the local Azure Blob Storage emulator).

Keeping this behind a small interface means the rest of the codebase never
talks to azure-storage-blob directly -- swapping Azurite for real Azure
Blob Storage in production is a one-line connection-string change.
"""
import io
import logging

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BlobStorage:
    def __init__(self) -> None:
        self._client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        self._container_name = settings.azure_container_name
        self._ensure_container()

    def _ensure_container(self) -> None:
        try:
            self._client.create_container(self._container_name)
            logger.info("Created blob container '%s'", self._container_name)
        except ResourceExistsError:
            pass

    def upload(self, blob_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        container = self._client.get_container_client(self._container_name)
        container.upload_blob(
            name=blob_path,
            data=io.BytesIO(data),
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type) if content_type else None,
        )
        return blob_path

    def download(self, blob_path: str) -> bytes:
        container = self._client.get_container_client(self._container_name)
        return container.download_blob(blob_path).readall()

    def delete(self, blob_path: str) -> None:
        container = self._client.get_container_client(self._container_name)
        container.delete_blob(blob_path)


_storage_singleton: BlobStorage | None = None


def get_storage() -> BlobStorage:
    """Lazily construct the BlobStorage client (avoids connecting at import time)."""
    global _storage_singleton
    if _storage_singleton is None:
        _storage_singleton = BlobStorage()
    return _storage_singleton