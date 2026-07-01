"""
Celery application factory.

Redis backs both the task broker (queueing) and the result backend
(status/return values). `task_acks_late` + `worker_prefetch_multiplier=1`
mean a worker that crashes mid-document doesn't silently drop the job --
it gets redelivered to another worker instead.
"""
from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "credit_ocr",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=60 * 60 * 24,  # 24h
)
