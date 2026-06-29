import os
import logging

from celery import Celery


CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/0")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis_broker:6379/0")

celery_app = Celery(
    "defect_detection_tasks",
    broker=CELERY_BROKER,
    backend=CELERY_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(bind=True)
def train_autoencoder_task(self, config: dict):
    """
    Background autoencoder training task. Skeleton only — the real training
    (model build via NetworkFactory, dataset load, early stopping, per-epoch
    progress, best-weight saving) is brought over from src/training_service.py on Monday.

    Args:
        self: Bound Celery task instance (used for progress updates).
        config: Training configuration (network_type, dataset_type, epochs, ...).

    Returns:
        dict: Result summary once training finishes.
    """
    logging.info("Starting background autoencoder training task")
    self.update_state(state="PROGRESS", meta={"status": "Initializing"})
    return {"status": "SUCCESS", "message": "placeholder — training not implemented yet"}