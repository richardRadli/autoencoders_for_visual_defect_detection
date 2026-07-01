import logging
import os

from celery import Celery

from services.defect_detection.app.core.utility_services.training_service import TrainAutoEncoder


CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis_broker:6379/0")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis_broker:6379/0")

celery_app = Celery("defect_detection_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)

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
    Background autoencoder training task.

    Args:
        self: Bound Celery task instance.
        config: Effective training config from the API.

    Returns:
        dict: The training result summary.
    """
    logging.info("Starting autoencoder training task")
    self.update_state(state="PROGRESS", meta={"status": "Training in progress"})
    try:
        return TrainAutoEncoder(config).fit()
    except Exception as e:
        logging.exception("Training task failed")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise