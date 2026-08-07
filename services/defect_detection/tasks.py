import logging
import os

from celery import Celery

from services.defect_detection.app.core.utility_services.testing_service import TestAutoEncoder
from services.defect_detection.app.core.utility_services.training_service import TrainAutoEncoder
from services.defect_detection.app.core.utility_services.tuning_service import TuneAutoEncoder

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


def _progress_reporter(task):
    """
    Build a progress callback that forwards processed-item counts to Celery.

    The returned callable matches the progress_callback signature used by the
    training and testing services, and reports each update as a PROGRESS state
    with a {current, total, phase} meta the frontend renders as a progress bar.

    Args:
        task: The bound Celery task whose state should carry the progress.

    Returns:
        Callable(current, total, phase): Reports one PROGRESS state update.
    """
    def report(current: int, total: int, phase: str) -> None:
        task.update_state(
            state="PROGRESS",
            meta={"current": current, "total": total, "phase": phase},
        )

    return report


@celery_app.task(bind=True)
def train_autoencoder_task(self, config: dict):
    """
    Background autoencoder training task.

    Args:
        self: Bound Celery task instance.
        config: Effective training config from the API.

    Returns:
        dict: The training result summary (status, best valid loss, weights path).
    """
    logging.info("Starting autoencoder training task")
    self.update_state(
        state="PROGRESS",
        meta={
            "status": "Training in progress",
            "current": 0,
            "total": config["epochs"],
            "phase": "epochs",
        },
    )
    return TrainAutoEncoder(config).fit(progress_callback=_progress_reporter(self))

@celery_app.task(bind=True)
def test_autoencoder_task(self, config: dict):
    """
    Background autoencoder evaluation task.

    Args:
        self: Bound Celery task instance.
        config: Effective testing config from the API.

    Returns:
        dict: The evaluation result summary (metrics and output paths).
    """
    logging.info("Starting autoencoder testing task")
    self.update_state(
        state="PROGRESS",
        meta={"current": 0, "total": 100, "phase": "starting"},
    )
    return TestAutoEncoder(config).run(progress_callback=_progress_reporter(self))

@celery_app.task(bind=True)
def tune_autoencoder_task(self, config: dict):
    """
    Background Optuna hyperparameter tuning task.

    Args:
        self: Bound Celery task instance.
        config: Effective tuning config from the API.

    Returns:
        dict: The tuning result (status, best params, best valid loss).
    """
    logging.info("Starting autoencoder tuning task")
    self.update_state(
        state="PROGRESS",
        meta={
            "status": "Tuning in progress",
            "current": 0,
            "total": config["n_trials"] * config["epochs_per_trial"],
            "phase": f"trial 1/{config['n_trials']}",
        },
    )
    return TuneAutoEncoder(config).run(progress_callback=_progress_reporter(self))