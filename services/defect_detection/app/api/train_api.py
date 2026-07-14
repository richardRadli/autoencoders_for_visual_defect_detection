import logging

from enum import Enum
from fastapi import APIRouter, HTTPException, Query, status

from services.defect_detection.app.core.utility_services.config_service.training_config_service import TrainingConfigService
from shared.core.enums import DatasetType
from shared.core.path_bindings import config_paths
from services.defect_detection.tasks import celery_app, train_autoencoder_task



train_router = APIRouter(
    prefix="/train",
    tags=["Training"],
)


class AEType(str, Enum):
    """Autoencoder type: standard or denoising."""

    plain = "plain"
    denoising = "denoising"


class ModelSize(str, Enum):
    """Model size: base or extended (deeper)."""

    base = "base"
    extended = "extended"


NETWORK_TYPE_MAP = {
    ("plain", "base"): "AE",
    ("plain", "extended"): "AEE",
    ("denoising", "base"): "DAE",
    ("denoising", "extended"): "DAEE",
}


@train_router.post("/run")
async def run_training(
    dataset_type: DatasetType = Query(..., description="Dataset to train on"),
    ae_type: AEType = Query(AEType.plain, description="plain = standard AE, denoising = noisy→clean (DAE)"),
    model_size: ModelSize = Query(ModelSize.base, description="base = standard, extended = deeper network"),
    validation_split: float | None = Query(None, gt=0, lt=1, description="Validation fraction, decimal 0–1, e.g. 0.2 — empty = json default"),
    epochs: int | None = Query(None, ge=1, description="Number of epochs, whole number, e.g. 200 — empty = json default"),
    batch_size: int | None = Query(None, ge=1, description="Batch size, whole number, e.g. 128 — empty = json default"),
    learning_rate: float | None = Query(None, gt=0, description="Learning rate as a decimal, e.g. 0.0002 (= 2e-4) — empty = json default"),
    decrease_learning_rate: bool | None = Query(None, description="Decay the learning rate during training (true/false) — empty = json default"),
    step_size: int | None = Query(None, ge=1, description="LR scheduler step size in epochs, whole number, e.g. 15 — empty = json default"),
    gamma: float | None = Query(None, gt=0, description="LR decay factor, decimal, e.g. 0.5 — empty = json default"),
    grayscale: bool | None = Query(None, description="Grayscale (true) or color/RGB (false) — empty = json default"),
    latent_space_dimension: int | None = Query(None, ge=1, description="Latent space size, whole number, e.g. 100 — empty = json default"),
    vis_during_training: bool | None = Query(None, description="Save visualizations during training (true/false) — empty = json default"),
    vis_interval: int | None = Query(None, ge=1, description="Visualization interval in epochs, whole number, e.g. 10 — empty = json default"),
    early_stopping: int | None = Query(None, ge=1, description="Early-stopping patience in epochs, whole number, e.g. 10 — empty = json default"),
    seed: int | None = Query(None, ge=0, description="Random seed, whole number; empty = json default (no fixed seed)"),
):
    """
    Resolve the training setup for the selected network.

    The two dropdowns choose the network: ae_type (plain/denoising) and
    model_size (base/extended) combine into the network type
    (AE / AEE / DAE / DAEE). Every other parameter overrides the matching field
    in training_config.json; left empty, each falls back to the JSON default.

    NOTE: for now this only resolves and echoes the effective config — the
    actual training is wired in later (background task).

    Args:
        dataset_type: Dataset to train on.
        ae_type: Standard (plain) or denoising autoencoder.
        model_size: Base or extended architecture.
        validation_split: Optional override for the validation fraction.
        epochs: Optional override for the number of epochs.
        batch_size: Optional override for the batch size.
        learning_rate: Optional override for the learning rate.
        decrease_learning_rate: Optional override for LR decay on/off.
        step_size: Optional override for the LR scheduler step size.
        gamma: Optional override for the LR decay factor.
        grayscale: Optional override for grayscale vs RGB.
        latent_space_dimension: Optional override for the latent space size.
        vis_during_training: Optional override for training visualizations.
        vis_interval: Optional override for the visualization interval.
        early_stopping: Optional override for the early-stopping patience.
        seed: Optional override for the random seed.

    Returns:
        dict: The resolved network type and the effective training parameters.
    """
    training_cfg_path = config_paths().get("training_config")
    config = TrainingConfigService.load(training_cfg_path)

    network_type = NETWORK_TYPE_MAP[(ae_type.value, model_size.value)]

    effective_config = {
        "network_type": network_type,
        "dataset_type": dataset_type.value,
        "validation_split": config.validation_split if validation_split is None else validation_split,
        "epochs": config.epochs if epochs is None else epochs,
        "batch_size": config.batch_size if batch_size is None else batch_size,
        "learning_rate": config.learning_rate if learning_rate is None else learning_rate,
        "decrease_learning_rate": config.decrease_learning_rate if decrease_learning_rate is None else decrease_learning_rate,
        "step_size": config.step_size if step_size is None else step_size,
        "gamma": config.gamma if gamma is None else gamma,
        "grayscale": config.grayscale if grayscale is None else grayscale,
        "latent_space_dimension": config.latent_space_dimension if latent_space_dimension is None else latent_space_dimension,
        "vis_during_training": config.vis_during_training if vis_during_training is None else vis_during_training,
        "vis_interval": config.vis_interval if vis_interval is None else vis_interval,
        "early_stopping": config.early_stopping if early_stopping is None else early_stopping,
        "seed": config.seed if seed is None else seed,
    }

    task = train_autoencoder_task.delay(effective_config)
    logging.info(f"Queued training task {task.id}")

    return {"task_id": task.id, "status": "QUEUED"}

@train_router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """
    Return the state and progress/result of a training task.

    Args:
        task_id: The Celery task id returned by /train/run.

    Returns:
        dict: The task state and its progress info or final result.
    """
    result = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": result.state, "info": None}
    if result.state == "PROGRESS":
        response["info"] = result.info
    elif result.state == "SUCCESS":
        response["info"] = result.get()
    elif result.state == "FAILURE":
        response["info"] = str(result.info)
    return response

@train_router.post("/stop/{task_id}")
async def stop_training(task_id: str):
    """
    Stop a running training task.

    Args:
        task_id: The Celery task id returned by /train/run.

    Returns:
        dict: Confirmation that the stop signal was sent.
    """
    try:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")
        return {
            "status": "ABORTED",
            "message": f"Task {task_id} has been stopped"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop task {task_id} due to {e}"
        )