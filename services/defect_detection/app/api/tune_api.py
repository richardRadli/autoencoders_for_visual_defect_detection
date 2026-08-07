import logging

from enum import Enum
from fastapi import APIRouter, HTTPException, Query, status

from services.defect_detection.app.core.utility_services.config_service.training_config_service import TrainingConfigService
from services.defect_detection.app.core.utility_services.config_service.tuning_config_service import TuningConfigService
from shared.core.enums import DatasetType
from shared.core.path_bindings import config_paths
from services.defect_detection.tasks import celery_app, tune_autoencoder_task



tune_router = APIRouter(
    prefix="/tune",
    tags=["Tuning"],
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


@tune_router.post("/run")
async def run_tuning(
    dataset_type: DatasetType = Query(..., description="Dataset to tune on"),
    ae_type: AEType = Query(AEType.plain, description="plain = standard AE, denoising = noisy→clean (DAE)"),
    model_size: ModelSize = Query(ModelSize.base, description="base = standard, extended = deeper network"),
    n_trials: int | None = Query(None, ge=1, description="Number of Optuna trials, whole number, e.g. 20 — empty = tuning_config.json default"),
    epochs_per_trial: int | None = Query(None, ge=1, description="Epochs trained per trial, whole number, e.g. 5 — empty = tuning_config.json default"),
    learning_rate_min: float | None = Query(None, gt=0, description="Lowest learning rate to search, decimal, e.g. 0.00001 — empty = json default"),
    learning_rate_max: float | None = Query(None, gt=0, description="Highest learning rate to search, decimal, e.g. 0.01 — empty = json default"),
    latent_space_dimension_min: int | None = Query(None, ge=1, description="Smallest latent size to search, whole number, e.g. 16 — empty = json default"),
    latent_space_dimension_max: int | None = Query(None, ge=1, description="Largest latent size to search, whole number, e.g. 256 — empty = json default"),
    step_size_min: int | None = Query(None, ge=1, description="Smallest LR step size to search, whole number, e.g. 5 — empty = json default"),
    step_size_max: int | None = Query(None, ge=1, description="Largest LR step size to search, whole number, e.g. 30 — empty = json default"),
    gamma_min: float | None = Query(None, gt=0, description="Smallest LR decay factor to search, decimal, e.g. 0.1 — empty = json default"),
    gamma_max: float | None = Query(None, gt=0, description="Largest LR decay factor to search, decimal, e.g. 0.9 — empty = json default"),
    batch_size_values: list[int] | None = Query(None, description="Batch sizes to choose from, whole numbers, e.g. 32 64 128 — empty = json default"),
):
    """
    Queue an Optuna hyperparameter tuning run.

    The two dropdowns choose the network: ae_type (plain/denoising) and
    model_size (base/extended) combine into the network type
    (AE / AEE / DAE / DAEE), fixed for the whole run. Exactly five
    hyperparameters are searched — learning_rate, latent_space_dimension,
    step_size and gamma within [min, max], and batch_size from the given list.
    Each tuning field left empty falls back to tuning_config.json. Every other
    training field (validation_split, grayscale, early_stopping, seed) is read
    from training_config.json and kept fixed. The run stays fully in memory and
    produces no weights, params.json or TensorBoard log.

    Args:
        dataset_type: Dataset to tune on.
        ae_type: Standard (plain) or denoising autoencoder.
        model_size: Base or extended architecture.
        n_trials: Optional override for the number of Optuna trials.
        epochs_per_trial: Optional override for epochs per trial.
        learning_rate_min: Optional override for the learning-rate lower bound.
        learning_rate_max: Optional override for the learning-rate upper bound.
        latent_space_dimension_min: Optional override for the latent-size lower bound.
        latent_space_dimension_max: Optional override for the latent-size upper bound.
        step_size_min: Optional override for the step-size lower bound.
        step_size_max: Optional override for the step-size upper bound.
        gamma_min: Optional override for the gamma lower bound.
        gamma_max: Optional override for the gamma upper bound.
        batch_size_values: Optional override for the discrete batch sizes.

    Returns:
        dict: The queued task id and its status.
    """
    tuning_cfg = TuningConfigService.load(config_paths().get("tuning_config"))
    training_cfg = TrainingConfigService.load(config_paths().get("training_config"))

    resolved_n_trials = tuning_cfg.n_trials if n_trials is None else n_trials
    resolved_epochs_per_trial = tuning_cfg.epochs_per_trial if epochs_per_trial is None else epochs_per_trial
    resolved_learning_rate_min = tuning_cfg.learning_rate_min if learning_rate_min is None else learning_rate_min
    resolved_learning_rate_max = tuning_cfg.learning_rate_max if learning_rate_max is None else learning_rate_max
    resolved_latent_min = tuning_cfg.latent_space_dimension_min if latent_space_dimension_min is None else latent_space_dimension_min
    resolved_latent_max = tuning_cfg.latent_space_dimension_max if latent_space_dimension_max is None else latent_space_dimension_max
    resolved_step_min = tuning_cfg.step_size_min if step_size_min is None else step_size_min
    resolved_step_max = tuning_cfg.step_size_max if step_size_max is None else step_size_max
    resolved_gamma_min = tuning_cfg.gamma_min if gamma_min is None else gamma_min
    resolved_gamma_max = tuning_cfg.gamma_max if gamma_max is None else gamma_max
    resolved_batch_size_values = tuning_cfg.batch_size_values if batch_size_values is None else batch_size_values

    if resolved_n_trials < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="n_trials must be greater than or equal to 1",
        )
    if resolved_epochs_per_trial < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="epochs_per_trial must be greater than or equal to 1",
        )
    if resolved_learning_rate_min > resolved_learning_rate_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="learning_rate_min must be less than or equal to learning_rate_max",
        )
    if resolved_latent_min > resolved_latent_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="latent_space_dimension_min must be less than or equal to latent_space_dimension_max",
        )
    if resolved_step_min > resolved_step_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="step_size_min must be less than or equal to step_size_max",
        )
    if resolved_gamma_min > resolved_gamma_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="gamma_min must be less than or equal to gamma_max",
        )
    if not resolved_batch_size_values:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="batch_size_values must contain at least one value",
        )
    if any(value < 1 for value in resolved_batch_size_values):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="batch_size_values must all be greater than or equal to 1",
        )

    network_type = NETWORK_TYPE_MAP[(ae_type.value, model_size.value)]

    effective_config = {
        "network_type": network_type,
        "dataset_type": dataset_type.value,
        "n_trials": resolved_n_trials,
        "epochs_per_trial": resolved_epochs_per_trial,
        "learning_rate_min": resolved_learning_rate_min,
        "learning_rate_max": resolved_learning_rate_max,
        "latent_space_dimension_min": resolved_latent_min,
        "latent_space_dimension_max": resolved_latent_max,
        "step_size_min": resolved_step_min,
        "step_size_max": resolved_step_max,
        "gamma_min": resolved_gamma_min,
        "gamma_max": resolved_gamma_max,
        "batch_size_values": resolved_batch_size_values,
        "validation_split": training_cfg.validation_split,
        "grayscale": training_cfg.grayscale,
        "early_stopping": training_cfg.early_stopping,
        "seed": training_cfg.seed,
    }

    task = tune_autoencoder_task.delay(effective_config)
    logging.info(f"Queued tuning task {task.id}")

    return {"task_id": task.id, "status": "QUEUED"}

@tune_router.get("/status/{task_id}")
async def get_tuning_status(task_id: str):
    """
    Return the state and progress/result of a tuning task.

    Args:
        task_id: The Celery task id returned by /tune/run.

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

@tune_router.post("/stop/{task_id}")
async def stop_tuning(task_id: str):
    """
    Stop a running tuning task.

    Args:
        task_id: The Celery task id returned by /tune/run.

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