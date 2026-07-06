import logging

from enum import Enum
from fastapi import APIRouter, HTTPException, Query, status

from services.defect_detection.app.core.utility_services.config_service.testing_config_service import TestingConfigService
from shared.core.enums import DatasetType
from shared.core.path_bindings import config_paths
from services.defect_detection.tasks import celery_app, test_autoencoder_task


test_router = APIRouter(
    prefix="/test",
    tags=["Testing"],
)


class AEType(str, Enum):
    """Autoencoder type: standard or denoising."""

    plain = "plain"
    denoising = "denoising"


class ModelSize(str, Enum):
    """Model size: base or extended (deeper)."""

    base = "base"
    extended = "extended"


class ImageSize(int, Enum):
    """Allowed test image sizes."""

    px_256 = 256
    px_512 = 512


class CropSize(int, Enum):
    """Allowed patch (crop) sizes."""

    px_64 = 64
    px_128 = 128


class Stride(int, Enum):
    """Allowed sliding-window strides."""

    s_4 = 4
    s_8 = 8
    s_16 = 16
    s_32 = 32
    s_64 = 64


class SubtestFolder(str, Enum):
    """Test subset: texture -> defective; cpu -> added/contamination/missing."""

    defective = "defective"
    added = "added"
    contamination = "contamination"
    missing = "missing"


NETWORK_TYPE_MAP = {
    ("plain", "base"): "AE",
    ("plain", "extended"): "AEE",
    ("denoising", "base"): "DAE",
    ("denoising", "extended"): "DAEE",
}


VALID_SUBTESTS = {
    "texture_1": {"defective"},
    "texture_2": {"defective"},
    "cpu": {"added", "contamination", "missing"},
}


@test_router.post("/run")
async def run_testing(
    dataset_type: DatasetType = Query(..., description="Dataset to evaluate on"),
    ae_type: AEType = Query(AEType.plain, description="plain = standard AE, denoising = noisy→clean (DAE)"),
    model_size: ModelSize = Query(ModelSize.base, description="base = standard, extended = deeper network"),
    subtest_folder: SubtestFolder = Query(SubtestFolder.defective, description="texture → defective; cpu → added/contamination/missing"),
    img_size: ImageSize = Query(ImageSize.px_256, description="Image size in pixels (256 or 512)"),
    crop_size: CropSize = Query(CropSize.px_128, description="Patch size in pixels (64 or 128)"),
    stride: Stride = Query(Stride.s_32, description="Sliding-window stride (4/8/16/32/64)"),
    num_of_steps: int | None = Query(None, ge=1, description="Number of threshold steps, whole number — empty = json default"),
    threshold_init: float | None = Query(None, ge=0, description="Threshold range start, decimal ≥ 0 — empty = json default"),
    threshold_end: float | None = Query(None, le=2, description="Threshold range end, decimal ≤ 2 — empty = json default"),
    vis_results: bool | None = Query(None, description="Save per-image result visualizations (true/false) — empty = json default"),
    vis_reconstruction: bool | None = Query(None, description="Reconstruction-only mode, no metrics (true/false) — empty = json default"),
):
    """
    Resolve the evaluation setup and queue a testing task.

    The two dropdowns choose the network: ae_type (plain/denoising) and
    model_size (base/extended) combine into the network type
    (AE / AEE / DAE / DAEE). grayscale is NOT set here — the test reads it from
    the trained run's params.json so it always matches the weights.

    Args:
        dataset_type: Dataset to evaluate on.
        ae_type: Standard (plain) or denoising autoencoder.
        model_size: Base or extended architecture.
        subtest_folder: Which test subset to use (validated against the dataset).
        img_size: Image size (dropdown).
        crop_size: Patch size (dropdown).
        stride: Sliding-window stride (dropdown).
        num_of_steps: Optional override for the number of threshold steps.
        threshold_init: Optional override for the threshold range start.
        threshold_end: Optional override for the threshold range end.
        vis_results: Optional override for per-image visualizations.
        vis_reconstruction: Optional override for reconstruction-only mode.

    Returns:
        dict: The queued task id and status.
    """
    if subtest_folder.value not in VALID_SUBTESTS[dataset_type.value]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid subtest_folder '{subtest_folder.value}' for dataset '{dataset_type.value}'. "
                   f"Allowed: {sorted(VALID_SUBTESTS[dataset_type.value])}",
        )

    img = int(img_size)
    crop = int(crop_size)
    strd = int(stride)

    if crop >= img:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"crop_size ({crop}) must be smaller than img_size ({img})",
        )
    if img - crop < strd:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"img_size - crop_size ({img - crop}) must be >= stride ({strd})",
        )
    if (img - crop) % strd != 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"(img_size - crop_size) ({img - crop}) must be divisible by stride ({strd})",
        )

    testing_cfg_path = config_paths().get("testing_config")
    config = TestingConfigService.load(testing_cfg_path)

    network_type = NETWORK_TYPE_MAP[(ae_type.value, model_size.value)]

    resolved_init = config.threshold_init if threshold_init is None else threshold_init
    resolved_end = config.threshold_end if threshold_end is None else threshold_end
    if resolved_init >= resolved_end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"threshold_init ({resolved_init}) must be smaller than threshold_end ({resolved_end})",
        )

    effective_config = {
        "network_type": network_type,
        "dataset_type": dataset_type.value,
        "subtest_folder": subtest_folder.value,
        "img_size": img,
        "crop_size": crop,
        "stride": strd,
        "num_of_steps": config.num_of_steps if num_of_steps is None else num_of_steps,
        "threshold_init": resolved_init,
        "threshold_end": resolved_end,
        "vis_results": config.vis_results if vis_results is None else vis_results,
        "vis_reconstruction": config.vis_reconstruction if vis_reconstruction is None else vis_reconstruction,
    }

    task = test_autoencoder_task.delay(effective_config)
    logging.info(f"Queued testing task {task.id}")

    return {"task_id": task.id, "status": "QUEUED"}


@test_router.get("/status/{task_id}")
async def get_testing_status(task_id: str):
    """
    Return the state and progress/result of a testing task.

    Args:
        task_id: The Celery task id returned by /test/run.

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


@test_router.post("/stop/{task_id}")
async def stop_testing(task_id: str):
    """
    Stop a running testing task.

    Args:
        task_id: The Celery task id returned by /test/run.

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