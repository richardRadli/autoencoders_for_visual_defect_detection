import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from services.data_operations.app.core.aug_config_service import AugmentationConfigService
from services.data_operations.app.core.augmentation_service import AugmentationService
from shared.core.path_bindings import config_paths


augmentation_router = APIRouter(
    prefix="/augmentation",
    tags=["Augmentation"],
)

VALID_DATASETS = {"texture_1", "texture_2", "cpu"}


@augmentation_router.post("/run")
async def run_augmentation(
    dataset_type: str = Query(..., description="Dataset to process: texture_1, texture_2 or cpu")
):
    """
    Generate the augmented training set for the selected dataset.

    Args:
        dataset_type: Dataset to process (texture_1, texture_2 or cpu).

    Returns:
        dict: Summary with the image counts and the used paths.
    """
    if dataset_type not in VALID_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset_type: {dataset_type}. Allowed: {sorted(VALID_DATASETS)}",
        )

    aug_cfg_path = config_paths().get("augmentation_config")

    config = AugmentationConfigService.load(aug_cfg_path)
    result = await run_in_threadpool(AugmentationService.run, dataset_type, config)

    logging.info(f"Augmentation finished for dataset: {dataset_type}")

    return {
        "status": "success",
        "dataset_type": result.dataset_type,
        "source_images": result.source_images,
        "augmented_images": result.augmented_images,
        "source_dir": str(result.source_dir),
        "target_dir": str(result.target_dir),
    }