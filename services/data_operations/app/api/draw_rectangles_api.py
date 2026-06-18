import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from services.data_operations.app.core.aug_config_service import AugmentationConfigService
from services.data_operations.app.core.draw_rectangles_service import DrawRectanglesService
from shared.core.path_bindings import config_paths


draw_rectangles_router = APIRouter(
    prefix="/draw-rectangles",
    tags=["Draw Rectangles"],
)

VALID_DATASETS = {"texture_1", "texture_2", "cpu"}


@draw_rectangles_router.post("/run")
async def run_draw_rectangles(
    dataset_type: str = Query(..., description="Dataset to process: texture_1, texture_2 or cpu")
):
    """
    Generate noise images for the selected dataset.

    Args:
        dataset_type: Dataset to process (texture_1, texture_2 or cpu).

    Returns:
        dict: Summary with the processed image count and the used paths.
    """
    if dataset_type not in VALID_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset_type: {dataset_type}. Allowed: {sorted(VALID_DATASETS)}",
        )

    config = AugmentationConfigService.load(config_paths("augmentation_config"))
    result = await run_in_threadpool(DrawRectanglesService.run, dataset_type, config)

    logging.info(f"Draw rectangles finished for dataset: {dataset_type}")

    return {
        "status": "success",
        "dataset_type": result.dataset_type,
        "processed_images": result.processed_images,
        "source_dir": str(result.source_dir),
        "target_dir": str(result.target_dir),
    }