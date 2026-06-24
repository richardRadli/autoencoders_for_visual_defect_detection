import logging

from dataclasses import replace
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from services.data_operations.app.core.aug_config_service import AugmentationConfigService
from services.data_operations.app.core.draw_rectangles_service import DrawRectanglesService
from shared.core.enums import DatasetType
from shared.core.path_bindings import config_paths


draw_rectangles_router = APIRouter(
    prefix="/draw-rectangles",
    tags=["Draw Rectangles"],
)


@draw_rectangles_router.post("/run")
async def run_draw_rectangles(
    dataset_type: DatasetType = Query(..., description="Dataset to process"),
    size_of_cover: int | None = Query(None, ge=4, le=64, description="Square size to draw (4–64) — empty = config default"),
):
    """
    Generate noise images for the selected dataset.

    Args:
        dataset_type: Dataset to process.
        size_of_cover: Optional override for the drawn square size (4–64).

    Returns:
        dict: Summary with the processed image count and the used paths.
    """
    aug_cfg_path = config_paths().get("augmentation_config")
    config = AugmentationConfigService.load(aug_cfg_path)

    cover = config.size_of_cover if size_of_cover is None else size_of_cover
    config = replace(config, size_of_cover=cover)

    result = await run_in_threadpool(DrawRectanglesService.run, dataset_type.value, config)

    logging.info(f"Draw rectangles finished for dataset: {dataset_type.value}")

    return {
        "status": "success",
        "dataset_type": result.dataset_type,
        "processed_images": result.processed_images,
        "source_dir": str(result.source_dir),
        "target_dir": str(result.target_dir),
    }