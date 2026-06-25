import logging

from enum import Enum
from dataclasses import replace
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from services.data_operations.app.core.aug_config_service import AugmentationConfigService
from services.data_operations.app.core.draw_rectangles_service import DrawRectanglesService
from shared.core.enums import DatasetType
from shared.core.path_bindings import config_paths
from utils.system_utils import file_reader


draw_rectangles_router = APIRouter(
    prefix="/draw-rectangles",
    tags=["Draw Rectangles"],
)


class SourceImage(str, Enum):
    """Source folder to draw the rectangles on."""

    good = "good"
    aug = "aug"


@draw_rectangles_router.post("/run")
async def run_draw_rectangles(
    dataset_type: DatasetType = Query(..., description="Dataset to process"),
    source: SourceImage = Query(SourceImage.good, description="good = clean / training images, aug = images from the LATEST augmentation run"),
    size_of_cover: int | None = Query(None, ge=4, le=64, description="Square size to draw (4–64) — empty = config default"),
):
    """
    Generate noise images for the selected dataset.

    When source is 'aug', the images are taken from the most recent augmentation
    run (the latest timestamp folder inside the dataset's aug directory).

    Args:
        dataset_type: Dataset to process.
        source: Source folder to draw on — good, or aug (latest augmentation run).
        size_of_cover: Optional override for the drawn square size (4–64).

    Returns:
        dict: Summary with the processed image count and the used paths.
    """
    aug_cfg_path = config_paths().get("augmentation_config")
    config = AugmentationConfigService.load(aug_cfg_path)

    cover = config.size_of_cover if size_of_cover is None else size_of_cover
    config = replace(config, size_of_cover=cover)

    try:
        source_dir = DrawRectanglesService.resolve_source_dir(dataset_type.value, source.value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"No augmented run found for dataset '{dataset_type.value}'. Run augmentation first.",
        )

    if not file_reader(str(source_dir), "png", "jpg"):
        raise HTTPException(
            status_code=422,
            detail=f"No source images found in the '{source.value}' folder: {source_dir}",
        )

    request_params = {
        "dataset_type": dataset_type.value,
        "source": source.value,
        "size_of_cover": cover,
    }

    result = await run_in_threadpool(
        DrawRectanglesService.run,
        dataset_type.value,
        source.value,
        source_dir,
        config,
        request_params,
    )

    logging.info(f"Draw rectangles finished for dataset: {dataset_type.value}")

    return {
        "status": "stopped" if result.stopped else "success",
        "dataset_type": result.dataset_type,
        "source": source.value,
        "processed_images": result.processed_images,
        "source_dir": str(result.source_dir),
        "target_dir": str(result.target_dir),
    }


@draw_rectangles_router.post("/stop")
async def stop_draw_rectangles():
    """
    Signal a running draw-rectangles job to stop as soon as possible.

    Returns:
        dict: Confirmation that the stop signal was sent.
    """
    DrawRectanglesService.request_stop()
    return {"status": "stopping", "service": "draw_rectangles"}