import logging

from enum import Enum
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from services.data_operations.app.core.aug_config_service import AugmentationConfigService
from services.data_operations.app.core.augmentation_service import AugmentationService
from shared.core.enums import DatasetType
from shared.core.path_bindings import config_paths


augmentation_router = APIRouter(
    prefix="/augmentation",
    tags=["Augmentation"],
)

MIN_TOTAL = 5000
MAX_TOTAL = 20000


class ImageSize(int, Enum):
    """Allowed source image sizes."""

    px_512 = 512
    px_256 = 256


class CropSize(int, Enum):
    """Allowed crop sizes."""

    px_256 = 256
    px_128 = 128
    px_64 = 64


@augmentation_router.post("/run")
async def run_augmentation(
    dataset_type: DatasetType = Query(..., description="Dataset to process"),
    img_size: ImageSize = Query(ImageSize.px_256, description="Source image size"),
    crop_size: CropSize = Query(CropSize.px_128, description="Crop size; must be smaller than img_size and divide it evenly"),
    rotate_count: int | None = Query(None, ge=0, description="Rotated images — empty = config default, 0 = none"),
    horizontal_flip_count: int | None = Query(None, ge=0, description="Horizontally flipped images — empty = config default, 0 = none"),
    vertical_flip_count: int | None = Query(None, ge=0, description="Vertically flipped images — empty = config default, 0 = none"),
):
    """
    Generate the base crops and the count-based augmented set for a dataset.

    The augmented total (rotate + horizontal flip + vertical flip) must be
    between 5000 and 20000. An invalid size combination does not fail the
    request: it falls back to the default crop size and reports a warning.

    Args:
        dataset_type: Dataset to process.
        img_size: Source image size (512 or 256).
        crop_size: Crop size (256, 128 or 64).
        rotate_count: Optional override for the number of rotated images.
        horizontal_flip_count: Optional override for the horizontally flipped images.
        vertical_flip_count: Optional override for the vertically flipped images.

    Returns:
        dict: Summary with the image counts, the used paths and an optional warning.
    """
    aug_cfg_path = config_paths().get("augmentation_config")
    config = AugmentationConfigService.load(aug_cfg_path)

    img = int(img_size)
    crop = int(crop_size)
    warning = None
    if crop >= img or img % crop != 0:
        warning = (
            f"Invalid size combination: crop_size={crop}, img_size={img}. "
            f"crop_size must be smaller than img_size and divide it evenly. "
            f"Used default crop_size={config.crop_size} instead."
        )
        crop = config.crop_size
        logging.warning(warning)

    rotate = config.rotate_count if rotate_count is None else rotate_count
    hflip = config.horizontal_flip_count if horizontal_flip_count is None else horizontal_flip_count
    vflip = config.vertical_flip_count if vertical_flip_count is None else vertical_flip_count

    total = rotate + hflip + vflip
    if total < MIN_TOTAL or total > MAX_TOTAL:
        raise HTTPException(
            status_code=422,
            detail=f"Total augmented count must be between {MIN_TOTAL} and {MAX_TOTAL}, got {total}.",
        )

    overrides = {
        "img_size": img,
        "crop_size": crop,
        "rotate_count": rotate,
        "horizontal_flip_count": hflip,
        "vertical_flip_count": vflip,
    }

    result = await run_in_threadpool(AugmentationService.run, dataset_type.value, config, overrides)

    logging.info(f"Augmentation finished for dataset: {dataset_type.value}")

    response = {
        "status": "success",
        "dataset_type": result.dataset_type,
        "source_images": result.source_images,
        "augmented_images": result.augmented_images,
        "source_dir": str(result.source_dir),
        "target_dir": str(result.target_dir),
    }
    if warning:
        response["warning"] = warning

    return response