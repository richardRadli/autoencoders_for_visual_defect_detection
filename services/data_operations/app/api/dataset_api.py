from enum import Enum
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from services.data_operations.app.core.dataset_service import DatasetService
from shared.core.enums import DatasetType


ops_dataset_router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
)


class PreviewType(str, Enum):
    """Image set to preview."""

    good = "good"
    aug = "aug"
    noise = "noise"


@ops_dataset_router.get("/readiness")
async def get_readiness(
    dataset_type: DatasetType = Query(..., description="Dataset to report on"),
):
    """
    Report whether augmentation and draw-rectangles have a finished run.

    Args:
        dataset_type: Dataset to report on.

    Returns:
        dict: Flat readiness facts for the workflow bar.
    """
    return await run_in_threadpool(DatasetService.get_readiness, dataset_type.value)


@ops_dataset_router.get("/preview")
async def get_preview(
    dataset_type: DatasetType = Query(..., description="Dataset to preview"),
    preview_type: PreviewType = Query(..., description="Which image set: good / aug / noise"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of sampled images"),
):
    """
    List evenly sampled preview images for a dataset image set.

    Args:
        dataset_type: Dataset to preview.
        preview_type: Which image set to sample (good / aug / noise).
        limit: Maximum number of images to return.

    Returns:
        dict: The preview type and the root-relative image paths.
    """
    images = await run_in_threadpool(
        DatasetService.get_preview_list, dataset_type.value, preview_type.value, limit
    )
    return {
        "dataset_type": dataset_type.value,
        "preview_type": preview_type.value,
        "images": images,
    }


@ops_dataset_router.get("/preview/image")
async def get_preview_image(
    dataset_type: DatasetType = Query(..., description="Dataset the image belongs to"),
    preview_type: PreviewType = Query(..., description="Which image set: good / aug / noise"),
    name: str = Query(
        ...,
        description="First call GET /dataset/preview — it returns an 'images' list. "
                    "Pick one entry from that list and paste it here, keeping the extension. "
                    "Example: if the list contains '000.png', then name = '000.png'. "
                    "(aug/noise entries look like '2026-07-13_10-20-30/img1_crop.jpg'.)",
    ),
):
    """
    Serve a single preview image file.

    Args:
        dataset_type: Dataset the image belongs to.
        preview_type: Which image set the image is from.
        name: One entry taken from the GET /dataset/preview 'images' list, pasted as-is (e.g. list has '000.png' → name = '000.png').

    Returns:
        FileResponse: The image file.
    """
    try:
        path = await run_in_threadpool(
            DatasetService.resolve_preview_image, dataset_type.value, preview_type.value, name
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return FileResponse(path)