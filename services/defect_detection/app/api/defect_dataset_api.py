from enum import Enum
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from services.defect_detection.app.core.utility_services.defect_dataset_service import DatasetService
from shared.core.enums import DatasetType


defect_dataset_router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
)


class SubtestFolder(str, Enum):
    """Test subset: texture -> defective; cpu -> added/contamination/missing."""

    defective = "defective"
    added = "added"
    contamination = "contamination"
    missing = "missing"


VALID_SUBTESTS = {
    "texture_1": {"defective"},
    "texture_2": {"defective"},
    "cpu": {"added", "contamination", "missing"},
}


def _validate_subtest(dataset_type: str, subtest_folder: str) -> None:
    """Raise 422 if the subtest folder is not valid for the dataset."""
    if subtest_folder not in VALID_SUBTESTS[dataset_type]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid subtest_folder '{subtest_folder}' for dataset '{dataset_type}'. "
                   f"Allowed: {sorted(VALID_SUBTESTS[dataset_type])}",
        )


@defect_dataset_router.get("/readiness")
async def get_readiness(
    dataset_type: DatasetType = Query(..., description="Dataset to report on"),
):
    """
    Report training inputs and which networks have trained weights.

    Args:
        dataset_type: Dataset to report on.

    Returns:
        dict: Flat readiness facts for the workflow bar.
    """
    return await run_in_threadpool(DatasetService.get_readiness, dataset_type.value)


@defect_dataset_router.get("/preview")
async def get_preview(
    dataset_type: DatasetType = Query(..., description="Dataset to preview"),
    subtest_folder: SubtestFolder = Query(..., description="Test subset: texture → defective; cpu → added/contamination/missing"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of sampled images"),
):
    """
    List evenly sampled test images for a subtest folder.

    Args:
        dataset_type: Dataset to preview.
        subtest_folder: Which test subset to sample.
        limit: Maximum number of images to return.

    Returns:
        dict: The subtest folder and the root-relative image paths.
    """
    _validate_subtest(dataset_type.value, subtest_folder.value)

    images = await run_in_threadpool(
        DatasetService.get_preview_list, dataset_type.value, subtest_folder.value, limit
    )
    return {
        "dataset_type": dataset_type.value,
        "subtest_folder": subtest_folder.value,
        "images": images,
    }


@defect_dataset_router.get("/preview/image")
async def get_preview_image(
    dataset_type: DatasetType = Query(..., description="Dataset the image belongs to"),
    subtest_folder: SubtestFolder = Query(..., description="Test subset the image is from"),
    name: str = Query(
        ...,
        description="Copy one entry exactly as it appears in the 'images' list "
                    "returned by GET /dataset/preview (keep the extension). "
                    "You do not build this string yourself — just paste one entry. "
                    "Example: '000.png'.",
    ),
):
    """
    Serve a single test preview image file.

    Args:
        dataset_type: Dataset the image belongs to.
        subtest_folder: Test subset the image is from.
        name: One entry copied from the GET /dataset/preview 'images' list (keep the extension).

    Returns:
        FileResponse: The image file.
    """
    _validate_subtest(dataset_type.value, subtest_folder.value)

    try:
        path = await run_in_threadpool(
            DatasetService.resolve_preview_image, dataset_type.value, subtest_folder.value, name
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return FileResponse(path)