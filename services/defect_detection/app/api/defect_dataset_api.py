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


class PreviewType(str, Enum):
    """
    Which image set a preview refers to.

    - test: the input test images of a subtest folder.
    - reconstruction / reconstruction_vis / roc_plot: outputs of a test run,
      stored per network type.
    """

    test = "test"
    reconstruction = "reconstruction"
    reconstruction_vis = "reconstruction_vis"
    roc_plot = "roc_plot"


class SubtestFolder(str, Enum):
    """Test subset: texture -> defective; cpu -> added / contamination / missing."""

    defective = "defective"
    added = "added"
    contamination = "contamination"
    missing = "missing"


class NetworkType(str, Enum):
    """Autoencoder network type (plain/denoising x base/extended)."""

    AE = "AE"
    AEE = "AEE"
    DAE = "DAE"
    DAEE = "DAEE"


# Which subtest folders each dataset actually has.
VALID_SUBTESTS = {
    "texture_1": {"defective"},
    "texture_2": {"defective"},
    "cpu": {"added", "contamination", "missing"},
}


def _validate_preview_params(
    dataset_type: str,
    preview_type: PreviewType,
    subtest_folder: SubtestFolder | None,
    network_type: NetworkType | None,
) -> None:
    """
    Ensure the parameter a preview type needs is present and valid.

    'test' previews are keyed by a subtest folder (validated against the
    dataset); the output previews (reconstruction / reconstruction_vis /
    roc_plot) are keyed by a network type.

    Args:
        dataset_type: Selected dataset name.
        preview_type: The requested preview type.
        subtest_folder: Provided subtest folder, if any.
        network_type: Provided network type, if any.

    Returns:
        None

    Raises:
        HTTPException: 422 if the required parameter is missing, or the subtest
            folder is not valid for the dataset.
    """
    if preview_type == PreviewType.test:
        if subtest_folder is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="subtest_folder is required for preview_type=test",
            )
        if subtest_folder.value not in VALID_SUBTESTS[dataset_type]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid subtest_folder '{subtest_folder.value}' for dataset "
                    f"'{dataset_type}'. Allowed: {sorted(VALID_SUBTESTS[dataset_type])}"
                ),
            )
    elif network_type is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"network_type is required for preview_type={preview_type.value}",
        )


@defect_dataset_router.get("/readiness")
async def get_readiness(
    dataset_type: DatasetType = Query(..., description="Dataset to report readiness for"),
):
    """
    Report training inputs and which networks have trained weights.

    Returns flat facts (aug/noise image counts and the trained networks); the
    frontend derives per-network training/testing readiness from them.

    Args:
        dataset_type: Dataset to report readiness for.

    Returns:
        dict: Readiness facts (aug + noise counts and the trained_networks list).
    """
    return await run_in_threadpool(DatasetService.get_readiness, dataset_type.value)


@defect_dataset_router.get("/preview")
async def get_preview(
    dataset_type: DatasetType = Query(..., description="Dataset to preview"),
    preview_type: PreviewType = Query(
        PreviewType.test,
        description="test = test input images; reconstruction / reconstruction_vis / roc_plot = outputs of a test run",
    ),
    subtest_folder: SubtestFolder | None = Query(
        None,
        description="Required for preview_type=test (texture → defective; cpu → added / contamination / missing)",
    ),
    network_type: NetworkType | None = Query(
        None,
        description="Required for preview_type=reconstruction / reconstruction_vis / roc_plot",
    ),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of evenly sampled images to return"),
):
    """
    List evenly sampled preview images for a preview type.

    The first and last images are always included; the returned paths are
    relative to the preview type's root, ready to pass back to
    /dataset/preview/image.

    Args:
        dataset_type: Dataset to preview.
        preview_type: Which image set to sample.
        subtest_folder: Test subset; required for preview_type=test.
        network_type: Network type; required for the output preview types.
        limit: Maximum number of images to return.

    Returns:
        dict: The echoed selectors and the list of root-relative image paths.

    Raises:
        HTTPException: 422 if the parameter required by the preview type is
            missing or invalid.
    """
    _validate_preview_params(dataset_type.value, preview_type, subtest_folder, network_type)

    images = await run_in_threadpool(
        DatasetService.get_preview_list,
        dataset_type.value,
        preview_type.value,
        subtest_folder.value if subtest_folder else None,
        network_type.value if network_type else None,
        limit,
    )

    return {
        "dataset_type": dataset_type.value,
        "preview_type": preview_type.value,
        "subtest_folder": subtest_folder.value if subtest_folder else None,
        "network_type": network_type.value if network_type else None,
        "images": images,
    }


@defect_dataset_router.get("/preview/image")
async def get_preview_image(
    dataset_type: DatasetType = Query(..., description="Dataset the image belongs to"),
    preview_type: PreviewType = Query(
        PreviewType.test,
        description="Which image set the image is from",
    ),
    name: str = Query(
        ...,
        description=(
            "Copy one entry exactly as it appears in the 'images' list "
            "returned by GET /dataset/preview (keep the extension). "
            "You do not build this string yourself — just paste one entry. "
            "Example: '000.png'."
        ),
    ),
    subtest_folder: SubtestFolder | None = Query(
        None,
        description="Required for preview_type=test",
    ),
    network_type: NetworkType | None = Query(
        None,
        description="Required for preview_type=reconstruction / reconstruction_vis / roc_plot",
    ),
):
    """
    Serve a single preview image file.

    Validates the selectors, resolves the image safely under its root (guarding
    against path traversal) and streams the file back.

    Args:
        dataset_type: Dataset the image belongs to.
        preview_type: Which image set the image is from.
        name: One entry copied exactly from the /dataset/preview 'images' list.
        subtest_folder: Test subset; required for preview_type=test.
        network_type: Network type; required for the output preview types.

    Returns:
        FileResponse: The image file with its detected content type.

    Raises:
        HTTPException: 422 if a required selector is missing/invalid; 404 if the
            image cannot be resolved (escapes the root, wrong type, or missing).
    """
    _validate_preview_params(dataset_type.value, preview_type, subtest_folder, network_type)

    try:
        path = await run_in_threadpool(
            DatasetService.resolve_preview_image,
            dataset_type.value,
            preview_type.value,
            name,
            subtest_folder.value if subtest_folder else None,
            network_type.value if network_type else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return FileResponse(path)