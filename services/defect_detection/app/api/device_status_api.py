import torch

from fastapi import APIRouter


device_status_router = APIRouter(
    prefix="/device-status",
    tags=["Device Status"],
)


@device_status_router.get("")
async def get_device_status():
    """
    Report the compute device the service can use (GPU if available, else CPU).

    Returns:
        dict: CUDA availability, the selected device and, on GPU, its name and VRAM.
    """
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "cuda_available": False,
            "message": "No GPU available, running on CPU",
        }

    free, total = torch.cuda.mem_get_info(0)
    return {
        "device": "cuda",
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "free_vram_gb": round(free / 1024 ** 3, 2),
        "total_vram_gb": round(total / 1024 ** 3, 2),
        "usage" : round((total - free)/1024 ** 3, 2),
    }