import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI

from services.defect_detection.app.api.defect_dataset_api import defect_dataset_router
from services.defect_detection.app.api.test_api import test_router
from shared.core.data_paths import init_all_paths
from utils.system_utils import setup_logger
from services.defect_detection.app.api.device_status_api import device_status_router
from services.defect_detection.app.api.train_api import train_router
from services.defect_detection.app.api.tune_api import tune_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize logging and verify storage on startup; log shutdown.

    Args:
        app: The FastAPI application instance.

    Returns:
        None
    """
    setup_logger()
    logging.info("Starting defect_detection service...")
    try:
        init_all_paths()
        logging.info("Storage/dataset paths verified.")
    except Exception as e:
        logging.error(f"Failed to initialize paths: {e}")
    yield
    logging.info("Shutting down defect_detection service...")


app = FastAPI(title="Defect Detection Service", lifespan=lifespan)

app.include_router(device_status_router)
app.include_router(train_router)
app.include_router(test_router)
app.include_router(defect_dataset_router)
app.include_router(tune_router)


@app.get("/")
async def health_check():
    """
    Report that the service is online.

    Returns:
        dict: The service status.
    """
    return {"status": "online", "service": "defect_detection"}