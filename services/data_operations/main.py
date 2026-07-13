import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI

from shared.core.data_paths import init_all_paths
from services.data_operations.app.api.draw_rectangles_api import draw_rectangles_router
from utils.system_utils import setup_logger
from services.data_operations.app.api.augmentation_api import augmentation_router
from services.data_operations.app.api.ops_dataset_api import ops_dataset_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared storage on startup and log shutdown.

    Args:
        app: The FastAPI application instance.

    Returns:
        None
    """
    setup_logger()
    logging.info("Starting data_operations service...")
    init_all_paths()
    yield
    logging.info("Shutting down data_operations service...")


app = FastAPI(title="Data Operations Service", lifespan=lifespan)

app.include_router(draw_rectangles_router)
app.include_router(augmentation_router)
app.include_router(ops_dataset_router)

@app.get("/")
async def health_check():
    """
    Report that the service is online.

    Returns:
        dict: The service status.
    """
    return {"status": "online", "service": "data_operations"}