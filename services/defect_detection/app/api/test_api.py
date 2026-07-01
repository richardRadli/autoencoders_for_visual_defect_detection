from fastapi import APIRouter

test_router = APIRouter(
    prefix="/test",
    tags=["Testing"],
)