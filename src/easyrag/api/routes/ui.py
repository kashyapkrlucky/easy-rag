"""The web UI: one static page and its assets, shipped inside the package (no build step)."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
ASSETS_PATH = "/assets"

router = APIRouter(tags=["ui"])


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")


def assets() -> StaticFiles:
    return StaticFiles(directory=WEB_DIR / "assets")
