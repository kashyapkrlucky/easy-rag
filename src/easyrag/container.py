"""Composition root.

The only place that decides which concrete implementations the application uses.
Built once per application in the FastAPI lifespan and closed on shutdown.
"""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.config import Settings
from easyrag.storage.database import create_engine


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    engine: AsyncEngine

    async def aclose(self) -> None:
        await self.engine.dispose()


def build_container(settings: Settings) -> AppContainer:
    return AppContainer(settings=settings, engine=create_engine(settings))
