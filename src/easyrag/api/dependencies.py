"""FastAPI dependency providers.

Routes receive collaborators through these functions rather than importing them,
so tests can substitute them via ``app.dependency_overrides``.
"""

from typing import Annotated, cast

from fastapi import Depends, Request

from easyrag.container import AppContainer


def get_container(request: Request) -> AppContainer:
    return cast(AppContainer, request.app.state.container)


ContainerDep = Annotated[AppContainer, Depends(get_container)]
