from fastapi.testclient import TestClient

from easyrag.config import Settings
from easyrag.main import create_app


def test_liveness_returns_ok() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_route_returns_404() -> None:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        assert client.get("/does-not-exist").status_code == 404
