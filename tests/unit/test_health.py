from fastapi.testclient import TestClient

from easyrag.main import create_app
from tests.support import SettingsFactory


def test_liveness_returns_ok_even_when_database_is_unreachable(
    make_settings: SettingsFactory,
) -> None:
    with TestClient(create_app(make_settings())) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_route_returns_404(make_settings: SettingsFactory) -> None:
    with TestClient(create_app(make_settings())) as client:
        assert client.get("/does-not-exist").status_code == 404
