import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import make_url

from easyrag.config import with_password
from easyrag.container import build_container
from easyrag.main import create_app
from easyrag.models import ModelPuller
from easyrag.ollama import OllamaRequestError, OllamaUnavailableError, PullProgress, pull_model
from tests.support import SettingsFactory

Handler = Callable[[httpx2.Request], httpx2.Response]


class FakeOllama:
    """/api/show and /api/pull over a set of present models."""

    def __init__(self, present: set[str], *, pull_error: str | None = None) -> None:
        self.present = present
        self.pull_error = pull_error
        self.pulled: list[str] = []
        self.down = 0  # fail this many requests with a connection error first

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if self.down:
            self.down -= 1
            raise httpx2.ConnectError("refused", request=request)
        model = json.loads(request.content)["model"]
        if request.url.path == "/api/show":
            return httpx2.Response(200 if model in self.present else 404, json={})
        assert request.url.path == "/api/pull"
        lines: list[dict[str, object]]
        if self.pull_error:
            lines = [{"status": "pulling manifest"}, {"error": self.pull_error}]
        else:
            self.pulled.append(model)
            self.present.add(model)
            lines = [
                {"status": "pulling manifest"},
                {"status": "downloading", "digest": "a", "total": 100, "completed": 40},
                {"status": "downloading", "digest": "b", "total": 300, "completed": 300},
                {"status": "downloading", "digest": "a", "total": 100, "completed": 100},
                {"status": "success"},
            ]
        body = "\n".join(json.dumps(line) for line in lines) + "\n"
        return httpx2.Response(200, content=body.encode())


def client(handler: Handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(base_url="http://ollama", transport=httpx2.MockTransport(handler))


# -- pull_model ------------------------------------------------------------------------


async def test_pull_reports_progress_over_all_layers() -> None:
    updates: list[PullProgress] = []
    async with client(FakeOllama(set())) as c:
        await pull_model(c, "m", updates.append)

    assert [(u.completed, u.total) for u in updates] == [
        (0, 0),
        (40, 100),
        (340, 400),
        (400, 400),
        (400, 400),
    ]


async def test_pull_error_in_the_stream_raises() -> None:
    async with client(FakeOllama(set(), pull_error="file does not exist")) as c:
        with pytest.raises(OllamaRequestError, match="file does not exist"):
            await pull_model(c, "nope")


async def test_pull_without_success_raises() -> None:
    def truncated(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b'{"status": "pulling manifest"}\n')

    async with client(truncated) as c:
        with pytest.raises(OllamaRequestError, match="without success"):
            await pull_model(c, "m")


async def test_pull_http_error_raises() -> None:
    async with client(lambda r: httpx2.Response(500, json={"error": "boom"})) as c:
        with pytest.raises(OllamaRequestError, match="HTTP 500"):
            await pull_model(c, "m")


async def test_unreachable_ollama_is_unavailable() -> None:
    fake = FakeOllama(set())
    fake.down = 1
    async with client(fake) as c:
        with pytest.raises(OllamaUnavailableError):
            await pull_model(c, "m")


# -- ModelPuller ---------------------------------------------------------------------


async def test_only_missing_models_are_pulled_once() -> None:
    fake = FakeOllama({"embed"})
    async with client(fake) as c:
        await ModelPuller(c, ["embed", "chat", "chat"]).ensure_all()

    assert fake.pulled == ["chat"]


async def test_background_pull_retries_until_ollama_is_up(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="easyrag.models")
    sleeps: list[float] = []

    async def no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    fake = FakeOllama(set())
    fake.down = 3
    async with client(fake) as c:
        puller = ModelPuller(c, ["embed", "chat"], initial_retry_seconds=2, sleep=no_sleep)
        puller.start()
        await puller.wait()

    assert fake.pulled == ["embed", "chat"]
    assert sleeps == [2, 4, 8]  # exponential backoff
    assert "Pulled model chat" in caplog.text


async def test_failed_pull_backs_off_up_to_the_maximum() -> None:
    sleeps: list[float] = []
    fake = FakeOllama(set(), pull_error="dial tcp: no route to host")
    attempts = 0

    async def stop_after_five(seconds: float) -> None:
        nonlocal attempts
        sleeps.append(seconds)
        attempts += 1
        if attempts == 5:
            fake.pull_error = None  # internet is back

    async with client(fake) as c:
        puller = ModelPuller(
            c, ["m"], initial_retry_seconds=10, max_retry_seconds=60, sleep=stop_after_five
        )
        puller.start()
        await puller.wait()

    assert sleeps == [10, 20, 40, 60, 60]
    assert fake.pulled == ["m"]


async def test_readiness_reports_a_model_being_pulled(make_settings: SettingsFactory) -> None:
    container = build_container(make_settings(llm_model="chat"))
    try:
        container.models._pulling = "chat"  # simulate an ongoing pull
        assert await container.readiness_checks["llm_model"]() == "pulling"
    finally:
        await container.aclose()


def test_pull_runs_at_startup_only_when_enabled(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(ModelPuller, "start", lambda self: started.append(True))

    for enabled in (False, True):
        with TestClient(create_app(make_settings(ollama_pull_models=enabled))):
            pass

    assert started == [True]


# -- Database password file ------------------------------------------------------------


def test_password_file_is_added_to_the_url(make_settings: SettingsFactory, tmp_path: Path) -> None:
    secret = tmp_path / "password"
    secret.write_text("p@ss:/w rd\n")

    settings = make_settings(
        database_url="postgresql+psycopg://easyrag@db:5432/easyrag", database_password_file=secret
    )

    url = make_url(settings.database_url.get_secret_value())
    assert (url.username, url.password, url.host, url.database) == (
        "easyrag", "p@ss:/w rd", "db", "easyrag",
    )  # fmt: skip


def test_password_file_problems_are_reported(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    url = "postgresql+psycopg://easyrag@db/easyrag"
    with pytest.raises(ValueError, match="cannot read"):
        make_settings(database_url=url, database_password_file=tmp_path / "missing")
    (tmp_path / "empty").write_text(" \n")
    with pytest.raises(ValueError, match="is empty"):
        make_settings(database_url=url, database_password_file=tmp_path / "empty")
    (tmp_path / "ok").write_text("x")
    with pytest.raises(ValueError, match="already contains a password"):
        make_settings(
            database_url="postgresql+psycopg://u:p@db/x", database_password_file=tmp_path / "ok"
        )


def test_with_password_keeps_the_rest_of_the_url() -> None:
    assert (
        with_password("postgresql+psycopg://u@h:5/d?sslmode=require", "s")
        == "postgresql+psycopg://u:s@h:5/d?sslmode=require"
    )
