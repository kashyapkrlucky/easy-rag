import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyrag.api.routes.ui import WEB_DIR
from tests.support import SettingsFactory
from tests.unit.test_api_chat import StubRag, client_with, make_answer

ROOT = Path(__file__).resolve().parents[2]
KEY = "k" * 24


def client(make_settings: SettingsFactory, **overrides: object) -> TestClient:
    return client_with(make_settings(**overrides), StubRag(make_answer()))


def test_page_is_served_with_a_strict_policy(make_settings: SettingsFactory) -> None:
    with client(make_settings) as c:
        response = c.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert '<script type="module" src="/assets/app.js">' in response.text
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "script-src 'self'" in csp
    assert "'unsafe-inline'" not in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/assets/app.js", "javascript"),
        ("/assets/render.js", "javascript"),
        ("/assets/style.css", "text/css"),
    ],
)
def test_assets_are_served(make_settings: SettingsFactory, path: str, content_type: str) -> None:
    with client(make_settings) as c:
        response = c.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert "content-security-policy" in response.headers


def test_assets_cannot_escape_their_directory(make_settings: SettingsFactory) -> None:
    with client(make_settings) as c:
        response = c.get("/assets/%2e%2e/%2e%2e/api/routes/ui.py")

    assert response.status_code == 404
    assert "WEB_DIR" not in response.text


def test_api_responses_get_basic_headers_but_not_the_ui_policy(
    make_settings: SettingsFactory,
) -> None:
    with client(make_settings) as c:
        response = c.post("/api/v1/chat", json={"question": "q"})
        docs = c.get("/docs")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in response.headers
    assert "content-security-policy" not in docs.headers  # Swagger UI loads from a CDN


def test_page_and_assets_are_public_but_the_api_is_not(make_settings: SettingsFactory) -> None:
    with client(make_settings, api_key=KEY) as c:
        assert c.get("/").status_code == 200
        assert c.get("/assets/app.js").status_code == 200
        assert c.post("/api/v1/chat", json={"question": "q"}).status_code == 401
        assert c.get("/assetsX").status_code == 401  # prefix match is on "/assets/"


def test_ui_can_be_disabled(make_settings: SettingsFactory) -> None:
    with client(make_settings, ui_enabled=False) as c:
        assert c.get("/").status_code == 404
        assert c.get("/assets/app.js").status_code == 404


def test_chat_sources_include_their_text(make_settings: SettingsFactory) -> None:
    with client(make_settings) as c:
        body = c.post("/api/v1/chat", json={"question": "q"}).json()

    assert body["sources"][0]["content"] == "Run docker compose up."


# -- Static checks on the front-end source -----------------------------------------------


def test_page_loads_nothing_from_other_origins() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)
    # Every script is an external same-origin file (the CSP forbids inline scripts).
    assert re.findall(r"<script[^>]*>", html) == ['<script type="module" src="/assets/app.js">']
    assert "style=" not in html  # inline styles would be blocked too


@pytest.mark.parametrize("name", ["app.js", "render.js"])
def test_scripts_never_insert_html(name: str) -> None:
    source = (WEB_DIR / "assets" / name).read_text(encoding="utf-8")

    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in source, f"{name} uses {sink}: model output must stay text"


def test_javascript_unit_tests() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run(
        [node, "--test", str(ROOT / "tests" / "web" / "render.test.mjs")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_key_dialog_has_a_single_submit_button() -> None:
    # Pressing Enter submits with the form's first submit button; a submit "Cancel" placed
    # before "Save" silently discarded the key (found in the live check).
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    dialog = html[html.index("<dialog") : html.index("</dialog>")]
    buttons = re.findall(r"<button[^>]*>", dialog)

    assert [b for b in buttons if 'type="button"' not in b] == ['<button value="save">']
