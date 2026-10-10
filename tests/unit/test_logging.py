import json
import logging
import sys

from easyrag.observability.context import bound, index_run_id, request_id
from easyrag.observability.logging import ContextFilter, JsonFormatter, configure_logging


def record(message: str = "hello %s", *args: object, **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord(
        "easyrag.test", logging.INFO, __file__, 1, message, args or ("world",), None
    )
    for key, value in extra.items():
        setattr(rec, key, value)
    ContextFilter().filter(rec)
    return rec


def test_json_lines_have_the_standard_fields() -> None:
    entry = json.loads(JsonFormatter().format(record()))

    assert entry["message"] == "hello world"
    assert entry["level"] == "INFO" and entry["logger"] == "easyrag.test"
    assert entry["timestamp"].endswith("+00:00")
    assert "request_id" not in entry  # None values are omitted


def test_extra_fields_become_json_keys_and_odd_values_do_not_crash() -> None:
    entry = json.loads(JsonFormatter().format(record(status=200, odd=object())))

    assert entry["status"] == 200
    assert entry["odd"].startswith("<object object")


def test_context_ids_are_attached() -> None:
    with bound(request_id, "req-1"), bound(index_run_id, "run-9"):
        entry = json.loads(JsonFormatter().format(record()))

    assert (entry["request_id"], entry["index_run_id"]) == ("req-1", "run-9")


def test_exceptions_are_serialised() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        rec = record()
        rec.exc_info = sys.exc_info()

    entry = json.loads(JsonFormatter().format(rec))

    assert "ValueError: boom" in entry["exception"]


def test_text_format_shows_context_inline() -> None:
    configure_logging("INFO", "text")
    handler = logging.getLogger().handlers[0]
    with bound(request_id, "req-2"):
        rec = record()
        handler.filter(rec)
        line = handler.format(rec)

    assert line.endswith("hello world [request_id=req-2]")


def test_uvicorn_access_log_is_replaced_and_errors_are_routed() -> None:
    configure_logging("INFO", "json")

    assert logging.getLogger("uvicorn.access").handlers == []
    assert logging.getLogger("uvicorn.error").handlers == logging.getLogger().handlers
