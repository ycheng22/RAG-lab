"""One JSON log line per request, including what the request cost in OpenAI tokens.

Written as plain ASGI middleware (not BaseHTTPMiddleware) so that streamed responses are logged when
the last chunk has been sent, with the tokens the whole stream used.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid

from raglab import llm

logger = logging.getLogger("raglab.service")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_json(**fields) -> None:
    logger.info(json.dumps(fields, ensure_ascii=False))


class JsonLogMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        meter = llm.start_meter()  # every OpenAI call made for this request is recorded here
        state = {"status": 500, "logged": False}

        def write_log() -> None:
            if state["logged"]:
                return
            state["logged"] = True
            log_json(
                request_id=request_id,
                method=scope["method"],
                path=scope["path"],
                status=state["status"],
                latency_ms=round((time.perf_counter() - started) * 1000),
                **llm.meter_totals(meter),
            )

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                write_log()

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            write_log()  # also covers errors and disconnects
