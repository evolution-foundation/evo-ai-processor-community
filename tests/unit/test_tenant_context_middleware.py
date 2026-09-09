"""TenantContextMiddleware: a refusal raised by the runtime_context extension
point reaches the client; a plain resolution failure still degrades to "no
tenant bound" instead of dropping the request."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src import evo_extension_points
from src.middleware.tenant_context import TenantContextMiddleware


class _Context:
    def __init__(self, resolve):
        self._resolve = resolve
        self.bound: list[str] = []

    def current_context_id(self, source):
        return self._resolve()

    def with_context(self, context_id, fn):
        return fn()

    @asynccontextmanager
    async def bind_context(self, context_id):
        self.bound.append(context_id)
        yield


@pytest.fixture(autouse=True)
def _reset_runtime_context():
    evo_extension_points.reset("runtime_context")
    yield
    evo_extension_points.reset("runtime_context")


def build_client(resolve):
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    app.add_middleware(TenantContextMiddleware)
    ctx = _Context(resolve)
    evo_extension_points.replace("runtime_context", ctx)
    return TestClient(app), ctx


def test_http_exception_from_the_extension_point_is_returned_as_is():
    def refuse():
        raise HTTPException(status_code=403, detail="user is not a member of tenant")

    client, ctx = build_client(refuse)
    response = client.get("/ping")

    assert response.status_code == 403
    assert response.json() == {
        "error": "Forbidden",
        "code": "ERR_FORBIDDEN",
        "message": "user is not a member of tenant",
    }
    assert ctx.bound == [], "a refused request must not reach bind_context"


def test_resolution_failure_still_runs_the_request_unbound():
    def blow_up():
        raise RuntimeError("membership lookup unavailable")

    client, ctx = build_client(blow_up)
    response = client.get("/ping")

    assert response.status_code == 200
    assert ctx.bound == []


def test_resolved_context_is_bound_around_the_request():
    client, ctx = build_client(lambda: "ctx-a")
    response = client.get("/ping")

    assert response.status_code == 200
    assert ctx.bound == ["ctx-a"]


def test_non_standard_status_code_is_still_returned():
    def refuse():
        raise HTTPException(status_code=499, detail="client closed request")

    client, _ = build_client(refuse)
    response = client.get("/ping")

    assert response.status_code == 499
    assert response.json()["code"] == "ERR_ERROR"
