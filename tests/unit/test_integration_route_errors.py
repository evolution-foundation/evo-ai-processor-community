"""Integration routes answer with the error they mean, never with a crash or the exception text."""

import importlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src.utils.response import map_status_to_error_code

AGENT_ID = "0b6f7c1e-3f55-4a51-9a49-7d0d0c2b4a10"
MARKER = "upstream-detail-7f3a"

MCP_PROVIDERS = [
    "asana",
    "atlassian",
    "canva",
    "github",
    "hubspot",
    "linear",
    "monday",
    "notion",
    "paypal",
    "supabase",
]
GOOGLE_PROVIDERS = ["google_calendar", "google_sheets"]


class FakeService:
    """Every service call fails with MARKER unless the test says otherwise."""

    def __init__(self, **overrides):
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return AsyncMock(side_effect=RuntimeError(MARKER))


def _client(provider, service):
    router = importlib.import_module(f"src.api.{provider}_routes").router
    app = FastAPI()
    app.include_router(router)
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        for dep in route.dependant.dependencies:
            if "service" in getattr(dep.call, "__name__", ""):
                app.dependency_overrides[dep.call] = lambda: service
            else:
                app.dependency_overrides[dep.call] = lambda: None
    return TestClient(app, raise_server_exceptions=False), router.prefix.format(
        agent_id=AGENT_ID
    )


def _method_of(provider, suffix):
    router = importlib.import_module(f"src.api.{provider}_routes").router
    for route in router.routes:
        if route.path == router.prefix + suffix:
            return next(iter(route.methods))
    raise AssertionError(f"{provider} has no {suffix!r} route")


def _assert_error(response, status_code):
    assert response.status_code == status_code, response.text
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == map_status_to_error_code(status_code)
    assert MARKER not in response.text


@pytest.mark.parametrize("provider", MCP_PROVIDERS)
def test_discover_failure_is_a_generic_500(provider):
    client, base = _client(provider, FakeService())

    response = client.request(_method_of(provider, "/discover"), f"{base}/discover")

    _assert_error(response, 500)


@pytest.mark.parametrize("provider", MCP_PROVIDERS)
def test_config_read_failure_is_a_generic_500(provider):
    client, base = _client(provider, FakeService())

    with patch(
        "src.api.mcp_integration_base.get_integration_config",
        AsyncMock(side_effect=RuntimeError(MARKER)),
    ):
        response = client.get(base)

    _assert_error(response, 500)


@pytest.mark.parametrize("provider", MCP_PROVIDERS)
def test_config_read_returns_the_sanitized_config(provider):
    client, base = _client(provider, FakeService())

    stored = {"access_token": MARKER, "tools": ["list_issues"]}
    with patch(
        "src.api.mcp_integration_base.get_agent_integrations",
        AsyncMock(return_value=[{"provider": provider, "config": stored}]),
    ):
        response = client.get(base)

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["connected"] is True
    assert data["tools"] == ["list_issues"]
    assert MARKER not in response.text


def test_listing_all_configs_failure_is_a_generic_500():
    client, base = _client("integrations", FakeService())

    with patch(
        "src.services.agent_service.get_agent_integrations",
        AsyncMock(side_effect=RuntimeError(MARKER)),
    ):
        response = client.get(base, headers={"Authorization": "Bearer t"})

    _assert_error(response, 500)


@pytest.mark.parametrize("provider", MCP_PROVIDERS)
def test_saving_config_without_a_connected_integration_is_404(provider):
    client, base = _client(
        provider, FakeService(_load_credentials=AsyncMock(return_value=None))
    )

    with patch(
        "src.api.mcp_integration_base.get_integration_config",
        AsyncMock(return_value=None),
    ):
        response = client.put(base, json={"config": {}})

    _assert_error(response, 404)


@pytest.mark.parametrize("provider", MCP_PROVIDERS)
def test_saving_config_failure_is_a_generic_500(provider):
    client, base = _client(provider, FakeService())

    with patch(
        "src.api.mcp_integration_base.get_integration_config",
        AsyncMock(side_effect=RuntimeError(MARKER)),
    ):
        response = client.put(base, json={"config": {}})

    _assert_error(response, 500)


@pytest.mark.parametrize("provider", MCP_PROVIDERS + GOOGLE_PROVIDERS)
def test_disconnect_failure_is_a_generic_500(provider):
    client, base = _client(provider, FakeService())

    response = client.delete(base)

    _assert_error(response, 500)


@pytest.mark.parametrize(
    "method,suffix,body",
    [
        ("PUT", "", {}),
        ("POST", "/availability", {"calendarId": "primary", "start": "a", "end": "b"}),
        (
            "POST",
            "/events",
            {"calendarId": "primary", "summary": "s", "start": "a", "end": "b"},
        ),
    ],
)
def test_google_calendar_body_route_failure_is_a_generic_500(method, suffix, body):
    # These handlers name their pydantic body `request`, which error_response cannot read.
    client, base = _client("google_calendar", FakeService())

    response = client.request(method, base + suffix, json=body)

    _assert_error(response, 500)


def test_google_sheets_save_failure_is_a_generic_500():
    client, base = _client("google_sheets", FakeService())

    response = client.put(base, json={})

    _assert_error(response, 500)


@pytest.mark.parametrize(
    "provider,suffix", [("google_calendar", "/calendars"), ("google_sheets", "/spreadsheets")]
)
def test_google_listing_failure_is_a_generic_500(provider, suffix):
    client, base = _client(provider, FakeService())

    response = client.get(base + suffix)

    _assert_error(response, 500)
