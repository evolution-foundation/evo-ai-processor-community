"""CRM-3: the Google services reach core-service under the request's tenant, and
the Calendar routes answer their own errors instead of a relabelled 503."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from src.api.google_calendar_routes import get_calendars, get_google_calendar_service
from src.api.google_sheets_routes import get_google_sheets_service, get_spreadsheets
from src.services.google_calendar_service import GoogleCalendarService
from src.services.google_sheets_service import GoogleSheetsService

CREDENTIALS = {
    "client_id": "cid",
    "client_secret": "secret",
    "redirect_uri": "http://localhost:5173/callback",
}


def _request(token: str | None = None) -> Request:
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers, "state": {}})


def _service(cls, tenant_id):
    return cls(
        client_id="cid",
        client_secret="secret",
        redirect_uri="http://localhost:5173/callback",
        core_service_url="http://core-service:5555/api/v1",
        user_token="tok",
        tenant_id=tenant_id,
    )


@pytest.mark.parametrize("cls", [GoogleCalendarService, GoogleSheetsService])
def test_service_sends_the_tenant_header_only_when_bound(cls):
    assert _service(cls, "t-1").http_client.headers["X-Evo-Tenant-Id"] == "t-1"
    assert "X-Evo-Tenant-Id" not in _service(cls, None).http_client.headers


@pytest.mark.parametrize(
    "dependency, getter",
    [
        (get_google_calendar_service, "get_google_calendar_credentials"),
        (get_google_sheets_service, "get_google_sheets_credentials"),
    ],
)
def test_dependency_threads_the_request_tenant_into_the_service(dependency, getter):
    config_service = MagicMock()
    setattr(config_service, getter, AsyncMock(return_value=CREDENTIALS))

    with patch(
        "src.services.global_config_service.get_global_config_service",
        return_value=config_service,
    ), patch(
        "src.evo_extension_points.runtime_context.current_context_id",
        return_value="t-1",
    ):
        service = asyncio.run(dependency(_request(token="tok")))

    assert service.http_client.headers["X-Evo-Tenant-Id"] == "t-1"


def test_calendar_route_error_is_answered_not_raised():
    service = MagicMock()
    service.get_calendars = AsyncMock(side_effect=RuntimeError("core-service down"))

    response = asyncio.run(
        get_calendars(agent_id="a-1", request=_request(), service=service, db=None, _permission=None, _={})
    )

    assert response.status_code == 500
    assert json.loads(response.body)["success"] is False


def test_sheets_route_error_is_answered_not_raised():
    service = MagicMock()
    service.get_spreadsheets = AsyncMock(side_effect=RuntimeError("core-service down"))

    response = asyncio.run(
        get_spreadsheets(agent_id="a-1", request=_request(), service=service, db=None, _permission=None, _={})
    )

    assert response.status_code == 500
    assert json.loads(response.body)["success"] is False
