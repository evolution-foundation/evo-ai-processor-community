"""CRM-3: a missing Google OAuth credential is a configuration error, not a 500."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api.google_calendar_routes import get_google_calendar_service
from src.api.google_sheets_routes import get_google_sheets_service
from src.core.error_codes import BUSINESS_RULE_VIOLATION


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/", "headers": [], "state": {}})


@pytest.mark.parametrize(
    "dependency, getter",
    [
        (get_google_calendar_service, "get_google_calendar_credentials"),
        (get_google_sheets_service, "get_google_sheets_credentials"),
    ],
)
def test_missing_google_credential_is_a_422_configuration_error(dependency, getter):
    config_service = MagicMock()
    setattr(config_service, getter, AsyncMock(return_value={}))

    with patch(
        "src.services.global_config_service.get_global_config_service",
        return_value=config_service,
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(dependency(_request()))

    assert exc.value.status_code == 422
    assert exc.value.detail["error_code"] == BUSINESS_RULE_VIOLATION
    assert "Client ID and Client Secret" in exc.value.detail["error"]
