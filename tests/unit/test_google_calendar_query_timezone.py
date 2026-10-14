"""CRM-3: the busy-event query sends the window in the agent timezone.

The tool receives naive times in the agent timezone. The query appended a bare 'Z',
so Google read them as UTC and the window slid by the offset: in America/Sao_Paulo a
10:00-11:00 meeting fell outside an 08:00-12:00 search and was offered as free.
The other availability tests mock check_availability whole, so they never saw the
query; these patch one level lower, at the Google service.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.services.adk.tools.google_calendar import create_check_availability_tool
from src.services.adk.tool_builder import ToolBuilder
from src.services.adk.tools.google_calendar.base import GoogleCalendarClient

CREDS = {
    "refresh_token": "1//refresh",
    "client_id": "client-id",
    "client_secret": "secret",
    "token": "ya29.token",
    "token_uri": "https://oauth2.googleapis.com/token",
}


def _google_service():
    service = MagicMock()
    service.events.return_value.list.return_value.execute.return_value = {"items": []}
    return service


def _query_window(service):
    kwargs = service.events.return_value.list.call_args.kwargs
    return kwargs["timeMin"], kwargs["timeMax"]


def test_naive_times_are_read_in_the_given_timezone():
    service = _google_service()
    with patch.object(GoogleCalendarClient, "get_calendar_service", return_value=service):
        asyncio.run(
            GoogleCalendarClient().check_availability(
                CREDS, datetime(2026, 10, 15, 8), datetime(2026, 10, 15, 12), "primary", "America/Sao_Paulo"
            )
        )

    assert _query_window(service) == ("2026-10-15T08:00:00-03:00", "2026-10-15T12:00:00-03:00")


def test_aware_times_keep_their_offset():
    service = _google_service()
    with patch.object(GoogleCalendarClient, "get_calendar_service", return_value=service):
        asyncio.run(
            GoogleCalendarClient().check_availability(
                CREDS,
                datetime(2026, 10, 15, 11, tzinfo=timezone.utc),
                datetime(2026, 10, 15, 15, tzinfo=timezone.utc),
            )
        )

    assert _query_window(service) == ("2026-10-15T11:00:00+00:00", "2026-10-15T15:00:00+00:00")


def test_slot_search_queries_in_the_agent_configured_timezone():
    tool = create_check_availability_tool(
        agent_id="a-1",
        calendar_config={"settings": {"timezone": "America/Manaus", "minAdvanceTime": 0, "maxDuration": 0}},
        credentials_config=CREDS,
        db=None,
    )
    service = _google_service()
    with patch.object(GoogleCalendarClient, "get_calendar_service", return_value=service):
        asyncio.run(
            tool(start_date="2026-10-15T08:00:00", end_date="2026-10-15T12:00:00", find_slots=True, slot_duration=60)
        )

    assert _query_window(service) == ("2026-10-15T08:00:00-04:00", "2026-10-15T12:00:00-04:00")


def _built_availability_tool(agent_config):
    tools = ToolBuilder().build_tools(agent_config, db=None, agent_id="a-1")
    return next(t for t in tools if getattr(t, "__name__", "") == "check_calendar_availability")


def _agent_config(calendar_settings, timezone=None):
    config = {
        "integrations": {
            "google-calendar": {"connected": True, "settings": calendar_settings},
            "google-calendar-credentials": CREDS,
        }
    }
    if timezone:
        config["timezone"] = timezone
    return config


def _slot_search_window(tool):
    service = _google_service()
    with patch.object(GoogleCalendarClient, "get_calendar_service", return_value=service):
        asyncio.run(
            tool(start_date="2026-10-15T08:00:00", end_date="2026-10-15T12:00:00", find_slots=True, slot_duration=60)
        )
    return _query_window(service)


def test_built_tools_fall_back_to_the_agent_timezone():
    # The calendar dialog never writes settings.timezone; the agent's own one has to reach the tools.
    tool = _built_availability_tool(_agent_config({"minAdvanceTime": 0}, timezone="Europe/Lisbon"))

    assert _slot_search_window(tool) == ("2026-10-15T08:00:00+01:00", "2026-10-15T12:00:00+01:00")


def test_built_tools_keep_a_timezone_set_on_the_calendar():
    tool = _built_availability_tool(
        _agent_config({"minAdvanceTime": 0, "timezone": "America/Manaus"}, timezone="Europe/Lisbon")
    )

    assert _slot_search_window(tool) == ("2026-10-15T08:00:00-04:00", "2026-10-15T12:00:00-04:00")
