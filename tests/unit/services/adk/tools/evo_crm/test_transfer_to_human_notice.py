from unittest.mock import AsyncMock, patch

import pytest

from src.services.adk.tools.evo_crm.transfer_to_human import create_transfer_to_human_tool


async def _transfer_with_failing_assignment(notice_error: Exception | None):
    """Runs transfer_to_human with a customer notice configured, an
    assignment POST that always fails, and a notice POST that fails only
    when notice_error is given."""
    tool = create_transfer_to_human_tool()

    async def post_side_effect(endpoint, json_data=None, **kwargs):
        if endpoint.endswith("/messages"):
            if notice_error:
                raise notice_error
            return {"success": True}
        if endpoint.endswith("/assignments"):
            raise RuntimeError("500 Internal Server Error")
        return {"success": True}

    with patch(
        "src.services.adk.tools.evo_crm.transfer_to_human.EvoCrmClient.post",
        new=AsyncMock(side_effect=post_side_effect),
    ):
        return await tool.func(
            conversation_id="conv-1",
            assignee_id="user-1",
            message_to_customer="Você será transferido em instantes.",
        )


@pytest.mark.asyncio
async def test_error_response_reports_notice_already_sent():
    """The customer notice and the assignment are two separate, already-
    committed POSTs. If the notice succeeds but the assignment then fails,
    the error response must say so — otherwise a caller/model retry has no
    way to know the customer was already told a transfer was coming, and
    could resend the same notice."""
    result = await _transfer_with_failing_assignment(notice_error=None)

    assert result["status"] == "error"
    assert result["message_to_customer_sent"] is True
    assert result["message_to_customer_error"] is None
    assert "already sent" in result["message"]


@pytest.mark.asyncio
async def test_error_response_reports_notice_not_sent_when_it_also_failed():
    result = await _transfer_with_failing_assignment(notice_error=RuntimeError("notice failed"))

    assert result["status"] == "error"
    assert result["message_to_customer_sent"] is False
    assert result["message_to_customer_error"] == "notice failed"
    assert "already sent" not in result["message"]
