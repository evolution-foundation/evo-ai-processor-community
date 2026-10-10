from unittest.mock import AsyncMock, patch

import pytest

from src.services.adk.tools.evo_crm.transfer_to_human import create_transfer_to_human_tool


def _post_mock(return_value=None):
    return AsyncMock(return_value=return_value or {"success": True, "data": {}})


@pytest.mark.asyncio
async def test_keyword_match_skips_a_rule_missing_its_target_id():
    """A rule can win keyword scoring on its instructions text alone while
    having no userId/teamId for its transferTo — selecting it would leave
    both effective_assignee_id and effective_team_id unset. The tool should
    skip it and fall through to the next usable rule instead of erroring."""
    rules = [
        {
            "transferTo": "team",
            "teamId": None,  # misconfigured: no team to assign to
            "instructions": "quando o cliente pedir reembolso urgente",
        },
        {
            "transferTo": "human",
            "userId": "user-42",
            "instructions": "quando o cliente quiser falar sobre pagamento",
        },
    ]
    tool = create_transfer_to_human_tool(transfer_rules=rules)
    fn = tool.func

    with patch(
        "src.services.adk.tools.evo_crm.transfer_to_human.EvoCrmClient.post",
        new=_post_mock(),
    ) as mock_post:
        result = await fn(
            conversation_id="conv-1",
            reason="cliente pedir reembolso urgente",  # matches rule 1's instructions best
        )

    assert result["status"] == "success"
    # Routed to rule 2 (the only rule with a usable target), not rule 1.
    assign_call = next(c for c in mock_post.call_args_list if "assignments" in c.kwargs["endpoint"])
    assert assign_call.kwargs["json_data"] == {"assignee_id": "user-42"}


@pytest.mark.asyncio
async def test_errors_when_every_rule_lacks_a_valid_target():
    rules = [{"transferTo": "team", "teamId": None, "instructions": "algo"}]
    tool = create_transfer_to_human_tool(transfer_rules=rules)
    fn = tool.func

    with patch(
        "src.services.adk.tools.evo_crm.transfer_to_human.EvoCrmClient.post",
        new=_post_mock(),
    ):
        result = await fn(conversation_id="conv-1", reason="algo")

    assert result["status"] == "error"
    assert "assignee_id or team_id is required" in result["message"]


@pytest.mark.asyncio
async def test_explicit_rule_index_still_errors_on_its_own_malformed_rule():
    """An explicit rule_index is the model's own choice — if THAT specific
    rule is malformed, this stays a reported contract failure rather than
    silently substituting a different rule (unlike the keyword/fallback
    paths, which have no such explicit signal to honor)."""
    rules = [
        {"transferTo": "team", "teamId": None, "instructions": "malformed rule"},
        {"transferTo": "human", "userId": "user-42", "instructions": "valid rule"},
    ]
    tool = create_transfer_to_human_tool(transfer_rules=rules)
    fn = tool.func

    with patch(
        "src.services.adk.tools.evo_crm.transfer_to_human.EvoCrmClient.post",
        new=_post_mock(),
    ):
        result = await fn(conversation_id="conv-1", rule_index=1)

    assert result["status"] == "error"
    assert "Transfer rule #1" in result["message"]
    assert "missing a valid" in result["message"]
