"""The EvoCRM tools act on the turn's conversation/contact/pipeline item, never on an id the model picked."""

import pytest

from src.services.adk.tools.evo_crm import context_ids
from src.services.adk.tools.evo_crm.base import EvoCrmClient
from src.services.adk.tools.evo_crm.context_ids import (
    extract_contact_id,
    extract_conversation_id,
    extract_pipeline_item_id,
    resolve_id,
)
from src.services.adk.tools.evo_crm.link_product_to_pipeline_item import (
    create_link_product_to_pipeline_item_tool,
)
from src.services.adk.tools.evo_crm.manage_conversation_labels import (
    create_manage_conversation_labels_tool,
)
from src.services.adk.tools.evo_crm.send_private_message import (
    create_send_private_message_tool,
)
from src.services.adk.tools.evo_crm.transfer_to_human import (
    create_transfer_to_human_tool,
)
from src.services.adk.tools.evo_crm.update_contact import create_update_contact_tool

CONV_ID = "16e5207a-8be2-419e-9f31-d9b70a39a307"
CONTACT_ID = "c6d3efd5-9d2b-42b7-b1cf-93d190255618"
ITEM_ID = "0b6f7c1e-3f55-4a51-9a49-7d0d0c2b4a10"
MODEL_ID = "9f1d2e3c-0000-4000-8000-000000000bad"


class _Ctx:
    """Minimal ToolContext stand-in: the tools only read `.state`."""

    def __init__(self, state):
        self.state = state


def _turn_ctx():
    return _Ctx(
        {
            "evoai_crm_data": {
                "conversation_id": CONV_ID,
                "conversation": {"id": CONV_ID, "display_id": 2},
                "contactId": CONTACT_ID,
                "pipeline_item_id": ITEM_ID,
            }
        }
    )


class TestExtractors:
    def test_read_the_ids_of_the_turn(self):
        ctx = _turn_ctx()

        assert extract_conversation_id(ctx) == CONV_ID
        assert extract_contact_id(ctx) == CONTACT_ID
        assert extract_pipeline_item_id(ctx) == ITEM_ID

    def test_conversation_dot_id_is_the_uuid(self):
        ctx = _Ctx(
            {"evoai_crm_data": {"conversation": {"id": CONV_ID, "display_id": 2}}}
        )

        assert extract_conversation_id(ctx) == CONV_ID

    def test_a_key_present_with_none_is_not_the_string_none(self):
        ctx = _Ctx(
            {"conversation_id": None, "pipeline_item_id": None, "contact_id": None}
        )

        assert extract_conversation_id(ctx) is None
        assert extract_pipeline_item_id(ctx) is None
        assert extract_contact_id(ctx) is None

    def test_an_empty_source_falls_through_to_the_next(self):
        ctx = _Ctx(
            {"evoai_crm_data": {"conversation_id": "  "}, "conversationId": CONV_ID}
        )

        assert extract_conversation_id(ctx) == CONV_ID

    def test_no_context_yields_none(self):
        for extract in (
            extract_conversation_id,
            extract_contact_id,
            extract_pipeline_item_id,
        ):
            assert extract(None) is None
            assert extract(_Ctx({})) is None


class TestResolveId:
    @pytest.fixture
    def warnings(self, monkeypatch):
        # The project logger does not propagate, so caplog never sees it.
        seen = []
        monkeypatch.setattr(
            context_ids.logger, "warning", lambda msg, *a, **k: seen.append(msg)
        )
        return seen

    def test_the_context_wins_and_the_divergence_is_logged(self, warnings):
        assert resolve_id("conversation_id", CONV_ID, MODEL_ID) == CONV_ID

        assert len(warnings) == 1
        assert MODEL_ID in warnings[0]

    def test_no_warning_when_the_model_agrees_or_stays_silent(self, warnings):
        assert resolve_id("conversation_id", CONV_ID, CONV_ID) == CONV_ID
        assert resolve_id("conversation_id", CONV_ID, None) == CONV_ID

        assert warnings == []

    def test_the_model_decides_only_when_the_context_is_silent(self):
        assert resolve_id("conversation_id", None, MODEL_ID) == MODEL_ID
        assert resolve_id("conversation_id", None, "   ") is None
        assert resolve_id("conversation_id", None, None) is None


@pytest.fixture
def crm_calls(monkeypatch):
    """Records every request the tools send to the CRM instead of sending it."""
    calls = []

    def recorder(method):
        async def fake(self, endpoint, json_data=None, **_):
            calls.append((method, endpoint, json_data))
            return {"payload": []}

        return fake

    for method in ("get", "post", "put", "patch", "delete"):
        monkeypatch.setattr(EvoCrmClient, method, recorder(method))
    return calls


def _endpoints(calls):
    return [endpoint for _, endpoint, _ in calls]


def _assert_every_call_targets(calls, expected_id):
    assert calls, "the tool sent nothing to the CRM"
    assert all(expected_id in endpoint for endpoint in _endpoints(calls))


def _assert_only_the_turns_id_reached_the_crm(calls, expected_id):
    _assert_every_call_targets(calls, expected_id)
    for _, endpoint, body in calls:
        assert MODEL_ID not in endpoint
        assert MODEL_ID not in str(body)


@pytest.mark.asyncio
class TestEachToolActsOnTheTurnNotOnTheModelsId:
    async def test_update_contact(self, crm_calls):
        await create_update_contact_tool().func(
            name="Ana", contact_id=MODEL_ID, tool_context=_turn_ctx()
        )

        _assert_only_the_turns_id_reached_the_crm(crm_calls, CONTACT_ID)

    async def test_send_private_message(self, crm_calls):
        await create_send_private_message_tool()(
            content="lembrete", conversation_id=MODEL_ID, tool_context=_turn_ctx()
        )

        _assert_only_the_turns_id_reached_the_crm(crm_calls, CONV_ID)

    async def test_transfer_to_human(self, crm_calls):
        await create_transfer_to_human_tool().func(
            assignee_id="agent-1", conversation_id=MODEL_ID, tool_context=_turn_ctx()
        )

        _assert_only_the_turns_id_reached_the_crm(crm_calls, CONV_ID)

    async def test_manage_conversation_labels(self, crm_calls):
        await create_manage_conversation_labels_tool().func(
            action="list", conversation_id=MODEL_ID, tool_context=_turn_ctx()
        )

        _assert_only_the_turns_id_reached_the_crm(crm_calls, CONV_ID)

    async def test_link_product_to_pipeline_item(self, crm_calls):
        await create_link_product_to_pipeline_item_tool().func(
            product_id="prod-1",
            quantity=1,
            pipeline_item_id=MODEL_ID,
            tool_context=_turn_ctx(),
        )

        _assert_only_the_turns_id_reached_the_crm(crm_calls, ITEM_ID)


@pytest.mark.asyncio
class TestTheModelsIdStandsWhenTheContextIsSilent:
    """Outside a conversation (playground, direct API calls) the argument is all there is."""

    async def test_update_contact(self, crm_calls):
        await create_update_contact_tool().func(
            name="Ana", contact_id=MODEL_ID, tool_context=_Ctx({})
        )

        assert _endpoints(crm_calls) == [f"/contacts/{MODEL_ID}"]

    async def test_send_private_message(self, crm_calls):
        await create_send_private_message_tool()(
            content="lembrete", conversation_id=MODEL_ID, tool_context=_Ctx({})
        )

        _assert_every_call_targets(crm_calls, MODEL_ID)

    async def test_transfer_to_human(self, crm_calls):
        await create_transfer_to_human_tool().func(
            assignee_id="agent-1", conversation_id=MODEL_ID, tool_context=_Ctx({})
        )

        _assert_every_call_targets(crm_calls, MODEL_ID)

    async def test_manage_conversation_labels(self, crm_calls):
        await create_manage_conversation_labels_tool().func(
            action="list", conversation_id=MODEL_ID, tool_context=_Ctx({})
        )

        _assert_every_call_targets(crm_calls, MODEL_ID)

    async def test_link_product_to_pipeline_item(self, crm_calls):
        await create_link_product_to_pipeline_item_tool().func(
            product_id="prod-1",
            quantity=1,
            pipeline_item_id=MODEL_ID,
            tool_context=_Ctx({}),
        )

        assert _endpoints(crm_calls) == [f"/pipeline_items/{MODEL_ID}/products"]
