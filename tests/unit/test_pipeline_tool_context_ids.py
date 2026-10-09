"""CRM-237: the context's conversation/contact must win over the ids the model sends."""

import pytest

from src.services.adk.tools.evo_crm.base import EvoCrmClient
from src.services.adk.tools.evo_crm.context_ids import (
    extract_contact_id,
    extract_conversation_id,
)
from src.services.adk.tools.evo_crm.pipeline_manipulation import create_pipeline_manipulation_tool

CONV_ID = "16e5207a-8be2-419e-9f31-d9b70a39a307"
CONTACT_ID = "c6d3efd5-9d2b-42b7-b1cf-93d190255618"


class _Ctx:
    """Minimal ToolContext stand-in: the tool only reads `.state`."""

    def __init__(self, state):
        self.state = state


def _ctx_with_conversation():
    return _Ctx({
        "evoai_crm_data": {
            "conversation_id": CONV_ID,
            "conversation": {"id": CONV_ID, "display_id": 2},
            "contactId": CONTACT_ID,
        }
    })


class TestExtractionFromMetadata:
    def test_reads_the_conversation_id(self):
        assert extract_conversation_id(_ctx_with_conversation()) == CONV_ID

    def test_reads_the_contact_id(self):
        assert extract_contact_id(_ctx_with_conversation()) == CONTACT_ID

    def test_the_two_ids_are_not_interchangeable(self):
        ctx = _ctx_with_conversation()

        assert extract_conversation_id(ctx) != extract_contact_id(ctx)

    def test_no_context_yields_none(self):
        assert extract_conversation_id(None) is None
        assert extract_contact_id(None) is None
        assert extract_conversation_id(_Ctx({})) is None


@pytest.mark.asyncio
class TestContextWinsOverTheModel:
    """The regression: the model passing a wrong id must not reach the CRM."""

    async def _call(self, monkeypatch, **kwargs):
        seen = {}

        async def fake_move(client, pipeline_id, conversation_id, stage_id, notes,
                            pipeline_rules, stage_name=None):
            seen["conversation_id"] = conversation_id
            seen["stage_id"] = stage_id
            return {"status": "success", "action": "move_to_stage"}

        monkeypatch.setattr(
            "src.services.adk.tools.evo_crm.pipeline_manipulation._move_to_stage", fake_move
        )

        tool = create_pipeline_manipulation_tool()
        await tool.func(**kwargs)
        return seen

    async def test_the_contact_id_the_model_sent_never_reaches_the_crm(self, monkeypatch):
        # Exactly the live failure: model passes the CONTACT id as conversation_id.
        seen = await self._call(
            monkeypatch,
            action="move_to_stage",
            conversation_id=CONTACT_ID,
            pipeline_id="pipe-1",
            stage_id="stg-1",
            tool_context=_ctx_with_conversation(),
        )

        assert seen["conversation_id"] == CONV_ID, "the context's conversation must win"
        assert seen["conversation_id"] != CONTACT_ID

    async def test_context_is_used_when_the_model_omits_the_id(self, monkeypatch):
        seen = await self._call(
            monkeypatch,
            action="move_to_stage",
            pipeline_id="pipe-1",
            stage_id="stg-1",
            tool_context=_ctx_with_conversation(),
        )

        assert seen["conversation_id"] == CONV_ID

    async def test_the_model_still_decides_when_the_context_is_silent(self, monkeypatch):
        # Outside a conversation the context has nothing, so the argument stands.
        seen = await self._call(
            monkeypatch,
            action="move_to_stage",
            conversation_id="explicit-conv",
            pipeline_id="pipe-1",
            stage_id="stg-1",
            tool_context=_Ctx({}),
        )

        assert seen["conversation_id"] == "explicit-conv"


@pytest.mark.asyncio
class TestAddToPipelineSendsTheContextIds:
    """add_to_pipeline is where the live failure happened and the only action where both ids meet."""

    async def _posted_body(self, monkeypatch, **kwargs):
        posted = {}

        async def fake_post(self, endpoint, json_data=None, **_):
            posted["endpoint"] = endpoint
            posted["body"] = json_data
            return {"id": "item-1"}

        monkeypatch.setattr(EvoCrmClient, "post", fake_post)
        tool = create_pipeline_manipulation_tool()
        await tool.func(action="add_to_pipeline", pipeline_id="pipe-1", stage_id="stg-1", **kwargs)
        return posted["body"]

    async def test_the_conversation_from_the_context_is_added_whatever_ids_the_model_sends(self, monkeypatch):
        body = await self._posted_body(
            monkeypatch,
            conversation_id=CONTACT_ID,
            contact_id="model-contact",
            tool_context=_ctx_with_conversation(),
        )

        assert body["type"] == "conversation"
        assert body["item_id"] == CONV_ID

    async def test_the_context_contact_overrides_the_contact_id_the_model_sends(self, monkeypatch):
        # No conversation in the context, so the contact is the item: it must be the context's.
        body = await self._posted_body(
            monkeypatch,
            contact_id="model-contact",
            tool_context=_Ctx({"evoai_crm_data": {"contactId": CONTACT_ID}}),
        )

        assert body["type"] == "contact"
        assert body["item_id"] == CONTACT_ID
