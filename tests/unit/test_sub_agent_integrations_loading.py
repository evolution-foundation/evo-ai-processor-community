"""Integrations stored in `evo_core_agent_integrations` must reach every LLM agent.

Only the root agent used to get `_integrations` attached (RunnerUtils.get_and_build_agent).
Sub agents come out of get_agent() without it, so integration backed tools such as
knowledge_nexus_search were never built for them.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.adk.agents.llm_agent_builder import LlmAgentBuilder
from src.services.adk.tool_builder import ToolBuilder

FETCH = "src.services.agent_service.get_agent_integrations"

NEXUS_ROW = {
    "provider": "knowledge_nexus",
    "config": {
        "connected": True,
        "nexus_base_url": "https://nexus.example.io",
        "nexus_api_key": "evo_k_prefix.secret",
        "space_id": "space-1",
    },
}


class _StopBuild(Exception):
    """Aborts _create_llm_agent right after the tools are built."""


def _agent(**attrs):
    return SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="sub_agent",
        type="llm",
        config={},
        **attrs,
    )


async def _config_seen_by_tool_builder(agent) -> dict:
    builder = LlmAgentBuilder(MagicMock())
    captured = {}

    def capture(config, db=None, agent_id=None):
        captured["config"] = config
        raise _StopBuild

    builder.tool_builder.build_tools = capture
    with pytest.raises(_StopBuild):
        await builder._create_llm_agent(agent)
    return captured["config"]


@pytest.mark.asyncio
async def test_sub_agent_without_attached_integrations_loads_them_from_the_database():
    sub_agent = _agent()

    with patch(FETCH, new=AsyncMock(return_value=[NEXUS_ROW])) as fetch:
        config = await _config_seen_by_tool_builder(sub_agent)

    fetch.assert_awaited_once()
    assert config["integrations"]["knowledge-nexus"]["space_id"] == "space-1"


@pytest.mark.asyncio
async def test_knowledge_nexus_search_tool_is_built_for_the_sub_agent():
    sub_agent = _agent()

    with patch(FETCH, new=AsyncMock(return_value=[NEXUS_ROW])):
        config = await _config_seen_by_tool_builder(sub_agent)

    tools = ToolBuilder().build_tools(config, None, sub_agent.id)

    assert "knowledge_nexus_search" in [tool.func.__name__ for tool in tools]


@pytest.mark.asyncio
async def test_root_agent_integrations_attached_by_the_runner_are_not_fetched_again():
    root_agent = _agent(_integrations=[NEXUS_ROW])

    with patch(FETCH, new=AsyncMock(return_value=[])) as fetch:
        config = await _config_seen_by_tool_builder(root_agent)

    fetch.assert_not_awaited()
    assert "knowledge-nexus" in config["integrations"]


@pytest.mark.asyncio
async def test_an_empty_attached_list_is_respected_and_not_refetched():
    root_agent = _agent(_integrations=[])

    with patch(FETCH, new=AsyncMock(return_value=[NEXUS_ROW])) as fetch:
        config = await _config_seen_by_tool_builder(root_agent)

    fetch.assert_not_awaited()
    assert "knowledge-nexus" not in config["integrations"]
