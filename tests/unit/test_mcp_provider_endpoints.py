"""The runtime sends an integration's Bearer only to the provider's own MCP endpoint."""

import importlib
import re
from pathlib import Path

import pytest

from src.services import mcp_provider_endpoints
from src.services.mcp_provider_endpoints import (
    PROVIDER_MCP_URLS,
    is_provider_mcp_url,
    trusted_mcp_url,
)

FOREIGN = "https://forged.invalid/mcp"
SRC = Path(__file__).resolve().parents[2] / "src"


def test_a_provider_endpoint_is_kept_with_or_without_trailing_slash():
    assert trusted_mcp_url("github", "https://api.githubcopilot.com/mcp", "x") == (
        "https://api.githubcopilot.com/mcp"
    )
    assert trusted_mcp_url("canva", "https://mcp.canva.com/mcp/", "x") == (
        "https://mcp.canva.com/mcp/"
    )


@pytest.mark.parametrize("provider", sorted(PROVIDER_MCP_URLS))
@pytest.mark.parametrize("stored", [FOREIGN, "http://localhost:8000/mcp", "", None])
def test_anything_else_falls_back_to_the_provider_default(provider, stored):
    default = PROVIDER_MCP_URLS[provider][0]

    assert trusted_mcp_url(provider, stored, default) == default


def test_a_lookalike_host_is_not_the_provider():
    assert not is_provider_mcp_url(
        "notion", "https://mcp.notion.com.forged.invalid/mcp"
    )
    assert not is_provider_mcp_url("notion", "https://mcp.notion.com/mcp?x=1")


def test_the_provider_name_is_matched_case_insensitively():
    assert trusted_mcp_url("Notion", FOREIGN, "x") == "x"


def test_custom_servers_keep_their_url():
    assert trusted_mcp_url("my-custom-server", FOREIGN, "x") == FOREIGN


def _service_defaults():
    services = {
        "github": "github_service.GitHubService",
        "notion": "notion_service.NotionService",
        "asana": "asana_service.AsanaService",
        "monday": "monday_service.MondayService",
        "atlassian": "atlassian_service.AtlassianService",
        "supabase": "supabase_service.SupabaseService",
        "hubspot": "hubspot_service.HubSpotService",
        "linear": "linear_service.LinearService",
        "canva": "canva_service.CanvaService",
    }
    for provider, path in services.items():
        module, cls = path.rsplit(".", 1)
        yield provider, getattr(
            importlib.import_module(f"src.services.{module}"), cls
        ).DEFAULT_MCP_URL
    from src.services.paypal_service import PayPalService

    yield "paypal", PayPalService.DEFAULT_MCP_URL_PRODUCTION
    yield "paypal", PayPalService.DEFAULT_MCP_URL_SANDBOX


@pytest.mark.parametrize("provider,url", list(_service_defaults()))
def test_every_service_default_is_a_provider_endpoint(provider, url):
    # A default left out of the table would cut a working connection.
    assert is_provider_mcp_url(provider, url)


@pytest.mark.parametrize("provider", sorted(PROVIDER_MCP_URLS))
def test_every_runtime_config_default_is_a_provider_endpoint(provider):
    source = (SRC / "services/adk/mcp_servers" / provider / "config.py").read_text()
    defaults = re.findall(r'default_url\s*=\s*"([^"]+)"', source) + re.findall(
        r'"mcp_url"\),\s*"([^"]+)"', source
    )

    assert defaults, f"{provider}: no default URL found in the runtime config"
    assert all(is_provider_mcp_url(provider, url) for url in defaults), defaults


def test_every_agent_builder_default_is_a_provider_endpoint():
    from src.services.adk.agents.llm_agent_builder import INTEGRATION_MCP_URLS

    for provider, url in INTEGRATION_MCP_URLS.items():
        if provider in PROVIDER_MCP_URLS:
            assert is_provider_mcp_url(provider, url), (provider, url)


@pytest.mark.parametrize("provider", ["notion", "linear", "github", "asana", "hubspot"])
def test_the_agent_builder_loads_tools_from_the_provider_endpoint(provider):
    from src.services.adk.agents.llm_agent_builder import (
        INTEGRATION_MCP_URLS,
        integration_mcp_url_for,
    )

    url = integration_mcp_url_for(provider, {"mcp_url": FOREIGN, "connected": True})

    assert url == INTEGRATION_MCP_URLS[provider]


@pytest.mark.asyncio
async def test_tool_discovery_never_calls_a_foreign_stored_url(monkeypatch):
    from src.api import mcp_integration_base

    async def stored(*args, **kwargs):
        return {"access_token": "tok", "mcp_url": FOREIGN}

    seen = {}

    async def discover(mcp_config):
        seen.update(mcp_config)
        return []

    monkeypatch.setattr(mcp_integration_base, "get_integration_config", stored)
    monkeypatch.setattr(mcp_integration_base, "_discover_async", discover)

    await mcp_integration_base.discover_tools_endpoint(
        agent_id="agent-1",
        provider="notion",
        mcp_url="https://mcp.notion.com/mcp",
        db=None,
    )

    assert seen["url"] == "https://mcp.notion.com/mcp"


async def _no_integrations(*args, **kwargs):
    return []


def _integration(provider, mcp_url):
    async def fetch(db, agent_id):
        return [
            {
                "provider": provider,
                "config": {"access_token": "tok", "mcp_url": mcp_url, "tools": []},
            }
        ]

    return fetch


READS_STORED_CONFIG = [
    "asana",
    "canva",
    "monday",
    "paypal",
    "supabase",
    "atlassian",
    "hubspot",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", READS_STORED_CONFIG)
async def test_a_foreign_stored_url_never_receives_the_bearer(provider, monkeypatch):
    module = importlib.import_module(f"src.services.adk.mcp_servers.{provider}.config")
    monkeypatch.setattr(
        module, "get_agent_integrations", _integration(provider, FOREIGN), raising=False
    )
    monkeypatch.setattr(
        "src.services.agent_service.get_agent_integrations",
        _integration(provider, FOREIGN),
    )
    get_config = getattr(module, f"get_{provider}_mcp_config")

    config = await get_config(agent_id="agent-1", db=object())

    assert config["url"] != FOREIGN
    assert is_provider_mcp_url(provider, config["url"])
    assert config["headers"].get("Authorization") == "Bearer tok"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["github", "notion", "linear"])
async def test_a_foreign_url_handed_to_the_runtime_is_replaced(provider, monkeypatch):
    module = importlib.import_module(f"src.services.adk.mcp_servers.{provider}.config")
    get_config = getattr(module, f"get_{provider}_mcp_config")

    config = await get_config(agent_id="agent-1", mcp_url=FOREIGN, db=None)

    assert config["url"] != FOREIGN
    assert is_provider_mcp_url(provider, config["url"])


def test_a_rejected_url_is_logged_without_the_token(caplog, monkeypatch):
    seen = []
    monkeypatch.setattr(mcp_provider_endpoints.logger, "warning", seen.append)

    trusted_mcp_url("notion", FOREIGN, "https://mcp.notion.com/mcp")

    assert len(seen) == 1 and "Bearer" not in seen[0]
