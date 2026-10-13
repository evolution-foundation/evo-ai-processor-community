"""
The MCP endpoints the built-in integrations are allowed to talk to.

The integration config stored for an agent can be written outside the OAuth
callback (the configuration PUT merges whatever the body brings, and configs
written before the callback was fixed may hold anything), and the runtime sends
the integration's Bearer token to the URL it resolves. For the built-in
providers that URL is never taken from storage unless it is one of the
provider's own endpoints. Custom MCP servers are not in this table and keep
their configured URL.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Every spelling the code uses today, so a working connection keeps its URL.
PROVIDER_MCP_URLS = {
    "github": ("https://api.githubcopilot.com/mcp/",),
    "notion": ("https://mcp.notion.com/mcp",),
    "asana": ("https://mcp.asana.com/mcp",),
    "monday": ("https://mcp.monday.com/mcp",),
    "atlassian": ("https://mcp.atlassian.com/mcp",),
    "supabase": ("https://mcp.supabase.com/mcp",),
    "hubspot": ("https://mcp.hubspot.com/mcp",),
    "linear": ("https://mcp.linear.app/mcp",),
    "canva": ("https://mcp.canva.com", "https://mcp.canva.com/mcp"),
    "paypal": (
        "https://mcp.paypal.com",
        "https://mcp.paypal.com/mcp",
        "https://mcp.sandbox.paypal.com",
        "https://mcp.sandbox.paypal.com/mcp",
    ),
}


def _normalized(url: str) -> str:
    return url.strip().rstrip("/")


def is_provider_mcp_url(provider: str, url: Optional[str]) -> bool:
    allowed = PROVIDER_MCP_URLS.get((provider or "").lower())
    if not allowed or not url:
        return False
    return _normalized(url) in {_normalized(u) for u in allowed}


def trusted_mcp_url(provider: str, url: Optional[str], fallback: str) -> str:
    """`url` when it is one of the provider's own endpoints, else `fallback`.

    Providers outside the table (custom MCP servers) return `url` unchanged.
    """
    if (provider or "").lower() not in PROVIDER_MCP_URLS:
        return url or fallback
    if is_provider_mcp_url(provider, url):
        return url
    if url:
        logger.warning(
            f"[{provider}] Ignoring a stored MCP URL that is not the provider's "
            f"endpoint; using {fallback}"
        )
    return fallback
