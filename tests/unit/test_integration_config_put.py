"""The configuration PUT of an integration writes only what the user edits; credentials come from the OAuth callback."""

import importlib
import inspect
import json

import pytest
from pydantic import BaseModel

import src.api.mcp_integration_base as mcp_integration_base
import src.services.agent_service as agent_service

AGENT_ID = "0b6f7c1e-3f55-4a51-9a49-7d0d0c2b4a10"

MCP_PROVIDERS = [
    "asana",
    "atlassian",
    "canva",
    "github",
    "hubspot",
    "linear",
    "monday",
    "notion",
    "paypal",
    "supabase",
]

STORED = {
    "provider": "x",
    "connected": True,
    "mcp_url": "https://mcp.provider.example/mcp",
    "username": "alice",
    "email": "alice@example.com",
    "access_token": "stored-access",
    "refresh_token": "stored-refresh",
    "client_id": "stored-client-id",
    "client_secret": "stored-client-secret",
    "pkce_verifiers": {"state-1": "stored-verifier"},
    "oauth_metadata": {"token_endpoint": "https://auth.provider.example/token"},
    "headers": {"Authorization": "Bearer stored-access"},
    "tools": ["search"],
}

FORGED = {
    "access_token": "forged-access",
    "refresh_token": "forged-refresh",
    "client_id": "forged-client-id",
    "client_secret": "forged-client-secret",
    "pkce_verifiers": {"state-1": "forged-verifier"},
    "code_verifier": "forged-verifier",
    "oauth_metadata": {"token_endpoint": "https://attacker.example/token"},
    "mcp_url": "https://attacker.example/mcp",
    "headers": {"Authorization": "Bearer forged-access"},
    "expires_at": "2099-01-01T00:00:00",
    "connected": False,
}


class _Service:
    async def _load_credentials(self, agent_id):
        return None


def _put_endpoint(module_name):
    router = importlib.import_module(f"src.api.{module_name}_routes").router
    for route in router.routes:
        if "PUT" in route.methods:
            return route.endpoint
    raise AssertionError(f"{module_name} has no PUT route")


@pytest.fixture
def saved(monkeypatch):
    writes = []

    async def fake_get_integration_config(db, agent_id, provider, load_from_service=None):
        return {**json.loads(json.dumps(STORED)), "provider": provider}

    async def fake_upsert(db, agent_id, provider, config):
        writes.append((provider, config))
        return True

    monkeypatch.setattr(mcp_integration_base, "get_integration_config", fake_get_integration_config)
    monkeypatch.setattr(agent_service, "upsert_agent_integration", fake_upsert)
    return writes


async def _put(provider, body):
    endpoint = _put_endpoint(provider)
    kwargs = {
        "agent_id": AGENT_ID,
        "config": {"config": body},
        "db": object(),
        "service": _Service(),
        "_permission": None,
        "_": {},
    }
    if "request" in inspect.signature(endpoint).parameters:
        kwargs["request"] = None
    return await endpoint(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", MCP_PROVIDERS)
async def test_put_does_not_overwrite_credentials_or_oauth_fields(provider, saved):
    response = await _put(provider, {**FORGED, "tools": ["search", "create_page"]})

    assert response.status_code == 200
    (written_provider, written), = saved
    assert written_provider == provider
    for field in FORGED:
        if field == "connected":
            continue
        assert written.get(field) == STORED.get(field), field
    assert written["connected"] is True
    assert written["tools"] == ["search", "create_page"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", MCP_PROVIDERS)
async def test_put_saves_the_editable_fields(provider, saved):
    # What the config dialogs send: the sanitized GET config plus the selected tools.
    echoed = {k: v for k, v in STORED.items() if k not in ("access_token", "refresh_token", "client_id", "client_secret", "pkce_verifiers")}
    response = await _put(provider, {**echoed, "tools": []})

    assert response.status_code == 200
    (_, written), = saved
    assert written == {**STORED, "provider": provider, "tools": []}


@pytest.mark.parametrize("provider", MCP_PROVIDERS)
def test_every_put_provider_has_a_closed_field_list(provider):
    assert mcp_integration_base.EDITABLE_CONFIG_FIELDS[provider] == ("tools",)


class _GoogleService:
    def __init__(self):
        self.saved = []

    async def save_configuration(self, agent_id, config):
        self.saved.append(config)
        return True


GOOGLE = {
    "google_calendar": ("calendars", [{"id": "primary", "name": "Agenda", "email": None, "primary": True, "selected": True}]),
    "google_sheets": ("spreadsheets", [{"id": "sheet-1", "name": "Leads", "url": None, "selected": True}]),
}


async def _google_put(provider, body):
    endpoint = _put_endpoint(provider)
    params = inspect.signature(endpoint).parameters
    # Calendar names the body `request`; Sheets names it `config_request` and also takes the Starlette `request`.
    body_param = next(
        name for name, p in params.items()
        if isinstance(p.annotation, type) and issubclass(p.annotation, BaseModel)
    )
    model = params[body_param].annotation
    service = _GoogleService()
    kwargs = {"agent_id": AGENT_ID, body_param: model(**body), "service": service, "_permission": None, "_": {}}
    if "request" in params and body_param != "request":
        kwargs["request"] = None
    response = await endpoint(**kwargs)
    return response, service.saved


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(GOOGLE))
async def test_google_put_drops_credential_fields(provider):
    field, value = GOOGLE[provider]
    body = {**FORGED, "token": "forged-google-token", "provider": provider, field: value, "settings": {"timezone": "America/Sao_Paulo"}}

    response, saved = await _google_put(provider, body)

    assert response.status_code == 200
    (written,) = saved
    for forged in (*FORGED, "token"):
        if forged == "connected":
            continue
        assert forged not in written, forged
    assert written[field] == value
    assert written["settings"] == {"timezone": "America/Sao_Paulo"}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(GOOGLE))
async def test_google_put_saves_the_editable_fields(provider):
    field, value = GOOGLE[provider]
    body = {"provider": provider, "email": "alice@example.com", "connected": True, field: value, "settings": {"notify": True}}

    response, saved = await _google_put(provider, body)

    assert response.status_code == 200
    (written,) = saved
    assert written[field] == value
    assert written["settings"] == {"notify": True}
    assert written["email"] == "alice@example.com"
