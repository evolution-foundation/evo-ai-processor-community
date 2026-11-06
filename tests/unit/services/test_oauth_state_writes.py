"""OAuth state (PKCE verifiers, client credentials, tokens) is written to the database, never through the core API."""

import httpx
import pytest
from google.oauth2.credentials import Credentials

import src.services.agent_service as agent_service
from src.services.asana_service import AsanaService
from src.services.atlassian_service import AtlassianService
from src.services.canva_service import CanvaService
from src.services.github_service import GitHubService
from src.services.google_calendar_service import GoogleCalendarService
from src.services.google_sheets_service import GoogleSheetsService
from src.services.hubspot_service import HubSpotService
from src.services.linear_service import LinearService
from src.services.monday_service import MondayService
from src.services.notion_service import NotionService
from src.services.paypal_service import PayPalService
from src.services.supabase_service import SupabaseService

AGENT_ID = "0b6f7c1e-3f55-4a51-9a49-7d0d0c2b4a10"
COMMON = {
    "redirect_uri": "https://app.example/callback",
    "core_service_url": "http://core.invalid",
    "user_token": "t",
}
CREDS = {"client_id": "cid", "client_secret": "csecret"}

MCP_SERVICES = {
    "asana": lambda: AsanaService(**COMMON, **CREDS),
    "atlassian": lambda: AtlassianService(**COMMON, **CREDS),
    "canva": lambda: CanvaService(**COMMON, **CREDS),
    "github": lambda: GitHubService(**CREDS, **COMMON),
    "hubspot": lambda: HubSpotService(**CREDS, **COMMON),
    "linear": lambda: LinearService(**COMMON, **CREDS),
    "monday": lambda: MondayService(**COMMON, **CREDS),
    "notion": lambda: NotionService(**COMMON, **CREDS),
    "paypal": lambda: PayPalService(**COMMON, **CREDS),
    "supabase": lambda: SupabaseService(**COMMON),
}

STORED = {"access_token": "stored-access", "pkce_verifiers": {"old": "old-verifier"}, "tools": ["search"]}


class _NoHttp:
    async def _fail(self, *args, **kwargs):
        raise AssertionError("OAuth state went through the core API")

    get = post = put = delete = request = _fail


@pytest.fixture
def store(monkeypatch):
    rows = {}
    writes = []

    def fake_read(db, agent_id, provider):
        return dict(rows.get(provider, {}))

    async def fake_write(db, agent_id, provider, config):
        writes.append((provider, config))
        rows[provider] = config

    monkeypatch.setattr(agent_service, "read_integration_config", fake_read)
    monkeypatch.setattr(agent_service, "write_integration_config", fake_write)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _NoHttp())
    return rows, writes


def _service(provider):
    service = MCP_SERVICES[provider]()
    service.http_client = _NoHttp()
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(MCP_SERVICES))
async def test_pkce_verifier_is_stored_and_deleted_in_the_database(provider, store):
    rows, writes = store
    service = _service(provider)
    rows[service.provider_name] = dict(STORED)

    await service._store_pkce_verifier(AGENT_ID, "new", "new-verifier")
    assert rows[service.provider_name]["pkce_verifiers"] == {"old": "old-verifier", "new": "new-verifier"}
    assert rows[service.provider_name]["access_token"] == "stored-access"

    await service._delete_pkce_verifier(AGENT_ID, "new")
    assert rows[service.provider_name]["pkce_verifiers"] == {"old": "old-verifier"}
    assert [p for p, _ in writes] == [service.provider_name, service.provider_name]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(MCP_SERVICES))
async def test_client_credentials_are_stored_in_the_database(provider, store):
    rows, _ = store
    service = _service(provider)
    rows[service.provider_name] = dict(STORED)

    await service._store_client_credentials(AGENT_ID, service.mcp_url, token_endpoint_auth_method="none")

    stored = rows[service.provider_name]
    assert stored["client_id"] == service.client_id
    assert stored["pkce_verifiers"] == {"old": "old-verifier"}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(MCP_SERVICES))
async def test_tokens_are_stored_in_the_database(provider, store):
    rows, _ = store
    service = _service(provider)

    await service._store_credentials(
        agent_id=AGENT_ID,
        mcp_url=service.mcp_url,
        access_token="new-access",
        username="alice",
        email=None,
        user_data={},
        oauth_metadata={},
        refresh_token="new-refresh",
        expires_in=3600,
    )

    stored = rows[service.provider_name]
    assert stored["access_token"] == "new-access"
    assert stored["refresh_token"] == "new-refresh"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_class, provider",
    [(GoogleCalendarService, "google_calendar_credentials"), (GoogleSheetsService, "google_sheets_credentials")],
)
async def test_google_tokens_are_stored_in_the_database(service_class, provider, store):
    rows, _ = store
    service = service_class(**CREDS, **COMMON)
    service.http_client = _NoHttp()
    credentials = Credentials(
        token="new-token",
        refresh_token="new-refresh",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csecret",
    )

    await service._store_credentials(AGENT_ID, credentials, "alice@example.com")

    assert rows[provider]["token"] == "new-token"
    assert rows[provider]["refresh_token"] == "new-refresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(MCP_SERVICES))
async def test_a_failed_database_write_surfaces(provider, monkeypatch):
    async def failing_upsert(db, agent_id, provider, config):
        return False

    monkeypatch.setattr(agent_service, "upsert_agent_integration", failing_upsert)
    service = _service(provider)

    with pytest.raises(RuntimeError):
        await service._store_credentials(
            agent_id=AGENT_ID,
            mcp_url=service.mcp_url,
            access_token="new-access",
            username=None,
            email=None,
            user_data={},
            oauth_metadata={},
            db=object(),
        )


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Session:
    def __init__(self, row=None, error=None):
        self.row, self.error, self.closed = row, error, False

    def execute(self, *args, **kwargs):
        if self.error:
            raise self.error
        return _Result(self.row)

    def close(self):
        self.closed = True


def test_read_returns_the_stored_config_or_empty():
    assert agent_service.read_integration_config(_Session(row=({"a": 1},)), AGENT_ID, "notion") == {"a": 1}
    assert agent_service.read_integration_config(_Session(row=None), AGENT_ID, "notion") == {}


def test_read_raises_on_a_database_error():
    # Read-modify-write must not mistake a failed read for an empty config.
    with pytest.raises(RuntimeError):
        agent_service.read_integration_config(_Session(error=RuntimeError("db down")), AGENT_ID, "notion")


def test_read_without_a_session_opens_and_closes_its_own(monkeypatch):
    import src.config.database as database

    session = _Session(row=({"a": 1},))
    monkeypatch.setattr(database, "SessionLocal", lambda: session)

    assert agent_service.read_integration_config(None, AGENT_ID, "notion") == {"a": 1}
    assert session.closed
