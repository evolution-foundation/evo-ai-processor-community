"""An inline secret the user unlinked stays unlinked.

The CRM migration copies every inline auth secret into the vault and keeps the
inline value. Once the installation records the migration, a consumer with no
vault reference is one whose credential was unlinked, so its inline secret must
not go out. Before the migration, and when the record cannot be read, the
inline value is still the fallback.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services import custom_mcp_server_service, custom_tool_service
from src.services.adk import mcp_service
from src.services.adk.agents import external_agent_builder
from src.services.adk.custom_tools import CustomToolBuilder
from src.services.adk.integration_credentials import (
    apply_vault_credential,
    installation_integration_migrated,
    integration_migrated,
    migration_reader,
    retire_unreferenced_auth_headers,
)
from src.services.adk.tool_builder import ToolBuilder

MIGRATED = "src.services.adk.integration_credentials.installation_integration_migrated"
RESOLVE = "src.services.adk.integration_credentials.resolve_credential_refs"


def _db_answering(row=None, raises=None):
    db = MagicMock()
    if raises is not None:
        db.execute.side_effect = raises
    else:
        db.execute.return_value.fetchone.return_value = row
    return db


class TestIntegrationMigrated:
    def test_a_json_true_record_is_migrated(self):
        assert integration_migrated(_db_answering((True,))) is True

    @pytest.mark.parametrize("value", [False, "true", 1, None, {}])
    def test_anything_but_json_true_is_not(self, value):
        assert integration_migrated(_db_answering((value,))) is False

    def test_no_record_is_not_migrated(self):
        assert integration_migrated(_db_answering(None)) is False

    def test_a_failed_read_is_not_migrated(self):
        assert integration_migrated(_db_answering(raises=RuntimeError("boom"))) is False

    def test_reads_the_flag_the_crm_migration_writes(self):
        db = _db_answering((True,))
        integration_migrated(db)

        sql, params = db.execute.call_args.args
        # InstallationConfig#value= stores {"value": <value>} in serialized_value.
        assert "FROM installation_configs" in str(sql)
        assert "serialized_value -> 'value'" in str(sql)
        assert params == {"name": "INTEGRATION_CREDENTIALS_MIGRATED"}

    def test_a_session_that_cannot_open_is_not_migrated(self):
        with patch("src.config.database.SessionLocal", side_effect=RuntimeError("down")):
            assert installation_integration_migrated() is False

    def test_a_reader_asks_the_database_once(self):
        with patch(MIGRATED, return_value=True) as read:
            migrated = migration_reader()
            assert [migrated(), migrated(), migrated()] == [True, True, True]

        assert read.call_count == 1


class TestRetireUnreferencedAuthHeaders:
    HEADERS = {"Authorization": "Bearer antigo", "Content-Type": "application/json"}

    def test_drops_the_unlinked_auth_header_once_migrated(self):
        assert retire_unreferenced_auth_headers(self.HEADERS, {}, lambda: True) == {
            "Content-Type": "application/json"
        }

    def test_keeps_it_on_an_installation_that_never_migrated(self):
        assert retire_unreferenced_auth_headers(self.HEADERS, {}, lambda: False) == self.HEADERS

    def test_keeps_a_header_a_credential_still_covers(self):
        headers = retire_unreferenced_auth_headers(
            self.HEADERS, {"Authorization": "cred-1"}, lambda: True
        )
        assert headers == self.HEADERS

    def test_decides_per_header(self):
        headers = retire_unreferenced_auth_headers(
            {"Authorization": "a", "X-Api-Key": "b"}, {"Authorization": "cred-1"}, lambda: True
        )
        assert headers == {"Authorization": "a"}

    @pytest.mark.parametrize("name", ["authorization", "X-API-KEY", "Api-Key", "apikey", "X-Auth-Token"])
    def test_every_auth_header_name_the_migration_imports(self, name):
        assert retire_unreferenced_auth_headers({name: "segredo"}, {}, lambda: True) == {}

    def test_an_empty_reference_does_not_cover_the_header(self):
        assert retire_unreferenced_auth_headers(
            {"Authorization": "a"}, {"Authorization": ""}, lambda: True
        ) == {}

    def test_does_not_read_the_record_without_an_unlinked_auth_header(self):
        def must_not_ask():
            raise AssertionError("read the migration record for nothing")

        retire_unreferenced_auth_headers({"Content-Type": "application/json"}, {}, must_not_ask)
        retire_unreferenced_auth_headers(
            {"Authorization": "a"}, {"Authorization": "cred-1"}, must_not_ask
        )


def _http_config(headers, credential_refs=None):
    config = {
        "name": "consulta",
        "description": "x",
        "endpoint": "https://api.example.com/x",
        "method": "GET",
        "headers": headers,
    }
    if credential_refs is not None:
        config["credential_refs"] = credential_refs
    return config


def _sent_headers(tool):
    with patch("requests.request") as request:
        request.return_value.status_code = 200
        request.return_value.json.return_value = {}
        tool.func()
    return request.call_args.kwargs["headers"]


@pytest.mark.parametrize("builder_cls", [CustomToolBuilder, ToolBuilder])
class TestHttpToolHeaders:
    def test_migrated_tool_without_reference_sends_no_inline_auth(self, builder_cls):
        with patch(MIGRATED, return_value=True):
            tool = builder_cls()._create_http_tool(
                _http_config({"Authorization": "Bearer antigo", "Accept": "json"}, {})
            )

        assert _sent_headers(tool) == {"Accept": "json"}

    def test_never_migrated_tool_keeps_the_inline_auth(self, builder_cls):
        with patch(MIGRATED, return_value=False):
            tool = builder_cls()._create_http_tool(
                _http_config({"Authorization": "Bearer antigo"}, {})
            )

        assert _sent_headers(tool) == {"Authorization": "Bearer antigo"}

    def test_linked_header_still_comes_from_the_vault(self, builder_cls):
        def from_vault(headers, refs, vault, decrypt):
            return {**headers, **{name: "do-cofre" for name in refs}}

        with patch(MIGRATED, return_value=True), patch(RESOLVE, side_effect=from_vault), patch(
            "src.config.database.SessionLocal"
        ):
            tool = builder_cls()._create_http_tool(
                _http_config({"Authorization": "Bearer antigo"}, {"Authorization": "cred-1"})
            )

        assert _sent_headers(tool) == {"Authorization": "do-cofre"}


def _catalog_tool(**overrides):
    tool = SimpleNamespace(
        id=uuid.uuid4(),
        name="consulta",
        description="x",
        method="GET",
        endpoint="https://api.example.com/x",
        headers={"Authorization": "Bearer antigo"},
        credential_refs={},
        path_params={},
        query_params={},
        body_params={},
        error_handling={},
        values={},
    )
    for key, value in overrides.items():
        setattr(tool, key, value)
    return tool


class TestToolAttachedById:
    """The catalog row has to carry its references to the builder: without them
    every linked header would look unlinked."""

    def _build(self, tool):
        with patch.object(custom_tool_service, "get_custom_tool", return_value=tool):
            return CustomToolBuilder().build_tools({"custom_tool_ids": [str(tool.id)]}, db=MagicMock())

    def test_a_linked_header_reaches_the_vault(self):
        def from_vault(headers, refs, vault, decrypt):
            return {**headers, **{name: "do-cofre" for name in refs}}

        tool = _catalog_tool(credential_refs={"Authorization": "cred-1"})
        with patch(MIGRATED, return_value=True), patch(RESOLVE, side_effect=from_vault), patch(
            "src.config.database.SessionLocal"
        ):
            (built,) = self._build(tool)

        assert _sent_headers(built) == {"Authorization": "do-cofre"}

    def test_an_unlinked_header_is_not_sent_once_migrated(self):
        with patch(MIGRATED, return_value=True):
            (built,) = self._build(_catalog_tool())

        assert _sent_headers(built) == {}

    def test_an_agent_with_many_tools_reads_the_record_once(self):
        tools = [_catalog_tool(name=f"consulta_{n}") for n in range(3)]
        with patch(MIGRATED, return_value=True) as read, patch.object(
            custom_tool_service, "get_custom_tool", side_effect=tools
        ):
            ToolBuilder().build_tools(
                {
                    "custom_tool_ids": [str(t.id) for t in tools],
                    "http_tools": [
                        {**_http_config({"Authorization": "Bearer antigo"}, {}), "name": "inline"}
                    ],
                },
                db=MagicMock(),
            )

        assert read.call_count == 1

    def test_the_agent_config_expansion_carries_the_references(self):
        tool = _catalog_tool(credential_refs={"Authorization": "cred-1"})

        assert custom_tool_service.convert_to_http_tool(tool)["credential_refs"] == {
            "Authorization": "cred-1"
        }


class TestCustomMcpHeaders:
    def _server(self, credential_refs):
        return SimpleNamespace(
            headers={"X-Api-Key": "antiga", "Accept": "text/event-stream"},
            credential_refs=credential_refs,
        )

    def test_migrated_server_without_reference_sends_no_inline_auth(self):
        with patch(MIGRATED, return_value=True):
            headers = mcp_service._resolve_mcp_headers(self._server({}), MagicMock())

        assert headers == {"Accept": "text/event-stream"}

    def test_never_migrated_server_keeps_the_inline_auth(self):
        with patch(MIGRATED, return_value=False):
            headers = mcp_service._resolve_mcp_headers(self._server({}), MagicMock())

        assert headers["X-Api-Key"] == "antiga"

    def test_a_failed_read_keeps_the_inline_auth(self):
        with patch("src.config.database.SessionLocal") as session_local:
            session_local.return_value.execute.side_effect = RuntimeError("boom")
            headers = mcp_service._resolve_mcp_headers(self._server({}), MagicMock())

        assert headers["X-Api-Key"] == "antiga"


class TestCustomMcpFromAgentConfig:
    """get_agent expands custom_mcp_server_ids into custom_mcp_servers entries,
    and build_lazy_tools connects to those too."""

    def _expanded(self, credential_refs):
        return custom_mcp_server_service.convert_to_mcp_server_config(
            SimpleNamespace(
                url="https://mcp.example.com/mcp",
                headers={"Authorization": "Bearer antigo"},
                credential_refs=credential_refs,
            )
        )

    def _discovered_headers(self, config):
        seen = []

        @asynccontextmanager
        async def capture(server, timeout=None):
            seen.append(dict(server.get("headers") or {}))
            yield []

        cache = MagicMock()
        cache.get_server_tools = AsyncMock(return_value=None)
        cache.get_agent_tools = AsyncMock(return_value=None)
        with patch.object(mcp_service, "mcp_context", capture), patch.object(
            mcp_service, "mcp_tool_cache", cache
        ):
            asyncio.run(mcp_service.MCPService().build_lazy_tools(config, MagicMock(), None))
        return seen

    def test_the_expansion_carries_the_references(self):
        assert self._expanded({"Authorization": "cred-1"})["credential_refs"] == {
            "Authorization": "cred-1"
        }

    def test_unlinked_entry_sends_no_inline_auth_once_migrated(self):
        with patch(MIGRATED, return_value=True):
            seen = self._discovered_headers({"custom_mcp_servers": [self._expanded({})]})

        assert [h.get("Authorization") for h in seen] == [None]

    def test_unlinked_entry_keeps_the_inline_auth_when_never_migrated(self):
        with patch(MIGRATED, return_value=False):
            seen = self._discovered_headers({"custom_mcp_servers": [self._expanded({})]})

        assert [h.get("Authorization") for h in seen] == ["Bearer antigo"]

    def test_linked_entry_comes_from_the_vault(self):
        def from_vault(headers, refs, vault, decrypt):
            return {**headers, **{name: "do-cofre" for name in refs}}

        with patch(MIGRATED, return_value=True), patch(RESOLVE, side_effect=from_vault):
            seen = self._discovered_headers(
                {"custom_mcp_servers": [self._expanded({"Authorization": "cred-1"})]}
            )

        assert [h.get("Authorization") for h in seen] == ["do-cofre"]

    def test_ids_and_expanded_entries_share_one_read(self):
        row = SimpleNamespace(
            url="https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer antigo"},
            credential_refs={},
        )
        config = {
            "custom_mcp_server_ids": [str(uuid.uuid4())],
            "custom_mcp_servers": [self._expanded({}), self._expanded({})],
        }
        with patch(MIGRATED, return_value=True) as read, patch.object(
            custom_mcp_server_service, "get_custom_mcp_server", return_value=row
        ):
            seen = self._discovered_headers(config)

        assert [h.get("Authorization") for h in seen] == [None, None, None]
        assert read.call_count == 1


class TestExternalAgentSecret:
    CONFIG = {"apiKey": "app-antiga", "apiUrl": "https://dify.example.com"}

    def _resolve(self, provider, config, migrated):
        return apply_vault_credential(
            provider,
            config,
            vault=MagicMock(),
            decrypt=lambda _: pytest.fail("must not decrypt"),
            migrated=lambda: migrated,
        )

    def test_unlinked_secret_fails_loudly_once_migrated(self):
        with pytest.raises(ValueError, match="no credential linked"):
            self._resolve("dify", self.CONFIG, migrated=True)

    def test_unlinked_pair_fails_loudly_once_migrated(self):
        with pytest.raises(ValueError, match="no credential linked"):
            self._resolve("n8n", {"basicAuthUser": "u", "basicAuthPass": "p"}, migrated=True)

    def test_never_migrated_keeps_the_inline_secret(self):
        assert self._resolve("dify", self.CONFIG, migrated=False)["apiKey"] == "app-antiga"

    def test_nothing_inline_is_left_to_fail_where_it_did(self):
        assert self._resolve("dify", {"apiUrl": "https://dify.example.com"}, migrated=True) == {
            "apiUrl": "https://dify.example.com"
        }

    def test_typebot_holds_no_secret(self):
        assert self._resolve("typebot", {"apiUrl": "x"}, migrated=True) == {"apiUrl": "x"}

    def test_the_builder_reads_the_installation_record(self):
        """The builder relies on the default: a default that never reads the
        record would keep every unlinked secret alive."""
        import inspect

        default = inspect.signature(apply_vault_credential).parameters["migrated"].default
        assert default is installation_integration_migrated
        assert external_agent_builder.apply_vault_credential is apply_vault_credential


class TestKnowledgeNexusKey:
    def _config(self, **extra):
        return {
            "integrations": {
                "knowledge-nexus": {
                    "connected": True,
                    "nexus_base_url": "https://nexus.example.com",
                    "nexus_api_key": "chave-antiga",
                    "space_id": "space-1",
                    **extra,
                }
            }
        }

    def _nexus_tools(self, config, migrated):
        with patch(MIGRATED, return_value=migrated), patch(
            "src.services.adk.tools.knowledge_nexus.create_knowledge_nexus_search_tool"
        ) as create:
            ToolBuilder().build_tools(config)
        return create

    def test_unlinked_key_builds_no_tool_once_migrated(self):
        assert self._nexus_tools(self._config(), migrated=True).call_count == 0

    def test_never_migrated_keeps_the_inline_key(self):
        create = self._nexus_tools(self._config(), migrated=False)

        assert create.call_args.kwargs["nexus_api_key"] == "chave-antiga"

    def test_a_linked_key_is_not_retired(self):
        with patch(RESOLVE, return_value={"nexus_api_key": "do-cofre"}), patch(
            "src.config.database.SessionLocal"
        ):
            create = self._nexus_tools(self._config(credential_id="cred-1"), migrated=True)

        assert create.call_args.kwargs["nexus_api_key"] == "do-cofre"
