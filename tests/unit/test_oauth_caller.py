"""The OAuth flow is bound to the caller of the request that starts and completes it."""

import importlib
from types import SimpleNamespace

import pytest

from src.api import oauth_caller as oauth_caller_module
from src.api.oauth_caller import binds_oauth_caller, oauth_caller

PROVIDERS = [
    "github",
    "notion",
    "asana",
    "monday",
    "atlassian",
    "supabase",
    "paypal",
    "hubspot",
    "linear",
    "canva",
    "google_sheets",
    "google_calendar",
]

_WRAPPER_CODE = binds_oauth_caller(lambda request: None).__code__


def _request(user_context):
    return SimpleNamespace(state=SimpleNamespace(user_context=user_context))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_provider_service_dependency_binds_the_caller(provider):
    module = importlib.import_module(f"src.api.{provider}_routes")
    dependency = getattr(module, f"get_{provider}_service")

    assert dependency.__code__ is _WRAPPER_CODE
    assert dependency.__wrapped__.__name__ == f"get_{provider}_service"


def test_the_caller_is_the_user_and_the_account_of_the_request(monkeypatch):
    monkeypatch.setattr(
        oauth_caller_module.runtime_context,
        "current_context_id",
        lambda request: "account-1",
    )

    assert oauth_caller(_request({"user_id": "user-1"})) == {
        "user": "user-1",
        "context": "account-1",
    }


def test_an_install_without_accounts_has_an_empty_context():
    assert oauth_caller(_request({"user_id": "user-1"}))["context"] == ""


@pytest.mark.parametrize("user_context", [None, {}, {"user_id": ""}])
def test_a_request_without_identity_yields_an_empty_user(user_context):
    assert oauth_caller(_request(user_context))["user"] == ""


@pytest.mark.asyncio
async def test_the_wrapper_records_the_caller_on_the_built_service():
    built = SimpleNamespace()

    async def build(request):
        return built

    service = await binds_oauth_caller(build)(_request({"user_id": "user-1"}))

    assert service is built
    assert built.oauth_caller == {"user": "user-1", "context": ""}
