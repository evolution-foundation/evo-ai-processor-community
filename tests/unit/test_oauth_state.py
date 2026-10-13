"""The OAuth state of the agent integrations is issued and verified only by this server."""

import base64
import json
import time

import pytest

from src.services import oauth_state
from src.services.oauth_state import InvalidOAuthState, issue_state, verify_state

AGENT_ID = "0b6f7c1e-3f55-4a51-9a49-7d0d0c2b4a10"
CALLER = {"user": "user-1", "context": "account-1"}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", "test-encryption-key-for-oauth-state")


def _decode(state):
    return json.loads(base64.urlsafe_b64decode(state.encode()).decode())


def _encode(payload):
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def test_an_issued_state_verifies_for_its_agent():
    assert (
        verify_state(issue_state(AGENT_ID, CALLER), AGENT_ID, CALLER)["agent_id"]
        == AGENT_ID
    )


def test_the_frontend_still_reads_agent_id_with_a_plain_decode():
    # vendor/crm CallbackPage decodes the state itself to find the agent.
    assert _decode(issue_state(AGENT_ID, CALLER))["agent_id"] == AGENT_ID


def test_the_state_carries_no_url():
    payload = _decode(issue_state(AGENT_ID, CALLER))

    assert set(payload) == {"agent_id", "issued_at", "nonce", "sig"}


def test_two_states_for_the_same_agent_differ():
    assert issue_state(AGENT_ID, CALLER) != issue_state(AGENT_ID, CALLER)


def test_the_unsigned_format_is_rejected():
    legacy = _encode(
        {
            "agent_id": AGENT_ID,
            "mcp_url": "https://mcp.example/mcp",
            "timestamp": "2026-10-13T00:00:00",
        }
    )

    with pytest.raises(InvalidOAuthState):
        verify_state(legacy, AGENT_ID, CALLER)


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_id", "another-agent"),
        ("issued_at", 0),
        ("nonce", "x"),
        ("mcp_url", "https://other.example"),
    ],
)
def test_any_change_to_a_signed_state_is_rejected(field, value):
    payload = _decode(issue_state(AGENT_ID, CALLER))
    payload[field] = value

    with pytest.raises(InvalidOAuthState):
        verify_state(_encode(payload), payload["agent_id"], CALLER)


def test_a_state_for_another_agent_is_rejected():
    with pytest.raises(InvalidOAuthState):
        verify_state(issue_state(AGENT_ID, CALLER), "another-agent", CALLER)


def test_an_expired_state_is_rejected(monkeypatch):
    state = issue_state(AGENT_ID, CALLER)
    now = time.time()
    monkeypatch.setattr(
        oauth_state.time, "time", lambda: now + oauth_state.STATE_TTL_SECONDS + 1
    )

    with pytest.raises(InvalidOAuthState):
        verify_state(state, AGENT_ID, CALLER)


def test_a_state_from_the_future_is_rejected(monkeypatch):
    now = time.time()
    monkeypatch.setattr(
        oauth_state.time, "time", lambda: now + oauth_state.CLOCK_SKEW_SECONDS + 60
    )
    state = issue_state(AGENT_ID, CALLER)
    monkeypatch.setattr(oauth_state.time, "time", lambda: now)

    with pytest.raises(InvalidOAuthState):
        verify_state(state, AGENT_ID, CALLER)


def test_a_state_signed_with_another_key_is_rejected(monkeypatch):
    state = issue_state(AGENT_ID, CALLER)
    monkeypatch.setenv("ENCRYPTION_KEY", "some-other-key")

    with pytest.raises(InvalidOAuthState):
        verify_state(state, AGENT_ID, CALLER)


def test_the_key_does_not_come_from_the_jwt_secret(monkeypatch):
    # JWT_SECRET_KEY defaults to a random value per process: a state signed by
    # one worker would not verify on another.
    monkeypatch.setenv("JWT_SECRET_KEY", "a")
    state = issue_state(AGENT_ID, CALLER)
    monkeypatch.setenv("JWT_SECRET_KEY", "b")

    assert verify_state(state, AGENT_ID, CALLER)


def test_nothing_is_issued_or_verified_without_a_key(monkeypatch):
    state = issue_state(AGENT_ID, CALLER)
    monkeypatch.delenv("ENCRYPTION_KEY")

    with pytest.raises(InvalidOAuthState):
        issue_state(AGENT_ID, CALLER)
    with pytest.raises(InvalidOAuthState):
        verify_state(state, AGENT_ID, CALLER)


@pytest.mark.parametrize(
    "garbage",
    [None, "", "not-base64!!", _encode(["a list"]), _encode({"agent_id": AGENT_ID})],
)
def test_malformed_states_are_rejected(garbage):
    with pytest.raises(InvalidOAuthState):
        verify_state(garbage, AGENT_ID, CALLER)


def test_the_ttl_is_at_most_fifteen_minutes():
    assert oauth_state.STATE_TTL_SECONDS <= 15 * 60


@pytest.mark.parametrize(
    "other",
    [
        {"user": "user-2", "context": "account-1"},
        {"user": "user-1", "context": "account-2"},
        {"user": "user-1", "context": ""},
    ],
)
def test_a_state_completed_by_another_user_or_account_is_rejected(other):
    state = issue_state(AGENT_ID, CALLER)

    with pytest.raises(InvalidOAuthState):
        verify_state(state, AGENT_ID, other)


def test_the_caller_does_not_travel_in_the_state():
    raw = base64.urlsafe_b64decode(issue_state(AGENT_ID, CALLER).encode()).decode()

    assert "user-1" not in raw and "account-1" not in raw


@pytest.mark.parametrize(
    "empty", [None, {}, {"user": ""}, {"user": None, "context": "account-1"}]
)
def test_nothing_is_issued_or_accepted_without_a_caller_identity(empty):
    with pytest.raises(InvalidOAuthState):
        issue_state(AGENT_ID, empty)

    state = issue_state(AGENT_ID, CALLER)
    with pytest.raises(InvalidOAuthState):
        verify_state(state, AGENT_ID, empty)


def test_an_install_without_accounts_binds_the_user_alone():
    caller = {"user": "user-1", "context": ""}

    assert (
        verify_state(issue_state(AGENT_ID, caller), AGENT_ID, caller)["agent_id"]
        == AGENT_ID
    )
