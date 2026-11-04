"""
OAuth `state` for the agent integrations.

The state travels through the provider and comes back on the callback, so it is
signed: a callback is only accepted for a state this server issued, for that
agent, to the same user and account that started the flow, recently. The caller
is bound through the signature only, so no user or account id travels in the
state. It carries no URL either; the endpoints of the flow are resolved on the
server. `agent_id` stays at the top level because the frontend callback page
reads it from the state.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Mapping, Optional

STATE_TTL_SECONDS = 15 * 60
# Allowance for clock drift between workers, not a validity window.
CLOCK_SKEW_SECONDS = 60
_KEY_LABEL = b"evo-ai-processor/oauth-state/v1"


class InvalidOAuthState(Exception):
    """The callback state was not issued by this server, or no longer valid."""


def _signing_key() -> bytes:
    # Derived rather than used raw: the encryption key keeps a single purpose
    # per derivation. Read at call time so every worker signs with the same key.
    secret = os.getenv("ENCRYPTION_KEY")
    if not secret:
        raise InvalidOAuthState("ENCRYPTION_KEY is not configured")
    return hmac.new(secret.encode(), _KEY_LABEL, hashlib.sha256).digest()


def _caller_key(caller: Optional[Mapping[str, Any]]) -> list:
    # No caller is a wiring bug, not an anonymous flow: refuse to sign or accept.
    user = str((caller or {}).get("user") or "")
    if not user:
        raise InvalidOAuthState("no caller identity")
    return [user, str((caller or {}).get("context") or "")]


def _signature(payload: Dict[str, Any], caller: Optional[Mapping[str, Any]]) -> str:
    signed = {"payload": payload, "caller": _caller_key(caller)}
    canonical = json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_signing_key(), canonical, hashlib.sha256).hexdigest()


def issue_state(agent_id: str, caller: Optional[Mapping[str, Any]]) -> str:
    """`caller` is {"user": ..., "context": ...} of who starts the flow."""
    payload = {
        "agent_id": str(agent_id),
        "issued_at": int(time.time()),
        "nonce": secrets.token_urlsafe(16),
    }
    payload["sig"] = _signature(payload, caller)
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def verify_state(
    state: Optional[str], agent_id: str, caller: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Returns the payload of a state this server issued for `agent_id` to `caller`, or raises."""
    try:
        padded = (state or "") + "=" * (-len(state or "") % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise InvalidOAuthState("malformed state") from exc

    if not isinstance(payload, dict):
        raise InvalidOAuthState("malformed state")

    received = payload.pop("sig", None)
    if not isinstance(received, str) or not hmac.compare_digest(
        received, _signature(payload, caller)
    ):
        raise InvalidOAuthState("bad signature")

    if payload.get("agent_id") != str(agent_id):
        raise InvalidOAuthState("agent mismatch")

    issued_at = payload.get("issued_at")
    now = time.time()
    if (
        not isinstance(issued_at, int)
        or issued_at > now + CLOCK_SKEW_SECONDS
        or now - issued_at > STATE_TTL_SECONDS
    ):
        raise InvalidOAuthState("expired state")

    return payload
