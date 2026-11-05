"""DifyService: streaming for chat apps and Dify-issued conversation ids."""

import json

import httpx
import pytest

from src.services.providers import dify_service
from src.services.providers.dify_service import DifyService


def _sse(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


class _FakeClient:
    """Stands in for httpx.AsyncClient; replays queued responses."""

    def __init__(self, responses, calls):
        self._responses = responses
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self._calls.append({"url": url, "json": json})
        status, body = self._responses.pop(0)
        request = httpx.Request("POST", url)
        return httpx.Response(status, text=body, request=request)


@pytest.fixture
def fake_dify(monkeypatch):
    calls, responses = [], []
    monkeypatch.setattr(
        dify_service.httpx, "AsyncClient", lambda **kw: _FakeClient(responses, calls)
    )
    return calls, responses


def _service(bot_type="chatBot"):
    return DifyService({"apiUrl": "https://dify.test/v1", "apiKey": "app-x", "botType": bot_type})


@pytest.mark.asyncio
@pytest.mark.parametrize("bot_type,event", [("chatBot", "message"), ("agent", "agent_message")])
async def test_chat_apps_stream_and_return_dify_conversation_id(fake_dify, bot_type, event):
    calls, responses = fake_dify
    responses.append((200, _sse(
        {"event": event, "answer": "Oi", "conversation_id": "dify-conv"},
        {"event": event, "answer": "!", "conversation_id": "dify-conv"},
        {"event": "message_end", "conversation_id": "dify-conv"},
    )))

    result = await _service(bot_type).send_message("oi", "evo-session", {"remoteJid": ""})

    assert result == {"text": "Oi!", "conversation_id": "dify-conv"}
    sent = calls[0]["json"]
    assert calls[0]["url"].endswith("/chat-messages")
    assert sent["response_mode"] == "streaming"
    # The Evo session id must never be sent as the Dify conversation id.
    assert "conversation_id" not in sent
    assert sent["user"] == "evo-session"


@pytest.mark.asyncio
async def test_reuses_stored_conversation_id(fake_dify):
    calls, responses = fake_dify
    responses.append((200, _sse({"event": "message", "answer": "ok", "conversation_id": "dify-conv"})))

    await _service().send_message("oi", "evo-session", {"providerConversationId": "dify-conv"})

    assert calls[0]["json"]["conversation_id"] == "dify-conv"


@pytest.mark.asyncio
async def test_stale_conversation_starts_a_new_one(fake_dify):
    calls, responses = fake_dify
    responses.append((404, json.dumps({"code": "not_found", "message": "Conversation Not Exists.", "status": 404})))
    responses.append((200, _sse({"event": "message", "answer": "ok", "conversation_id": "new-conv"})))

    result = await _service().send_message("oi", "evo-session", {"providerConversationId": "gone"})

    assert result["conversation_id"] == "new-conv"
    assert calls[0]["json"]["conversation_id"] == "gone"
    assert "conversation_id" not in calls[1]["json"]


@pytest.mark.asyncio
async def test_api_error_surfaces_dify_message(fake_dify):
    _, responses = fake_dify
    responses.append((400, json.dumps({"code": "invalid_param", "message": "bad input", "status": 400})))

    with pytest.raises(Exception, match="Dify API error: 400 - bad input"):
        await _service().send_message("oi", "evo-session", {})


@pytest.mark.asyncio
async def test_stream_error_event_raises(fake_dify):
    _, responses = fake_dify
    responses.append((200, _sse({"event": "error", "status": 400, "message": "quota exceeded"})))

    with pytest.raises(Exception, match="quota exceeded"):
        await _service().send_message("oi", "evo-session", {})


@pytest.mark.asyncio
async def test_text_generator_stays_blocking(fake_dify):
    calls, responses = fake_dify
    responses.append((200, json.dumps({"answer": "texto"})))

    result = await _service("textGenerator").send_message("oi", "evo-session", {})

    assert result == {"text": "texto", "conversation_id": None}
    assert calls[0]["url"].endswith("/completion-messages")
    assert calls[0]["json"]["response_mode"] == "blocking"
    assert calls[0]["json"]["inputs"]["query"] == "oi"
