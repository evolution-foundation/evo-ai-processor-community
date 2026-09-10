"""EVO-2227 (Fase 2): transcribe audio with the agent's own multimodal model.

Covers the provider routing (OpenAI STT vs chat input_audio), the best-effort
contract (any failure -> None so the turn survives), and the input shaping.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.services.adk.runners.audio_transcription as at

B64 = base64.b64encode(b"opus-bytes").decode()


def _run(coro):
    return asyncio.run(coro)


def _agent(model="gemini/gemini-2.0-flash", agent_id="agent-1"):
    return SimpleNamespace(id=agent_id, model=model, api_key_id=None, config={})


# ---- _is_openai_family ------------------------------------------------------


@pytest.mark.parametrize(
    "model,provider,expected",
    [
        ("gpt-4o", None, True),
        ("gpt-4.1-mini", None, True),
        ("o3-mini", None, True),
        ("openai/gpt-4o", None, True),
        ("anything", "openai", True),
        ("gemini/gemini-2.0-flash", None, False),
        ("gemini-1.5-pro", None, False),
        ("gpt-4o", "openrouter", False),  # openrouter never hits OpenAI STT
        ("claude-3-5-sonnet", None, False),
    ],
)
def test_is_openai_family(model, provider, expected):
    assert at._is_openai_family(model, provider) is expected


# ---- routing ----------------------------------------------------------------


def test_openai_family_routes_to_whisper(monkeypatch):
    monkeypatch.setattr(at, "get_api_key", AsyncMock(return_value=("sk-x", "openai")))
    atranscription = AsyncMock(return_value=SimpleNamespace(text="olá do whisper"))
    monkeypatch.setattr(at.litellm, "atranscription", atranscription)
    monkeypatch.setattr(
        at.litellm,
        "acompletion",
        AsyncMock(side_effect=AssertionError("should not be called")),
    )

    text = _run(
        at.transcribe_audio(
            MagicMock(), _agent(model="gpt-4o"), "audio/ogg", "v.ogg", B64
        )
    )

    assert text == "olá do whisper"
    assert atranscription.await_count == 1
    kwargs = atranscription.await_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["api_key"] == "sk-x"


def test_gemini_routes_to_chat_input_audio(monkeypatch):
    monkeypatch.setattr(at, "get_api_key", AsyncMock(return_value=("gm-key", None)))
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="olá do gemini"))]
    )
    acompletion = AsyncMock(return_value=resp)
    monkeypatch.setattr(at.litellm, "acompletion", acompletion)
    monkeypatch.setattr(
        at.litellm,
        "atranscription",
        AsyncMock(side_effect=AssertionError("should not be called")),
    )

    text = _run(
        at.transcribe_audio(
            MagicMock(),
            _agent(model="gemini/gemini-2.0-flash"),
            "audio/webm;codecs=opus",
            "v.webm",
            B64,
        )
    )

    assert text == "olá do gemini"
    msg = acompletion.await_args.kwargs["messages"][0]
    part = msg["content"][1]
    assert part["type"] == "input_audio"
    # mime parameter stripped for the data-uri header
    assert part["input_audio"]["data"].startswith("data:audio/webm;base64,")
    assert part["input_audio"]["format"] == "audio/webm"


def test_opus_label_is_mapped_to_ogg_container_for_chat(monkeypatch):
    # WhatsApp sends "audio/opus" but the bytes are an OGG container; Gemini
    # knows "audio/ogg", not "audio/opus".
    monkeypatch.setattr(at, "get_api_key", AsyncMock(return_value=("gm-key", None)))
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
    acompletion = AsyncMock(return_value=resp)
    monkeypatch.setattr(at.litellm, "acompletion", acompletion)

    _run(at.transcribe_audio(MagicMock(), _agent(), "audio/opus", "v.ogg", B64))

    part = acompletion.await_args.kwargs["messages"][0]["content"][1]
    assert part["input_audio"]["data"].startswith("data:audio/ogg;base64,")
    assert part["input_audio"]["format"] == "audio/ogg"


def test_openrouter_uses_chat_path_with_api_base(monkeypatch):
    monkeypatch.setattr(
        at, "get_api_key", AsyncMock(return_value=("or-key", "openrouter"))
    )
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="via openrouter"))]
    )
    acompletion = AsyncMock(return_value=resp)
    monkeypatch.setattr(at.litellm, "acompletion", acompletion)

    text = _run(
        at.transcribe_audio(
            MagicMock(), _agent(model="openai/gpt-4o"), "audio/ogg", "v.ogg", B64
        )
    )

    assert text == "via openrouter"
    kwargs = acompletion.await_args.kwargs
    # model has a vendor segment -> prefixed verbatim (EVO-1684).
    assert kwargs["model"] == "openrouter/openai/gpt-4o"
    assert kwargs["api_base"] == "https://openrouter.ai/api/v1"


# ---- best-effort contract ---------------------------------------------------


def test_llm_failure_returns_none(monkeypatch):
    monkeypatch.setattr(at, "get_api_key", AsyncMock(return_value=("gm-key", None)))
    monkeypatch.setattr(
        at.litellm, "acompletion", AsyncMock(side_effect=RuntimeError("boom"))
    )
    assert (
        _run(at.transcribe_audio(MagicMock(), _agent(), "audio/ogg", "v.ogg", B64))
        is None
    )


def test_blank_transcript_returns_none(monkeypatch):
    monkeypatch.setattr(at, "get_api_key", AsyncMock(return_value=("gm-key", None)))
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
    )
    monkeypatch.setattr(at.litellm, "acompletion", AsyncMock(return_value=resp))
    assert (
        _run(at.transcribe_audio(MagicMock(), _agent(), "audio/ogg", "v.ogg", B64))
        is None
    )


def test_missing_api_key_returns_none(monkeypatch):
    monkeypatch.setattr(at, "get_api_key", AsyncMock(side_effect=ValueError("no key")))
    assert (
        _run(at.transcribe_audio(MagicMock(), _agent(), "audio/ogg", "v.ogg", B64))
        is None
    )


def test_undecodable_audio_returns_none():
    assert (
        _run(
            at.transcribe_audio(
                MagicMock(), _agent(), "audio/ogg", "v.ogg", "!!!not-base64!!!"
            )
        )
        is None
    )


# ---- transcribe_audio_file (db / agent guards) ------------------------------


def test_file_helper_none_db_returns_none():
    assert (
        _run(at.transcribe_audio_file(None, "agent-1", "audio/ogg", "v.ogg", B64))
        is None
    )


def test_file_helper_missing_agent_returns_none(monkeypatch):
    monkeypatch.setattr(at, "get_agent", AsyncMock(return_value=None))
    assert (
        _run(
            at.transcribe_audio_file(MagicMock(), "agent-1", "audio/ogg", "v.ogg", B64)
        )
        is None
    )


def test_file_helper_delegates_to_transcribe(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(at, "get_agent", AsyncMock(return_value=agent))
    monkeypatch.setattr(at, "transcribe_audio", AsyncMock(return_value="delegated"))
    text = _run(
        at.transcribe_audio_file(MagicMock(), "agent-1", "audio/ogg", "v.ogg", B64)
    )
    assert text == "delegated"
