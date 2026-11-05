"""EVO-2227: before_model_callback that strips raw audio from a poisoned history.

Conversations that received audio before the transcription fix persisted user
events carrying the raw audio Blob. google-adk turns those into an `audio_url`
content part that litellm rejects -> a 500 on EVERY later turn. The callback
drops the audio bytes (keeping the turn's text) so the turn survives.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from google.genai import types

from src.services.adk.agents.llm_agent_builder import (
    strip_unsupported_audio_from_history,
)


def _run(coro):
    return asyncio.run(coro)


def _req(*contents):
    return SimpleNamespace(contents=list(contents))


def _audio_part():
    return types.Part(inline_data=types.Blob(mime_type="audio/opus", data=b"oggbytes"))


def _image_part():
    return types.Part(inline_data=types.Blob(mime_type="image/png", data=b"pngbytes"))


def test_audio_part_dropped_text_kept():
    content = types.Content(
        role="user",
        parts=[types.Part(text="Analyze the provided files"), _audio_part()],
    )
    _run(strip_unsupported_audio_from_history(None, _req(content)))
    assert [p.text for p in content.parts] == ["Analyze the provided files"]
    assert all(p.inline_data is None for p in content.parts)


def test_audio_only_content_gets_text_placeholder():
    content = types.Content(role="user", parts=[_audio_part()])
    _run(strip_unsupported_audio_from_history(None, _req(content)))
    assert len(content.parts) == 1
    assert content.parts[0].text == "[audio]"
    assert content.parts[0].inline_data is None


def test_image_and_text_are_untouched():
    content = types.Content(role="user", parts=[types.Part(text="veja"), _image_part()])
    _run(strip_unsupported_audio_from_history(None, _req(content)))
    assert content.parts[0].text == "veja"
    assert content.parts[1].inline_data.mime_type == "image/png"


def test_mixed_history_only_audio_turns_are_cleaned():
    audio_turn = types.Content(
        role="user", parts=[types.Part(text="oi"), _audio_part()]
    )
    clean_turn = types.Content(role="model", parts=[types.Part(text="olá")])
    _run(strip_unsupported_audio_from_history(None, _req(audio_turn, clean_turn)))
    assert all(p.inline_data is None for p in audio_turn.parts)
    assert clean_turn.parts[0].text == "olá"


def test_empty_or_missing_contents_is_safe():
    _run(strip_unsupported_audio_from_history(None, _req()))
    _run(strip_unsupported_audio_from_history(None, SimpleNamespace(contents=None)))
