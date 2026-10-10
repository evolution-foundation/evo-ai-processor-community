"""EVO-2181: every incoming file the model can read must reach the model.

process_files used to append the file part to `file_parts` only `if is_audio`, so
images were blobbed + saved to artifacts but never sent to the LLM, and
create_content("", file_parts) returned None -> "No content to process".

The other half of the contract: what the model layer *cannot* carry must stay out
of the content parts. google-adk's LiteLlm raises ValueError on a mime type it
does not know, the runner turns that into a 500, and the user loses the whole
turn -- their text included. A plain WhatsApp document (docx/zip) takes exactly
that path, and a caller that omits `mimeType` gets application/octet-stream from
a2a_routes.extract_files_from_message.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.models.lite_llm import _get_content
from google.genai.types import Blob, Part

from src.schemas.chat import FileData
from src.services.adk.runners.runner_utils import RunnerUtils

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _utils():
    # Bypass __init__ (which builds AgentBuilder(db)); the methods under test use
    # neither self.db nor self.agent_builder.
    return RunnerUtils.__new__(RunnerUtils)


def _artifacts():
    a = MagicMock()
    a.save_artifact = AsyncMock()
    return a


def _file(name, ctype):
    return FileData(
        filename=name, content_type=ctype, data=base64.b64encode(b"bytes-" + name.encode()).decode()
    )


def _run(coro):
    return asyncio.run(coro)


def test_image_is_appended_to_file_parts():
    parts, transcribed = _run(
        _utils().process_files([_file("photo.png", "image/png")], _artifacts(), "agent", "ext", "sess")
    )
    assert len(parts) == 1
    assert parts[0].inline_data.mime_type == "image/png"
    assert transcribed == []


def test_audio_is_not_inlined_as_a_model_part():
    # EVO-2227: audio never travels as a raw model part (google-adk emits an
    # audio_url part litellm rejects -> 500). It is transcribed instead. Without
    # a db (self.db unset here) transcription is a no-op, so parts AND transcript
    # are both empty -- but the file is still archived.
    artifacts = _artifacts()
    parts, transcribed = _run(
        _utils().process_files([_file("voice.ogg", "audio/ogg")], artifacts, "a", "e", "s")
    )
    assert parts == []
    assert transcribed == []
    artifacts.save_artifact.assert_awaited_once()  # still kept for reference


def test_only_the_image_is_inlined_next_to_audio():
    # The image inlines natively; the audio is routed to transcription (a no-op
    # here) and never becomes a model part.
    parts, _ = _run(
        _utils().process_files(
            [_file("p.png", "image/png"), _file("v.ogg", "audio/ogg")], _artifacts(), "a", "e", "s"
        )
    )
    assert [p.inline_data.mime_type for p in parts] == ["image/png"]


def test_audio_transcript_is_returned_and_not_inlined(monkeypatch):
    # With a working transcriber, the audio yields transcribed text (which the
    # runner folds into the message) and still no raw model part.
    from unittest.mock import AsyncMock

    import src.services.adk.runners.runner_utils as ru

    monkeypatch.setattr(ru, "transcribe_audio_file", AsyncMock(return_value="olá do áudio"))
    utils = _utils()
    utils.db = MagicMock()  # transcribe_audio_file is mocked; db is only passed through
    parts, transcribed = _run(
        utils.process_files([_file("voice.ogg", "audio/ogg")], _artifacts(), "agent", "ext", "sess")
    )
    assert parts == []
    assert transcribed == ["olá do áudio"]


def test_create_content_with_image_only_is_not_none():
    utils = _utils()
    parts, _ = _run(utils.process_files([_file("photo.png", "image/png")], _artifacts(), "a", "e", "s"))
    content = utils.create_content("", parts)
    assert content is not None  # regression guard for "No content to process"
    assert content.role == "user"
    assert len(content.parts) == 2  # empty text part + the image file part


def test_create_content_empty_is_none():
    assert _utils().create_content("", []) is None


@pytest.mark.parametrize("content_type", [DOCX, "application/zip", "application/octet-stream", ""])
def test_unreadable_file_stays_out_of_the_content_parts(content_type):
    artifacts = _artifacts()
    parts, _ = _run(
        _utils().process_files([_file("file.bin", content_type)], artifacts, "a", "e", "s")
    )
    assert parts == []
    artifacts.save_artifact.assert_awaited_once()  # still kept for reference


def test_unreadable_file_never_costs_the_text_reply():
    # The regression this guards: forwarding the docx raises ValueError inside
    # LiteLlm, the runner answers 500 and the caption goes unanswered.
    utils = _utils()
    parts, _ = _run(utils.process_files([_file("planilha.docx", DOCX)], _artifacts(), "a", "e", "s"))
    content = utils.create_content("segue o documento", parts)
    assert content is not None
    assert content.parts[0].text == "segue o documento"
    assert all(p.inline_data is None for p in content.parts)


def test_unreadable_file_does_not_drop_the_image_next_to_it():
    parts, _ = _run(
        _utils().process_files(
            [_file("a.zip", "application/zip"), _file("p.png", "image/png")], _artifacts(), "a", "e", "s"
        )
    )
    assert [p.inline_data.mime_type for p in parts] == ["image/png"]


def test_pdf_and_text_still_reach_the_model():
    parts, _ = _run(
        _utils().process_files(
            [_file("r.pdf", "application/pdf"), _file("n.txt", "text/plain")], _artifacts(), "a", "e", "s"
        )
    )
    assert [p.inline_data.mime_type for p in parts] == ["application/pdf", "text/plain"]


def test_mime_parameters_do_not_break_audio_detection():
    # "audio/webm;codecs=opus" (what WhatsApp/browsers actually send) must still
    # be recognized as audio -> routed to transcription, never inlined as a part.
    parts, transcribed = _run(
        _utils().process_files([_file("v.webm", "audio/webm;codecs=opus")], _artifacts(), "a", "e", "s")
    )
    assert parts == []
    assert transcribed == []  # no db here -> transcription is a no-op


# The guard must judge the mime exactly as ADK will, because ADK is what raises.
# A check that normalized first would pass these through and turn them into a 500
# in _get_content -- losing the caption along with the file.
@pytest.mark.parametrize(
    "content_type",
    [
        "IMAGE/PNG",  # case-sensitive startswith in _get_content
        "Image/png",
        "application/pdf; charset=binary",  # exact-match set in _get_content
        "application/json;charset=utf-8",
    ],
)
def test_mime_adk_would_reject_is_skipped_not_forwarded(content_type):
    artifacts = _artifacts()
    parts, _ = _run(_utils().process_files([_file("f.bin", content_type)], artifacts, "a", "e", "s"))
    assert parts == []
    artifacts.save_artifact.assert_awaited_once()  # still kept for reference


def test_every_skipped_mime_would_really_have_broken_adk():
    """The mirror is only worth having if it matches ADK's real behaviour.

    Feeds ADK exactly what the guard rejects and asserts it raises, so a future
    ADK bump that starts accepting these shows up as a failure here instead of as
    a guard that silently drops media the model could have read.
    """
    utils = _utils()
    for content_type in ["IMAGE/PNG", "application/pdf; charset=binary", DOCX]:
        part = Part(inline_data=Blob(mime_type=content_type, data=b"bytes"))
        with pytest.raises(ValueError):
            _get_content([part])


def test_image_survives_an_artifact_store_failure():
    artifacts = MagicMock()
    artifacts.save_artifact = AsyncMock(side_effect=RuntimeError("artifact store down"))
    parts, _ = _run(_utils().process_files([_file("p.png", "image/png")], artifacts, "a", "e", "s"))
    assert [p.inline_data.mime_type for p in parts] == ["image/png"]


def test_oversized_file_is_not_inlined(monkeypatch):
    monkeypatch.setattr(RunnerUtils, "MAX_INLINE_FILE_BYTES", 4)
    artifacts = _artifacts()
    parts, _ = _run(_utils().process_files([_file("big.png", "image/png")], artifacts, "a", "e", "s"))
    assert parts == []
    artifacts.save_artifact.assert_awaited_once()


def test_inline_budget_is_per_request(monkeypatch):
    # first.png decodes to 15 bytes, second.png to 16 -> only the first fits.
    monkeypatch.setattr(RunnerUtils, "MAX_INLINE_REQUEST_BYTES", 20)
    parts, _ = _run(
        _utils().process_files(
            [_file("first.png", "image/png"), _file("second.png", "image/png")], _artifacts(), "a", "e", "s"
        )
    )
    assert len(parts) == 1


def test_every_forwarded_part_is_accepted_by_adk_and_litellm():
    """Pins the allowlist to what BOTH layers accept.

    `_get_content` (ADK) converts the parts; litellm's
    `validate_chat_completion_user_messages` then guards the request. The old
    test only exercised ADK, which happily produces an `audio_url` part -- and
    litellm rejects that downstream with a 500. Audio is therefore no longer a
    forwarded part (it is transcribed); the samples here are exactly what may
    still travel as raw model parts, and both layers must accept them.
    """
    from litellm.utils import validate_chat_completion_user_messages

    utils = _utils()
    samples = [
        "image/png",
        "image/jpeg",
        "image/webp",
        "video/mp4",
        "text/plain",
        "application/pdf",
        "application/json",
    ]
    parts, _ = _run(
        utils.process_files(
            [_file(f"f{i}.bin", ctype) for i, ctype in enumerate(samples)], _artifacts(), "a", "e", "s"
        )
    )
    assert len(parts) == len(samples)
    content = _get_content(utils.create_content("", parts).parts)  # ADK: must not raise
    # litellm layer: must not raise either (this is what audio_url tripped).
    validate_chat_completion_user_messages([{"role": "user", "content": content}])


def test_audio_inlined_as_a_part_would_be_rejected_by_litellm():
    """The reason audio is transcribed instead of inlined (EVO-2227).

    ADK converts an audio Blob into an `audio_url` content part, which is NOT in
    litellm's ValidUserMessageContentTypes -> a 500 that costs the whole turn.
    This pins that fact: if a future litellm starts accepting `audio_url`, this
    fails and we can revisit native audio forwarding.
    """
    from litellm.utils import validate_chat_completion_user_messages

    audio_part = Part(inline_data=Blob(mime_type="audio/ogg", data=b"bytes"))
    content = _get_content([audio_part])  # ADK does not raise here...
    with pytest.raises(Exception):  # ...but litellm does.
        validate_chat_completion_user_messages([{"role": "user", "content": content}])
