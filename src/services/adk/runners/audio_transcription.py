"""EVO-2227 (Fase 2): transcribe an incoming audio attachment with the agent's
own multimodal model, so the transcript can be folded into the message text and
understood by ANY answering LLM.

Why not inline the raw audio to the agent instead? google-adk 1.19.0 emits an
``audio_url`` content part for audio (``lite_llm.py`` ``_get_content``), and
litellm 1.68.2 rejects ``audio_url`` -- it is not in
``ValidUserMessageContentTypes`` (``text``/``image_url``/``input_audio``/
``document``/``video_url``/``file``). The result is a 500 that costs the whole
turn. So audio never travels as a raw model part; it is transcribed here and the
text takes its place. Images still inline natively via ``image_url``.

Provider routing (both accept WhatsApp opus/ogg without transcoding):
- OpenAI family -> ``litellm.atranscription`` (whisper-1), a dedicated STT
  endpoint that ingests opus/ogg/m4a/mp3/wav/webm directly.
- Everything else that can hear audio in chat (Gemini) -> ``litellm.acompletion``
  with an ``input_audio`` part; litellm's Gemini transform consumes it
  (``vertex_ai/gemini/transformation.py``).

Best-effort by contract: any failure (unsupported provider, network, quota)
returns ``None`` so the turn survives on whatever text it already carries.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import litellm
from sqlalchemy.orm import Session

from src.models.models import Agent
from src.services.adk.agents.agent_utils import get_api_key
from src.services.agent_service import get_agent
from src.utils.llm_model_routing import normalize_model_for_provider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Kept short and directive so the model returns just the words, no preamble.
_TRANSCRIBE_PROMPT = (
    "Transcribe the following audio verbatim. Return only the transcription "
    "text in the audio's own language, with no extra commentary."
)

# Model families whose keys route through OpenAI's STT endpoint. Matched against
# the lowercased model identifier. Anything else is tried via chat input_audio.
_OPENAI_MODEL_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt", "whisper")

# OpenAI's dedicated speech-to-text model. Reachable with the same key the agent
# already uses; ingests opus/ogg directly (unlike chat input_audio, which is
# wav/mp3 only).
_OPENAI_TRANSCRIBE_MODEL = "whisper-1"

# WhatsApp voice notes arrive labeled "audio/opus", but the bytes are an OGG
# container (Opus codec) and Gemini's accepted audio MIME set lists "audio/ogg",
# not "audio/opus". Map the codec label to the container so the chat provider
# recognizes it. Only used on the chat (input_audio) path; whisper reads the file
# regardless of the label.
_CHAT_AUDIO_MIME_ALIASES = {
    "audio/opus": "audio/ogg",
    "audio/x-opus": "audio/ogg",
}


def _is_openai_family(model: str, provider: Optional[str]) -> bool:
    """Whether to route transcription through OpenAI's STT endpoint.

    OpenRouter keys never hit OpenAI's STT endpoint directly, so they fall to the
    chat path regardless of the underlying vendor.
    """
    if provider == "openrouter":
        return False
    if provider == "openai":
        return True
    model_l = (model or "").lower()
    if model_l.startswith("openai/"):
        return True
    return model_l.startswith(_OPENAI_MODEL_PREFIXES)


async def transcribe_audio_file(
    db: Optional[Session],
    agent_id: str,
    content_type: str,
    filename: str,
    data_b64: str,
) -> Optional[str]:
    """Resolve the agent's model/key and transcribe the base64 audio.

    Returns the transcript text, or ``None`` when transcription is impossible or
    fails (the caller keeps the turn either way).
    """
    if db is None or not agent_id:
        return None
    try:
        agent = await get_agent(db, agent_id)
    except Exception as e:  # get_agent can raise on a bad id / db hiccup
        logger.warning(f"[AudioTranscription] could not load agent {agent_id}: {e}")
        return None
    if agent is None:
        return None
    return await transcribe_audio(db, agent, content_type, filename, data_b64)


async def transcribe_audio(
    db: Session, agent: Agent, content_type: str, filename: str, data_b64: str
) -> Optional[str]:
    """Transcribe one audio attachment with the agent's configured model."""
    try:
        raw_bytes = base64.b64decode(data_b64)
    except Exception as e:
        logger.warning(f"[AudioTranscription] undecodable audio {filename}: {e}")
        return None
    if not raw_bytes:
        return None

    try:
        api_key, provider = await get_api_key(db, agent)
    except Exception as e:
        logger.warning(
            f"[AudioTranscription] no usable API key for agent {agent.id}: {e}"
        )
        return None

    model = agent.model or ""
    try:
        if _is_openai_family(model, provider):
            text = await _transcribe_via_openai(
                api_key, content_type, filename, raw_bytes
            )
        else:
            text = await _transcribe_via_chat(
                model, provider, api_key, content_type, raw_bytes
            )
    except Exception as e:
        logger.warning(
            f"[AudioTranscription] transcription failed for {filename}"
            f" (model={model!r}, provider={provider!r}): {e}"
        )
        return None

    text = (text or "").strip()
    if not text:
        logger.info(f"[AudioTranscription] empty transcript for {filename}")
        return None
    logger.info(
        f"[AudioTranscription] transcribed {filename} ({len(raw_bytes)} bytes)"
        f" -> {len(text)} chars"
    )
    return text


async def _transcribe_via_openai(
    api_key: str, content_type: str, filename: str, raw_bytes: bytes
) -> Optional[str]:
    """OpenAI STT (whisper-1). Ingests opus/ogg directly."""
    audio = io.BytesIO(raw_bytes)
    # The SDK derives the format from the file name's extension; keep the real
    # one so an .ogg/.opus is not mistaken for something the endpoint rejects.
    audio.name = filename or "audio.ogg"
    resp = await litellm.atranscription(
        model=_OPENAI_TRANSCRIBE_MODEL,
        file=audio,
        api_key=api_key,
    )
    # litellm returns a TranscriptionResponse with a .text attribute.
    return getattr(resp, "text", None)


async def _transcribe_via_chat(
    model: str,
    provider: Optional[str],
    api_key: str,
    content_type: str,
    raw_bytes: bytes,
) -> Optional[str]:
    """Chat completion with an ``input_audio`` part (Gemini et al.)."""
    norm_model, extra_kwargs = normalize_model_for_provider(model, provider)
    # Normalize "audio/webm;codecs=opus" -> "audio/webm" for the data-uri header,
    # then map codec labels (audio/opus) to the container MIME the provider knows.
    mime = (content_type or "audio/ogg").split(";")[0].strip().lower() or "audio/ogg"
    mime = _CHAT_AUDIO_MIME_ALIASES.get(mime, mime)
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    resp = await litellm.acompletion(
        model=norm_model,
        api_key=api_key,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _TRANSCRIBE_PROMPT},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data_uri, "format": mime},
                    },
                ],
            }
        ],
        **extra_kwargs,
    )
    return resp.choices[0].message.content
