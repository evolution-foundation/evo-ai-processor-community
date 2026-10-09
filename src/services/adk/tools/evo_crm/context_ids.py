"""
Ids of the current turn, read from the tool context.

The conversation, contact and pipeline item an EvoCRM tool acts on are facts of
the turn, not choices of the model: the ContactInfo block puts other UUIDs in
the prompt and the model copies them into tool arguments. So when the context
carries an id it wins over the argument; the argument only stands when the
context is silent (playground and direct API calls).
"""

from typing import Any, Optional

from google.adk.tools import ToolContext

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _value(raw: Any) -> Optional[str]:
    """A usable id, or None for a missing/None/blank value (never the string "None")."""
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _first(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        value = _value(candidate)
        if value:
            return value
    return None


def _state(tool_context: Optional[ToolContext]) -> Optional[Any]:
    if not tool_context or not hasattr(tool_context, "state"):
        return None
    return tool_context.state


def _dict(raw: Any) -> dict:
    return raw if isinstance(raw, dict) else {}


def extract_conversation_id(tool_context: Optional[ToolContext]) -> Optional[str]:
    """The current conversation's UUID.

    Sources, in order: evoai_crm_data.conversation_id, evoai_crm_data.conversation.id
    (also the UUID: the CRM sends display_id in a separate field), then the direct
    state keys conversation_id / conversationId.
    """
    state = _state(tool_context)
    if state is None:
        return None

    crm_data = _dict(state.get("evoai_crm_data"))
    return _first(
        crm_data.get("conversation_id"),
        _dict(crm_data.get("conversation")).get("id"),
        state.get("conversation_id"),
        state.get("conversationId"),
    )


def extract_contact_id(tool_context: Optional[ToolContext]) -> Optional[str]:
    """The current conversation's contact UUID."""
    state = _state(tool_context)
    if state is None:
        return None

    crm_data = _dict(state.get("evoai_crm_data"))
    return _first(
        _dict(state.get("contact")).get("id"),
        _dict(crm_data.get("contact")).get("id"),
        crm_data.get("contactId"),
        crm_data.get("contact_id"),
        state.get("contactId"),
        state.get("contact_id"),
    )


def extract_pipeline_item_id(tool_context: Optional[ToolContext]) -> Optional[str]:
    """The current conversation's pipeline item UUID."""
    state = _state(tool_context)
    if state is None:
        return None

    crm_data = _dict(state.get("evoai_crm_data"))
    return _first(
        crm_data.get("pipeline_item_id"),
        _dict(crm_data.get("pipeline_item")).get("id"),
        state.get("pipeline_item_id"),
        state.get("pipelineItemId"),
    )


def resolve_id(
    name: str, from_context: Optional[str], from_model: Optional[str]
) -> Optional[str]:
    """The id a tool acts on: the context's when present, else the model's argument."""
    from_model = _value(from_model)
    if not from_context:
        return from_model

    if from_model and from_model != from_context:
        logger.warning(
            f"ignoring {name}={from_model!r} from the model; the current one is {from_context}"
        )
    return from_context
