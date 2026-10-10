"""
Manage Conversation Labels Tool

This tool allows agents to list, add and remove labels on the current
conversation. Backend endpoints used:

- GET  /api/v1/conversations/{id}/labels  -> list current labels
- POST /api/v1/conversations/{id}/labels  -> replace labels (with `{labels: [...]}`)

The upstream POST is destructive (it replaces the full label list), so this
tool reads the current labels first and computes the union/difference before
writing back, preserving labels the user did not explicitly remove.
"""

from typing import Any, Dict, List, Optional

from google.adk.tools import FunctionTool, ToolContext

from src.services.adk.tools.evo_crm.base import EvoCrmClient
from src.services.adk.tools.evo_crm.context_ids import extract_conversation_id, resolve_id
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _normalize_labels(raw: Any) -> List[str]:
    """Normalize a labels payload coming from the API to a flat list of titles."""
    if raw is None:
        return []

    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]

    if isinstance(raw, dict):
        for key in ("data", "labels", "payload"):
            if key in raw:
                return _normalize_labels(raw[key])

    return []


async def _fetch_catalog_labels(client: EvoCrmClient) -> "tuple[bool, List[str]]":
    """Fetch the account's real Label catalog (Settings > Labels), not the
    free-form conversation tag list. Conversation tagging (acts_as_taggable_on)
    accepts any string with no relation to this catalog, so without this
    check the model could tag conversations with labels that were never
    created in Settings — invisible there, uncolored, and absent from any
    label-based filter (EVO-2248).

    Returns (fetch_succeeded, titles). A legitimately empty catalog
    (fetch_succeeded=True, titles=[]) must be distinguishable from a failed
    fetch (fetch_succeeded=False, titles=[]) — otherwise both look identical
    to the caller and a genuinely empty account gets misreported as a fetch
    failure instead of "no labels exist to apply".
    """
    try:
        response = await client.get(endpoint="/labels", params={"per_page": 200})
    except Exception as api_error:
        logger.error(f"Failed to load label catalog: {api_error}")
        return False, []

    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list):
        logger.error(f"Unexpected label catalog response shape: {response!r}")
        return False, []

    titles = []
    for item in data:
        if isinstance(item, dict) and item.get("title"):
            titles.append(str(item["title"]))
    return True, titles


def _coerce_input_list(value: Any) -> List[str]:
    """Accept either a single string or a list, return a deduped list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = list(value)
    else:
        items = [value]

    seen = set()
    result: List[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def create_manage_conversation_labels_tool() -> FunctionTool:
    """Create the manage_conversation_labels tool.

    The tool exposes three actions on the current conversation:
    - ``list``: returns the current labels.
    - ``add``: appends one or more labels, preserving existing ones.
    - ``remove``: removes one or more labels, preserving the rest.
    """

    client = EvoCrmClient()

    async def manage_conversation_labels(
        action: str,
        labels: Optional[Any] = None,
        conversation_id: Optional[str] = None,
        tool_context: Optional[ToolContext] = None,
    ) -> Dict[str, Any]:
        """Manage labels on the current conversation.

        Use this tool when:
        - You need to tag the conversation for routing or reporting
          (e.g. ``vip``, ``aguardando-pagamento``, ``followup``).
        - You need to remove a label that no longer applies.
        - You want to inspect which labels are currently attached.

        Important: ``add`` and ``remove`` are idempotent and additive — the
        tool reads the current labels first and merges your request, so it
        never erases labels the user did not ask to remove.

        Args:
            action: One of ``list``, ``add``, ``remove``.
            labels: For ``add`` / ``remove``: a label title (string) or a list
                of titles. Ignored when action is ``list``.
            conversation_id: DO NOT SET. The conversation context supplies it and
                overrides anything passed here.
            tool_context: Provided automatically by the runtime.

        Returns:
            Dictionary with the executed action and the resulting label list:
            ``{"status": "success"|"error", "message": "...",
               "conversation_id": "...", "action": "...",
               "labels": ["label-a", "label-b"], ...}``.
        """
        effective_conversation_id = resolve_id(
            "conversation_id", extract_conversation_id(tool_context), conversation_id
        )

        if not effective_conversation_id:
            return {
                "status": "error",
                "message": "No conversation in context: this tool only runs inside a conversation.",
                "conversation_id": None,
                "action": action,
            }

        normalized_action = (action or "").strip().lower()
        if normalized_action not in {"list", "add", "remove"}:
            return {
                "status": "error",
                "message": "action must be one of: list, add, remove.",
                "conversation_id": effective_conversation_id,
                "action": action,
            }

        endpoint = f"/conversations/{effective_conversation_id}/labels"

        try:
            current_labels_raw = await client.get(endpoint=endpoint)
            current_labels = _normalize_labels(current_labels_raw)
            # The CRM /labels endpoint replaces (not appends) the whole set on POST,
            # so this read is what protects existing labels. A 204/empty body or an
            # intermittent auth failure makes the read look like "no labels", and a
            # subsequent add would then POST only the requested labels — WIPING every
            # existing label, including the `atendimento_ia` gate that keeps the bot
            # active. Detect a read that returned nothing usable and refuse to do a
            # destructive add rather than silently clobbering the conversation.
            read_is_empty = not current_labels
            read_is_unreliable = (
                isinstance(current_labels_raw, dict)
                and "status_code" in current_labels_raw
                and current_labels_raw.get("status_code") != 200
            )
        except Exception as api_error:
            logger.error(
                f"Failed to load labels for conversation {effective_conversation_id}: {api_error}"
            )
            return {
                "status": "error",
                "message": f"Failed to load current labels: {api_error}",
                "conversation_id": effective_conversation_id,
                "action": normalized_action,
                "error": str(api_error),
            }

        if normalized_action == "list":
            return {
                "status": "success",
                "message": f"Conversation has {len(current_labels)} label(s).",
                "conversation_id": effective_conversation_id,
                "action": "list",
                "labels": current_labels,
            }

        requested = _coerce_input_list(labels)
        if not requested:
            return {
                "status": "error",
                "message": "Provide at least one label for 'add' or 'remove'.",
                "conversation_id": effective_conversation_id,
                "action": normalized_action,
            }

        existing_set = {label.lower(): label for label in current_labels}

        if normalized_action == "add":
            # Defense-in-depth: refuse a destructive add. If the read came back empty
            # *because it was unreliable* (non-200/204 with no body), POSTing only the
            # requested labels would replace and wipe every existing label, including
            # the `atendimento_ia` gate. In that case abort instead of clobbering.
            if read_is_empty and read_is_unreliable:
                logger.error(
                    "manage_conversation_labels: refusing to add labels to "
                    f"{effective_conversation_id} — the labels read was unreliable "
                    f"(raw={current_labels_raw}); a replace would wipe existing labels."
                )
                return {
                    "status": "error",
                    "message": (
                        "Could not read the conversation's current labels reliably, "
                        "so the add was skipped to avoid removing existing labels. "
                        "Please retry."
                    ),
                    "conversation_id": effective_conversation_id,
                    "action": "add",
                }

            # Only labels that already exist in the account's Label catalog
            # (Settings > Labels) may be applied — conversation tagging accepts
            # any free-form string with no relation to that catalog, so
            # without this check the model could invent labels that are
            # invisible in Settings, uncolored, and absent from label filters
            # (EVO-2248).
            catalog_fetch_ok, catalog_labels = await _fetch_catalog_labels(client)

            if not catalog_fetch_ok:
                # Fetch genuinely failed — distinct from a successfully
                # fetched, legitimately empty catalog. Don't silently reject
                # everything; surface the failure instead.
                return {
                    "status": "error",
                    "message": (
                        "Could not load the account's label catalog to validate the "
                        "requested label(s), so nothing was added. Please retry."
                    ),
                    "conversation_id": effective_conversation_id,
                    "action": "add",
                }

            catalog_by_lower = {title.lower(): title for title in catalog_labels}

            valid_requested = []
            rejected = []
            for label in requested:
                catalog_title = catalog_by_lower.get(label.lower())
                if catalog_title:
                    valid_requested.append(catalog_title)
                else:
                    rejected.append(label)

            merged = list(current_labels)
            added: List[str] = []
            already_present: List[str] = []
            for label in valid_requested:
                if label.lower() not in existing_set:
                    merged.append(label)
                    existing_set[label.lower()] = label
                    added.append(label)
                else:
                    already_present.append(label)

            if not added:
                if rejected and already_present:
                    message = (
                        f"Already present: {', '.join(already_present)}. Rejected "
                        f"(not in the account's label catalog): {', '.join(rejected)}. "
                        f"Only pre-existing labels can be applied — create them in "
                        f"Settings > Labels first."
                    )
                elif rejected:
                    message = (
                        f"None of the requested label(s) exist in the account's label "
                        f"catalog: {', '.join(rejected)}. Only pre-existing labels can "
                        f"be applied — create them in Settings > Labels first."
                    )
                else:
                    message = "All requested labels were already present; nothing to update."
                return {
                    "status": "success" if not rejected else "error",
                    "message": message,
                    "conversation_id": effective_conversation_id,
                    "action": "add",
                    "labels": current_labels,
                    "added": [],
                    "rejected": rejected,
                }

            payload = {"labels": merged}
        else:
            removed_lookup = {label.lower() for label in requested}
            merged = [label for label in current_labels if label.lower() not in removed_lookup]
            removed = [label for label in current_labels if label.lower() in removed_lookup]

            if not removed:
                return {
                    "status": "success",
                    "message": "None of the requested labels were present; nothing to update.",
                    "conversation_id": effective_conversation_id,
                    "action": "remove",
                    "labels": current_labels,
                    "removed": [],
                }

            payload = {"labels": merged}

        try:
            response = await client.post(endpoint=endpoint, json_data=payload)
            resulting_labels = _normalize_labels(response) or payload["labels"]
        except Exception as api_error:
            error_message = str(api_error)
            if "404" in error_message or "not found" in error_message.lower():
                error_message = (
                    f"Conversation {effective_conversation_id} not found. "
                    "Please verify the ID is correct."
                )
            elif "401" in error_message or "unauthorized" in error_message.lower():
                error_message = "Authentication failed. Please check EVOAI_CRM_API_TOKEN configuration."
            elif "400" in error_message or "bad request" in error_message.lower():
                error_message = (
                    f"Invalid request when updating labels on conversation "
                    f"{effective_conversation_id}. Labels: {payload['labels']}"
                )

            logger.error(f"Failed to update conversation labels: {error_message}")
            return {
                "status": "error",
                "message": error_message,
                "conversation_id": effective_conversation_id,
                "action": normalized_action,
                "labels": current_labels,
                "error": str(api_error),
            }

        if normalized_action == "add":
            logger.info(
                f"Added labels {added} to conversation {effective_conversation_id}; "
                f"now has {resulting_labels}"
            )
            message = f"Added {len(added)} label(s) to the conversation."
            if rejected:
                message += (
                    f" Skipped label(s) not in the account's catalog: "
                    f"{', '.join(rejected)}."
                )
            return {
                "status": "success",
                "message": message,
                "conversation_id": effective_conversation_id,
                "action": "add",
                "labels": resulting_labels,
                "added": added,
                "rejected": rejected,
            }

        logger.info(
            f"Removed labels {removed} from conversation {effective_conversation_id}; "
            f"now has {resulting_labels}"
        )
        return {
            "status": "success",
            "message": f"Removed {len(removed)} label(s) from the conversation.",
            "conversation_id": effective_conversation_id,
            "action": "remove",
            "labels": resulting_labels,
            "removed": removed,
        }

    manage_conversation_labels.__name__ = "manage_conversation_labels"
    manage_conversation_labels.__doc__ = """Manage labels (tags) on the current conversation.

    Actions:
      - list:   returns the labels currently attached to the conversation
      - add:    appends one or more labels, preserving existing ones
      - remove: removes one or more labels, preserving the rest

    IMPORTANT: `add` only accepts labels that already exist in the account's
    label catalog (Settings > Labels) — it will NOT invent a new label. If a
    requested title doesn't match an existing label (case-insensitive), it is
    skipped and reported back in `rejected`. If you're unsure which labels
    exist, ask the user or check with your operator; do not guess a title
    that wasn't explicitly configured.

    Args:
        action: "list" | "add" | "remove"
        labels: label title or list of titles (required for add/remove)
        conversation_id: DO NOT SET. Supplied by the conversation context, which
            overrides anything passed here.

    Returns:
        Dictionary with action result and the resulting label list. For
        `add`, also includes `rejected`: titles that were skipped because
        they don't exist in the account's label catalog.
    """

    return FunctionTool(func=manage_conversation_labels)
