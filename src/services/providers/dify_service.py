"""
Dify provider service for external agent integration.
"""

import json
import logging
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class DifyService:
    """Service for integrating with Dify AI platform."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Dify service.

        Args:
            config: Configuration dictionary with:
                - apiUrl: Base URL of Dify API (e.g., https://api.dify.ai/v1)
                - apiKey: API key for authentication
                - botType: Type of bot ('chatBot', 'textGenerator', or 'agent')
        """
        self.api_url = config.get("apiUrl")
        self.api_key = config.get("apiKey")
        self.bot_type = config.get("botType", "chatBot")
        
        if not self.api_url:
            raise ValueError("Dify apiUrl is required")
        if not self.api_key:
            raise ValueError("Dify apiKey is required")
        if self.bot_type not in ["chatBot", "textGenerator", "agent"]:
            raise ValueError(f"Invalid botType: {self.bot_type}. Must be 'chatBot', 'textGenerator', or 'agent'")

    async def send_message(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to Dify and get the response.

        Args:
            message: User message
            session_id: Evo session identifier (used as the Dify ``user``)
            context: Optional context variables (remoteJid, pushName, etc.).
                ``providerConversationId`` carries the Dify conversation_id
                returned by a previous call in the same Evo session.

        Returns:
            ``{"text": answer, "conversation_id": dify_conversation_id}``
        """
        if not self.api_url or not self.api_key:
            raise ValueError("Dify apiUrl and apiKey are required")

        context = context or {}
        conversation_id = context.get("providerConversationId") or None

        try:
            return await self._post(message, session_id, context, conversation_id)
        except _DifyConversationNotFound:
            # The stored conversation was deleted on the Dify side (or came from
            # another app): start a fresh one instead of failing the turn.
            logger.warning(
                f"Dify conversation {conversation_id} not found, starting a new one"
            )
            return await self._post(message, session_id, context, None)

    async def _post(
        self,
        message: str,
        session_id: str,
        context: Dict[str, Any],
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        # textGenerator → completion app; chatBot/agent → chat-style apps.
        is_completion = self.bot_type == "textGenerator"
        endpoint = f"{self.api_url}/{'completion-messages' if is_completion else 'chat-messages'}"

        payload: Dict[str, Any] = {
            "inputs": {
                "remoteJid": context.get("remoteJid", ""),
                "pushName": context.get("pushName", ""),
                "instanceName": context.get("instanceName", ""),
                "serverUrl": context.get("serverUrl", ""),
                "apiKey": context.get("apiKey", ""),
            },
            "query": message,
            # Chat-style apps always stream: Agent Chat apps reject blocking
            # mode with 400, and streaming works for every chat app type.
            "response_mode": "blocking" if is_completion else "streaming",
            "user": context.get("remoteJid") or session_id,
        }
        # conversation_id must be one Dify issued — never the Evo session id,
        # which Dify rejects with 404 "Conversation Not Exists".
        if conversation_id and not is_completion:
            payload["conversation_id"] = conversation_id

        if is_completion:
            payload["inputs"]["query"] = message
            del payload["query"]

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()

                if is_completion:
                    return {"text": response.json().get("answer", ""), "conversation_id": None}
                return self._process_streaming_response(response.text)
        except httpx.HTTPStatusError as e:
            body = e.response.text
            if e.response.status_code == 404 and conversation_id and "Conversation Not Exists" in body:
                raise _DifyConversationNotFound() from e
            logger.error(f"Dify API error: {e.response.status_code} - {body}")
            raise Exception(f"Dify API error: {e.response.status_code} - {_error_message(body)}")
        except Exception as e:
            logger.error(f"Error calling Dify: {e}")
            raise

    def _process_streaming_response(self, response_text: str) -> Dict[str, Any]:
        """
        Process a streaming SSE response from a Dify chat app.

        Chatbot/Chatflow apps stream ``message`` events and Agent apps stream
        ``agent_message`` events; both carry an ``answer`` chunk and the
        ``conversation_id``. An ``error`` event aborts the turn.
        """
        answer = ""
        conversation_id = None

        for line in response_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse SSE event: {line}")
                continue

            kind = event.get("event")
            if kind in ("message", "agent_message"):
                answer += event.get("answer", "")
                conversation_id = conversation_id or event.get("conversation_id")
            elif kind == "message_end":
                conversation_id = conversation_id or event.get("conversation_id")
            elif kind == "error":
                raise Exception(f"Dify API error: {event.get('status', '')} - {event.get('message', '')}")

        return {"text": answer, "conversation_id": conversation_id}


class _DifyConversationNotFound(Exception):
    """The conversation_id sent to Dify no longer exists."""


def _error_message(body: str) -> str:
    try:
        return json.loads(body).get("message", body)
    except (ValueError, AttributeError):
        return body
