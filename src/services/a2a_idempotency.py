"""Persistent idempotency for A2A requests produced by the CRM message queue."""

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.models.models import A2AIdempotencyRecord

RETENTION = timedelta(hours=24)
PROCESSING_LEASE = timedelta(minutes=5)
MAX_KEY_LENGTH = 128


def request_fingerprint(request_params: Dict[str, Any]) -> str:
    """Hash canonical execution inputs; never persist raw payload or PII."""
    encoded = json.dumps(
        request_params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def claim_or_replay(
    db: Session,
    *,
    agent_id: uuid.UUID,
    context_id: str,
    idempotency_key: str,
    fingerprint: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return ownership/replay state and an optional cached A2A task response."""
    now = datetime.now(timezone.utc)
    # Opportunistic bounded cleanup: only a small batch per accepted request.
    # Sample cleanup so steady-state request cost remains independent of the
    # number of retained records. Expired rows are also replaced below.
    if uuid.uuid4().int % 100 == 0:
        expired_ids = select(A2AIdempotencyRecord.id).where(
            A2AIdempotencyRecord.created_at < now - RETENTION
        ).limit(1000)
        db.execute(
            delete(A2AIdempotencyRecord)
            .where(A2AIdempotencyRecord.id.in_(expired_ids))
            .execution_options(synchronize_session=False)
        )
    statement = (
        insert(A2AIdempotencyRecord)
        .values(
            id=uuid.uuid4(),
            agent_id=agent_id,
            context_id=context_id,
            idempotency_key=idempotency_key,
            request_hash=fingerprint,
            state="processing",
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_nothing(
            constraint="uq_a2a_idempotency_agent_context_key"
        )
        .returning(A2AIdempotencyRecord.id)
    )
    claimed_id = db.execute(statement).scalar_one_or_none()
    db.commit()
    if claimed_id is not None:
        return "owner", None

    record = db.execute(
        select(A2AIdempotencyRecord).where(
            A2AIdempotencyRecord.agent_id == agent_id,
            A2AIdempotencyRecord.context_id == context_id,
            A2AIdempotencyRecord.idempotency_key == idempotency_key,
        )
    ).scalar_one_or_none()
    if record is None:
        return "in_progress", None
    if record.request_hash != fingerprint:
        return "conflict", None
    if record.state == "completed" and record.response is not None:
        return "replay", record.response
    if record.created_at < now - RETENTION:
        db.execute(
            delete(A2AIdempotencyRecord).where(A2AIdempotencyRecord.id == record.id)
        )
        db.commit()
        return claim_or_replay(
            db,
            agent_id=agent_id,
            context_id=context_id,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
        )

    if record.updated_at < now - PROCESSING_LEASE:
        # The prior owner may have completed an external tool immediately
        # before crashing. Never automatically repeat potentially side-effecting
        # agent execution after lease expiry; require explicit operator review.
        return "expired", None

    db.rollback()  # Close the read transaction before returning a retryable conflict.

    return "in_progress", None


def complete(
    db: Session,
    *,
    agent_id: uuid.UUID,
    context_id: str,
    idempotency_key: str,
    fingerprint: str,
    response: Dict[str, Any],
) -> None:
    updated = db.query(A2AIdempotencyRecord).filter(
        A2AIdempotencyRecord.agent_id == agent_id,
        A2AIdempotencyRecord.context_id == context_id,
        A2AIdempotencyRecord.idempotency_key == idempotency_key,
        A2AIdempotencyRecord.request_hash == fingerprint,
        A2AIdempotencyRecord.state == "processing",
    ).update(
        {
            A2AIdempotencyRecord.state: "completed",
            A2AIdempotencyRecord.response: response,
            A2AIdempotencyRecord.updated_at: datetime.now(timezone.utc),
        },
        synchronize_session=False,
    )
    if updated != 1:
        db.rollback()
        raise RuntimeError("A2A idempotency claim was lost before completion")
    db.commit()


def release(
    db: Session,
    *,
    agent_id: uuid.UUID,
    context_id: str,
    idempotency_key: str,
    fingerprint: str,
) -> None:
    """Remove an uncompleted claim so transient agent errors may be retried."""
    db.execute(
        delete(A2AIdempotencyRecord).where(
            A2AIdempotencyRecord.agent_id == agent_id,
            A2AIdempotencyRecord.context_id == context_id,
            A2AIdempotencyRecord.idempotency_key == idempotency_key,
            A2AIdempotencyRecord.request_hash == fingerprint,
            A2AIdempotencyRecord.state == "processing",
        )
    )
    db.commit()
