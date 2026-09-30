import uuid
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock

from src.models.models import A2AIdempotencyRecord

# Loading this leaf module directly avoids src.services.__init__ eagerly
# importing the full ADK runtime for a database-service unit test.
_module_spec = spec_from_file_location(
    "a2a_idempotency_under_test",
    Path(__file__).parents[2] / "src/services/a2a_idempotency.py",
)
a2a_idempotency = module_from_spec(_module_spec)
_module_spec.loader.exec_module(a2a_idempotency)


def _record(*, request_hash="hash", state="processing", created_at=None, updated_at=None, response=None):
    return A2AIdempotencyRecord(
        id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        context_id="conversation-1",
        idempotency_key="message-1",
        request_hash=request_hash,
        state=state,
        created_at=created_at or datetime.now(timezone.utc),
        updated_at=updated_at or datetime.now(timezone.utc),
        response=response,
    )


def test_claim_creates_a_persistent_record_without_storing_message_content():
    db = MagicMock()
    db.execute.side_effect = [MagicMock(scalar_one_or_none=MagicMock(return_value=uuid.uuid4()))]
    message = {"messageId": "message-1", "parts": [{"type": "text", "text": "private text"}]}

    result = a2a_idempotency.claim_or_replay(
        db,
        agent_id=uuid.uuid4(),
        context_id="conversation-1",
        idempotency_key="message-1",
        fingerprint=a2a_idempotency.request_fingerprint(message),
    )

    assert result == ("owner", None)
    insert_statement = db.execute.call_args.args[0]
    assert "private text" not in repr(insert_statement.compile().params)
    db.commit.assert_called_once()


def test_request_fingerprint_covers_tool_context_as_well_as_message_parts():
    common = {"message": {"parts": [{"type": "text", "text": "hello"}]}}
    with_context_a = {**common, "metadata": {"conversation_id": "conversation-a"}}
    with_context_b = {**common, "metadata": {"conversation_id": "conversation-b"}}

    assert a2a_idempotency.request_fingerprint(with_context_a) != a2a_idempotency.request_fingerprint(
        with_context_b
    )


def test_claim_replays_completed_response():
    response = {"id": "task-1", "status": {"state": "completed"}}
    record = _record(state="completed", response=response)
    db = MagicMock()
    db.execute.return_value.scalar_one_or_none.side_effect = [None, record]

    claim = a2a_idempotency.claim_or_replay(
        db,
        agent_id=record.agent_id,
        context_id=record.context_id,
        idempotency_key=record.idempotency_key,
        fingerprint=record.request_hash,
    )

    assert claim == ("replay", response)


def test_claim_rejects_reuse_with_different_payload():
    record = _record(request_hash="original-hash")
    db = MagicMock()
    db.execute.return_value.scalar_one_or_none.side_effect = [None, record]

    claim = a2a_idempotency.claim_or_replay(
        db,
        agent_id=record.agent_id,
        context_id=record.context_id,
        idempotency_key=record.idempotency_key,
        fingerprint="different-hash",
    )

    assert claim == ("conflict", None)


def test_fresh_processing_claim_is_not_taken_over():
    now = datetime.now(timezone.utc)
    record = _record(created_at=now, updated_at=now)
    db = MagicMock()
    db.execute.return_value.scalar_one_or_none.side_effect = [None, record]

    claim = a2a_idempotency.claim_or_replay(
        db,
        agent_id=record.agent_id,
        context_id=record.context_id,
        idempotency_key=record.idempotency_key,
        fingerprint=record.request_hash,
    )

    assert claim == ("in_progress", None)
    db.query.assert_not_called()


def test_stale_processing_claim_is_not_retried_automatically():
    now = datetime.now(timezone.utc)
    record = _record(updated_at=now - timedelta(minutes=6))
    db = MagicMock()
    db.execute.return_value.scalar_one_or_none.side_effect = [None, record]
    claim = a2a_idempotency.claim_or_replay(
        db,
        agent_id=record.agent_id,
        context_id=record.context_id,
        idempotency_key=record.idempotency_key,
        fingerprint=record.request_hash,
    )

    assert claim == ("expired", None)
    db.query.assert_not_called()


def test_completed_response_is_saved_only_for_matching_processing_claim():
    db = MagicMock()
    db.query.return_value.filter.return_value.update.return_value = 1
    a2a_idempotency.complete(
        db,
        agent_id=uuid.uuid4(),
        context_id="conversation-1",
        idempotency_key="message-1",
        fingerprint="hash",
        response={"id": "task-1"},
    )

    update_values = db.query.return_value.filter.return_value.update.call_args.args[0]
    assert update_values[A2AIdempotencyRecord.state] == "completed"
    assert update_values[A2AIdempotencyRecord.response] == {"id": "task-1"}
    db.commit.assert_called_once()


def test_complete_refuses_to_report_success_if_claim_disappeared():
    db = MagicMock()
    db.query.return_value.filter.return_value.update.return_value = 0

    try:
        a2a_idempotency.complete(
            db,
            agent_id=uuid.uuid4(),
            context_id="conversation-1",
            idempotency_key="message-1",
            fingerprint="hash",
            response={"id": "task-1"},
        )
    except RuntimeError as error:
        assert "claim was lost" in str(error)
    else:
        raise AssertionError("completion without the processing claim must fail")
    db.rollback.assert_called_once()
    db.commit.assert_not_called()


def test_failure_releases_only_matching_processing_claim():
    db = MagicMock()
    a2a_idempotency.release(
        db,
        agent_id=uuid.uuid4(),
        context_id="conversation-1",
        idempotency_key="message-1",
        fingerprint="hash",
    )

    db.execute.assert_called_once()
    db.commit.assert_called_once()
