"""Add durable idempotency records for CRM A2A message/send requests."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260929090000"
down_revision: Union[str, None] = "26a14ac7025d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evo_ai_a2a_idempotency_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("context_id", sa.String(length=255), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("response", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("state IN ('processing', 'completed')", name="check_a2a_idempotency_state"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_id",
            "context_id",
            "idempotency_key",
            name="uq_a2a_idempotency_agent_context_key",
        ),
    )
    op.create_index(
        "ix_a2a_idempotency_created_at",
        "evo_ai_a2a_idempotency_records",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_a2a_idempotency_created_at", table_name="evo_ai_a2a_idempotency_records")
    op.drop_table("evo_ai_a2a_idempotency_records")
