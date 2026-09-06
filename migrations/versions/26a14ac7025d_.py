"""

Revision ID: 26a14ac7025d
Revises: 
Create Date: 2025-08-21 09:23:27.584955

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '26a14ac7025d'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

metadata = sa.MetaData()

execution_metrics = sa.Table(
    'evo_agent_processor_execution_metrics',
    metadata,
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column('session_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('llm_model', sa.String(), nullable=False),
    sa.Column('prompt_tokens', sa.Integer(), nullable=False),
    sa.Column('candidate_tokens', sa.Integer(), nullable=False),
    sa.Column('cost', sa.Float(), nullable=False),
    sa.Column('total_tokens', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['evo_core_agents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
)

TABLE = 'evo_agent_processor_execution_metrics'


def _table_exists() -> bool:
    return op.get_bind().execute(
        sa.text("SELECT to_regclass(:t)"), {"t": TABLE}
    ).scalar() is not None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent (CRM-543): on a database where this table already exists —
    created by a prior run, or by ``Base.metadata.create_all`` on an older image
    — while this alembic's ``alembic_version`` is empty, a bare ``CREATE TABLE``
    raises ``DuplicateTable`` and crash-loops the processor boot before it ever
    reaches ``create_all``/the enterprise chain. Skip the create when the table
    is already there; the revision is still stamped, reconciling the drift.
    """
    if _table_exists():
        return
    op.create_table(
        TABLE,
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('llm_model', sa.String(), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False),
        sa.Column('candidate_tokens', sa.Integer(), nullable=False),
        sa.Column('cost', sa.Float(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['evo_core_agents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema. Idempotent for symmetry with upgrade (CRM-543)."""
    if _table_exists():
        op.drop_table(TABLE)
