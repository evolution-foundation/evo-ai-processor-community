"""The single legacy alembic migration (26a14ac7025d) must be idempotent.

A bare ``CREATE TABLE`` crash-loops the processor boot when the table already
exists but this alembic's ``alembic_version`` is empty — created by a prior run
or by ``Base.metadata.create_all`` on an older image (CRM-543). This runs in the
unit lane (no DB): it loads the migration by path and drives ``upgrade``/
``downgrade`` with a mocked bind, asserting the ``to_regclass`` guard gates the
DDL. The real crash→fix is proven end-to-end against Postgres (see the PR).
"""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_migration():
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations" / "versions" / "26a14ac7025d_.py"
    )
    spec = importlib.util.spec_from_file_location("mig_26a14ac7025d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bind(table_exists):
    bind = MagicMock()
    # _table_exists() reads op.get_bind().execute(text).scalar(): to_regclass
    # returns the relname when present, NULL (None) otherwise.
    bind.execute.return_value.scalar.return_value = (
        "evo_agent_processor_execution_metrics" if table_exists else None
    )
    return bind


def test_upgrade_skips_create_when_table_already_exists():
    mod = _load_migration()
    with patch.object(mod.op, "get_bind", return_value=_bind(True)), \
         patch.object(mod.op, "create_table") as create_table:
        mod.upgrade()
    create_table.assert_not_called()


def test_upgrade_creates_when_table_absent():
    mod = _load_migration()
    with patch.object(mod.op, "get_bind", return_value=_bind(False)), \
         patch.object(mod.op, "create_table") as create_table:
        mod.upgrade()
    create_table.assert_called_once()


def test_downgrade_skips_drop_when_table_absent():
    mod = _load_migration()
    with patch.object(mod.op, "get_bind", return_value=_bind(False)), \
         patch.object(mod.op, "drop_table") as drop_table:
        mod.downgrade()
    drop_table.assert_not_called()
