"""DB connection pool size/overflow must be configurable via env vars.

Without this, SQLAlchemy falls back to its bare defaults (pool_size=5,
max_overflow=10 => 15 total connections), which is too small for this
service's real concurrency (many stage-inactivity AI calls held open for
30-90s each, plus live chat traffic) and exhausts under load
(`QueuePool limit of size 5 overflow 10 reached, connection timed out`),
with no recovery until the process is restarted.
"""

import importlib


def _reload_engine():
    # importlib.import_module (not `import ... as`) to get the actual submodule
    # object: src/config/__init__.py does `from .settings import settings`,
    # which shadows the `settings` attribute on the `src.config` package with
    # the Settings *instance* — an `import x.y as z` resolves through that
    # attribute, so it would bind `z` to the instance instead of the module.
    settings_module = importlib.import_module("src.config.settings")
    importlib.reload(settings_module)
    database_module = importlib.import_module("src.config.database")
    importlib.reload(database_module)
    return database_module.engine


def test_pool_size_and_max_overflow_are_configurable(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "20")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "40")

    engine = _reload_engine()

    assert engine.pool.size() == 20
    assert engine.pool._max_overflow == 40


def test_pool_size_and_max_overflow_have_higher_than_library_defaults(monkeypatch):
    monkeypatch.delenv("DB_POOL_SIZE", raising=False)
    monkeypatch.delenv("DB_MAX_OVERFLOW", raising=False)

    engine = _reload_engine()

    # SQLAlchemy's bare defaults (5/10) are exactly what exhausted in production;
    # the service's own default must be higher, not merely "configurable".
    assert engine.pool.size() > 5
    assert engine.pool._max_overflow > 10
