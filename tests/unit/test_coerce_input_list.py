"""Regression test for `_coerce_input_list`'s stringified-array handling.

2026-09-22 incident: GLM (via OpenRouter) called `manage_conversation_labels`
with `labels='["qualificado_ntba", "reuniao_confirmada", "etapa_concluida"]'`
— a single string, not a real JSON array — instead of the expected
`labels=["qualificado_ntba", "reuniao_confirmada", "etapa_concluida"]`. Left
unhandled, `_coerce_input_list` treated the whole bracketed string as one
literal label, which never matched the catalog and was silently rejected —
surfacing to the model (and then the user, via its own private note) as
"these labels don't exist", when they were never actually looked up
correctly at all.
"""

from src.services.adk.tools.evo_crm.manage_conversation_labels import (
    _coerce_input_list,
)


def test_accepts_a_real_list():
    assert _coerce_input_list(["a", "b"]) == ["a", "b"]


def test_accepts_a_single_string():
    assert _coerce_input_list("a") == ["a"]


def test_parses_a_stringified_json_array():
    """The actual live regression."""
    stringified = '["qualificado_ntba", "reuniao_confirmada", "etapa_concluida"]'
    assert _coerce_input_list(stringified) == [
        "qualificado_ntba",
        "reuniao_confirmada",
        "etapa_concluida",
    ]


def test_a_malformed_bracketed_string_falls_back_to_a_literal_label():
    """Not valid JSON — must not crash, and must not silently vanish either."""
    malformed = "[not valid json]"
    assert _coerce_input_list(malformed) == ["[not valid json]"]


def test_deduplicates_and_strips_after_parsing_the_array():
    stringified = '[" a ", "a", "b"]'
    assert _coerce_input_list(stringified) == ["a", "b"]


def test_none_returns_empty_list():
    assert _coerce_input_list(None) == []
