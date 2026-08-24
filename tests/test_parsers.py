"""Tests for recovering JSON from unreliable model output."""

from __future__ import annotations

import pytest

from prompt_guard.parsers import (
    ParseError,
    extract_json,
    repair_json,
    strip_code_fences,
)


class TestStripCodeFences:
    def test_removes_json_labelled_fence(self) -> None:
        assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_removes_unlabelled_fence(self) -> None:
        assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_leaves_unfenced_text_alone(self) -> None:
        assert strip_code_fences('{"a": 1}') == '{"a": 1}'


class TestExtractJson:
    def test_parses_bare_object(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_parses_bare_array(self) -> None:
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_parses_fenced_object(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_ignores_leading_prose(self) -> None:
        assert extract_json('Here is the result:\n{"a": 1}') == {"a": 1}

    def test_ignores_trailing_prose(self) -> None:
        assert extract_json('{"a": 1}\nHope that helps!') == {"a": 1}

    def test_handles_nested_objects(self) -> None:
        assert extract_json('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}

    def test_handles_braces_inside_strings(self) -> None:
        # Naive brace counting truncates here; the parser tracks string state.
        assert extract_json('{"note": "use {curly} braces", "n": 2}') == {
            "note": "use {curly} braces",
            "n": 2,
        }

    def test_handles_escaped_quotes_inside_strings(self) -> None:
        assert extract_json(r'{"quote": "she said \"hi\"", "n": 1}') == {
            "quote": 'she said "hi"',
            "n": 1,
        }

    def test_repairs_trailing_comma_in_object(self) -> None:
        assert extract_json('{"a": 1,}') == {"a": 1}

    def test_repairs_trailing_comma_in_array(self) -> None:
        assert extract_json('{"a": [1, 2,]}') == {"a": [1, 2]}

    def test_repairs_python_literals(self) -> None:
        assert extract_json('{"a": None, "b": True, "c": False}') == {
            "a": None,
            "b": True,
            "c": False,
        }

    def test_repair_can_be_disabled(self) -> None:
        with pytest.raises(ParseError):
            extract_json('{"a": 1,}', attempt_repair=False)

    def test_empty_response_raises(self) -> None:
        with pytest.raises(ParseError, match="empty"):
            extract_json("")

    def test_whitespace_only_response_raises(self) -> None:
        with pytest.raises(ParseError, match="empty"):
            extract_json("   \n  ")

    def test_prose_without_json_raises(self) -> None:
        with pytest.raises(ParseError, match="no decodable JSON"):
            extract_json("I cannot help with that request.")

    def test_unterminated_object_raises(self) -> None:
        with pytest.raises(ParseError):
            extract_json('{"a": 1')


class TestRepairJson:
    def test_strips_trailing_commas(self) -> None:
        assert repair_json('{"a": 1,}') == '{"a": 1}'

    def test_converts_python_none(self) -> None:
        assert "null" in repair_json('{"a": None}')

    def test_leaves_valid_json_unchanged(self) -> None:
        assert repair_json('{"a": 1}') == '{"a": 1}'
