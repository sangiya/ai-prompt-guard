"""Recover JSON from unreliable model output.

Models rarely return bare JSON. They wrap it in markdown fences, prefix it with
prose, append explanations, or emit near-JSON with trailing commas and Python
literals. This module extracts and repairs the payload before validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Final

__all__ = ["ParseError", "extract_json", "repair_json", "strip_code_fences"]

logger = logging.getLogger(__name__)

_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?:json|JSON)?\s*\n?(?P<body>.*?)```", re.DOTALL
)
_TRAILING_COMMA: Final[re.Pattern[str]] = re.compile(r",\s*(?=[}\]])")
_OPENING = {"{": "}", "[": "]"}


class ParseError(ValueError):
    """Raised when no valid JSON object can be recovered from a response."""


def strip_code_fences(text: str) -> str:
    """Return the contents of the first fenced block, or ``text`` unchanged."""
    match = _FENCE_PATTERN.search(text)
    return match.group("body").strip() if match else text.strip()


def _find_balanced_span(text: str) -> str | None:
    """Return the first complete ``{...}`` or ``[...]`` span, respecting strings.

    Brace counting alone breaks on JSON containing braces inside string values,
    so quoted regions and escape sequences are tracked explicitly.
    """
    start = next(
        (i for i, char in enumerate(text) if char in _OPENING),
        None,
    )
    if start is None:
        return None

    opening = text[start]
    closing = _OPENING[opening]
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def repair_json(text: str) -> str:
    """Apply conservative fixes for the malformations models produce most often."""
    repaired = _TRAILING_COMMA.sub("", text)
    # Python literals leak in when a model reasons in Python rather than JSON.
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    return repaired


def extract_json(text: str, *, attempt_repair: bool = True) -> Any:
    """Extract and decode the first JSON value embedded in ``text``.

    Args:
        text: Raw model output, which may include prose or markdown fencing.
        attempt_repair: Retry with :func:`repair_json` when strict decoding fails.

    Returns:
        The decoded JSON value.

    Raises:
        ParseError: If no decodable JSON value is present.
    """
    if not text or not text.strip():
        raise ParseError("response was empty")

    candidate = strip_code_fences(text)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    span = _find_balanced_span(candidate)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            if attempt_repair:
                try:
                    return json.loads(repair_json(span))
                except json.JSONDecodeError:
                    pass

    if attempt_repair:
        try:
            return json.loads(repair_json(candidate))
        except json.JSONDecodeError:
            pass

    preview = text[:200].replace("\n", " ")
    raise ParseError(f"no decodable JSON found in response: {preview!r}")
