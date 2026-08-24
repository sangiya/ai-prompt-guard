"""Shared fixtures and test doubles."""

from __future__ import annotations

from typing import Any

import pytest

from prompt_guard.providers import GenerationConfig


class ScriptedClient:
    """Replays a fixed sequence of responses and records the prompts it saw.

    Lets the extraction repair loop be tested deterministically: queue a bad
    response followed by a good one and assert the loop recovers.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(
        self,
        messages: Any,
        config: GenerationConfig | None = None,
        system: str | None = None,
    ) -> str:
        self.prompts.append(messages if isinstance(messages, str) else str(messages))
        self.systems.append(system)
        if not self.responses:
            raise AssertionError("ScriptedClient exhausted: more calls than responses")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.prompts)


@pytest.fixture
def valid_ticket_json() -> str:
    return """{
        "summary": "Login fails after password reset",
        "category": "authentication",
        "priority": "high",
        "sentiment": "negative",
        "requires_human": true,
        "customer": {"name": "Ada Lovelace", "email": "ada@example.com", "company": "Analytical"}
    }"""
