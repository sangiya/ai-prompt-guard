"""Schema-enforced extraction with a validation-repair loop.

Asking a model for JSON gets JSON-shaped text, not a guarantee. This module
closes the gap: it derives the contract from a Pydantic model, validates every
response against it, and feeds concrete validation errors back to the model so
the next attempt is a correction rather than a re-roll.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .injection import InjectionDetector, RiskLevel, wrap_untrusted
from .parsers import ParseError, extract_json
from .providers import GenerationConfig

__all__ = ["ExtractionError", "ExtractionResult", "StructuredExtractor", "SupportsComplete"]

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_SYSTEM_PROMPT = (
    "You extract structured data from documents. Return a single JSON object "
    "that satisfies the supplied schema. Output raw JSON only, with no prose "
    "and no markdown fences. Use null for any field the document does not state; "
    "never invent values."
)


class SupportsComplete(Protocol):
    """Minimal client contract, so any provider or test double can be injected."""

    def complete(
        self,
        messages: Any,
        config: GenerationConfig | None = ...,
        system: str | None = ...,
    ) -> str: ...


class ExtractionError(RuntimeError):
    """Raised when no attempt produced a response satisfying the schema."""


@dataclass(slots=True)
class ExtractionResult(Generic[ModelT]):
    """A validated extraction plus the trail of how it was obtained."""

    data: ModelT
    attempts: int
    raw_responses: list[str] = field(repr=False)
    injection: Any = None

    @property
    def required_repair(self) -> bool:
        """True when the first attempt failed validation."""
        return self.attempts > 1


class StructuredExtractor(Generic[ModelT]):
    """Extract a validated instance of ``schema`` from unstructured text."""

    def __init__(
        self,
        client: SupportsComplete,
        schema: type[ModelT],
        *,
        max_attempts: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        screen_for_injection: bool = True,
        max_injection_risk: RiskLevel = RiskLevel.LOW,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        self.client = client
        self.schema = schema
        self.max_attempts = max_attempts
        self.screen_for_injection = screen_for_injection
        self.max_injection_risk = max_injection_risk
        self._detector = InjectionDetector()
        # Temperature defaults to 0: extraction is a deterministic transcription
        # task, and sampling only introduces field-level variance.
        self._config = GenerationConfig(temperature=temperature, max_tokens=max_tokens)

    def _schema_block(self) -> str:
        return json.dumps(self.schema.model_json_schema(), indent=2)

    def _initial_prompt(self, document: str) -> str:
        return (
            f"Extract data matching this JSON Schema:\n\n{self._schema_block()}\n\n"
            f"{wrap_untrusted(document, tag='document')}\n\n"
            "Respond with the JSON object only."
        )

    @staticmethod
    def _repair_prompt(previous: str, error: str) -> str:
        return (
            f"Your previous response failed validation.\n\n"
            f"Response:\n{previous}\n\n"
            f"Errors:\n{error}\n\n"
            "Return corrected JSON that resolves every error listed above. "
            "Output the JSON object only."
        )

    def extract(self, document: str) -> ExtractionResult[ModelT]:
        """Extract structured data from ``document``.

        Args:
            document: Untrusted source text to extract from.

        Returns:
            An :class:`ExtractionResult` holding the validated model.

        Raises:
            ValueError: If ``document`` is empty, or screening rejects it.
            ExtractionError: If every attempt failed to satisfy the schema.
        """
        if not document or not document.strip():
            raise ValueError("document must not be empty")

        detection = None
        if self.screen_for_injection:
            detection = self._detector.scan(document)
            order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
            if order.index(detection.risk) > order.index(self.max_injection_risk):
                raise ValueError(
                    f"document rejected: injection risk {detection.risk.value} "
                    f"(score {detection.score}) exceeds "
                    f"{self.max_injection_risk.value}"
                )

        prompt = self._initial_prompt(document)
        responses: list[str] = []
        last_error = ""

        for attempt in range(1, self.max_attempts + 1):
            response = self.client.complete(
                prompt, config=self._config, system=_SYSTEM_PROMPT
            )
            responses.append(response)

            try:
                payload = extract_json(response)
                validated = self.schema.model_validate(payload)
            except ParseError as exc:
                last_error = f"Response was not valid JSON: {exc}"
                logger.info("Attempt %d/%d: %s", attempt, self.max_attempts, last_error)
            except ValidationError as exc:
                last_error = "\n".join(
                    f"- field '{'.'.join(str(p) for p in err['loc']) or '<root>'}': {err['msg']}"
                    for err in exc.errors()
                )
                logger.info(
                    "Attempt %d/%d failed validation with %d error(s)",
                    attempt, self.max_attempts, len(exc.errors()),
                )
            else:
                return ExtractionResult(
                    data=validated,
                    attempts=attempt,
                    raw_responses=responses,
                    injection=detection,
                )

            prompt = self._repair_prompt(response, last_error)

        raise ExtractionError(
            f"failed to extract valid {self.schema.__name__} after "
            f"{self.max_attempts} attempt(s). Last error:\n{last_error}"
        )
