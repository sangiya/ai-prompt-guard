"""Provider-agnostic LLM client.

Supports Anthropic, OpenAI and Ollama behind one interface, plus a deterministic
offline backend so the toolkit is fully usable without credentials.

When ``provider`` is ``"auto"`` the first available backend wins, in order:
``ANTHROPIC_API_KEY`` -> ``OPENAI_API_KEY`` -> reachable Ollama -> ``offline``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "GenerationConfig",
    "LLMClient",
    "Message",
    "ProviderError",
    "resolve_provider",
]

logger = logging.getLogger(__name__)

ProviderName = Literal["auto", "anthropic", "openai", "ollama", "offline"]

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
    "offline": "offline-1",
}

_OLLAMA_PROBE_TIMEOUT = 1.5
_OLLAMA_REQUEST_TIMEOUT = 120


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class ProviderError(RuntimeError):
    """Raised when a backend is misconfigured, unavailable, or unknown."""


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat turn."""

    role: Literal["user", "assistant"]
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class GenerationConfig:
    """Decoding parameters. Backends silently ignore what they cannot express."""

    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    seed: int | None = None

    def validate(self) -> None:
        """Raise :class:`ValueError` if any parameter is outside its valid range."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(
            f"{_ollama_host()}/api/tags", timeout=_OLLAMA_PROBE_TIMEOUT
        ) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def resolve_provider(requested: ProviderName = "auto") -> str:
    """Return the concrete backend name for ``requested``."""
    if requested != "auto":
        return requested
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if _ollama_reachable():
        return "ollama"
    logger.info("No provider credentials found; using the offline backend.")
    return "offline"


class LLMClient:
    """Uniform chat-completion interface across providers."""

    def __init__(
        self,
        provider: ProviderName = "auto",
        model: str | None = None,
    ) -> None:
        self.provider = resolve_provider(provider)
        if self.provider not in DEFAULT_MODELS:
            raise ProviderError(
                f"unknown provider {self.provider!r}; expected one of {sorted(DEFAULT_MODELS)}"
            )
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        backends: dict[str, Callable[..., str]] = {
            "anthropic": self._call_anthropic,
            "openai": self._call_openai,
            "ollama": self._call_ollama,
            "offline": self._call_offline,
        }
        self._backend = backends[self.provider]
        logger.debug("LLMClient ready: provider=%s model=%s", self.provider, self.model)

    def complete(
        self,
        messages: Iterable[Message] | str,
        config: GenerationConfig | None = None,
        system: str | None = None,
    ) -> str:
        """Generate a completion.

        Args:
            messages: A prompt string, or an iterable of :class:`Message` turns.
            config: Decoding parameters; defaults are used when omitted.
            system: Optional system instruction.

        Returns:
            The assistant's reply as plain text.

        Raises:
            ValueError: If ``config`` holds an out-of-range parameter.
            ProviderError: If the backend is unavailable or its SDK is missing.
        """
        turns = [Message("user", messages)] if isinstance(messages, str) else list(messages)
        if not turns:
            raise ValueError("messages must not be empty")

        resolved = config or GenerationConfig()
        resolved.validate()
        return self._backend(turns, resolved, system)

    # -- backends --------------------------------------------------------

    def _call_anthropic(
        self, messages: list[Message], config: GenerationConfig, system: str | None
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError("Anthropic SDK missing; run: pip install anthropic") from exc

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": [m.as_dict() for m in messages],
        }
        if config.top_k is not None:
            payload["top_k"] = config.top_k
        if config.stop:
            payload["stop_sequences"] = config.stop
        if system:
            # Cached so repeated calls reusing this system prompt bill at a discount.
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        response = anthropic.Anthropic().messages.create(**payload)
        return "".join(block.text for block in response.content if block.type == "text")

    def _call_openai(
        self, messages: list[Message], config: GenerationConfig, system: str | None
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError("OpenAI SDK missing; run: pip install openai") from exc

        turns: list[dict[str, str]] = [m.as_dict() for m in messages]
        if system:
            turns.insert(0, {"role": "system", "content": system})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        if config.stop:
            payload["stop"] = config.stop
        if config.seed is not None:
            payload["seed"] = config.seed

        response = OpenAI().chat.completions.create(**payload)
        return response.choices[0].message.content or ""

    def _call_ollama(
        self, messages: list[Message], config: GenerationConfig, system: str | None
    ) -> str:
        turns: list[dict[str, str]] = [m.as_dict() for m in messages]
        if system:
            turns.insert(0, {"role": "system", "content": system})

        options: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        }
        if config.top_k is not None:
            options["top_k"] = config.top_k
        if config.stop:
            options["stop"] = config.stop
        if config.seed is not None:
            options["seed"] = config.seed

        body = json.dumps(
            {"model": self.model, "messages": turns, "stream": False, "options": options}
        ).encode()
        request = urllib.request.Request(
            f"{_ollama_host()}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=_OLLAMA_REQUEST_TIMEOUT) as response:
                return str(json.loads(response.read())["message"]["content"])
        except urllib.error.URLError as exc:
            raise ProviderError(f"Ollama unreachable at {_ollama_host()}: {exc}") from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Unexpected Ollama response format: {exc}") from exc

    def _call_offline(
        self, messages: list[Message], config: GenerationConfig, system: str | None
    ) -> str:
        """Deterministic local backend used when no provider is configured."""
        prompt = messages[-1].content
        header = f"[offline | temperature={config.temperature} top_p={config.top_p}]"
        return f"{header} {prompt[:400]}"
