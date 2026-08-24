"""Trustworthy LLM I/O: schema-enforced outputs and prompt-injection defence.

Two halves of the same problem. :mod:`extraction` guarantees what comes *out*
of a model conforms to a declared contract; :mod:`injection` hardens what goes
*in* when the source is untrusted.
"""

from __future__ import annotations

from .extraction import ExtractionError, ExtractionResult, StructuredExtractor
from .injection import (
    DetectionResult,
    InjectionDetector,
    RiskLevel,
    Signal,
    SignalCategory,
    normalize,
    wrap_untrusted,
)
from .parsers import ParseError, extract_json, repair_json, strip_code_fences
from .providers import GenerationConfig, LLMClient, Message, ProviderError
from .schemas import Contact, Invoice, LineItem, Priority, Sentiment, SupportTicket

__version__ = "1.0.0"

__all__ = [
    "Contact",
    "DetectionResult",
    "ExtractionError",
    "ExtractionResult",
    "GenerationConfig",
    "InjectionDetector",
    "Invoice",
    "LLMClient",
    "LineItem",
    "Message",
    "ParseError",
    "Priority",
    "ProviderError",
    "RiskLevel",
    "Sentiment",
    "Signal",
    "SignalCategory",
    "StructuredExtractor",
    "SupportTicket",
    "__version__",
    "extract_json",
    "normalize",
    "repair_json",
    "strip_code_fences",
    "wrap_untrusted",
]
