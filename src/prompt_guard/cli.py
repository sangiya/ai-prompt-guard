"""Command-line interface.

Subcommands:
    ``scan``     score text for prompt-injection indicators
    ``extract``  pull schema-validated structured data out of a document
    ``wrap``     print untrusted text fenced in its defensive envelope
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from .extraction import ExtractionError, StructuredExtractor
from .injection import InjectionDetector, RiskLevel, wrap_untrusted
from .providers import LLMClient, ProviderError
from .schemas import Contact, Invoice, SupportTicket

__all__ = ["main"]

SCHEMAS: dict[str, type[BaseModel]] = {
    "ticket": SupportTicket,
    "invoice": Invoice,
    "contact": Contact,
}
PROVIDER_CHOICES = ("auto", "anthropic", "openai", "ollama", "offline")

_RISK_EXIT_CODES = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _read_input(text: str | None, path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    if text is not None:
        return text
    return sys.stdin.read()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-guard",
        description="Schema-enforced LLM outputs and prompt-injection defence.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)

    source = argparse.ArgumentParser(add_help=False)
    group = source.add_mutually_exclusive_group()
    group.add_argument("--text", help="input text (reads stdin when omitted)")
    group.add_argument("--file", help="read input from this path")
    source.add_argument("--json", action="store_true", help="emit JSON")

    scan = subcommands.add_parser("scan", parents=[source], help="score injection risk")
    scan.add_argument(
        "--fail-on",
        choices=[level.value for level in RiskLevel],
        default="medium",
        help="exit non-zero at or above this risk level",
    )

    extract = subcommands.add_parser(
        "extract", parents=[source], help="extract validated structured data"
    )
    extract.add_argument("--schema", choices=sorted(SCHEMAS), required=True)
    extract.add_argument("--provider", default="auto", choices=PROVIDER_CHOICES)
    extract.add_argument("--model", default=None)
    extract.add_argument("--max-attempts", type=int, default=3)
    extract.add_argument(
        "--allow-suspicious",
        action="store_true",
        help="skip injection screening of the source document",
    )

    subcommands.add_parser("wrap", parents=[source], help="fence untrusted text")

    return parser


def _run_scan(args: argparse.Namespace, text: str) -> int:
    result = InjectionDetector().scan(text)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(f"risk  : {result.risk.value.upper()}  (score {result.score})")
        if result.signals:
            print(f"signals ({len(result.signals)}):")
            for signal in result.signals:
                print(f"  [{signal.category.value}] +{signal.weight}  {signal.evidence!r}")
        else:
            print("signals: none")

    order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    if order.index(result.risk) >= order.index(RiskLevel(args.fail_on)):
        return _RISK_EXIT_CODES[result.risk]
    return 0


def _run_extract(args: argparse.Namespace, text: str) -> int:
    client = LLMClient(provider=args.provider, model=args.model)
    extractor: StructuredExtractor[BaseModel] = StructuredExtractor(
        client=client,
        schema=SCHEMAS[args.schema],
        max_attempts=args.max_attempts,
        screen_for_injection=not args.allow_suspicious,
    )
    result = extractor.extract(text)

    if args.json:
        print(result.data.model_dump_json(indent=2))
    else:
        print(f"schema   : {args.schema}")
        print(f"attempts : {result.attempts}{' (repaired)' if result.required_repair else ''}")
        print("-" * 60)
        print(result.data.model_dump_json(indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        text = _read_input(args.text, args.file)
        if not text.strip():
            print("error: no input provided", file=sys.stderr)
            return 1

        if args.command == "scan":
            return _run_scan(args, text)
        if args.command == "extract":
            return _run_extract(args, text)

        print(wrap_untrusted(text))
        return 0

    except (ExtractionError, ProviderError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
