"""Target schemas for structured extraction.

Each model doubles as documentation of the contract the LLM must satisfy: field
descriptions are injected into the prompt, and constraints are enforced after
generation rather than trusted.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator

__all__ = ["Contact", "Invoice", "LineItem", "Priority", "Sentiment", "SupportTicket"]


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Contact(BaseModel):
    """A person referenced in a document."""

    name: str = Field(description="Full name as written")
    email: str | None = Field(default=None, description="Email address, if stated")
    company: str | None = Field(default=None, description="Employer or organisation")

    @field_validator("email")
    @classmethod
    def _reject_malformed_email(cls, value: str | None) -> str | None:
        if value is not None and ("@" not in value or "." not in value.split("@")[-1]):
            raise ValueError(f"malformed email address: {value!r}")
        return value


class SupportTicket(BaseModel):
    """A classified customer support message."""

    summary: str = Field(description="One-sentence summary of the issue")
    category: str = Field(description="Product area the issue belongs to")
    priority: Priority = Field(description="Urgency of the request")
    sentiment: Sentiment = Field(description="Customer's tone")
    requires_human: bool = Field(description="True if this needs a human agent")
    customer: Contact | None = Field(default=None, description="Requester, if identifiable")


class LineItem(BaseModel):
    """A single billed row on an invoice."""

    description: str
    quantity: float = Field(gt=0, description="Units billed, greater than zero")
    unit_price: float = Field(ge=0, description="Price per unit")

    @property
    def total(self) -> float:
        return round(self.quantity * self.unit_price, 2)


class Invoice(BaseModel):
    """A billing document reduced to its structured fields."""

    invoice_number: str = Field(description="Unique invoice identifier")
    issue_date: date | None = Field(default=None, description="ISO 8601 issue date")
    vendor: str = Field(description="Party issuing the invoice")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    line_items: list[LineItem] = Field(default_factory=list)
    total_amount: float = Field(ge=0, description="Invoice total including tax")

    @field_validator("currency")
    @classmethod
    def _normalise_currency(cls, value: str) -> str:
        code = value.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError(f"currency must be a 3-letter ISO 4217 code, got {value!r}")
        return code
