"""Shared data models for the support triage pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TicketFacts:
    raw_issue: str
    subject: str
    company_field: str
    issue_text: str
    user_text: str
    assistant_text: str
    language: str
    pii_detected: bool
    injection_detected: bool
    pure_injection: bool
    company_guess: str
    multi_issue: bool
    verified_identity: bool
    email: str | None = None
    phone: str | None = None
    user_identifier: str | None = None
    transaction_id: str | None = None
    amount: float | None = None


@dataclass
class TriageResult:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    confidence_score: float
    source_documents: list[str] = field(default_factory=list)
    risk_level: str = "low"
    pii_detected: bool = False
    language: str = "en"
    actions_taken: list[dict[str, Any]] = field(default_factory=list)

