"""Ticket parsing and coarse signal extraction."""

from __future__ import annotations

import json
import re
from typing import List

from models import TicketFacts
from text_utils import count_keyword_hits, detect_injection, detect_language, has_pii, normalize_text


SUPPORTED_COMPANIES = {"devplatform", "claude", "visa"}
MULTI_ISSUE_PHRASES = {
    "multiple issues",
    "three things",
    "also",
    "and also",
    "in addition",
    "separately",
    "another issue",
}


def parse_issue(raw_issue: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(raw_issue)
        if isinstance(parsed, list):
            cleaned = []
            for item in parsed:
                if isinstance(item, dict):
                    cleaned.append(
                        {
                            "role": str(item.get("role", "")),
                            "content": str(item.get("content", "")),
                        }
                    )
            return cleaned
    except json.JSONDecodeError:
        pass
    return [{"role": "user", "content": str(raw_issue)}]


def extract_facts(row: dict[str, str]) -> TicketFacts:
    issue_raw = row.get("Issue", row.get("issue", "")) or "[]"
    subject = row.get("Subject", row.get("subject", "")) or ""
    company_field = (row.get("Company", row.get("company", "")) or "").strip()
    messages = parse_issue(issue_raw)
    user_text_parts = [msg["content"] for msg in messages if msg.get("role") == "user" and msg.get("content")]
    assistant_text_parts = [msg["content"] for msg in messages if msg.get("role") != "user" and msg.get("content")]
    user_text = "\n".join(user_text_parts).strip()
    assistant_text = "\n".join(assistant_text_parts).strip()
    issue_text = "\n".join(part for part in [subject, user_text, assistant_text] if part).strip()
    language = detect_language(issue_text)
    pii_detected = has_pii(issue_text)
    injection_detected, pure_injection = detect_injection(issue_text)
    company_guess = infer_company(issue_text, company_field)
    multi_issue = any(phrase in normalize_text(issue_text) for phrase in MULTI_ISSUE_PHRASES)
    verified_identity = detect_verified_identity(messages)
    email = first_match(issue_text, r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    phone = first_match(issue_text, r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4})")
    user_identifier = first_match(
        issue_text,
        r"\b(?:user(?:name)?|account(?: id)?|customer|member|workspace|org(?:anization)?|team)\s*[:#-]?\s*([A-Za-z0-9._@-]{3,})\b",
        group=1,
    )
    transaction_id = first_match(
        issue_text,
        r"\b(?:txn_[A-Za-z0-9_-]+|cs_live_[A-Za-z0-9_-]+|pi_[A-Za-z0-9_-]+|ord(?:er)?[-_ ]?[A-Za-z0-9_-]{4,}|ref(?:erence)?\s*#?\s*[A-Za-z0-9_-]{4,})\b",
    )
    amount = first_amount(issue_text)
    return TicketFacts(
        raw_issue=issue_raw,
        subject=subject,
        company_field=company_field,
        issue_text=issue_text,
        user_text=user_text,
        assistant_text=assistant_text,
        language=language,
        pii_detected=pii_detected,
        injection_detected=injection_detected,
        pure_injection=pure_injection,
        company_guess=company_guess,
        multi_issue=multi_issue,
        verified_identity=verified_identity,
        email=email,
        phone=phone,
        user_identifier=user_identifier,
        transaction_id=transaction_id,
        amount=amount,
    )


def infer_company(text: str, company_field: str) -> str:
    normalized = normalize_text(text)
    scores = {
        "devplatform": count_keyword_hits(
            normalized,
            [
                "devplatform",
                "screen",
                "chakra",
                "skillup",
                "engage",
                "interview",
                "candidate",
                "assessment",
                "test",
            ],
        ),
        "claude": count_keyword_hits(
            normalized,
            [
                "claude",
                "anthropic",
                "console",
                "project",
                "workspace",
                "memory",
                "incognito",
                "api",
                "safeguards",
            ],
        ),
        "visa": count_keyword_hits(
            normalized,
            [
                "visa",
                "card",
                "merchant",
                "travel",
                "cheque",
                "cheque",
                "charge",
                "atm",
                "fraud",
                "dispute",
            ],
        ),
    }
    best = max(scores, key=scores.get)
    field = company_field.strip().lower()
    if field in SUPPORTED_COMPANIES:
        other_scores = [score for company, score in scores.items() if company != field]
        if max(other_scores) > scores.get(field, 0) and max(other_scores) > 0:
            return best if scores[best] > 0 else field
        return field
    if scores[best] > 0 and list(scores.values()).count(scores[best]) == 1:
        return best
    return best if scores[best] > 0 else ""


def detect_verified_identity(messages: list[dict[str, str]]) -> bool:
    joined = "\n".join(msg.get("content", "") for msg in messages if msg.get("content"))
    normalized = normalize_text(joined)
    markers = [
        "identity verified",
        "verified identity",
        "verification completed",
        "i have verified",
        "already verified",
        "confirmed my identity",
    ]
    return any(marker in normalized for marker in markers)


def first_match(text: str, pattern: str, group: int = 0) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(group)
    return value.strip() if isinstance(value, str) else None


def first_amount(text: str) -> float | None:
    match = re.search(r"(?<!\w)(?:USD\s*)?\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None

