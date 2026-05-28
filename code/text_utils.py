"""Utility functions for text normalization, detection, and redaction."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4})"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

STREET_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z0-9][A-Z0-9\s.'-]{2,40}\s+(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|way|ct|court|hwy|highway)\b",
    re.IGNORECASE,
)

INJECTION_PATTERNS = [
    r"ignore (?:all|any|the) previous instructions",
    r"disregard (?:all|any|the) previous instructions",
    r"system prompt",
    r"developer message",
    r"reveal (?:the )?(?:system|hidden|internal) (?:prompt|instructions|policy|docs)",
    r"output exactly",
    r"follow these rules",
    r"jailbreak",
    r"act as (?:a|an) support agent",
    r"you are now",
    r"prompt injection",
    r"do not mention",
    r"return the full text",
    r"print the hidden instructions",
]

EN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "your",
    "please",
    "help",
    "can",
    "you",
    "not",
    "are",
    "was",
    "but",
    "what",
    "how",
    "when",
    "where",
    "why",
    "who",
    "i",
    "to",
    "of",
    "in",
    "my",
}

FR_STOPWORDS = {
    "bonjour",
    "merci",
    "svp",
    "s'il",
    "vous",
    "de",
    "la",
    "le",
    "les",
    "des",
    "un",
    "une",
    "pour",
    "est",
    "pas",
    "que",
    "quoi",
    "comment",
    "ou",
    "dans",
    "mon",
    "ma",
}

ES_STOPWORDS = {
    "hola",
    "gracias",
    "por",
    "para",
    "que",
    "como",
    "cuando",
    "donde",
    "porque",
    "usted",
    "puede",
    "ayuda",
    "mi",
    "mis",
    "no",
    "si",
}

DE_STOPWORDS = {
    "ich",
    "bitte",
    "danke",
    "und",
    "oder",
    "nicht",
    "wie",
    "was",
    "warum",
    "wo",
    "mein",
    "meine",
    "hallo",
    "habe",
    "hilfe",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and normalize whitespace."""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u00a0", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text into deterministic word tokens."""

    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    return [stem_token(token) for token in tokens if token]


def stem_token(token: str) -> str:
    """Very small stemming step to improve recall without extra deps."""

    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def has_pii(text: str) -> bool:
    """Detect common PII patterns."""

    normalized = text or ""
    if EMAIL_RE.search(normalized):
        return True
    if SSN_RE.search(normalized):
        return True
    if IP_RE.search(normalized):
        return True
    if STREET_RE.search(normalized):
        return True
    if PHONE_RE.search(normalized):
        digits = re.sub(r"\D", "", PHONE_RE.search(normalized).group(0))
        if 10 <= len(digits) <= 15:
            return True
    for match in CARD_RE.finditer(normalized):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_check(digits):
            return True
    return False


def luhn_check(number: str) -> bool:
    """Return True if the number passes the Luhn checksum."""

    total = 0
    reverse_digits = list(reversed(number))
    for idx, digit in enumerate(reverse_digits):
        value = int(digit)
        if idx % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def redact_pii(text: str) -> str:
    """Redact common PII patterns from a string."""

    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = SSN_RE.sub("[REDACTED_SSN]", redacted)
    redacted = IP_RE.sub("[REDACTED_IP]", redacted)
    redacted = STREET_RE.sub("[REDACTED_ADDRESS]", redacted)

    def _card_repl(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_check(digits):
            return "[REDACTED_CARD]"
        return match.group(0)

    redacted = CARD_RE.sub(_card_repl, redacted)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    return redacted


def detect_language(text: str) -> str:
    """Detect a coarse ISO 639-1 language code."""

    if not text:
        return "en"

    normalized = normalize_text(text)
    tokens = set(tokenize(normalized))
    scores = {
        "en": len(tokens & EN_STOPWORDS),
        "fr": len(tokens & FR_STOPWORDS),
        "es": len(tokens & ES_STOPWORDS),
        "de": len(tokens & DE_STOPWORDS),
    }
    best_lang = max(scores, key=lambda lang: (scores[lang], lang == "en"))
    if scores[best_lang] == 0:
        return "en"
    return best_lang


def detect_injection(text: str) -> tuple[bool, bool]:
    """Return (injection_detected, pure_injection)."""

    lowered = normalize_text(text)
    matched = any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS)
    if not matched:
        return False, False

    support_signals = [
        "visa",
        "claude",
        "devplatform",
        "test",
        "account",
        "payment",
        "refund",
        "login",
        "password",
        "subscription",
        "card",
        "workspace",
        "interview",
        "candidate",
    ]
    has_support_context = any(signal in lowered for signal in support_signals)
    return True, not has_support_context


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    """Count keyword hits in normalized text."""

    haystack = normalize_text(text)
    return sum(1 for keyword in keywords if keyword in haystack)

