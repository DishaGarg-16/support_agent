"""Thin wrapper around the Groq SDK for LLM-powered response generation.

Provides a single `generate()` function. If the API key is missing or the
call fails for any reason, every function returns None so the caller can
fall back to the deterministic path with zero downtime.
"""

from __future__ import annotations

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK import — optional so the rest of the pipeline works without it
# ---------------------------------------------------------------------------
try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module-level client (lazy singleton)
# ---------------------------------------------------------------------------
_client: Optional["Groq"] = None
_initialised = False

MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3
SEED = 42
MAX_TOKENS = 512


def _get_client() -> Optional["Groq"]:
    """Return a Groq client if a key is available, else None."""
    global _client, _initialised
    if _initialised:
        return _client
    _initialised = True
    if Groq is None:
        logger.info("groq SDK not installed — LLM features disabled")
        return None
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        logger.info("GROQ_API_KEY not set — LLM features disabled, using deterministic fallback")
        return None
    try:
        _client = Groq(api_key=key)
    except Exception as exc:
        logger.warning("Failed to initialise Groq client: %s", exc)
        _client = None
    return _client


def is_available() -> bool:
    """Check whether the LLM backend is usable."""
    return _get_client() is not None


def generate(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Optional[str]:
    """Call the LLM and return the assistant message, or None on failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=SEED,
        )
        text = response.choices[0].message.content
        return text.strip() if text else None
    except Exception as exc:
        logger.warning("Groq API call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Output guardrail — validate LLM response before returning to user
# ---------------------------------------------------------------------------
_LEAKED_PATTERNS = [
    r"system prompt",
    r"you are a",
    r"internal instruction",
    r"corpus content",
    r"hidden doc",
    r"\bBM25\b",
    r"retrieval pipeline",
]

_HALLUCINATED_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_HALLUCINATED_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# Allowed domains that the LLM may reasonably cite
_ALLOWED_DOMAINS = {
    "anthropic.com", "claude.ai", "console.anthropic.com",
    "visa.com", "usa.visa.com",
    "hackerrank.com", "support.hackerrank.com",
    "help.hackerrank.com",
}


def validate_llm_response(text: str) -> bool:
    """Return True if the LLM response passes output guardrails."""
    if not text or len(text) < 15:
        return False
    if len(text) > 3000:
        return False
    lower = text.lower()
    # Check for leaked system prompt / internal details
    for pat in _LEAKED_PATTERNS:
        if re.search(pat, lower):
            return False
    # Check for hallucinated URLs (allow known support domains)
    for url_match in _HALLUCINATED_URL.finditer(text):
        url = url_match.group(0).rstrip(".,;:)")
        if not any(domain in url for domain in _ALLOWED_DOMAINS):
            return False
    # Check for hallucinated emails (only allow known support emails)
    for email_match in _HALLUCINATED_EMAIL.finditer(text):
        email = email_match.group(0).lower()
        if email not in {"help@hackerrank.com", "support@anthropic.com", "support@visa.com"}:
            return False
    return True
