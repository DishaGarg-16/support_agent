"""Async wrapper around the Groq SDK for LLM-powered response generation.

Provides a single async `generate()` function. If the API key is missing or
the call fails for any reason, every function returns None so the caller can
fall back to the deterministic path with zero downtime.

Rate-limiting: an asyncio.Semaphore caps concurrent in-flight requests to
avoid hitting Groq's free-tier RPM limit.
"""

from __future__ import annotations

import asyncio
import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK import — optional so the rest of the pipeline works without it
# ---------------------------------------------------------------------------
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module-level client (lazy singleton)
# ---------------------------------------------------------------------------
_client: Optional["AsyncGroq"] = None
_initialised = False

MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"  # Higher free-tier rate limits — used when primary hits daily/TPM cap
TEMPERATURE = 0.3
SEED = 42
MAX_TOKENS = 512

# Rate-limit guard — max concurrent LLM requests
_MAX_CONCURRENT = 20  # Allows enough concurrency for fast execution
_semaphore: Optional[asyncio.Semaphore] = None

# Retry config for 429 / transient errors
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds


def _get_semaphore() -> asyncio.Semaphore:
    """Return (and lazily create) the per-event-loop semaphore."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _semaphore


def _get_client() -> Optional["AsyncGroq"]:
    """Return an AsyncGroq client if a key is available, else None."""
    global _client, _initialised
    if _initialised:
        return _client
    _initialised = True
    if AsyncGroq is None:
        logger.info("groq SDK not installed — LLM features disabled")
        return None
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        logger.info("GROQ_API_KEY not set — LLM features disabled, using deterministic fallback")
        return None
    try:
        _client = AsyncGroq(api_key=key)
    except Exception as exc:
        logger.warning("Failed to initialise AsyncGroq client: %s", exc)
        _client = None
    return _client


def is_available() -> bool:
    """Check whether the LLM backend is usable."""
    return _get_client() is not None


async def generate(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Optional[str]:
    """Call the LLM and return the assistant message, or None on failure.

    Respects the concurrency semaphore and retries on transient / rate-limit
    errors with exponential backoff. If all retries on the primary model fail
    due to rate/token limits, automatically falls back to FALLBACK_MODEL.
    """
    client = _get_client()
    if client is None:
        return None

    sem = _get_semaphore()

    async def _call(model: str) -> Optional[str]:
        """Attempt a single call to the given model with retry logic."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with sem:
                    response = await client.chat.completions.create(
                        model=model,
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
                exc_str = str(exc).lower()
                is_rate_limit = "429" in exc_str or "rate" in exc_str or "token" in exc_str
                is_retryable = is_rate_limit or "timeout" in exc_str
                if is_retryable and attempt < _MAX_RETRIES:
                    wait = _BASE_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(
                        "Groq [%s] retryable error (attempt %d/%d): %s — retrying in %.1fs",
                        model, attempt, _MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    if is_rate_limit:
                        logger.warning(
                            "Groq [%s] rate/token limit exhausted (attempt %d/%d): %s",
                            model, attempt, _MAX_RETRIES, exc,
                        )
                    else:
                        logger.warning(
                            "Groq [%s] call failed (attempt %d/%d): %s",
                            model, attempt, _MAX_RETRIES, exc,
                        )
                    return None
        return None

    # Try primary model first
    result = await _call(MODEL)
    if result is not None:
        return result

    # Primary model failed — try fallback model (higher rate limits)
    if FALLBACK_MODEL != MODEL:
        logger.info("Primary model failed — falling back to %s", FALLBACK_MODEL)
        result = await _call(FALLBACK_MODEL)
        if result is not None:
            return result

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
