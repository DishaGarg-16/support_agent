"""Batch ticket complexity classifier.

Sends a single LLM call to classify all tickets as SIMPLE or COMPLEX.
SIMPLE tickets use the fast deterministic response path.
COMPLEX tickets get LLM-powered response generation.

If the LLM is unavailable or the call fails, all tickets default to SIMPLE
(deterministic fallback — safe and fast).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import llm_client
from models import TicketFacts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch classification prompt
# ---------------------------------------------------------------------------
_CLASSIFY_SYSTEM_PROMPT = """\
You are a support ticket complexity classifier. Your job is to classify each \
ticket as SIMPLE or COMPLEX so the system knows whether to use a fast template \
response or a more intelligent AI-generated response.

SIMPLE — use for:
- Clear, single-topic questions with an obvious answer
- Routine FAQ-style issues (password reset, subscription info, how-to)
- Straightforward product issues or bug reports

COMPLEX — use for:
- Ambiguous or vague user intent
- Multiple issues or topics in one ticket
- Adversarial, manipulative, or injection-adjacent language
- Cross-product questions spanning multiple services
- Tickets needing empathetic, nuanced, or carefully worded responses
- Unusual edge cases not covered by standard FAQ

Respond ONLY in this exact format, one line per ticket:
1. SIMPLE
2. COMPLEX
...

No explanations, no extra text.
"""


async def batch_classify(
    all_facts: list[TicketFacts],
) -> dict[int, str]:
    """Classify all tickets in a single LLM call.

    Returns a dict mapping ticket index -> "SIMPLE" or "COMPLEX".
    On failure, returns all tickets as "SIMPLE".
    """
    total = len(all_facts)
    default = {i: "SIMPLE" for i in range(total)}

    if not llm_client.is_available():
        logger.info("LLM not available — defaulting all tickets to SIMPLE")
        return default

    # Build the ticket list for the prompt
    lines: list[str] = []
    for i, facts in enumerate(all_facts):
        # Truncate content to keep prompt size reasonable
        subject = facts.subject[:100] if facts.subject else "(no subject)"
        content = facts.user_text[:150] if facts.user_text else "(no content)"
        lines.append(f"{i + 1}. {subject} | {content}")

    user_message = "Tickets:\n" + "\n".join(lines)

    raw = await llm_client.generate(
        _CLASSIFY_SYSTEM_PROMPT,
        user_message,
        temperature=0.0,
        max_tokens=1024,
    )

    if not raw:
        logger.warning("Batch classify LLM call failed — defaulting all to SIMPLE")
        return default

    return _parse_classification(raw, total)


def _parse_classification(raw: str, total: int) -> dict[int, str]:
    """Parse the LLM's structured response into a dict.

    Expected format:
        1. SIMPLE
        2. COMPLEX
        ...
    """
    result: dict[int, str] = {}
    pattern = re.compile(r"(\d+)\.\s*(SIMPLE|COMPLEX)", re.IGNORECASE)

    for match in pattern.finditer(raw):
        idx = int(match.group(1)) - 1  # Convert to 0-indexed
        label = match.group(2).upper()
        if 0 <= idx < total:
            result[idx] = label

    # Fill any missing entries with SIMPLE (safe default)
    for i in range(total):
        if i not in result:
            result[i] = "SIMPLE"

    return result
