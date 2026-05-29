"""Company routing, retrieval, and product-area inference."""

from __future__ import annotations

from pathlib import Path

from corpus import CorpusDoc, CorpusIndex, extract_answer_excerpt
import llm_client
from models import TicketFacts
from text_utils import normalize_text


# ---------------------------------------------------------------------------
# LLM routing prompt — used only for ambiguous edge cases
# ---------------------------------------------------------------------------
_ROUTE_SYSTEM_PROMPT = """\
You are a support ticket classifier. Given a support ticket, determine which 
company the ticket is about.

Respond with EXACTLY ONE of these words and nothing else:
- devplatform
- claude
- visa

If unsure, pick the most likely one.
"""


def _llm_route_fallback(facts: TicketFacts) -> str | None:
    """Ask the LLM to classify the company when BM25 is uncertain."""
    if not llm_client.is_available():
        return None
    user_msg = (
        f"Subject: {facts.subject}\n"
        f"Company field: {facts.company_field}\n"
        f"Content: {facts.user_text[:500]}"
    )
    raw = llm_client.generate(
        _ROUTE_SYSTEM_PROMPT,
        user_msg,
        temperature=0.0,
        max_tokens=10,
    )
    if not raw:
        return None
    answer = raw.strip().lower().split()[0] if raw.strip() else None
    if answer in {"devplatform", "claude", "visa"}:
        return answer
    return None


def choose_company_hint(facts: TicketFacts) -> str:
    if facts.company_guess in {"devplatform", "claude", "visa"}:
        return facts.company_guess
    field = facts.company_field.strip().lower()
    if field in {"devplatform", "claude", "visa"}:
        return field
    return ""


def find_doc(corpus: CorpusIndex, path: str) -> CorpusDoc | None:
    for doc in corpus.docs:
        if doc.path == path:
            return doc
    return None


def route_and_retrieve(
    corpus: CorpusIndex,
    facts: TicketFacts,
    company_hint: str,
    request_type: str,
    risk_level: str,
    *,
    skip_retrieval: bool = False,
) -> tuple[str, list[str], list[str]]:
    if skip_retrieval:
        return "", [], []

    if request_type == "invalid" and (facts.pure_injection or not facts.multi_issue):
        return "", [], []

    direct = direct_route(corpus, facts, company_hint)
    if direct:
        return direct

    if facts.multi_issue and mentions_multiple_companies(facts.issue_text):
        return "", [], []

    company_filter = company_hint or ""
    query = build_retrieval_query(facts, company_filter)
    docs = corpus.search(query, company_hint=company_filter or None, top_k=5)
    if not docs and company_filter:
        docs = corpus.search(query, company_hint=None, top_k=5)

    # --- LLM edge-case routing ---
    # If BM25 returned nothing or very low confidence, ask the LLM to help
    # disambiguate the company/product area so we can retry with a better hint.
    if not docs or docs[0][0] < 2.0:
        llm_hint = _llm_route_fallback(facts)
        if llm_hint and llm_hint != company_filter:
            retry_query = build_retrieval_query(facts, llm_hint)
            retry_docs = corpus.search(retry_query, company_hint=llm_hint, top_k=5)
            if retry_docs and (not docs or retry_docs[0][0] > docs[0][0]):
                docs = retry_docs
                company_filter = llm_hint

    if not docs:
        return "", [], []

    best_score = docs[0][0]
    second_score = docs[1][0] if len(docs) > 1 else 0.0
    top_docs = [doc for score, doc in docs if score >= max(1.0, best_score * 0.55)]
    if len(top_docs) > 2:
        top_docs = top_docs[:2]
    if facts.multi_issue and second_score >= best_score * 0.75:
        top_docs = top_docs[:2]
    source_docs = [doc.path for doc in top_docs]
    snippets = [extract_answer_excerpt(doc, query) for doc in top_docs]
    product_area = infer_product_area(facts, top_docs[0], query, request_type)
    return product_area, source_docs, snippets


def direct_route(corpus: CorpusIndex, facts: TicketFacts, company_filter: str) -> tuple[str, list[str], list[str]] | None:
    text = normalize_text(facts.issue_text + " " + facts.subject)
    if company_filter == "visa" and any(term in text for term in ["dispute", "merchant", "wrong product", "refund", "charge", "unauthorized"]):
        doc = find_doc(corpus, "data/visa/dispute-resolution-updated-2026.md")
        if doc:
            return "general_support", [doc.path], [extract_answer_excerpt(doc, "dispute charge merchant refund zero liability")]
    if company_filter == "visa" and any(term in text for term in ["lost", "stolen", "blocked", "traveling", "travel"]):
        doc = find_doc(corpus, "data/visa/support.md")
        if doc:
            return "general_support", [doc.path], [extract_answer_excerpt(doc, "lost stolen card travel support")]
    if company_filter == "claude" and any(term in text for term in ["seat", "workspace", "admin", "member", "organization", "owner"]):
        doc = find_doc(corpus, "data/claude/team-and-enterprise-plans/admin-management/13133750-manage-members-on-team-and-enterprise-plans.md")
        if doc:
            return "account_management", [doc.path], [extract_answer_excerpt(doc, "manage members seat organization admin workspace")]
    if company_filter == "devplatform" and any(term in text for term in ["mock interview", "refund"]):
        doc = find_doc(corpus, "data/devplatform/hackerrank_community/subscriptions-payments-and-billing/3282259518-purchase-mock-interviews.md")
        if doc:
            snippet = (
                "DevPlatform Community lets you purchase mock interview credits for practice, and those credits do not expire. "
                "If you accidentally make a purchase or are not satisfied with your mock interview, contact help@devplatform.com. "
                "The support team will promptly review your request."
            )
            return "screen", [doc.path], [snippet]
    if company_filter == "devplatform" and any(term in text for term in ["payment", "billing", "order id"]):
        doc = find_doc(corpus, "data/devplatform/hackerrank_community/subscriptions-payments-and-billing/9157064719-payments-and-billing-faqs.md")
        if doc:
            return "screen", [doc.path], [extract_answer_excerpt(doc, "mock interview payment refund")]
    return None


def build_retrieval_query(facts: TicketFacts, company_filter: str) -> str:
    text = normalize_text(facts.issue_text + " " + facts.subject)
    parts = [facts.subject, facts.user_text, facts.assistant_text, company_filter]
    if company_filter == "claude":
        if any(term in text for term in ["access", "seat", "workspace", "team", "admin", "member", "organization", "owner"]):
            parts.extend(
                [
                    "manage members on team and enterprise plans",
                    "roles and permissions",
                    "find and join a team or enterprise organization",
                    "configuring session security settings",
                    "team and enterprise plans",
                ]
            )
        if any(term in text for term in ["delete", "export", "privacy", "data", "conversation"]):
            parts.extend(["privacy", "conversation management", "delete conversation"])
        if any(term in text for term in ["login", "password", "session"]):
            parts.extend(["logging in", "active sessions", "security settings"])
        if any(term in text for term in ["api", "console", "bedrock", "pricing", "usage"]):
            parts.extend(["api and console", "usage and limits", "pricing and billing"])
        if any(term in text for term in ["web search", "project", "skill", "artifact", "plugin", "research"]):
            parts.extend(["features and capabilities", "projects", "skills"])
        if any(term in text for term in ["policy", "privacy", "hipaa", "gdpr", "law", "security", "prompt injection"]):
            parts.extend(["privacy and legal", "safeguards"])
    elif company_filter == "devplatform":
        if any(term in text for term in ["score dispute", "review my answers", "increase my score", "move me to the next round", "rejected"]):
            parts.extend(["viewing candidate test report", "scorecard", "candidate status", "interview reports"])
        if any(term in text for term in ["test", "assessment", "submission", "candidate", "interview", "score", "apply tab"]):
            parts.extend(["screen", "candidate", "test", "interview", "submission"])
        if any(term in text for term in ["mock interview", "refund", "payment", "billing", "order id"]):
            parts.extend(["purchase mock interview credits", "payments and billing faqs", "refund policy", "mock interview credits"])
        if any(term in text for term in ["login", "password", "account", "email", "team", "user", "billing", "subscription"]):
            parts.extend(["settings", "user account", "teams management", "billing"])
        if any(term in text for term in ["api", "integration", "ats", "sso", "scim", "webhook", "workspace"]):
            parts.extend(["integrations", "api", "sso", "scim"])
        if any(term in text for term in ["skillup", "certif", "employee", "manager", "learn"]):
            parts.extend(["skillup", "learn", "certifications"])
        if any(term in text for term in ["engage", "event", "campaign", "microsite"]):
            parts.extend(["engage", "event"])
    elif company_filter == "visa":
        if any(term in text for term in ["dispute", "wrong product", "charge", "merchant", "refund", "unauthorized"]):
            parts.extend(["how do i dispute a charge", "dispute resolution", "zero liability", "consumer support"])
        if any(term in text for term in ["card", "stolen", "lost", "blocked", "fraud"]):
            parts.extend(["lost or stolen card", "travel support", "fraud prevention", "consumer support"])
        if any(term in text for term in ["travel", "atm", "exchange", "cheque", "traveller", "travelers"]):
            parts.extend(["travel support", "travelers cheques"])
        if any(term in text for term in ["merchant", "small business", "accept", "interchange"]):
            parts.extend(["small business", "merchant"])
    return " ".join(part for part in parts if part)


def mentions_multiple_companies(text: str) -> bool:
    normalized = normalize_text(text)
    hits = sum(1 for company in {"devplatform", "claude", "visa"} if company in normalized)
    return hits > 1


def infer_product_area(facts: TicketFacts, doc: CorpusDoc | None, query: str, request_type: str) -> str:
    text = normalize_text(facts.issue_text + " " + facts.subject)
    if not doc:
        return product_area_from_text(facts.company_guess, text, query)

    if doc.company == "devplatform":
        return devplatform_area(text, doc)
    if doc.company == "claude":
        return claude_area(text, doc)
    if doc.company == "visa":
        return visa_area(text, doc)
    return product_area_from_text(facts.company_guess, text, query)


def product_area_from_text(company: str, text: str, query: str) -> str:
    if company == "devplatform":
        return devplatform_area(text, None)
    if company == "claude":
        return claude_area(text, None)
    if company == "visa":
        return visa_area(text, None)
    return ""


def devplatform_area(text: str, doc: CorpusDoc | None) -> str:
    if any(term in text for term in ["interview", "candidate", "proctor", "test", "assessment", "submission", "score", "apply tab"]):
        if "interview" in text and "mock" not in text:
            return "interview"
        return "screen"
    if any(term in text for term in ["subscription", "billing", "payment", "refund", "pause", "cancel"]):
        return "settings"
    if any(term in text for term in ["api", "integrat", "ats", "sso", "scim", "webhook", "workspace"]):
        return "integrations"
    if any(term in text for term in ["skillup", "certif", "employee", "manager", "learn"]):
        return "skillup"
    if any(term in text for term in ["engage", "event", "campaign"]):
        return "engage"
    if any(term in text for term in ["account", "password", "login", "security", "gdpr", "bias", "privacy"]):
        return "settings"
    return doc.category if doc else "general_help"


def claude_area(text: str, doc: CorpusDoc | None) -> str:
    if any(term in text for term in ["delete conversation", "rename conversation", "sharing", "incognito", "memory"]):
        return "conversation_management"
    if any(term in text for term in ["delete account", "export data", "personal data", "privacy", "who can view", "sensitive data", "baa", "hipaa", "gdpr"]):
        return "privacy"
    if any(term in text for term in ["seat", "workspace", "team", "organization", "owner", "admin", "member", "discoverable", "invite link", "join organization"]):
        return "account_management"
    if any(term in text for term in ["api", "console", "workspace", "bedrock", "pricing", "usage bundle", "key", "token"]):
        return "api_and_console"
    if any(term in text for term in ["login", "log in", "password", "active sessions", "session", "security settings"]):
        return "account_management"
    if any(term in text for term in ["usage", "limit", "plan", "subscription", "pro", "max", "team", "enterprise"]):
        return "usage_and_limits"
    if any(term in text for term in ["web search", "projects", "skills", "artifacts", "research", "plugins", "visual", "interactive", "cowork"]):
        return "features_and_capabilities"
    if any(term in text for term in ["error", "failing", "not responding", "troubleshoot", "blocked", "misleading"]):
        return "troubleshooting"
    if any(term in text for term in ["policy", "safety", "vulnerability", "prompt injection", "harm", "law enforcement"]):
        return "safeguards"
    return doc.category if doc else "general"


def visa_area(text: str, doc: CorpusDoc | None) -> str:
    if any(term in text for term in ["travel", "cheque", "traveler", "traveller", "exchange rate", "atm", "concierge", "china"]):
        return "travel_support"
    if any(term in text for term in ["merchant", "accept", "small business", "interchange", "rules", "regulations", "fraud prevention", "dispute resolution"]):
        return "small_business"
    return "general_support"
