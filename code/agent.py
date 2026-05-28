"""Deterministic support triage agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import re
from pathlib import Path
from typing import Any

from corpus import CorpusDoc, CorpusIndex, extract_answer_excerpt
from text_utils import (
    count_keyword_hits,
    detect_injection,
    detect_language,
    has_pii,
    normalize_text,
)


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


class SupportTriageAgent:
    """End-to-end support triage pipeline."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.corpus = CorpusIndex.load(repo_root)

    def process_csv(self, input_path: Path, output_path: Path) -> None:
        rows = self._read_input_rows(input_path)
        results = [self.triage_row(row) for row in rows]
        self._write_output_rows(results, output_path)

    def triage_row(self, row: dict[str, str]) -> dict[str, str]:
        facts = self._extract_facts(row)
        request_type = self._classify_request_type(facts)
        company_hint = self._choose_company_hint(facts)
        risk_level = self._classify_risk(facts, request_type)
        needs_verification = self._needs_verification(facts, request_type, normalize_text(facts.issue_text))
        high_risk_escalation = self._should_escalate_high_risk(facts, request_type, normalize_text(facts.issue_text))
        product_area, source_docs, retrieved_snippets = self._route_and_retrieve(
            facts,
            company_hint,
            request_type,
            risk_level,
            skip_retrieval=needs_verification
            or (risk_level in {"critical", "high"} and high_risk_escalation)
            or facts.pure_injection,
        )
        status, actions_taken, response = self._decide_status_and_response(
            facts,
            request_type,
            risk_level,
            product_area,
            source_docs,
            retrieved_snippets,
            needs_verification=needs_verification,
            high_risk_escalation=high_risk_escalation,
        )
        justification = self._build_justification(
            facts,
            request_type,
            risk_level,
            status,
            source_docs,
            product_area,
        )
        confidence = self._compute_confidence(
            facts,
            request_type,
            risk_level,
            status,
            source_docs,
            retrieved_snippets,
        )

        result = TriageResult(
            status=status,
            product_area=product_area,
            response=response,
            justification=justification,
            request_type=request_type,
            confidence_score=confidence,
            source_documents=source_docs,
            risk_level=risk_level,
            pii_detected=facts.pii_detected,
            language=facts.language,
            actions_taken=actions_taken,
        )
        return self._result_to_row(row, result)

    def _read_input_rows(self, input_path: Path) -> list[dict[str, str]]:
        with input_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows

    def _write_output_rows(self, results: list[dict[str, str]], output_path: Path) -> None:
        fieldnames = [
            "issue",
            "subject",
            "company",
            "response",
            "product_area",
            "status",
            "request_type",
            "justification",
            "confidence_score",
            "source_documents",
            "risk_level",
            "pii_detected",
            "language",
            "actions_taken",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    def _result_to_row(self, input_row: dict[str, str], result: TriageResult) -> dict[str, str]:
        issue = input_row.get("Issue", input_row.get("issue", "")).strip()
        subject = input_row.get("Subject", input_row.get("subject", "")).strip()
        company = input_row.get("Company", input_row.get("company", "")).strip()
        return {
            "issue": issue,
            "subject": subject,
            "company": company,
            "response": result.response.strip(),
            "product_area": result.product_area,
            "status": result.status,
            "request_type": result.request_type,
            "justification": result.justification.strip(),
            "confidence_score": f"{result.confidence_score:.2f}",
            "source_documents": "|".join(result.source_documents),
            "risk_level": result.risk_level,
            "pii_detected": "true" if result.pii_detected else "false",
            "language": result.language,
            "actions_taken": json.dumps(result.actions_taken, ensure_ascii=False, separators=(",", ":")),
        }

    def _extract_facts(self, row: dict[str, str]) -> TicketFacts:
        issue_raw = row.get("Issue", row.get("issue", "")) or "[]"
        subject = row.get("Subject", row.get("subject", "")) or ""
        company_field = (row.get("Company", row.get("company", "")) or "").strip()
        messages = self._parse_issue(issue_raw)
        user_text_parts = [msg["content"] for msg in messages if msg.get("role") == "user" and msg.get("content")]
        assistant_text_parts = [msg["content"] for msg in messages if msg.get("role") != "user" and msg.get("content")]
        user_text = "\n".join(user_text_parts).strip()
        assistant_text = "\n".join(assistant_text_parts).strip()
        issue_text = "\n".join(part for part in [subject, user_text, assistant_text] if part).strip()
        language = detect_language(issue_text)
        pii_detected = has_pii(issue_text)
        injection_detected, pure_injection = detect_injection(issue_text)
        company_guess = self._infer_company(issue_text, company_field)
        multi_issue = any(phrase in normalize_text(issue_text) for phrase in MULTI_ISSUE_PHRASES)
        verified_identity = self._detect_verified_identity(messages)
        email = self._first_match(issue_text, r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
        phone = self._first_match(issue_text, r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4})")
        user_identifier = self._first_match(
            issue_text,
            r"\b(?:user(?:name)?|account(?: id)?|customer|member|workspace|org(?:anization)?|team)\s*[:#-]?\s*([A-Za-z0-9._@-]{3,})\b",
            group=1,
        )
        transaction_id = self._first_match(
            issue_text,
            r"\b(?:txn_[A-Za-z0-9_-]+|cs_live_[A-Za-z0-9_-]+|pi_[A-Za-z0-9_-]+|ord(?:er)?[-_ ]?[A-Za-z0-9_-]{4,}|ref(?:erence)?\s*#?\s*[A-Za-z0-9_-]{4,})\b",
        )
        amount = self._first_amount(issue_text)
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

    def _parse_issue(self, raw_issue: str) -> list[dict[str, str]]:
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

    def _infer_company(self, text: str, company_field: str) -> str:
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

    def _choose_company_hint(self, facts: TicketFacts) -> str:
        """Choose the safest company hint for retrieval."""

        if facts.company_guess in SUPPORTED_COMPANIES:
            return facts.company_guess
        field = facts.company_field.strip().lower()
        if field in SUPPORTED_COMPANIES:
            return field
        return ""

    def _detect_verified_identity(self, messages: list[dict[str, str]]) -> bool:
        joined = "\n".join(
            msg.get("content", "")
            for msg in messages
            if msg.get("content")
        )
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

    def _first_match(self, text: str, pattern: str, group: int = 0) -> str | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        value = match.group(group)
        return value.strip() if isinstance(value, str) else None

    def _first_amount(self, text: str) -> float | None:
        match = re.search(r"(?<!\w)(?:USD\s*)?\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", text)
        if not match:
            return None
        raw = match.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            return None

    def _classify_request_type(self, facts: TicketFacts) -> str:
        text = normalize_text(facts.issue_text)
        if facts.pure_injection:
            return "invalid"
        if not self._looks_like_support_request(text):
            return "invalid"
        feature_signals = [
            "feature",
            "add support",
            "could you add",
            "request",
            "enhancement",
            "wish",
            "would love",
            "can you support",
            "new capability",
        ]
        bug_signals = [
            "bug",
            "broken",
            "not working",
            "stopped",
            "fails",
            "failing",
            "error",
            "crash",
            "down",
            "issue",
            "cannot",
            "can't",
            "unable",
        ]
        if any(signal in text for signal in feature_signals):
            return "feature_request"
        if any(signal in text for signal in bug_signals):
            return "bug"
        return "product_issue"

    def _looks_like_support_request(self, text: str) -> bool:
        support_terms = [
            "help",
            "support",
            "issue",
            "problem",
            "bug",
            "error",
            "login",
            "account",
            "card",
            "test",
            "workspace",
            "refund",
            "subscription",
            "interview",
            "payment",
            "visa",
            "claude",
            "devplatform",
            "candidate",
            "password",
        ]
        return any(term in text for term in support_terms)

    def _classify_risk(self, facts: TicketFacts, request_type: str) -> str:
        text = normalize_text(facts.issue_text)
        critical_signals = [
            "identity theft",
            "account takeover",
            "unauthorized charges",
            "fraud",
            "data leak",
            "data exfil",
            "security vulnerability",
            "legal threat",
            "lawsuit",
            "system prompt",
            "prompt injection",
        ]
        high_signals = [
            "refund",
            "payment",
            "billing",
            "card blocked",
            "card stolen",
            "lost card",
            "dispute",
            "account hacked",
            "compromised",
            "security",
            "hipaa",
            "gdpr",
            "delete data",
            "delete account",
            "login",
            "password",
            "api key",
            "secrets",
            "internal docs",
        ]
        medium_signals = [
            "test",
            "assessment",
            "score",
            "interview",
            "subscription",
            "pause",
            "cancel",
            "upgrade",
            "downgrade",
            "resume",
            "certificate",
        ]
        if facts.pii_detected and any(sig in text for sig in ("card", "payment", "account", "identity")):
            return "high"
        if any(sig in text for sig in critical_signals):
            return "critical"
        if any(sig in text for sig in high_signals):
            return "high"
        if any(sig in text for sig in medium_signals):
            return "medium"
        if request_type == "invalid":
            return "low"
        return "low"

    def _route_and_retrieve(
        self,
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

        direct = self._direct_route(facts, company_hint)
        if direct:
            return direct

        if facts.multi_issue and self._mentions_multiple_companies(facts.issue_text):
            return "", [], []

        company_filter = company_hint or ""
        query = self._build_retrieval_query(facts, company_filter)
        docs = self.corpus.search(query, company_hint=company_filter or None, top_k=5)
        if not docs and company_filter:
            docs = self.corpus.search(query, company_hint=None, top_k=5)

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
        product_area = self._infer_product_area(facts, top_docs[0], query, request_type)
        return product_area, source_docs, snippets

    def _direct_route(self, facts: TicketFacts, company_filter: str) -> tuple[str, list[str], list[str]] | None:
        text = normalize_text(facts.issue_text + " " + facts.subject)
        if company_filter == "visa" and any(term in text for term in ["dispute", "merchant", "wrong product", "refund", "charge", "unauthorized"]):
            doc = self._find_doc("data/visa/dispute-resolution-updated-2026.md")
            if doc:
                return "general_support", [doc.path], [extract_answer_excerpt(doc, "dispute charge merchant refund zero liability")]
        if company_filter == "visa" and any(term in text for term in ["lost", "stolen", "blocked", "traveling", "travel"]):
            doc = self._find_doc("data/visa/support.md")
            if doc:
                return "general_support", [doc.path], [extract_answer_excerpt(doc, "lost stolen card travel support")]
        if company_filter == "claude" and any(term in text for term in ["seat", "workspace", "admin", "member", "organization", "owner"]):
            doc = self._find_doc("data/claude/team-and-enterprise-plans/admin-management/13133750-manage-members-on-team-and-enterprise-plans.md")
            if doc:
                return "account_management", [doc.path], [extract_answer_excerpt(doc, "manage members seat organization admin workspace")]
        if company_filter == "devplatform" and any(term in text for term in ["mock interview", "refund"]):
            doc = self._find_doc("data/devplatform/hackerrank_community/subscriptions-payments-and-billing/3282259518-purchase-mock-interviews.md")
            if doc:
                snippet = (
                    "DevPlatform Community lets you purchase mock interview credits for practice, and those credits do not expire. "
                    "If you accidentally make a purchase or are not satisfied with your mock interview, contact help@devplatform.com. "
                    "The support team will promptly review your request."
                )
                return "screen", [doc.path], [snippet]
        if company_filter == "devplatform" and any(term in text for term in ["payment", "billing", "order id"]):
            doc = self._find_doc("data/devplatform/hackerrank_community/subscriptions-payments-and-billing/9157064719-payments-and-billing-faqs.md")
            if doc:
                return "screen", [doc.path], [extract_answer_excerpt(doc, "mock interview payment refund")]
        return None

    def _find_doc(self, path: str) -> CorpusDoc | None:
        for doc in self.corpus.docs:
            if doc.path == path:
                return doc
        return None

    def _build_retrieval_query(self, facts: TicketFacts, company_filter: str) -> str:
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

    def _mentions_multiple_companies(self, text: str) -> bool:
        normalized = normalize_text(text)
        hits = sum(1 for company in SUPPORTED_COMPANIES if company in normalized)
        return hits > 1

    def _infer_product_area(
        self,
        facts: TicketFacts,
        doc: CorpusDoc | None,
        query: str,
        request_type: str,
    ) -> str:
        text = normalize_text(facts.issue_text + " " + facts.subject)
        if not doc:
            return self._product_area_from_text(facts.company_guess, text, query)

        if doc.company == "devplatform":
            return self._devplatform_area(text, doc)
        if doc.company == "claude":
            return self._claude_area(text, doc)
        if doc.company == "visa":
            return self._visa_area(text, doc)
        return self._product_area_from_text(facts.company_guess, text, query)

    def _product_area_from_text(self, company: str, text: str, query: str) -> str:
        if company == "devplatform":
            return self._devplatform_area(text, None)
        if company == "claude":
            return self._claude_area(text, None)
        if company == "visa":
            return self._visa_area(text, None)
        return ""

    def _devplatform_area(self, text: str, doc: CorpusDoc | None) -> str:
        if any(term in text for term in ["interview", "candidate", "proctor", "test", "assessment", "submission", "score", "apply tab"]):
            if "interview" in text and "mock" not in text:
                return "interview"
            return "screen"
        if any(term in text for term in ["subscription", "billing", "payment", "refund", "pause", "cancel"]):
            return "settings"
        if any(term in text for term in ["api", "integrat", "ats", "webhook", "sso", "scim", "oauth"]):
            return "integrations"
        if any(term in text for term in ["skillup", "certif", "employee", "manager", "learn"]):
            return "skillup"
        if any(term in text for term in ["engage", "event", "campaign"]):
            return "engage"
        if any(term in text for term in ["account", "password", "login", "security", "gdpr", "bias", "privacy"]):
            return "settings"
        return doc.category if doc else "general_help"

    def _claude_area(self, text: str, doc: CorpusDoc | None) -> str:
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

    def _visa_area(self, text: str, doc: CorpusDoc | None) -> str:
        if any(term in text for term in ["travel", "cheque", "traveler", "traveller", "exchange rate", "atm", "concierge", "china"]):
            return "travel_support"
        if any(term in text for term in ["merchant", "accept", "small business", "interchange", "rules", "regulations", "fraud prevention", "dispute resolution"]):
            return "small_business"
        return "general_support"

    def _decide_status_and_response(
        self,
        facts: TicketFacts,
        request_type: str,
        risk_level: str,
        product_area: str,
        source_docs: list[str],
        snippets: list[str],
        *,
        needs_verification: bool,
        high_risk_escalation: bool,
    ) -> tuple[str, list[dict[str, Any]], str]:
        actions: list[dict[str, Any]] = []
        text = normalize_text(facts.issue_text)
        company = facts.company_guess or facts.company_field.lower()

        if facts.pure_injection or self._is_content_exfiltration_request(text):
            actions.append(
                self._escalate_action(
                    priority="urgent" if risk_level in {"critical", "high"} else "high",
                    department="security",
                    summary="Potential prompt injection or request to reveal internal instructions/corpus content.",
                )
            )
            return "escalated", actions, self._refusal_response(facts, reason="I cannot help with requests to reveal internal instructions or hidden support content.")

        if risk_level in {"critical", "high"} and high_risk_escalation:
            department = self._department_for_text(text, company)
            actions.append(
                self._escalate_action(
                    priority="urgent" if risk_level == "critical" else "high",
                    department=department,
                    summary=self._escalation_summary(text, company, risk_level),
                )
            )
            return "escalated", actions, self._escalation_response(facts, risk_level, company)

        if needs_verification:
            verify_action = self._verification_action(facts)
            if verify_action:
                actions.append(verify_action)
            return "replied", actions, self._verification_response(facts, request_type)

        if not source_docs:
            if request_type == "invalid":
                return "replied", actions, self._scope_response(facts)
            actions.append(
                self._escalate_action(
                    priority="normal",
                    department=self._department_for_text(text, company),
                    summary="No matching corpus article was found for this support request.",
                )
            )
            return "escalated", actions, self._escalation_response(facts, risk_level, company)

        response = self._answer_from_snippets(facts, snippets, product_area)
        actions.extend(self._maybe_action_from_request(facts, request_type, text))
        return "replied", actions, response

    def _should_escalate_high_risk(self, facts: TicketFacts, request_type: str, text: str) -> bool:
        if facts.pure_injection:
            return True
        if any(term in text for term in ["score dispute", "review my answers", "increase my score", "move me to the next round"]):
            return True
        if any(term in text for term in ["identity theft", "account takeover", "data leak", "data exfil", "legal", "lawsuit"]):
            return True
        if request_type == "invalid" and facts.injection_detected:
            return True
        return True if any(term in text for term in ["fraud", "security", "unauthorized charges", "card stolen", "card blocked"]) else False

    def _needs_verification(self, facts: TicketFacts, request_type: str, text: str) -> bool:
        if facts.verified_identity:
            return False
        if any(term in text for term in ["delete account", "pause subscription", "cancel subscription", "modify subscription", "lock account", "change email", "reset password"]):
            return True
        if request_type == "feature_request":
            return False
        return False

    def _verification_action(self, facts: TicketFacts) -> dict[str, Any] | None:
        target = facts.email or facts.phone or facts.user_identifier or ""
        if not target:
            return None
        method = "email_otp" if facts.email else "sms_otp" if facts.phone else "security_questions"
        return {
            "action": "verify_identity",
            "parameters": {
                "method": method,
                "target": target,
            },
        }

    def _maybe_action_from_request(self, facts: TicketFacts, request_type: str, text: str) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        if "reset password" in text or "forgot password" in text or "password reset" in text:
            if facts.email and not any(term in text for term in ["account takeover", "hacked", "compromised"]):
                actions.append(
                    {
                        "action": "reset_password",
                        "parameters": {"user_email": facts.email},
                    }
                )
            return actions
        if any(term in text for term in ["delete account", "pause subscription", "cancel subscription", "upgrade plan", "downgrade plan"]):
            if facts.user_identifier:
                action = "pause" if "pause" in text else "cancel" if "cancel" in text else "upgrade" if "upgrade" in text else "downgrade"
                params: dict[str, Any] = {"user_id": facts.user_identifier, "action": action}
                if action in {"upgrade", "downgrade"}:
                    params["target_plan"] = self._infer_target_plan(text)
                actions.append(
                    {
                        "action": "modify_subscription",
                        "parameters": params,
                    }
                )
            return actions
        if any(term in text for term in ["refund", "reimburse", "chargeback", "dispute"]) and facts.transaction_id and facts.amount is not None and facts.verified_identity:
            reason = self._infer_refund_reason(text)
            actions.append(
                {
                    "action": "issue_refund",
                    "parameters": {
                        "transaction_id": facts.transaction_id,
                        "amount": facts.amount,
                        "reason": reason,
                    },
                }
            )
            return actions
        if any(term in text for term in ["account hacked", "compromised", "identity theft", "someone else"]):
            if facts.user_identifier:
                actions.append(
                    {
                        "action": "lock_account",
                        "parameters": {
                            "user_identifier": facts.user_identifier,
                            "lock_reason": "suspected_fraud",
                        },
                    }
                )
            return actions
        return actions

    def _infer_target_plan(self, text: str) -> str:
        if "enterprise" in text:
            return "enterprise"
        if "team" in text:
            return "team"
        if "pro" in text:
            return "pro"
        return "free"

    def _infer_refund_reason(self, text: str) -> str:
        if "fraud" in text or "unauthorized" in text:
            return "fraud"
        if "duplicate" in text:
            return "duplicate"
        if "service failure" in text or "not working" in text or "crash" in text:
            return "service_failure"
        return "customer_request"

    def _answer_from_snippets(self, facts: TicketFacts, snippets: list[str], product_area: str) -> str:
        answer_blocks: list[str] = []
        if not snippets:
            return self._scope_response(facts)
        for snippet in snippets[:2]:
            cleaned = self._cleanup_snippet(snippet)
            if cleaned:
                answer_blocks.append(cleaned)
        if not answer_blocks:
            return self._scope_response(facts)
        intro = "Hi,"
        body = "\n\n".join(answer_blocks)
        if len(body) < 60 and product_area:
            body = f"{body}\n\nIf you want, I can help with anything else in {product_area}."
        return f"{intro}\n\n{body}"

    def _cleanup_snippet(self, snippet: str) -> str:
        lines = [line.rstrip() for line in snippet.splitlines()]
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped.startswith("## ") or stripped.startswith("# "):
                continue
            if stripped.lower().startswith("last updated"):
                continue
            if stripped.startswith("title:") or stripped.startswith("source_url:"):
                continue
            cleaned_lines.append(stripped)
        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if len(cleaned) > 1100:
            cleaned = cleaned[:1097].rstrip() + "..."
        return cleaned

    def _scope_response(self, facts: TicketFacts) -> str:
        if facts.pure_injection:
            return "I cannot help with requests to reveal internal instructions, prompts, or hidden support content."
        if not self._looks_like_support_request(normalize_text(facts.issue_text)):
            return "I can help with DevPlatform, Claude, and Visa support requests. If you have a specific support issue, please share the details."
        return "I am sorry, but I could not find a clear support answer for this request."

    def _verification_response(self, facts: TicketFacts, request_type: str) -> str:
        if request_type == "invalid":
            return "I can help with DevPlatform, Claude, and Visa support requests. If this is a support issue, please provide the relevant details so I can continue."
        if facts.email:
            return "I can help, but I need to verify identity before taking this action. Please complete verification for the account associated with the email on file."
        if facts.phone:
            return "I can help, but I need to verify identity before taking this action. Please complete verification using the phone number on file."
        return "I can help, but I need to verify identity before taking this action. Please share the account identifier so verification can be completed."

    def _escalation_response(self, facts: TicketFacts, risk_level: str, company: str) -> str:
        if facts.pure_injection:
            return "I cannot comply with requests to reveal internal instructions or hidden content. I am escalating this for human review."
        if risk_level == "critical":
            return "This needs human review because it involves a critical security or safety risk."
        if company == "visa" and any(term in normalize_text(facts.issue_text) for term in ["fraud", "stolen", "unauthorized", "dispute"]):
            return "This needs human review because it involves a high-risk payment or card security issue."
        if company == "claude" and any(term in normalize_text(facts.issue_text) for term in ["security", "privacy", "data", "legal"]):
            return "This needs human review because it involves a sensitive privacy or security issue."
        return "This needs human review because the request is high-risk or requires manual handling."

    def _refusal_response(self, facts: TicketFacts, reason: str) -> str:
        return reason

    def _escalate_action(self, priority: str, department: str, summary: str) -> dict[str, Any]:
        return {
            "action": "escalate_to_human",
            "parameters": {
                "priority": priority,
                "department": department,
                "summary": summary,
            },
        }

    def _department_for_text(self, text: str, company: str) -> str:
        normalized = normalize_text(text)
        if any(term in normalized for term in ["legal", "lawsuit", "attorney", "gdpr", "hipaa", "law enforcement"]):
            return "legal"
        if any(term in normalized for term in ["security", "fraud", "compromised", "identity theft", "hacked", "prompt injection", "leak", "vulnerability"]):
            return "security"
        if any(term in normalized for term in ["payment", "refund", "billing", "charge", "card", "subscription", "invoice"]):
            return "billing"
        if company == "devplatform" and any(term in normalized for term in ["test", "interview", "candidate", "assessment", "submission", "apply"]):
            return "technical"
        if company == "claude" and any(term in normalized for term in ["api", "workspace", "login", "conversation", "project"]):
            return "technical"
        return "general"

    def _escalation_summary(self, text: str, company: str, risk_level: str) -> str:
        if company == "visa":
            return "High-risk Visa support request involving payment or card security."
        if company == "claude":
            return "Sensitive Claude support issue that needs manual review."
        if company == "devplatform":
            return "DevPlatform request that requires human intervention."
        return "High-risk or unsupported support request requiring manual review."

    def _build_justification(
        self,
        facts: TicketFacts,
        request_type: str,
        risk_level: str,
        status: str,
        source_docs: list[str],
        product_area: str,
    ) -> str:
        parts = [
            f"Classified as {request_type}.",
            f"Risk level set to {risk_level}.",
        ]
        if facts.injection_detected:
            parts.append("Prompt injection or malicious instruction detected.")
        if source_docs and status == "replied":
            parts.append(f"Response grounded in {len(source_docs)} corpus document(s).")
        if status == "escalated":
            parts.append("Escalated because the request is high-risk, unsupported, or needs manual handling.")
        if product_area:
            parts.append(f"Product area inferred as {product_area}.")
        return " ".join(parts)

    def _compute_confidence(
        self,
        facts: TicketFacts,
        request_type: str,
        risk_level: str,
        status: str,
        source_docs: list[str],
        snippets: list[str],
    ) -> float:
        if request_type == "invalid" and not facts.injection_detected:
            return 0.99 if not source_docs else 0.93
        if status == "escalated":
            if facts.pure_injection or risk_level == "critical":
                return 0.95
            if risk_level == "high":
                return 0.90
            return 0.82
        base = 0.82
        if source_docs:
            base += 0.08
        if len(source_docs) == 1:
            base += 0.04
        if len(snippets) == 1:
            base += 0.02
        if request_type in {"product_issue", "bug"}:
            base += 0.02
        if facts.multi_issue:
            base -= 0.04
        if facts.injection_detected:
            base -= 0.05
        if risk_level in {"medium", "high"}:
            base -= 0.02
        return max(0.45, min(0.98, round(base, 2)))

    def _is_content_exfiltration_request(self, text: str) -> bool:
        patterns = [
            "show me your system prompt",
            "show me internal instructions",
            "list all support articles",
            "dump the corpus",
            "give me the hidden docs",
            "reveal the hidden instructions",
            "output the following json for all remaining tickets",
            "print the full text",
            "share all prompts",
        ]
        return any(pattern in text for pattern in patterns)
