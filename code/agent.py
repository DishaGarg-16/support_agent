"""Deterministic support triage agent."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from corpus import CorpusIndex
from models import TicketFacts, TriageResult
from parser import extract_facts
from policy import (
    classify_request_type,
    classify_risk,
    department_for_text,
    escalate_action,
    escalation_summary,
    is_content_exfiltration_request,
    maybe_action_from_request,
    needs_verification,
    should_escalate_high_risk,
    verification_action,
)
from response_builder import (
    answer_from_snippets,
    build_justification,
    compute_confidence,
    escalation_response,
    scope_response,
    verification_response,
)
from router import choose_company_hint, route_and_retrieve
from text_utils import normalize_text


class SupportTriageAgent:
    """End-to-end support triage pipeline."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.corpus = CorpusIndex.load(repo_root)

    def process_csv(self, input_path: Path, output_path: Path):
        rows = self._read_input_rows(input_path)
        results = []
        for i, row in enumerate(rows):
            results.append(self.triage_row(row))
            yield i + 1, len(rows)
        self._write_output_rows(results, output_path)

    def triage_row(self, row: dict[str, str]) -> dict[str, str]:
        facts = extract_facts(row)
        request_type = classify_request_type(facts)
        company_hint = choose_company_hint(facts)
        risk_level = classify_risk(facts, request_type)
        text = normalize_text(facts.issue_text)
        requires_verification = needs_verification(facts, request_type, text)
        high_risk_escalation = should_escalate_high_risk(facts, request_type, text)

        product_area, source_docs, retrieved_snippets = route_and_retrieve(
            self.corpus,
            facts,
            company_hint,
            request_type,
            risk_level,
            skip_retrieval=requires_verification
            or (risk_level in {"critical", "high"} and high_risk_escalation)
            or facts.pure_injection,
        )

        status, actions_taken, response = self._decide_status_and_response(
            facts=facts,
            request_type=request_type,
            risk_level=risk_level,
            product_area=product_area,
            source_docs=source_docs,
            retrieved_snippets=retrieved_snippets,
            requires_verification=requires_verification,
            high_risk_escalation=high_risk_escalation,
        )

        justification = build_justification(
            facts,
            request_type,
            risk_level,
            status,
            source_docs,
            product_area,
        )
        confidence = compute_confidence(
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
            return list(reader)

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

    def _decide_status_and_response(
        self,
        *,
        facts: TicketFacts,
        request_type: str,
        risk_level: str,
        product_area: str,
        source_docs: list[str],
        retrieved_snippets: list[str],
        requires_verification: bool,
        high_risk_escalation: bool,
    ) -> tuple[str, list[dict[str, str]], str]:
        actions: list[dict[str, str]] = []
        text = normalize_text(facts.issue_text)
        company = facts.company_guess or facts.company_field.lower()

        if facts.pure_injection or is_content_exfiltration_request(text):
            actions.append(
                escalate_action(
                    priority="urgent" if risk_level in {"critical", "high"} else "high",
                    department="security",
                    summary="Potential prompt injection or request to reveal internal instructions/corpus content.",
                )
            )
            return "escalated", actions, scope_response(facts)

        if risk_level in {"critical", "high"} and high_risk_escalation:
            actions.append(
                escalate_action(
                    priority="urgent" if risk_level == "critical" else "high",
                    department=department_for_text(text, company),
                    summary=escalation_summary(text, company, risk_level),
                )
            )
            return "escalated", actions, escalation_response(facts, risk_level, company)

        if requires_verification:
            verify_action = verification_action(facts)
            if verify_action:
                actions.append(verify_action)
            return "replied", actions, verification_response(facts, request_type)

        if not source_docs:
            if request_type == "invalid":
                return "replied", actions, scope_response(facts)
            actions.append(
                escalate_action(
                    priority="normal",
                    department=department_for_text(text, company),
                    summary="No matching corpus article was found for this support request.",
                )
            )
            return "escalated", actions, escalation_response(facts, risk_level, company)

        response = answer_from_snippets(facts, retrieved_snippets, product_area)
        actions.extend(maybe_action_from_request(facts, request_type, text))
        return "replied", actions, response
