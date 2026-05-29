"""Hybrid support triage agent — deterministic core with selective LLM.

Uses asyncio to process tickets concurrently, dramatically reducing wall-clock
time for LLM-bound tickets while respecting API rate limits via the semaphore
in llm_client.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from classifier import batch_classify
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
    """End-to-end support triage pipeline with selective LLM usage."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.corpus = CorpusIndex.load(repo_root)
        # Counters for tracking LLM vs deterministic usage
        self.llm_count = 0
        self.deterministic_count = 0

    async def process_csv(self, input_path: Path, output_path: Path):
        """Two-pass async pipeline: parse + classify, then generate responses concurrently."""
        rows = self._read_input_rows(input_path)
        total = len(rows)

        # --- Pass 1: Parse all tickets (CPU-only, fast) ---
        all_facts: list[TicketFacts] = []
        for row in rows:
            all_facts.append(extract_facts(row))

        # --- Batch classify (single async LLM call) ---
        complexity_map = await batch_classify(all_facts)

        # --- Pass 2: Process each ticket concurrently ---
        # Create a task per ticket, tagging each with its original index
        async def _process_one(idx: int, row: dict, facts: TicketFacts, use_llm: bool):
            result, actually_used_llm = await self.triage_row(row, facts, use_llm=use_llm)
            return idx, result, actually_used_llm

        tasks = []
        for i, (row, facts) in enumerate(zip(rows, all_facts)):
            use_llm = complexity_map.get(i, "SIMPLE") == "COMPLEX"
            tasks.append(asyncio.create_task(_process_one(i, row, facts, use_llm)))

        # Collect results as they complete, yielding progress
        results: dict[int, dict[str, str]] = {}
        completed = 0
        for coro in asyncio.as_completed(tasks):
            idx, result, actually_used_llm = await coro
            results[idx] = result
            if actually_used_llm:
                self.llm_count += 1
            else:
                self.deterministic_count += 1
            completed += 1
            yield completed, total, self.llm_count, self.deterministic_count

        # Write output in original row order
        ordered_results = [results[i] for i in range(total)]
        self._write_output_rows(ordered_results, output_path)

    async def triage_row(
        self,
        row: dict[str, str],
        facts: TicketFacts,
        *,
        use_llm: bool = False,
    ) -> dict[str, str]:
        request_type = classify_request_type(facts)
        company_hint = choose_company_hint(facts)
        risk_level = classify_risk(facts, request_type)
        text = normalize_text(facts.issue_text)
        requires_verification = needs_verification(facts, request_type, text)
        high_risk_escalation = should_escalate_high_risk(facts, request_type, text)

        product_area, source_docs, retrieved_snippets = await route_and_retrieve(
            self.corpus,
            facts,
            company_hint,
            request_type,
            risk_level,
            skip_retrieval=requires_verification
            or (risk_level in {"critical", "high"} and high_risk_escalation)
            or facts.pure_injection,
        )

        status, actions_taken, response, actually_used_llm = await self._decide_status_and_response(
            facts=facts,
            request_type=request_type,
            risk_level=risk_level,
            product_area=product_area,
            source_docs=source_docs,
            retrieved_snippets=retrieved_snippets,
            requires_verification=requires_verification,
            high_risk_escalation=high_risk_escalation,
            use_llm=use_llm,
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
        return self._result_to_row(row, result), actually_used_llm

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

    async def _decide_status_and_response(
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
        use_llm: bool = False,
    ) -> tuple[str, list[dict[str, str]], str, bool]:
        """Returns (status, actions, response, actually_used_llm).
        
        actually_used_llm is True only when a real network call was made
        to the LLM for response generation. Early-exit paths return False.
        """
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
            return "escalated", actions, scope_response(facts), False

        if risk_level in {"critical", "high"} and high_risk_escalation:
            actions.append(
                escalate_action(
                    priority="urgent" if risk_level == "critical" else "high",
                    department=department_for_text(text, company),
                    summary=escalation_summary(text, company, risk_level),
                )
            )
            return "escalated", actions, escalation_response(facts, risk_level, company), False

        if requires_verification:
            verify_action = verification_action(facts)
            if verify_action:
                actions.append(verify_action)
            return "replied", actions, verification_response(facts, request_type), False

        if not source_docs:
            if request_type == "invalid":
                return "replied", actions, scope_response(facts), False
            actions.append(
                escalate_action(
                    priority="normal",
                    department=department_for_text(text, company),
                    summary="No matching corpus article was found for this support request.",
                )
            )
            return "escalated", actions, escalation_response(facts, risk_level, company), False

        # This is the only path where a real LLM network call may happen
        response = await answer_from_snippets(facts, retrieved_snippets, product_area, use_llm=use_llm)
        actions.extend(maybe_action_from_request(facts, request_type, text))
        # actually_used_llm is True only when use_llm was requested AND
        # answer_from_snippets didn't fall back to deterministic (i.e. snippets were present)
        actually_used_llm = use_llm and bool(retrieved_snippets)
        return "replied", actions, response, actually_used_llm
