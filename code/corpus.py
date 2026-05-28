"""Corpus loading and lightweight BM25 retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import math
import os
import re
from pathlib import Path
from typing import Iterable

from text_utils import normalize_text, tokenize


@dataclass(frozen=True)
class CorpusDoc:
    path: str
    company: str
    title: str
    category: str
    content: str
    tokens: tuple[str, ...]
    block_texts: tuple[str, ...]


class CorpusIndex:
    """Simple BM25-style index over markdown support docs."""

    def __init__(self, docs: list[CorpusDoc]):
        self.docs = docs
        self.doc_count = len(docs)
        self.avg_doc_len = (
            sum(len(doc.tokens) for doc in docs) / len(docs) if docs else 0.0
        )
        self.doc_freq: dict[str, int] = defaultdict(int)
        for doc in docs:
            for token in set(doc.tokens):
                self.doc_freq[token] += 1

    @classmethod
    def load(cls, repo_root: Path) -> "CorpusIndex":
        docs: list[CorpusDoc] = []
        data_root = repo_root / "data"
        for path in sorted(data_root.rglob("*.md")):
            relative = path.relative_to(repo_root).as_posix()
            if path.name.lower() == "index.md":
                continue
            if "api_specs" in relative:
                continue
            company = path.parts[path.parts.index("data") + 1] if "data" in path.parts else ""
            text = path.read_text(encoding="utf-8", errors="ignore")
            content, title = _extract_markdown_content(text)
            if len(content.strip()) < 40:
                continue
            blocks = tuple(
                block.strip()
                for block in re.split(r"\n\s*\n", content)
                if block.strip()
            )
            category = _infer_category(relative)
            tokens = tuple(tokenize(title + "\n" + content + "\n" + category))
            if not tokens:
                continue
            docs.append(
                CorpusDoc(
                    path=relative,
                    company=company.lower(),
                    title=title.strip(),
                    category=category,
                    content=content.strip(),
                    tokens=tokens,
                    block_texts=blocks,
                )
            )
        return cls(docs)

    def search(
        self,
        query: str,
        *,
        company_hint: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[float, CorpusDoc]]:
        query_tokens = [token for token in tokenize(query) if token]
        if company_hint:
            query_tokens.append(company_hint.lower())
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scored: list[tuple[float, CorpusDoc]] = []
        for doc in self.docs:
            score = self._bm25_score(doc, query_counts)
            score += self._boost_for_company(doc, company_hint)
            score += self._boost_for_exact_terms(doc, query_tokens)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: (item[0], item[1].path), reverse=True)
        return scored[:top_k]

    def _bm25_score(self, doc: CorpusDoc, query_counts: Counter[str]) -> float:
        if not doc.tokens or not self.doc_count:
            return 0.0

        k1 = 1.4
        b = 0.72
        doc_len = len(doc.tokens)
        score = 0.0
        tf = Counter(doc.tokens)
        for token, qf in query_counts.items():
            if token not in tf:
                continue
            df = self.doc_freq.get(token, 0)
            if not df:
                continue
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1.0)))
            score += idf * (freq * (k1 + 1)) / denom * qf
        return score

    def _boost_for_company(self, doc: CorpusDoc, company_hint: str | None) -> float:
        if not company_hint:
            return 0.0
        if doc.company == company_hint.lower():
            return 1.4
        return 0.0

    def _boost_for_exact_terms(self, doc: CorpusDoc, query_tokens: Iterable[str]) -> float:
        haystack = normalize_text(doc.title + " " + doc.category + " " + doc.content[:500])
        boost = 0.0
        for term in query_tokens:
            if len(term) > 3 and term in haystack:
                boost += 0.05
        return boost


def _extract_markdown_content(raw: str) -> tuple[str, str]:
    lines = raw.splitlines()
    idx = 0
    title = ""
    if lines and lines[0].strip() == "---":
        idx = 1
        while idx < len(lines) and lines[idx].strip() != "---":
            idx += 1
        idx += 1
    for line in lines[idx:]:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break
    if not title:
        for line in lines[idx:]:
            stripped = line.strip()
            if stripped and not stripped.startswith(("title:", "source_url:", "last_updated", "breadcrumbs:")):
                title = stripped.lstrip("# ").strip()
                break

    content_lines: list[str] = []
    in_code = False
    for line in lines[idx:]:
        stripped = line.rstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if stripped.startswith("!") and "](" in stripped:
            continue
        if stripped.startswith("title:") or stripped.startswith("source_url:"):
            continue
        if stripped.startswith("last_updated") or stripped.startswith("last_modified"):
            continue
        if stripped.startswith("article_id:") or stripped.startswith("article_slug:"):
            continue
        if stripped.startswith("breadcrumbs:") or stripped.startswith("- "):
            content_lines.append(stripped)
            continue
        if stripped.startswith("---"):
            continue
        content_lines.append(stripped)

    content = "\n".join(content_lines)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", content)
    content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
    content = re.sub(r"\s+\n", "\n", content)
    content = content.strip()
    if not title:
        title = Path("untitled").stem
    return content, title


def _infer_category(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) < 3:
        return "general"
    # data/<company>/<top_level>/<...>
    category = parts[2]
    category = category.replace("-", "_").replace(" ", "_")
    return category


def split_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    return blocks


def extract_answer_excerpt(doc: CorpusDoc, query: str, max_chars: int = 1200) -> str:
    """Select the most relevant excerpt from a document."""

    query_tokens = set(tokenize(query))
    blocks = list(doc.block_texts) or split_blocks(doc.content)
    if not blocks:
        return doc.content[:max_chars]

    scored: list[tuple[float, int, str]] = []
    for idx, block in enumerate(blocks):
        block_tokens = set(tokenize(block))
        overlap = len(block_tokens & query_tokens)
        if overlap == 0:
            heading_match = 1 if any(term in normalize_text(block) for term in query_tokens if len(term) > 3) else 0
            if heading_match == 0:
                continue
            overlap = 1
        list_bonus = 0.2 if re.search(r"(?m)^(?:[-*]|\d+\.)\s+", block) else 0.0
        question_bonus = 0.2 if "?" in block else 0.0
        length_penalty = min(len(block) / 1400.0, 1.0) * 0.35
        table_penalty = 0.35 if block.count("|") >= 4 else 0.0
        numeral_penalty = 0.15 if len(re.findall(r"\d", block)) >= 18 else 0.0
        score = overlap + list_bonus + question_bonus - length_penalty - table_penalty - numeral_penalty
        scored.append((score, idx, block))

    if not scored:
        chosen = blocks[:2]
    else:
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        _, best_idx, best_block = scored[0]
        chosen = [best_block]
        if best_idx + 1 < len(blocks):
            next_block = blocks[best_idx + 1]
            if len(" ".join(chosen)) + len(next_block) < max_chars and (
                re.search(r"(?m)^(?:[-*]|\d+\.)\s+", next_block) or len(next_block) < 500
            ):
                chosen.append(next_block)

    excerpt = "\n\n".join(chosen)
    excerpt = re.sub(r"\n{3,}", "\n\n", excerpt).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt
