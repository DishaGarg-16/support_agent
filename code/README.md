# SUPA — Support Triage Pipeline Agent

A hybrid deterministic + LLM-powered terminal agent for the MLE Hiring Challenge.
Processes support tickets across **DevPlatform**, **Claude**, and **Visa** using a local BM25 corpus combined with Groq (Llama 3.3 70B) for intelligent response generation.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Copy the example env file and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free key at [console.groq.com](https://console.groq.com). The agent falls back to fully deterministic responses if no key is set.

---

## Run

From the **repository root**:

```bash
python code/main.py
```

This reads `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv`.

### Validate output format

```bash
python code/validate_output.py
```

---

## Module Map

| File | Purpose |
|---|---|
| `main.py` | CLI entry point — banner, progress bar, summary table via `rich` |
| `agent.py` | Two-pass async orchestrator — parses, classifies, then processes tickets concurrently |
| `classifier.py` | Single batch LLM call to label tickets as `SIMPLE` or `COMPLEX` |
| `llm_client.py` | Async Groq wrapper — `AsyncGroq`, semaphore rate limiting, exponential backoff, fallback model |
| `router.py` | BM25 retrieval routing with LLM company-disambiguation fallback |
| `corpus.py` | BM25 corpus indexer — loads and indexes all `data/**/*.md` files |
| `parser.py` | Parses ticket JSON, extracts facts, detects PII, language, and injection signals |
| `policy.py` | Deterministic rules — risk classification, escalation logic, action generation |
| `response_builder.py` | Async response assembly — LLM generation for COMPLEX tickets, deterministic fallback |
| `models.py` | Shared dataclasses — `TicketFacts`, `TriageResult` |
| `text_utils.py` | Text normalisation helpers |
| `validate_output.py` | Schema validation script (structure only, not quality) |

---

## Design Notes

- **Two-pass async pipeline:** All tickets are parsed and batch-classified first (one LLM call), then processed concurrently with `asyncio`. Results are collected and sorted by original row index before writing, ensuring deterministic CSV order.
- **Rate-limit safety:** An `asyncio.Semaphore(10)` caps concurrent Groq requests. Transient 429 errors trigger exponential backoff (up to 3 retries). If the primary model (`llama-3.3-70b-versatile`) exhausts its daily quota, the pipeline automatically falls back to `llama-3.1-8b-instant`.
- **Determinism:** Fixed `temperature=0.3` and `seed=42` on all LLM calls. Deterministic paths produce identical outputs on repeated runs.
- **Corpus-only:** All factual claims in responses are grounded in the `data/` corpus. The LLM is strictly prompted to never hallucinate beyond the retrieved snippets.
- **No hardcoded secrets:** All credentials are read from environment variables only.

---

## Future Scope

### 1. Recursive Sub-query Resolution
Currently, if a ticket contains multiple distinct questions, the agent handles it as a single unit and may conservatively escalate or partially answer. A future improvement would **decompose multi-intent tickets into individual sub-queries**, resolve each independently through the full safety → retrieval → generation loop, and merge the answers into a single coherent response. This would be implemented recursively — each sub-query is treated as its own ticket, with results aggregated and citations collected from all relevant documents before final output assembly.

### 2. Multilingual Prompt Injection Defense
The current injection detection focuses on English-language patterns, leaving a surface for attackers who embed instructions in other languages or mixed-language text. A more robust approach would include:
- **Language-agnostic intent detection** — running the injection classifier on a machine-translated version of the ticket alongside the original
- **Unicode homoglyph detection** — catching injections that substitute visually similar characters (e.g., Cyrillic `а` for Latin `a`) to bypass keyword filters
- **Mixed-language segmentation** — detecting when a ticket begins in one language and embeds malicious instructions in another

### 3. Confidence-Weighted Multi-Document Cross-Referencing
Instead of trusting the top BM25 document, future versions could cross-reference claims across multiple retrieved documents and automatically flag low-confidence responses when sources contradict each other. This would reduce hallucination risk on corpus conflicts and improve Brier score calibration.

### 4. Adaptive Complexity Threshold
The SIMPLE/COMPLEX classifier currently uses a fixed prompt with static criteria. A future improvement would dynamically tune the complexity boundary based on observed accuracy — routing more tickets to LLM only when the deterministic path's confidence score falls below a calibrated cutoff, reducing unnecessary API calls further while maintaining response quality.

### 5. Semantic Vector Retrieval (BM25 → Embeddings)
Replacing or augmenting the BM25 retriever with dense vector embeddings (e.g., `text-embedding-3-small`, `all-MiniLM-L6-v2`) would significantly improve retrieval accuracy for semantically similar but lexically different queries — for example, matching *"my payment didn't go through"* to a corpus document about *"transaction failures"* with no keyword overlap. A hybrid approach — BM25 for speed + vector re-ranking for precision — would offer the best of both worlds. The primary constraint is the strict 3-minute execution limit and the evaluation infrastructure's lack of GPU, making a lightweight ONNX-quantized embedding model the preferred implementation path.

