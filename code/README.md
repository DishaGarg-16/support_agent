# Code README

This directory contains a deterministic, local-first support triage pipeline for the MLE Hiring Challenge.

## What it does

- Reads `support_tickets/support_tickets.csv`
- Parses the JSON issue history for each ticket
- Applies safety and adversarial-input guardrails
- Retrieves the most relevant support corpus documents from `data/`
- Produces `support_tickets/output.csv` with all required columns

## Run

From the repository root:

```bash
python code/main.py
```

Then validate the output format:

```bash
python code/validate_output.py
```

## Design notes

- The pipeline is deterministic and uses only local corpus data.
- No external API calls are required for the baseline.
- Source attribution is written to the `source_documents` column using repository-relative file paths.
- Actions are emitted as JSON in `actions_taken` only when the conversation provides enough context.

