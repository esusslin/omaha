# Omaha

*The call you make at the line, after you see what's actually in front of you.*

Document intelligence and agent layer for NFL prediction — ingestion, hybrid retrieval,
and evaluated LLM extraction over the text that structured feeds throw away.

## The premise

Injury feeds give you `Questionable`. They don't give you *"limited Wednesday, full
Thursday, coach said game-time decision"* — and the trajectory is the signal. Omaha
ingests official practice reports, inactives, transactions, transcripts and depth charts,
then turns them into typed, cited context records.

**Agents produce evidence, never probabilities.** That constraint is enforced in the
output schema, not by convention. An LLM-produced probability has no calibration curve
and no way to tell whether a prompt change improved it or merely moved it. Numbers come
from statistical models that can be walk-forward validated; language models do extraction
and judgement over text, where there is no ground truth to regress against.

## What this repository deliberately does not contain

No picks, no betting strategy, no model weights, no user data, no edge. The prediction
system that consumes these context records is separate and private. What's here is the
document and agent infrastructure.

## Status

**Phase 1 — ingestion.** In progress.

| Phase | Scope | State |
|---|---|---|
| 0 | Foundations, schema, health | ✅ |
| 1 | Document ingestion, bitemporal store | 🚧 |
| 2 | Chunking, embeddings, hybrid retrieval | — |
| 3 | Gold set, eval harness, CI gate | — |
| 4 | LangGraph agents, typed context records | — |
| 5 | MCP servers | — |

## Design rules

**1. Bitemporal, always.** Every document carries `knowledge_time` — when *we* learned
it, not when it happened. Wednesday's report and Friday's are separate rows. Nothing may
be read that wasn't knowable at the time being reconstructed. Leakage is how these
systems lie to themselves.

**2. Provenance survives the pipeline.** Chunks keep span offsets so every downstream
claim cites exact source text. Groundedness is only measurable if the spans persist.

**3. Untrusted input.** Fetched text is hostile until proven otherwise. Prompt injection
in a scraped article is a real attack surface, and it's tested for.

**4. Evals before agents.** The gold set and eval harness (Phase 3) land before the agent
layer (Phase 4), so every subsequent change has a number attached.

## Setup

```bash
# Postgres 17 + pgvector
docker compose up -d

# dependencies
uv sync --extra dev

# schema
uv run alembic upgrade head

# run
uv run uvicorn omaha.api:app --reload
```

Then `curl localhost:8000/health` — it reports per-source staleness, not just process
liveness. It returns 200 while the process is alive and sets `ok: false` when a source is
overdue, so an orchestrator doesn't restart a healthy container over stale upstream data.
Alert on `ok`, not on the status code.

## Sources

Official club practice and injury reports, inactives, transactions, press conference
transcripts, published depth charts. Official and public feeds only — no paywalled
scraping.

PDFs are parsed with `pdfplumber`. `docling` is the better tool for complex layout but
depends on torch, which has no x86_64 macOS wheels — a portability constraint worth
knowing before you reach for it.

## Licence

MIT
