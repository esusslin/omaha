# Omaha

*The call you make at the line, after you see what's actually in front of you.*

Document intelligence for NFL prediction — ingestion, hybrid retrieval, and evaluated LLM
extraction over the text that structured feeds throw away.

## The premise

Injury feeds give you `Questionable`. They don't give you *"limited Wednesday, full
Thursday, coach said game-time decision"* — and the trajectory is the signal. Omaha
ingests official practice reports, inactives, transactions, transcripts and depth charts,
and turns them into typed, cited context records.

**Agents produce evidence, never probabilities.** That constraint is enforced in the
output schema, not by convention. An LLM-produced probability has no calibration curve
and no way to tell whether a prompt change improved it or merely moved it. Numbers come
from statistical models that can be walk-forward validated; language models do extraction
and judgement over text, where there is no ground truth to regress against.

## What this repository deliberately does not contain

No picks, no betting strategy, no model weights, no user data, no edge. The prediction
system that consumes these context records is separate and private. What's here is the
document and agent infrastructure.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Foundations, schema, health endpoint | ✅ |
| 1 | Ingestion — conditional fetch, parsers, bitemporal store, scheduler | ✅ |
| 2 | Chunking, embeddings, hybrid retrieval | 🚧 embeddings done, retrieval next |
| 3 | Gold set, eval harness, CI gate | — |
| 4 | Agent runtime, typed context records | — |
| 5 | MCP servers | — |

---

## Design rules

**1. Bitemporal, always.** Every document carries `knowledge_time` — when *we* learned it,
not when it happened. Nothing may be read that wasn't knowable at the time being
reconstructed. Leakage is how these systems lie to themselves.

**2. Provenance survives the pipeline.** Chunks keep span offsets and a section path so
every downstream claim cites exact source text. Groundedness is only measurable if the
spans persist.

**3. Untrusted input.** Fetched text is hostile until proven otherwise. Prompt injection
in a scraped article is a real attack surface, and it's tested for.

**4. Evals before agents.** The gold set and eval harness land before the agent layer, so
every subsequent change has a number attached.

---

## Setup

```bash
docker compose up -d              # Postgres 17 + pgvector on :5433
uv sync --extra dev
uv run alembic upgrade head
uv run uvicorn omaha.api:app --reload
```

`curl localhost:8000/health` reports per-source staleness and the last scheduled run —
not just process liveness. It returns 200 while the process is alive and sets
`ok: false` when a source is overdue, so an orchestrator doesn't restart a healthy
container over stale upstream data. **Alert on `ok`, not on the status code.**

### Ingestion

```bash
uv run python -m omaha.ingest.run add --name ne-injury --kind injury_report \
    --team NE --url https://www.patriots.com/team/injury-report/
uv run python -m omaha.ingest.run sweep          # only fetches what's due
uv run python -m omaha.ingest.run sweep --force  # ignore cadence
uv run python -m omaha.ingest.run list
uv run python -m omaha.ingest.run jobs
uv run python -m omaha.ingest.run asof --name ne-injury --when 2026-09-10T18:00:00Z
```

### Retrieval

```bash
make chunk               # host — pure Python
make worker              # build the Linux image (one-time)
make embed               # container — needs ONNX Runtime
make retrieve-status
```

---

## Architecture notes

These are the decisions that took thought. Each one has a reason that isn't obvious.

### Dedup is on parsed text, not raw bytes

Club pages carry build IDs, nonces and rotating tokens, so the bytes differ on every
fetch while the report is unchanged. Byte hashing produced a new row per poll — 24
near-duplicate snapshots a day, defeating the point of the store. The parsed text is
stable, so that's the identity that decides "is this new?"

`content_hash` is retained for raw-file provenance; `text_hash` decides whether a row is
written. Found by running against a live club site; a synthetic fixture would have hashed
identically forever and the bug would have surfaced in week six.

### A snapshot is written only when content changes

`knowledge_time` therefore means *when we first saw this version*, which is the moment
that matters for leakage. Reconstructing what was known at time T is "the latest document
for this source with `knowledge_time <= T`" — that's the `asof` command, and it's the
invariant everything else rests on.

### Cron runs in America/New_York

NFL policy requires practice reports by 4:00 pm Eastern on Wednesday, Thursday and
Friday. A UTC schedule drifts an hour across daylight saving — landing wrong in November,
exactly when reports start mattering. The sweep fires at 17:00 ET so late filers are
caught.

| Job | When | What |
|---|---|---|
| `injury_sweep` | Wed/Thu/Fri 17:00 ET | Practice reports |
| `hourly_sweep` | hourly at :07 | Everything else, gated per source |

### Cadence gating uses last attempt, not last success

A failing source would otherwise be retried on every tick, hammering an origin that's
already unhappy. Sources declare `cadence_seconds`; the scheduler fires often and the
sweep decides who's actually due.

### One chunk per player row

Practice reports are tables, and the atom is a player: *"K. Allen, WR, hamstring,
LIMITED, Wed"*. Windowing that into 400-token blocks smears three players across a
boundary and makes citation meaningless. Header labels are folded in, so a retrieved
chunk reads `Player: K. Allen | Pos: WR | Injury: Hamstring | Wed: Limited` — meaningful
alone and readable in a citation.

Tabular documents skip their prose entirely: a practice report's prose is the legend, and
embedding the same DNP/LP/FP definitions 32 times a week is pure retrieval noise.

### Embeddings run in a Linux container

The dev machine is x86_64 macOS, and upstream dropped those wheels — torch after 2.2,
onnxruntime shortly after. Rather than pin ancient versions and diverge from the server,
model work runs on `linux/amd64`, which is where it runs in production anyway. Embeddings
are an optional `[embed]` extra installed only in the image, so `uv sync --extra dev`
stays installable on any host.

```bash
docker compose --profile worker run --rm worker uv run python -m omaha.retrieve.run embed
```

### bge-base-en-v1.5, not bge-m3

M3 is 568M parameters and multilingual; base is 109M and English-only. Practice reports
are English, this runs on CPU, and base scores within a point or two on English retrieval
benchmarks. Five times the compute for no measurable gain.

Model and version are stamped on every chunk, so re-embedding is additive: write the new
version alongside the old, switch reads, drop the old. That's impossible if the vector has
no provenance.

BGE wants a prefix on *queries* but not on stored passages — `embed_query` and
`embed_passages` are separate functions because getting it backwards is a silent 5–10%
retrieval regression that no test catches.

### PDFs use pdfplumber, not docling

docling is better on complex layout but depends on torch — same portability wall.
Practice reports are plain tables and pdfplumber handles them. Revisit on the server if
gamebooks prove too much.

---

## Sources

Official club practice and injury reports, inactives, transactions, press conference
transcripts, published depth charts. **Official and public feeds only** — no paywalled
scraping.

## Stack

Python 3.13 · FastAPI · SQLAlchemy 2 · Alembic · Postgres 17 + pgvector · APScheduler ·
pdfplumber · trafilatura · fastembed (ONNX) · uv · ruff · mypy · pytest · Docker

## Licence

MIT
