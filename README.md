# Omaha

*The call you make at the line, after you see what's actually in front of you.*

Document intelligence for NFL prediction — ingestion, hybrid retrieval, and evaluated LLM
extraction over the text that structured feeds throw away.

## Try it

Running live, collecting on its own schedule:

| | |
|---|---|
| **[Hybrid search over the corpus](https://omaha-production-17e9.up.railway.app/ui)** | Ask it something. Every result is tagged with which retriever found it — `dense`, `lexical`, or `both`. |
| **[Typed injury records](https://omaha-production-17e9.up.railway.app/injuries?team=PHI)** | The machine-facing output. Note the `knowledge` field. |
| **[Health and per-source staleness](https://omaha-production-17e9.up.railway.app/health)** | Not a liveness probe — it reports which of 62 sources are overdue. |
| **[What's been extracted, by club](https://omaha-production-17e9.up.railway.app/injuries/summary)** | |

Rate-limited to 30 requests a minute per caller — it's one small instance, and the
limiter is a fixed window in process memory rather than anything clever. Clone the repo
if you want to hammer it.

Questions that show the two retrievers disagreeing:

- *"Which Eagles lineman is out with a foot injury?"* — lexical finds it, dense doesn't
- *"Why is the Commanders quarterback not playing?"* — dense finds it, lexical doesn't
- *"Trey Pipkins back injury"* — an exact name, which is where embeddings are weakest

## The premise

Injury feeds give you `Questionable`. They don't give you *"limited Wednesday, full
Thursday, coach said game-time decision"* — and the trajectory is the signal. Omaha
ingests official practice reports, inactives, transactions, transcripts and depth charts,
and turns them into typed, cited context records.

**That premise was tested before this was built, and it survived.** Across 90,467
injury rows from 2009–2025, walk-forward by season: among players listed `Questionable`,
the rate at which they actually took a snap runs 30% for those who didn't practise and
55% for those who practised fully. Adding practice status to a model that already knows
prior form, position and game designation improves AUC by **+0.054** on that panel, and
by +0.043 on players carrying no game designation at all — the two groups where the
club's own label says least. Mean absolute error on usage is flat to four decimal places,
so the feature predicts *whether* a player takes the field, not *how much* he's used.

That measurement lives in the private prediction repo, not here. It's stated because a
data pipeline built on an untested assumption is a hobby, and the honest version of this
README says which one this is.

**Agents produce evidence, never probabilities.** That constraint is enforced in the
output schema, not by convention. An LLM-produced probability has no calibration curve
and no way to tell whether a prompt change improved it or merely moved it. Numbers come
from statistical models that can be walk-forward validated; language models do extraction
and judgement over text, where there is no ground truth to regress against.

## Absence is not ignorance

The single thing this system does that a structured feed cannot: say *why* a player has
no records.

```jsonc
// GET /injuries?team=PHI
{
  "knowledge": "complete",   // sources fresh — an empty list means he isn't on the
                             // report, which is information: he's healthy
  "knowledge": "partial",    // some sources overdue — treat absence as unknown
  "knowledge": "unknown",    // nothing collected, or all stale — an empty list says
                             // nothing at all
  "knowledge": "as_of_historical"  // you asked about the past; present freshness
                                   // cannot speak to it
}
```

Those cases produce identical JSON everywhere else, and conflating them is expensive. The
consuming system's red-team agent downgraded 41% of picks when weather data was merely
*missing*, having read "no value" as "bad value". `knowledge` is computed from source
health rather than record counts, because counting records cannot tell the two apart.

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
| 2 | Chunking, embeddings, hybrid retrieval | ✅ |
| 3 | Gold set, eval harness | ✅ gated in CI |
| 4 | Typed context records — LLM extraction, `/injuries` | ✅ |
| 5 | MCP servers | — |

Corpus: 1,193 chunks and 1,402 typed records from the 2025 season. Deployed and
collecting unattended — index discovery hourly, practice reports at 17:00 ET on
Wednesday, Thursday and Friday, an hour after the league's filing deadline.

Coverage is uneven **by measurement, not by assumption**: of 32 clubs, four curate a
season archive of report articles. `run discover` reports what each index actually links
rather than guessing.

---

## Retrieval results

20 questions, top-10, measured with `make eval`.

| mode | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| lexical | 55% | 70% | 70% | 75% | 0.625 |
| dense | 45% | 70% | 75% | 85% | 0.578 |
| **hybrid (RRF)** | **65%** | **80%** | **90%** | **95%** | **0.747** |

The number that matters isn't the headline — it's that **the two retrievers fail on
disjoint questions**, which is the only condition under which fusing them is worth
anything.

Lexical misses the paraphrases: *"Why is the Commanders quarterback not playing?"* shares
almost no vocabulary with `Player: Jayden Daniels | Injury: Left Elbow`. Dense misses the
exact names: `Trey Pipkins III`, `Otito Ogbonnia` — rare proper nouns are precisely what a
768-dimension average smooths away. Lexical wins on MRR, dense wins on hit@10, fusion wins
on every column.

### What the harness caught immediately

The first run scored lexical at 15%, with an *identical* hit rate at k=1 and k=10. A
retriever returning ten results cannot have a flat hit rate — that's the signature of one
returning zero rows.

Cause: `websearch_to_tsquery` ANDs its terms. *"Is Jalen Carter playing against the
Commanders?"* became `jalen & carter & play & against & commander`, and the row it should
find contains neither "playing" nor "against", so Postgres matched nothing. Not a bad
ranking — an empty result. Switching to OR-of-terms with `ts_rank` doing the
discriminating took lexical from 0.150 to 0.625 MRR.

Worth stating plainly: without the eval this would have shipped. Hybrid search "worked",
returned plausible results, and was quietly running on one retriever.

That bug is now a **canary in CI**. `tests/test_retrieval_regression.py` asks *"Which
lineman is hurt with a back problem?"* — a question where "which", "lineman" and "hurt"
appear in no chunk in the corpus. Under the old AND semantics it matches zero rows; the
test asserts it returns something. Alongside it, a quality floor (hit@3 ≥ 100%, MRR ≥
0.60) and two `as_of` assertions that check a player known only on Friday is invisible to
a query dated Wednesday.

The gate was **mutation-tested**: the AND semantics were deliberately reintroduced and the
suite confirmed to go red before it was trusted. A passing test and a test that would
catch the bug aren't the same thing, and the only way to know which you have is to break
the code on purpose.

Two smaller notes on that suite, both mistakes worth not repeating. The tests use a
dedicated scratch database, because the first version ran against whatever `DATABASE_URL`
pointed at — empty on CI, the full corpus on a laptop — so it passed where nobody looked
and failed where they did. And CI fails if those tests *skip*: a skipped gate reports
green forever.

---

## Extraction results

1,193 chunks processed, 1,402 typed records.

| field | coverage |
|---|---|
| evidence span | 100% |
| team | 100% |
| position | 97% |
| injury | 90% |
| practice status | 82% |
| report day | 64% |
| game status | 26% |

`game_status` at 26% is correct, not a gap — clubs only designate on the final report.

**`report_day` at 64% is a story about my own bug.** It sat at 55%, and the first two
explanations I reached for were both wrong: that the model couldn't resolve "today"
without a publication date (supplying one changed nothing), and that the missing rows
were final-status rows that carry no day (the counts said 70% of them had a practice
status). The actual cause was a closed vocabulary of `("WED", "THU", "FRI")` — the
league's *filing deadline*, encoded as if it were the set of days clubs practise. Real
rows read `Day: Tuesday`; the model extracted `TUE` correctly and validation discarded
it. Widening to all seven weekdays recovered 119 records.

The rest genuinely aren't recoverable: a final report states the last practice status and
the game designation without labelling a day, and inferring one would be wrong often
enough to matter in short weeks.

### How the extractor is kept honest

Everything the model returns passes through `extract/schema.py` before it can reach a
row, and that module makes no API call — so the component most likely to be confidently
wrong is also the one that's free to test.

- **Grounding.** A record whose player isn't named in the source chunk is dropped. Ask a
  model about injuries and it will produce a plausible NFL player who isn't in the text;
  a fluent invention is more dangerous than a blank, because downstream it looks exactly
  like a real record.
- **Closed vocabularies.** `practice_status` is `DNP`/`LIMITED`/`FULL` or null.
  "Limited participation in practice" — the actual source wording — becomes **null, not
  LIMITED**. Mapping it would paper over a model not following the schema.
- **Partial survival.** A garbled position doesn't discard a good practice status.
- **`extractor_version` on every row.** Bump it and the corpus becomes unextracted, so a
  prompt change is a re-run over stored chunks rather than a migration — and the old rows
  survive long enough to answer "did v2 actually beat v1?" It has already been used to
  prove a prompt change did *nothing*: v1 and v2 output were byte-identical on the same
  chunks.

### The remaining miss

`calcaterra-illness` fails in all three modes, and it's instructive. Grant Calcaterra
appears four times with four different injuries, so retrieval must discriminate among his
own rows. The query — *"Which Eagles tight end missed practice with an illness?"* — offers
almost no help: "Eagles" matches 318 chunks and "practice" matches **every** structured
row, because `Practice:` is a folded-in column label. Only "illness" carries signal.

That's a real tension, not a bug: header folding makes chunks self-contained and citable,
helps dense retrieval, and dilutes lexical ranking. Documented rather than hidden.

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
make chunk                # host — pure Python
make worker               # build the Linux image (one-time)
make embed                # container — needs ONNX Runtime
make retrieve-status
make audit                # extraction quality, counted

make search q="which Eagles lineman is out with a foot injury"
make eval                 # compare lexical / dense / hybrid over the gold set
make eval-lexical         # host-only, no model needed
```

### The demo

```bash
make demo     # API in the container, so /ui gets dense retrieval
```

Then open **http://localhost:8000/ui** — a single-file search page with no build step.
Each result is tagged with which retriever found it (`dense`, `lexical`, or `both`) and
its rank in each list, so the fusion is visible rather than asserted. `/docs` has the
OpenAPI schema.

`make dev` runs the same app on the host, where `fastembed` isn't installed. Search
still works and degrades to lexical, and the response says so — `mode_used`,
`dense_available`, and a banner on the page. A demo that silently drops half its
retrieval is worse than one that admits it.

Point-in-time search — the reason the store is bitemporal:

```bash
uv run python -m omaha.retrieve.search_cli \
    --query "who is questionable" --as-of 2025-12-19T17:00:00Z
```

Ask that with and without `--as-of` and the answers differ, because Thursday's report and
Saturday's are separate rows and neither overwrote the other.

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

### RRF, not score blending

Cosine similarity and `ts_rank` are not on a comparable scale, and normalising them needs
per-query calibration that drifts as the corpus grows. Reciprocal Rank Fusion ignores
scores entirely and uses only rank position: each retriever contributes `1/(k + rank)` to
every chunk it returns. No tuning, and one retriever returning confident nonsense cannot
poison the merge.

`k` stays at 60, the value from Cormack et al. Fitting it to 20 questions would be fitting
it to noise.

`hybrid_search` takes the embedder as an argument rather than importing it, so the module
imports on a machine where `fastembed` won't install and degrades to lexical-only instead
of raising.

### The gold set matches on content, not chunk ids

The extractor changed four times in one week and chunk ids churned every time. A gold set
pinned to ids would have needed rewriting at each step — and would have measured nothing
in between. Questions assert on strings:

```json
{ "must_contain": ["Cameron Latu", "Thursday", "FULL"] }
```

That survives re-chunking, re-embedding and extractor changes. It's also deliberately
strict about *which* row: Latu appears three times with three different statuses, so
"right player, wrong day" scores as a miss. A looser predicate would report success while
answering the wrong question, which is how eval harnesses come to lie.

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

### The injury table is client-rendered — the archive is in the news articles

The obvious source, `/team/injury-report/` on a club site, returns the week selector and
the legend and **no player rows, ever**. The table is rendered client-side. Verified
against two clubs, in and out of season.

This is the failure mode worth dwelling on: a collector pointed at that URL fetches 200,
parses successfully, dedups cleanly, and reports healthy forever while storing page
furniture. Nothing alerts. You find out in week six.

What *is* server-rendered is each club's weekly injury-report news article — and it's
richer than the widget: one article carries both teams and the full Tuesday → Wednesday →
Thursday progression. So sources come in two shapes now. An **index source** is polled on
a cadence, and the articles it links are fetched once each and never re-polled, because an
article is immutable after publication. Snapshot sources ask *"has this URL changed?"*;
index-discovered articles ask *"is this URL new?"* — a different question needing a
different dedup path.

Coverage is uneven and worth knowing before relying on it: of 32 clubs, four curate a
season archive on that page (ATL 25 articles, PHI 25, SF 18, PIT 4). `run discover` reports
what each index actually links, storing nothing, so this is measured rather than assumed.

### Rows are reconstructed from prose, then handed to the chunker unchanged

Those articles contain no `<table>` markup — the data is lines under headings:

```
Thursday's Injury Report
Eagles Injury Report
Out
DT Jalen Carter (Shoulders/Did Not Participate)
```

`ingest/report.py` rebuilds rows from that text and emits **the same shape `parse_html`
produces for real tables**, so the row-per-player chunker consumes it with no changes at
all.

Team headings match against the 32 real club nicknames rather than a regex. Three
successive patterns got this wrong: `^[A-Z]` silently dropped every `49ers` heading,
end-anchoring missed `Injury Report Ahead of Week 12`, and neither handled Pittsburgh's
`Week 17 Injury Report (Browns)`. A closed vocabulary handles all three and — more
importantly — **cannot invent a team that doesn't exist**. A malformed heading yields
nothing rather than something plausible like "Week 17" that would never join to anything.

Roughly 8% of rows end up with no team, from blocks that genuinely name none. They're left
empty. A wrong team is worse than a missing one: it embeds, it ranks, it gets cited, and it
answers confidently about the wrong player.

### Extraction is frozen at ingest, so `reparse` exists

`parse()` runs once and its output is persisted. Improving a parser therefore changes
nothing for documents already collected — re-chunking just replays the stored tables. This
was discovered the hard way, by "fixing" an extractor three times and getting
byte-identical audit output each time.

`run reparse` re-runs the parsers over the original bytes in `raw_ref` and refreshes the
derived fields. `content_hash`, `knowledge_time` and the raw files are untouched.

Which is what `raw_ref` was always for — but storing the originals is only half of it. The
command to use them has to exist too.

### `audit` counts what sampling only hints at

Retrieval hides bad extraction: a chunk with the wrong team still embeds, still ranks,
still gets cited. `make audit` reports unattributed rows, prose fallbacks, and distinct
teams seen. That last one is a canary — there are 32 clubs, so a number far above it means
headings are being misread as team names.

### PDFs use pdfplumber, not docling

docling is better on complex layout but depends on torch — same portability wall.
Practice reports are plain tables and pdfplumber handles them. Revisit on the server if
gamebooks prove too much.

---

## Sources

Official club practice and injury reports, inactives, transactions, press conference
transcripts, published depth charts. **Official and public feeds only** — no paywalled
scraping.

Fetching is paced to one request per second per host. Conditional requests keep repeat
polls of a single URL cheap, but a backfill walks dozens of *new* article URLs per club
where ETags buy nothing — so the throttle is per host, and different clubs don't block
each other.

## Stack

Python 3.13 · FastAPI · SQLAlchemy 2 · Alembic · Postgres 17 + pgvector · APScheduler ·
pdfplumber · trafilatura · selectolax · fastembed (ONNX) · uv · ruff · mypy · pytest ·
Docker

## Licence

MIT
