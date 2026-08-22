# Working in this repo

Conventions and traps. The *why* behind the architecture lives in `README.md` and in
module docstrings — this is the operational layer.

## Run things

```bash
docker compose up -d                 # Postgres 17 + pgvector on :5433
uv sync --extra dev
uv run alembic upgrade head

make dev        # API on the host — search works, lexical only
make demo       # API in the container — hybrid search at /ui
make check      # lint + typecheck + tests
```

Model work (`embed`, `eval`, `search`, `demo`) runs in the `linux/amd64` container.
x86_64 macOS has no wheels for onnxruntime past ~1.19 or torch past 2.2, so `fastembed`
is an optional `[embed]` extra installed only in the image. Code that touches the model
must degrade rather than raise when it isn't importable — see `get_embedder()` in
`search_api.py` for the pattern.

## Traps that have cost real time

**Re-stage after any pre-commit failure.** Hooks that auto-fix modify the working tree
but cannot add to the commit in flight. The commit then ships the *unfixed* version while
your terminal shows it fixed. Run `make fmt` before `git add` so hooks have nothing to
change. This has bitten twice.

**Improving a parser does nothing to existing documents.** `parse()` runs once at ingest
and its output is persisted in `parsed_text` / `parsed_tables`; `chunk` reads those
columns. To apply a parser change:

```bash
uv run python -m omaha.ingest.run reparse       # re-run parsers over raw_ref bytes
uv run python -m omaha.retrieve.run chunk --rechunk
```

Skipping `reparse` produces byte-identical output and looks like the fix failed.

**Identical output usually means the operation didn't run.** Audit numbers matching to
the digit across a change is almost never "no effect" — it's "no execution". Check that
first.

**A green pipeline over an empty pipe is the house failure mode.** Three separate bugs
here had that shape: a client-rendered table that parsed to boilerplate, a sweep filtered
on a `kind` no source had, and a lexical retriever ANDing its terms so every query
matched nothing. All reported success. Prefer checks that count *content*
(`make audit`, `make eval`) over checks that count *runs*.

**Curly apostrophes.** Club CMSes emit both `U+2019` and ASCII `'`, so regexes must
accept both — written as `’` escapes, because ruff's RUF001 rejects the literal and
the two are indistinguishable on screen.

**No `psql "$DATABASE_URL"`.** The URL carries a SQLAlchemy dialect suffix
(`postgresql+psycopg://`) that `psql` won't parse. Use the CLI commands.

## Checking your work

```bash
make audit      # extraction quality: unattributed rows, prose fallbacks, teams seen
make eval       # retrieval: lexical vs dense vs hybrid over the gold set
uv run python -m omaha.retrieve.run sample --contains "Barkley"
```

`audit` and `eval` are the honest signals. Chunk counts prove something was produced,
not that it was produced correctly — an earlier bug emitted one chunk per *character*
and the counts looked impressive.

Run `make eval` before and after any change to chunking, embedding or fusion. The point
of a number is to watch it move.

## Data rules

- **Official and public feeds only.** No paywalled scraping.
- **Never backdate `knowledge_time`.** It records when *we* learned something. Backdating
  is precisely the leakage the schema exists to prevent.
- **A wrong value is worse than a missing one.** Unattributed rows stay empty rather than
  guessing — a wrong team still embeds, still ranks, and still gets cited.
- `data/raw/` is gitignored (someone else's copyrighted pages). `data/gold/` is not — it's
  the evaluation spec.
