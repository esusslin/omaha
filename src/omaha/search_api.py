"""Search endpoints and a small browser UI.

Split out of `api.py` so the health/observability surface stays readable — it does a
different job and has different failure modes.

**The embedding model may not be here.** `fastembed` installs only in the Linux image,
so an API served from a macOS host has no dense retrieval. Rather than 500, search
degrades to lexical and says so in the response: every result carries `found_by`, and the
payload reports `mode` and `dense_available`. A demo that silently drops half its
retrieval is worse than one that admits it.

The model loads once at startup rather than per request — first load is ~200 MB of ONNX
weights, which is fine once and unacceptable on every query.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from omaha.db.session import get_session
from omaha.retrieve.search import (
    DEFAULT_CANDIDATES,
    EmbedQueryFn,
    SearchHit,
    dense_search,
    hybrid_search,
    lexical_search,
)

logger = logging.getLogger(__name__)

DbSession = Annotated[Session, Depends(get_session)]

router = APIRouter()


@lru_cache(maxsize=1)
def get_embedder() -> EmbedQueryFn | None:
    """The query embedder, or None where the model isn't installed.

    Cached so the ONNX weights load once per process. Returning None instead of raising
    is what lets the same code serve a full hybrid demo in the container and a
    lexical-only one on a laptop.
    """
    try:
        from omaha.retrieve.embed import embed_query

        logger.info("embedding model available — hybrid search enabled")
        return embed_query
    except Exception as exc:
        logger.warning("no embedding model (%s) — search is lexical only", type(exc).__name__)
        return None


def _serialise(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "text": hit.text,
        "score": round(hit.score, 6),
        "found_by": hit.found_by,
        "dense_rank": hit.dense_rank,
        "lexical_rank": hit.lexical_rank,
        "section_path": hit.section_path,
        "doc_type": hit.doc_type,
        "team": hit.team,
        "source_url": hit.source_url,
        "knowledge_time": hit.knowledge_time.isoformat(),
        "published_time": hit.published_time.isoformat() if hit.published_time else None,
    }


@router.get("/search")
def search(
    session: DbSession,
    q: Annotated[str, Query(min_length=2, description="Natural language question")],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    candidates: Annotated[int, Query(ge=10, le=200)] = DEFAULT_CANDIDATES,
    mode: Annotated[str, Query(pattern="^(hybrid|lexical|dense)$")] = "hybrid",
    as_of: Annotated[
        str | None, Query(description="ISO timestamp, e.g. 2025-12-19T17:00:00Z")
    ] = None,
) -> dict[str, Any]:
    """Search the corpus.

    `as_of` restricts results to what was knowable at that instant — filtering on
    `knowledge_time`, when *we* learned something, not when the club published it. Ask
    the same question with and without it and the answers differ, which is the whole
    reason the store keeps every version instead of overwriting.
    """
    when = None
    if as_of:
        try:
            when = dt.datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(422, f"as_of is not a valid ISO timestamp: {as_of}") from exc
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.UTC)

    embedder = get_embedder()
    effective = mode
    if mode in ("hybrid", "dense") and embedder is None:
        effective = "lexical"

    if effective == "lexical":
        hits = lexical_search(session, q, limit=limit, as_of=when)
    elif effective == "dense" and embedder is not None:
        # The `is not None` is redundant at runtime — the block above already downgraded
        # dense to lexical when there's no model — but mypy can't follow that across two
        # statements, and an `assert` would be a runtime cost for a compile-time problem.
        hits = dense_search(session, embedder(q), limit=limit, as_of=when)
    else:
        hits = hybrid_search(
            session, q, limit=limit, candidates=candidates, as_of=when, embed_query_fn=embedder
        )

    return {
        "query": q,
        "mode_requested": mode,
        "mode_used": effective,
        "dense_available": embedder is not None,
        "as_of": when.isoformat() if when else None,
        "count": len(hits),
        "results": [_serialise(h) for h in hits],
    }


# The demo page. Deliberately one file with no build step: a portfolio repo that needs
# npm install before anyone can look at it mostly doesn't get looked at.
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Omaha — search</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0d1117; color:#e6edf3;
         font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }
  main { max-width: 940px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  h1 { margin:0 0 .25rem; font-size:1.6rem; letter-spacing:-.02em; }
  .sub { color:#8b949e; margin:0 0 1.75rem; font-size:.92rem; }
  form { display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.75rem; }
  input,select,button { background:#161b22; color:#e6edf3; border:1px solid #30363d;
         border-radius:7px; padding:.6rem .7rem; font:inherit; }
  input[name=q] { flex:1 1 22rem; }
  input[name=as_of] { flex:0 1 15rem; }
  button { background:#238636; border-color:#238636; cursor:pointer; font-weight:600; }
  button:hover { background:#2ea043; }
  .examples { margin:0 0 1.5rem; font-size:.85rem; color:#8b949e; }
  .examples a { color:#58a6ff; cursor:pointer; margin-right:.9rem; text-decoration:none; }
  .meta { color:#8b949e; font-size:.85rem; margin:1rem 0; }
  .hit { border:1px solid #30363d; border-radius:9px; padding:.85rem 1rem; margin-bottom:.65rem;
         background:#0f141a; }
  .hit .txt { white-space:pre-wrap; }
  .tags { margin-top:.5rem; font-size:.76rem; color:#8b949e;
          display:flex; gap:.45rem; flex-wrap:wrap; align-items:center; }
  .tag { border:1px solid #30363d; border-radius:20px; padding:.1rem .55rem; }
  .both { border-color:#2ea043; color:#3fb950; }
  .dense { border-color:#8957e5; color:#a371f7; }
  .lexical { border-color:#1f6feb; color:#58a6ff; }
  a.src { color:#58a6ff; text-decoration:none; }
  .warn { border:1px solid #9e6a03; background:#221a08; color:#d29922;
          border-radius:7px; padding:.6rem .8rem; font-size:.86rem; margin-bottom:1rem; }
</style>
</head>
<body><main>
  <h1>Omaha</h1>
  <p class="sub">Hybrid retrieval over NFL injury reports — dense vectors + Postgres
     full-text, fused with Reciprocal Rank Fusion.</p>

  <div id="warn"></div>

  <form id="f">
    <input name="q" placeholder="Which Eagles lineman is out with a foot injury?" required>
    <input name="as_of" placeholder="as of (optional ISO)">
    <select name="mode">
      <option value="hybrid">hybrid</option>
      <option value="lexical">lexical</option>
      <option value="dense">dense</option>
    </select>
    <button>Search</button>
  </form>

  <p class="examples">
    <a onclick="go('Which Eagles lineman is out with a foot injury?')">foot injury</a>
    <a onclick="go('Why is the Commanders quarterback not playing?')">paraphrased</a>
    <a onclick="go('Trey Pipkins back injury')">exact name</a>
    <a onclick="go('Did Cameron Latu practice fully on Thursday?')">specific day</a>
  </p>

  <div id="meta" class="meta"></div>
  <div id="out"></div>

<script>
const f = document.getElementById('f');
function go(q){ f.q.value = q; f.requestSubmit(); }

f.addEventListener('submit', async e => {
  e.preventDefault();
  const p = new URLSearchParams({ q: f.q.value, mode: f.mode.value, limit: 10 });
  if (f.as_of.value.trim()) p.set('as_of', f.as_of.value.trim());

  document.getElementById('meta').textContent = 'searching…';
  document.getElementById('out').innerHTML = '';

  const r = await fetch('/search?' + p);
  if (!r.ok) {
    document.getElementById('meta').textContent = 'error ' + r.status + ': ' + await r.text();
    return;
  }
  const d = await r.json();

  document.getElementById('warn').innerHTML = d.dense_available ? '' :
    '<div class="warn">No embedding model in this process — dense retrieval is ' +
    'unavailable and results are lexical only. Run the API inside the worker ' +
    'container for hybrid search.</div>';

  let meta = d.count + ' results · mode ' + d.mode_used;
  if (d.mode_used !== d.mode_requested) meta += ' (requested ' + d.mode_requested + ')';
  if (d.as_of) meta += ' · as of ' + d.as_of;
  document.getElementById('meta').textContent = meta;

  document.getElementById('out').innerHTML = d.results.map(h => `
    <div class="hit">
      <div class="txt">${esc(h.text)}</div>
      <div class="tags">
        <span class="tag ${h.found_by}">${h.found_by}</span>
        <span class="tag">score ${h.score.toFixed(4)}</span>
        ${h.dense_rank ? `<span class="tag">dense #${h.dense_rank}</span>` : ''}
        ${h.lexical_rank ? `<span class="tag">lexical #${h.lexical_rank}</span>` : ''}
        <span class="tag">${esc(h.doc_type)}</span>
        <span>knew at ${esc(h.knowledge_time.slice(0,16).replace('T',' '))}</span>
        <a class="src" href="${esc(h.source_url)}" target="_blank" rel="noopener">source</a>
      </div>
    </div>`).join('') || '<p class="meta">nothing matched</p>';
});

function esc(s){ return String(s ?? '').replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
</script>
</main></body></html>
"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    """A single-file search page. No build step, no npm, no CDN."""
    return _PAGE
