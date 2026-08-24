"""Hybrid retrieval: dense vectors, lexical text, and Reciprocal Rank Fusion.

**Why both halves.** The corpus is two very different shapes. Structured rows read
"Team: Eagles | Day: Thursday | Pos: DT | Player: Jalen Carter | Injury: Shoulders",
where the discriminating tokens are proper nouns; Atlanta's prose reads "Drake London is
questionable for Monday night", where meaning is carried by phrasing. Dense embeddings
are good at the second and mediocre at the first — a name is a rare token that a 768-dim
average tends to smooth away. Lexical search is the reverse. Neither alone is enough.

**Why RRF rather than score blending.** Cosine similarity and `ts_rank` are not on a
comparable scale, and normalising them requires per-query calibration that drifts as the
corpus grows. RRF ignores scores entirely and uses only rank position, so it needs no
tuning and cannot be broken by one retriever returning confident nonsense. It is fifteen
lines and it is the part of this file worth being able to explain.

**Why `as_of` matters more than the ranking.** Every search can be constrained to what
was knowable at a point in time. Ask "who was questionable?" with `as_of` set to Friday
evening and you get Friday's answer, not the one contaminated by Sunday's inactives
list. That is the whole reason the store is bitemporal, and it is the thing that makes
this corpus usable for backtesting rather than just for demos.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Select, desc, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from omaha.db.models import Chunk, Document

EmbedQueryFn = Callable[[str], Sequence[float]]
"""Injected rather than imported. See `hybrid_search` for why."""

# RRF's only knob. 60 is the value from Cormack et al. (2009) and is deliberately left
# alone: the point of RRF is that it works without tuning, and a k fitted to 30 questions
# would be fitted to noise.
RRF_K = 60

DEFAULT_CANDIDATES = 50
DEFAULT_TOP_K = 10


@dataclass
class SearchHit:
    """One chunk, with enough provenance to cite it and audit how it was found."""

    chunk_id: int
    document_id: int
    text: str
    section_path: str | None
    source_url: str
    doc_type: str
    team: str | None
    knowledge_time: dt.datetime
    published_time: dt.datetime | None

    score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def found_by(self) -> str:
        if self.dense_rank is not None and self.lexical_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        return "lexical"


def _base_query(as_of: dt.datetime | None) -> Select[tuple[Chunk, Document]]:
    """Chunks joined to their document, optionally restricted to what we knew by then.

    The filter is on `knowledge_time` — when *we* learned it — not `published_time`.
    A club can publish a report at 4pm and we might not fetch it until 6pm; pretending
    otherwise would let a backtest use information it did not have.
    """
    query = select(Chunk, Document).join(Document, Chunk.document_id == Document.id)
    if as_of is not None:
        query = query.where(Document.knowledge_time <= as_of)
    return query


def _to_hit(chunk: Chunk, document: Document) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.id,
        document_id=document.id,
        text=chunk.text,
        section_path=chunk.section_path,
        source_url=document.source_url,
        doc_type=document.doc_type,
        team=document.team,
        knowledge_time=document.knowledge_time,
        published_time=document.published_time,
    )


def dense_search(
    session: Session,
    query_vector: Sequence[float],
    *,
    limit: int = DEFAULT_CANDIDATES,
    as_of: dt.datetime | None = None,
) -> list[SearchHit]:
    """Nearest neighbours by cosine distance, via the HNSW index.

    `Sequence[float]` rather than `list[float]`: callers pass whatever the embedder
    returns, and pinning the concrete type here forces a copy at every call site for no
    benefit — nothing in this function mutates the vector.
    """
    statement = (
        _base_query(as_of)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(limit)
    )
    return [_to_hit(chunk, document) for chunk, document in session.execute(statement)]


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019.-]*")


def build_or_tsquery(query_text: str) -> ColumnElement[Any] | None:
    """An OR-of-terms tsquery, or None if the query has no usable words.

    **This is the whole ballgame for lexical retrieval, and getting it wrong is silent.**
    `websearch_to_tsquery` and `plainto_tsquery` both AND their terms, so the question
    "Is Jalen Carter playing against the Commanders?" becomes
    `jalen & carter & play & against & commander`. The row it should find reads
    "Player: Jalen Carter | Injury: Shoulders | Practice: DNP | Status: OUT" — no
    "playing", no "against" — so Postgres returns nothing at all. Not a bad ranking:
    an empty result. The first evaluation showed lexical at 15% with an identical hit
    rate at k=1 and k=10, which is the signature of a retriever returning zero rows.

    OR semantics is what a lexical retriever is supposed to do — match any term, then
    let `ts_rank` reward the rows matching more of them. Same principle as BM25.

    Each term goes through `websearch_to_tsquery` individually rather than being pasted
    into `to_tsquery`, which handles stemming, stopwords and apostrophes for free and
    cannot produce a syntax error on names like Adoree' Jackson.
    """
    terms = [t for t in _WORD.findall(query_text) if len(t) > 1]
    if not terms:
        return None

    combined: ColumnElement[Any] | None = None
    for term in terms:
        part = func.websearch_to_tsquery("english", term)
        combined = part if combined is None else combined.op("||")(part)
    return combined


def lexical_search(
    session: Session,
    query_text: str,
    *,
    limit: int = DEFAULT_CANDIDATES,
    as_of: dt.datetime | None = None,
) -> list[SearchHit]:
    """Full-text search over the generated tsvector, matching any term.

    `ts_rank` does the discriminating: a row containing "Jalen" and "Carter" outranks
    one that merely contains "the".
    """
    tsquery = build_or_tsquery(query_text)
    if tsquery is None:
        return []

    rank = func.ts_rank(Chunk.tsv, tsquery)
    statement = (
        _base_query(as_of).where(Chunk.tsv.op("@@")(tsquery)).order_by(desc(rank)).limit(limit)
    )
    return [_to_hit(chunk, document) for chunk, document in session.execute(statement)]


def reciprocal_rank_fusion(
    rankings: dict[str, list[SearchHit]],
    *,
    k: int = RRF_K,
    limit: int = DEFAULT_TOP_K,
) -> list[SearchHit]:
    """Merge ranked lists by rank position alone.

    Each list contributes 1/(k + rank) to every document it returns. A chunk both
    retrievers rank highly wins; a chunk only one of them found can still surface if it
    ranked near the top. Scores from the underlying retrievers are never compared,
    which is exactly why no normalisation is needed.
    """
    totals: dict[int, float] = defaultdict(float)
    merged: dict[int, SearchHit] = {}

    for name, hits in rankings.items():
        for position, hit in enumerate(hits, start=1):
            contribution = 1.0 / (k + position)
            totals[hit.chunk_id] += contribution

            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = hit
                existing = hit
            existing.contributions[name] = contribution

            if name == "dense":
                existing.dense_rank = position
            elif name == "lexical":
                existing.lexical_rank = position

    for chunk_id, total in totals.items():
        merged[chunk_id].score = total

    ordered = sorted(merged.values(), key=lambda h: (-h.score, h.chunk_id))
    return ordered[:limit]


def hybrid_search(
    session: Session,
    query_text: str,
    *,
    limit: int = DEFAULT_TOP_K,
    candidates: int = DEFAULT_CANDIDATES,
    as_of: dt.datetime | None = None,
    embed_query_fn: EmbedQueryFn | None = None,
) -> list[SearchHit]:
    """Dense + lexical, fused.

    `embed_query_fn` is injected so this module never imports the embedding model. The
    model only installs inside the Linux container, and a search path that can't be
    imported on the host is a search path that can't be unit tested.

    With no embedder available this degrades to lexical-only rather than failing, which
    is the right behaviour: partial results beat a stack trace, and the caller can see
    what happened from `found_by`.
    """
    rankings: dict[str, list[SearchHit]] = {
        "lexical": lexical_search(session, query_text, limit=candidates, as_of=as_of)
    }

    if embed_query_fn is not None:
        vector = embed_query_fn(query_text)
        rankings["dense"] = dense_search(session, vector, limit=candidates, as_of=as_of)

    return reciprocal_rank_fusion(rankings, limit=limit)
