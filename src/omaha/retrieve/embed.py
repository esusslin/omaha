"""Embedding pipeline.

**Why `fastembed` and not `sentence-transformers`.** sentence-transformers needs torch,
and torch dropped x86_64 macOS wheels after 2.2 — so the obvious choice doesn't install
on this machine. fastembed runs the same BGE models through ONNX Runtime with no torch
at all, and ONNX is faster on CPU, which is what the server runs anyway. The constraint
forced a better answer.

**Why `bge-base-en-v1.5` and not `bge-m3`.** M3 is 568M parameters and multilingual;
base is 109M and English-only. Practice reports are English, this runs on CPU, and base
scores within a point or two of M3 on English retrieval benchmarks. Five times the
compute for no measurable gain here.

**Model and version are stamped on every row.** Re-embedding is then additive: write the
new version alongside the old, switch reads, drop the old. Without provenance on the
vector you're doing a wipe-and-pray migration.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable, Sequence
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from omaha.db.models import Chunk

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_VERSION = "v1"
BATCH_SIZE = 64

# BGE models want a prefix on *queries* but not on stored passages. Getting this
# backwards is a quiet 5-10% retrieval regression that no test will catch.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():  # type: ignore[no-untyped-def]
    """Loaded once per process. First call downloads ~130 MB of ONNX weights."""
    from fastembed import TextEmbedding

    logger.info("loading embedding model %s", MODEL_NAME)
    return TextEmbedding(model_name=MODEL_NAME)


def embed_passages(texts: Sequence[str]) -> list[list[float]]:
    """Embed stored documents. No query prefix."""
    if not texts:
        return []
    return [vec.tolist() for vec in _model().embed(list(texts), batch_size=BATCH_SIZE)]


def embed_query(text: str) -> list[float]:
    """Embed a search query. Prefixed, per BGE's training."""
    vector: list[float] = next(iter(_model().query_embed([text]))).tolist()
    return vector


def pending_chunks(session: Session, *, limit: int = 500) -> Sequence[Chunk]:
    """Chunks not yet embedded at the current version."""
    return session.scalars(
        select(Chunk)
        .where((Chunk.embedding_version.is_(None)) | (Chunk.embedding_version != EMBEDDING_VERSION))
        .order_by(Chunk.id)
        .limit(limit)
    ).all()


def embed_pending(session: Session, *, limit: int = 500) -> int:
    """Embed a batch of outstanding chunks. Returns how many were embedded.

    Idempotent and resumable: it only picks up rows whose version doesn't match, so an
    interrupted run costs nothing but the work already done.
    """
    chunks = pending_chunks(session, limit=limit)
    if not chunks:
        return 0

    vectors = embed_passages([c.text for c in chunks])
    now = dt.datetime.now(dt.UTC)

    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
        chunk.embedding_model = MODEL_NAME
        chunk.embedding_version = EMBEDDING_VERSION
        chunk.embedded_at = now

    session.flush()
    logger.info("embedded %d chunks", len(chunks))
    return len(chunks)


def embed_all(session: Session, *, batch: int = 500, max_batches: int = 1000) -> int:
    """Drain the backlog."""
    total = 0
    for _ in range(max_batches):
        done = embed_pending(session, limit=batch)
        if done == 0:
            break
        total += done
    return total


def chunk_iter(chunks: Iterable[Chunk]) -> Iterable[str]:
    for chunk in chunks:
        yield chunk.text
