"""Chunking and embedding CLI.

uv run python -m omaha.retrieve.run chunk          # chunk unchunked documents
uv run python -m omaha.retrieve.run embed          # embed unembedded chunks
uv run python -m omaha.retrieve.run status         # coverage
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import func, select

from omaha.db.models import Chunk, Document
from omaha.db.session import session_scope
from omaha.retrieve.chunk import chunk_document
from omaha.retrieve.embed import EMBEDDING_VERSION, MODEL_NAME, embed_all


def cmd_chunk(args: argparse.Namespace) -> int:
    """Chunk documents that have no chunks yet.

    Documents are immutable once written, so chunking is a pure function of a document
    and can be re-run safely: we only touch documents with zero chunks.
    """
    with session_scope() as session:
        query = select(Document).order_by(Document.id).limit(args.limit)

        if args.rechunk:
            # Chunking is a pure function of an immutable document, so discarding and
            # recomputing is always safe — and it's the normal path whenever the
            # extractor improves. Embeddings go with them: a vector for text that no
            # longer exists is worse than no vector, because it still ranks.
            if args.doc_type:
                query = query.where(Document.doc_type == args.doc_type)
            targets = session.scalars(query).all()
            ids = [d.id for d in targets]
            if ids:
                deleted = (
                    session.query(Chunk)
                    .filter(Chunk.document_id.in_(ids))
                    .delete(synchronize_session=False)
                )
                print(f"cleared {deleted} chunks from {len(ids)} documents")
            documents = targets
        else:
            chunked_ids = select(Chunk.document_id).distinct().scalar_subquery()
            documents = session.scalars(query.where(Document.id.not_in(chunked_ids))).all()

        if not documents:
            print("nothing to chunk")
            return 0

        total = 0
        for document in documents:
            tables = (
                (document.parsed_tables or {}).get("tables") if document.parsed_tables else None
            )
            drafts = chunk_document(
                doc_type=document.doc_type,
                text=document.parsed_text or "",
                tables=tables,
            )
            for draft in drafts:
                session.add(
                    Chunk(
                        document_id=document.id,
                        ordinal=draft.ordinal,
                        text=draft.text,
                        section_path=draft.section_path,
                        span_start=draft.span_start,
                        span_end=draft.span_end,
                    )
                )
            total += len(drafts)
            print(f"doc {document.id} ({document.doc_type}): {len(drafts)} chunks")

        session.flush()
        print(f"-- {total} chunks from {len(documents)} documents")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    with session_scope() as session:
        count = embed_all(session, batch=args.batch)
    print(f"embedded {count} chunks with {MODEL_NAME} ({EMBEDDING_VERSION})")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Print chunk text so a human can check it says what it should.

    Counts prove something was produced, not that it was produced correctly — a chunker
    that emitted one chunk per character also produced impressive counts. This is the
    step where you read the output.
    """
    from sqlalchemy import func

    with session_scope() as session:
        query = select(Chunk).join(Document)
        if args.doc_type:
            query = query.where(Document.doc_type == args.doc_type)
        if args.contains:
            query = query.where(Chunk.text.ilike(f"%{args.contains}%"))

        chunks = session.scalars(query.order_by(func.random()).limit(args.limit)).all()
        if not chunks:
            print("no chunks matched")
            return 1

        for chunk in chunks:
            flag = "" if chunk.embedding_version else "  [unembedded]"
            print(f"\ndoc {chunk.document_id} #{chunk.ordinal}  {chunk.section_path}{flag}")
            print(f"  {chunk.text[:400]}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Extraction quality, counted rather than eyeballed.

    Retrieval hides bad extraction: a chunk with the wrong team still embeds, still
    ranks, and still gets cited — it just answers the question incorrectly. Sampling
    finds the obvious cases; this counts them.
    """
    from collections import Counter

    with session_scope() as session:
        chunks = session.scalars(
            select(Chunk).join(Document).where(Document.doc_type == "injury_report")
        ).all()

    structured = [c for c in chunks if c.section_path and "table" in c.section_path]
    prose = [c for c in chunks if c not in structured]

    missing_team = [c for c in structured if not c.text.startswith("Team: ")]
    missing_day = [c for c in structured if "| Day: " not in c.text]
    no_status = [
        c for c in structured if "| Practice: " not in c.text and "| Status: " not in c.text
    ]

    teams = Counter(
        c.text.split("|", 1)[0].removeprefix("Team: ").strip()
        for c in structured
        if c.text.startswith("Team: ")
    )

    total = len(structured) or 1
    print(f"injury_report chunks   {len(chunks)}")
    print(f"  structured (rows)    {len(structured)}")
    print(f"  prose fallback       {len(prose)}   <- extractor did not fire")
    print()
    print(f"  missing team         {len(missing_team):>5}  ({len(missing_team) / total:.1%})")
    print(f"  missing day          {len(missing_day):>5}  ({len(missing_day) / total:.1%})")
    print(f"  no practice/status   {len(no_status):>5}  ({len(no_status) / total:.1%})")
    print(f"\n  distinct teams seen  {len(teams)}")

    for team, count in teams.most_common(args.limit):
        print(f"    {team:<28} {count:>5}")

    if len(teams) > 40:
        print("\n  more than 40 distinct 'teams' — headings are being misread as team names")

    return 0


def cmd_status(_: argparse.Namespace) -> int:
    with session_scope() as session:
        docs = session.scalar(select(func.count()).select_from(Document)) or 0
        chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
        embedded = (
            session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.embedding_version == EMBEDDING_VERSION)
            )
            or 0
        )
        chunked_docs = session.scalar(select(func.count(func.distinct(Chunk.document_id)))) or 0

    print(f"documents        {docs}")
    print(f"  chunked        {chunked_docs}")
    print(f"chunks           {chunks}")
    print(f"  embedded ({EMBEDDING_VERSION})  {embedded}")
    if chunks:
        print(f"  coverage       {embedded / chunks:.1%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omaha.retrieve.run")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chunk = sub.add_parser("chunk", help="chunk documents that have no chunks")
    p_chunk.add_argument("--limit", type=int, default=200)
    p_chunk.add_argument(
        "--rechunk",
        action="store_true",
        help="discard existing chunks and recompute — use after changing the extractor",
    )
    p_chunk.add_argument("--doc-type", default=None, help="restrict --rechunk to one doc type")
    p_chunk.set_defaults(func=cmd_chunk)

    p_embed = sub.add_parser("embed", help="embed chunks missing the current version")
    p_embed.add_argument("--batch", type=int, default=500)
    p_embed.set_defaults(func=cmd_embed)

    p_sample = sub.add_parser("sample", help="print chunk text to eyeball it")
    p_sample.add_argument("--limit", type=int, default=5)
    p_sample.add_argument("--doc-type", default=None, help="injury_report | inactives | ...")
    p_sample.add_argument("--contains", default=None, help="only chunks containing this text")
    p_sample.set_defaults(func=cmd_sample)

    p_audit = sub.add_parser("audit", help="extraction quality, counted")
    p_audit.add_argument("--limit", type=int, default=40)
    p_audit.set_defaults(func=cmd_audit)

    sub.add_parser("status", help="chunk and embedding coverage").set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
