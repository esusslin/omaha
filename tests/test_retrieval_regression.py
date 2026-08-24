"""Retrieval quality, asserted against a live Postgres.

**Why this exists.** Every other test in this suite runs on pure functions, and the one
retrieval bug that actually mattered slipped past all of them: `websearch_to_tsquery`
ANDs its terms, so a natural-language question containing any word absent from the
corpus matched *nothing*. Lexical search returned zero rows, the API returned 200, the
UI rendered an empty list, and the eval printed an identical number at k=1 and k=10 —
the signature of a retriever that isn't retrieving. It took a hand-run evaluation to
notice.

**Why a separate database.** The first version of this file ran against whatever
`DATABASE_URL` pointed at, which on a developer machine is the real corpus. Seven
fixture rows then compete with twelve hundred real chunks and lose, so the test passed
on empty CI and failed locally — a gate that only works where nobody looks at it. It
now requires `OMAHA_TEST_DATABASE_URL` and skips without one, so it is either isolated
or absent, never quietly wrong.

    docker compose exec postgres createdb -U omaha omaha_test
    OMAHA_TEST_DATABASE_URL=postgresql+psycopg://omaha:omaha_local_dev@localhost:5433/omaha_test \
        uv run pytest tests/test_retrieval_regression.py
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session, sessionmaker

from omaha.db.models import Base, Chunk, Document, Source
from omaha.eval.score import Report, evaluate_one
from omaha.retrieve.search import build_or_tsquery, lexical_search

TEST_DATABASE_URL = os.getenv("OMAHA_TEST_DATABASE_URL")

# Two knowledge times, two days apart, so the `as_of` assertions have something to cut
# between. These are when *we* learned it — never derived from the article's own date.
WEDNESDAY = dt.datetime(2025, 12, 17, 22, 0, tzinfo=dt.UTC)
FRIDAY = dt.datetime(2025, 12, 19, 22, 0, tzinfo=dt.UTC)
BETWEEN = dt.datetime(2025, 12, 18, 12, 0, tzinfo=dt.UTC)


def _reachable(url: str | None) -> bool:
    if not url:
        return False
    try:
        engine = create_engine(url)
        with engine.connect() as connection:
            connection.execute(sql_text("select 1"))
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _reachable(TEST_DATABASE_URL),
    reason="set OMAHA_TEST_DATABASE_URL to a scratch database (see module docstring)",
)


# Rows shaped like real extractor output: "POSITION Player (detail)" with practice
# columns. Wording is drawn from actual club reports so the tsvector sees realistic
# tokens rather than lorem ipsum.
WEDNESDAY_ROWS = [
    "T Trey Pipkins (Back) | Practice: DNP | Injury: Back",
    "TE Cameron Latu (Ankle) | Practice: LIMITED | Injury: Ankle",
    "QB Jayden Daniels (Elbow) | Practice: DNP | Injury: Elbow",
]
FRIDAY_ROWS = [
    "T Trey Pipkins (Back) | Practice: FULL | Injury: Back | Game Status: QUESTIONABLE",
    "TE Cameron Latu (Ankle) | Practice: FULL | Injury: Ankle | Game Status: (-)",
    "QB Jayden Daniels (Elbow) | Practice: LIMITED | Injury: Elbow | Game Status: OUT",
    "DE Jalen Carter (Foot) | Practice: DNP | Injury: Foot | Game Status: OUT",
]


@pytest.fixture(scope="module")
def schema() -> Iterator[sessionmaker[Session]]:
    """Tables in the scratch database, created once.

    `create_all` rather than alembic: this is a disposable database and the models are
    the thing under test. The vector extension has to exist first — the `embedding`
    column's type doesn't exist without it, and the error names a missing type rather
    than a missing extension.
    """
    engine = create_engine(TEST_DATABASE_URL or "", pool_pre_ping=True)
    with engine.begin() as connection:
        connection.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def corpus(schema: sessionmaker[Session]) -> Iterator[Session]:
    """A throwaway corpus, rolled back afterwards.

    Everything happens inside one transaction that is never committed. The generated
    `tsv` column is computed by Postgres on insert, so rows are searchable without a
    commit — which is what lets this stay isolated between tests.
    """
    session = schema()
    try:
        source = Source(
            name="test-regression-source",
            kind="injury_report",
            url="https://example.test/team/injury-report/",
            team="PHI",
        )
        session.add(source)
        session.flush()

        for knowledge_time, rows in ((WEDNESDAY, WEDNESDAY_ROWS), (FRIDAY, FRIDAY_ROWS)):
            document = Document(
                source_id=source.id,
                source_url=source.url,
                doc_type="injury_report",
                team="PHI",
                knowledge_time=knowledge_time,
                fetch_time=knowledge_time,
                content_hash=f"hash-{knowledge_time.isoformat()}",
                parsed_text="\n".join(rows),
            )
            session.add(document)
            session.flush()

            for ordinal, row in enumerate(rows):
                session.add(Chunk(document_id=document.id, ordinal=ordinal, text=row))

        session.flush()
        yield session
    finally:
        session.rollback()
        session.close()


# --- the canary --------------------------------------------------------------------


def test_multi_word_question_is_not_anded_into_nothing(corpus: Session) -> None:
    """The regression that started this file.

    Not one chunk contains "which", "lineman" or "hurt". Under `websearch_to_tsquery`'s
    default AND semantics this question matches zero rows — silently, with a 200 and an
    empty list. Under OR semantics `ts_rank` sorts on the terms that *do* appear.

    If this ever fails, lexical retrieval is dead and nothing downstream will say so.
    """
    hits = lexical_search(corpus, "Which lineman is hurt with a back problem?", limit=10)
    assert hits, "lexical search returned nothing — terms are being ANDed again"
    assert "Pipkins" in hits[0].text


def test_a_query_with_no_usable_terms_returns_nothing_rather_than_everything() -> None:
    """`build_or_tsquery` yields None when nothing survives term extraction.

    Punctuation and single characters are dropped, so this query has no terms at all.
    None is the right answer and callers short-circuit on it — the failure mode worth
    guarding against is building an empty tsquery, which matches *every* row and turns
    a meaningless question into a full corpus dump ranked at random.
    """
    assert build_or_tsquery("? ! a") is None


def test_a_real_term_still_builds_a_query() -> None:
    """The other half of the above: term extraction must not be so aggressive that
    ordinary questions come back empty."""
    assert build_or_tsquery("Pipkins back") is not None


# --- quality floor -----------------------------------------------------------------


GOLD = [
    {
        "id": "pipkins-back",
        "question": "Trey Pipkins back injury",
        "must_contain": ["Pipkins", "Back"],
    },
    {
        "id": "latu-practice",
        "question": "Did Cameron Latu practice fully?",
        "must_contain": ["Latu", "FULL"],
    },
    {
        "id": "carter-foot",
        "question": "Which player is out with a foot injury?",
        "must_contain": ["Carter", "Foot"],
    },
    {
        "id": "daniels-elbow",
        "question": "Jayden Daniels elbow status",
        "must_contain": ["Daniels", "Elbow"],
    },
]

MIN_HIT_AT_3 = 1.0
MIN_MRR = 0.60


def test_lexical_retrieval_meets_the_floor(corpus: Session) -> None:
    """A threshold, not a snapshot.

    Exact scores move whenever chunking or the corpus changes, and a test that pins them
    gets updated reflexively until it means nothing. A floor only fires when retrieval
    genuinely got worse, which is the only time anyone should be interrupted.
    """
    results = [
        evaluate_one(
            entry["id"],
            entry["question"],
            entry["must_contain"],
            [h.text for h in lexical_search(corpus, entry["question"], limit=10)],
        )
        for entry in GOLD
    ]
    report = Report(results)

    assert report.hit_rate_at(3) >= MIN_HIT_AT_3, (
        f"hit@3 fell to {report.hit_rate_at(3):.0%}; missed: "
        f"{[m.question_id for m in report.misses]}"
    )
    assert report.mrr >= MIN_MRR, f"MRR fell to {report.mrr:.3f} (floor {MIN_MRR})"


# --- point-in-time -----------------------------------------------------------------


def test_as_of_hides_what_was_not_yet_known(corpus: Session) -> None:
    """The bitemporal claim, asserted rather than described.

    Jalen Carter appears only in Friday's report. Asking on Wednesday must not find him
    — that's the leakage the schema exists to prevent, and the demo the README promises.
    """
    later = lexical_search(corpus, "Jalen Carter foot", limit=10)
    assert any("Carter" in h.text for h in later)

    earlier = lexical_search(corpus, "Jalen Carter foot", limit=10, as_of=BETWEEN)
    assert not any(
        "Carter" in h.text for h in earlier
    ), "as_of returned a document whose knowledge_time is in the future"


def test_as_of_returns_the_status_current_at_that_moment(corpus: Session) -> None:
    """Pipkins is DNP on Wednesday and FULL on Friday. Both rows exist; which one comes
    back depends entirely on when you ask. If this fails, the store is overwriting
    snapshots instead of appending them."""
    early = lexical_search(corpus, "Trey Pipkins back", limit=10, as_of=BETWEEN)
    assert early and "DNP" in early[0].text
    assert not any("FULL" in h.text for h in early)

    late = lexical_search(corpus, "Trey Pipkins back", limit=10)
    assert late and any("FULL" in h.text for h in late)
