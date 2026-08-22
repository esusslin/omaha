"""Why do some rows have no team?

Run:  uv run python scripts/diagnose_headings.py

Prints, for documents that produced team-less rows, the lines the extractor saw around
them. Three regexes have now been written against imagined heading shapes; this reads
the real ones.
"""

from __future__ import annotations

from sqlalchemy import select

from omaha.db.models import Document
from omaha.db.session import session_scope
from omaha.ingest.report import (
    _DAY_HEADING,
    _GROUP_LABEL,
    _PLAYER_LINE,
    extract_injury_rows,
    parse_team_heading,
)


def classify(line: str) -> str:
    if _DAY_HEADING.match(line):
        return "DAY   "
    if _GROUP_LABEL.match(line):
        return "GROUP "
    if _PLAYER_LINE.match(line):
        return "PLAYER"
    if parse_team_heading(line):
        return "TEAM  "
    return "      "


def main() -> None:
    print(
        f"sanity: parse_team_heading('49ers Injury Report') -> {parse_team_heading('49ers Injury Report')!r}"
    )
    print(
        f"        parse_team_heading('Eagles Injury Report') -> {parse_team_heading('Eagles Injury Report')!r}\n"
    )

    with session_scope() as session:
        documents = session.scalars(
            select(Document).where(Document.doc_type == "injury_report").order_by(Document.id)
        ).all()

        offenders = []
        for document in documents:
            rows = extract_injury_rows(document.parsed_text or "")
            missing = sum(1 for r in rows if not r.team)
            if missing:
                offenders.append((document, rows, missing))

        print(f"{len(offenders)} of {len(documents)} documents produce team-less rows\n")

        for document, rows, missing in offenders[:3]:
            print("=" * 78)
            print(f"doc {document.id}  {missing}/{len(rows)} rows without a team")
            print(f"  {document.source_url}")
            print("=" * 78)

            lines = [ln.strip() for ln in (document.parsed_text or "").splitlines() if ln.strip()]
            shown = 0
            for line in lines:
                kind = classify(line)
                # show every non-player line, plus the first player after each heading
                if kind != "PLAYER":
                    print(f"  {kind} {line[:96]}")
                    shown = 0
                elif shown < 1:
                    print(f"  {kind} {line[:96]}")
                    shown += 1
            print()

        # what the extractor believes it saw, overall
        all_teams: dict[str, int] = {}
        none_count = 0
        for document in documents:
            for row in extract_injury_rows(document.parsed_text or ""):
                if row.team:
                    all_teams[row.team] = all_teams.get(row.team, 0) + 1
                else:
                    none_count += 1
        print(f"teams: {sorted(all_teams)}")
        print(f"rows without team: {none_count}")


if __name__ == "__main__":
    main()
