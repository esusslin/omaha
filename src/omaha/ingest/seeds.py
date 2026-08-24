"""The club source registry.

All 32 clubs run the same CMS, so the injury-report index lives at the same path on
every domain. What differs is what each club chooses to list there: some curate a full
season of report articles, others show generic latest-news. Coverage is therefore
uneven by design, and `omaha.ingest.run discover --dry-run` is the way to find out which
clubs are actually productive rather than assuming.

**These are index pages, not documents.** Fetching one yields the week selector and the
legend and nothing else — the table on that page is rendered client-side and never
appears in the HTML. Verified against two clubs. Registering these URLs as ordinary
sources is the trap: the collector reports healthy forever and stores page furniture.
They are useful only for the article links they carry.

Cadence is deliberately slow. An index gains an entry a few times a week during the
season; polling it hourly is impolite and buys nothing. The *articles* it points at are
immutable once published, so there is nothing to re-poll at all.
"""

from __future__ import annotations

from dataclasses import dataclass

INJURY_INDEX_PATH = "/team/injury-report/"
TRANSACTIONS_PATH = "/team/transactions/"
DEPTH_CHART_PATH = "/team/depth-chart/"


@dataclass(frozen=True)
class Club:
    abbr: str
    """nflverse-standard abbreviation, so joins against nflverse data are free."""
    name: str
    domain: str

    def url(self, path: str) -> str:
        return f"https://{self.domain}{path}"


CLUBS: tuple[Club, ...] = (
    Club("ARI", "Cardinals", "www.azcardinals.com"),
    Club("ATL", "Falcons", "www.atlantafalcons.com"),
    Club("BAL", "Ravens", "www.baltimoreravens.com"),
    Club("BUF", "Bills", "www.buffalobills.com"),
    Club("CAR", "Panthers", "www.panthers.com"),
    Club("CHI", "Bears", "www.chicagobears.com"),
    Club("CIN", "Bengals", "www.bengals.com"),
    Club("CLE", "Browns", "www.clevelandbrowns.com"),
    Club("DAL", "Cowboys", "www.dallascowboys.com"),
    Club("DEN", "Broncos", "www.denverbroncos.com"),
    Club("DET", "Lions", "www.detroitlions.com"),
    Club("GB", "Packers", "www.packers.com"),
    Club("HOU", "Texans", "www.houstontexans.com"),
    Club("IND", "Colts", "www.colts.com"),
    Club("JAX", "Jaguars", "www.jaguars.com"),
    Club("KC", "Chiefs", "www.chiefs.com"),
    Club("LA", "Rams", "www.therams.com"),
    Club("LAC", "Chargers", "www.chargers.com"),
    Club("LV", "Raiders", "www.raiders.com"),
    Club("MIA", "Dolphins", "www.miamidolphins.com"),
    Club("MIN", "Vikings", "www.vikings.com"),
    Club("NE", "Patriots", "www.patriots.com"),
    Club("NO", "Saints", "www.neworleanssaints.com"),
    Club("NYG", "Giants", "www.giants.com"),
    Club("NYJ", "Jets", "www.newyorkjets.com"),
    Club("PHI", "Eagles", "www.philadelphiaeagles.com"),
    Club("PIT", "Steelers", "www.steelers.com"),
    Club("SEA", "Seahawks", "www.seahawks.com"),
    Club("SF", "49ers", "www.49ers.com"),
    Club("TB", "Buccaneers", "www.buccaneers.com"),
    Club("TEN", "Titans", "www.tennesseetitans.com"),
    Club("WAS", "Commanders", "www.commanders.com"),
)

# Six hours: an index changes a few times a week in season, never in the small hours.
INDEX_CADENCE_SECONDS = 6 * 60 * 60


# Clubs whose CMS doesn't behave like the other thirty. Keyed by (abbr, kind), valued by
# the reason, because "why is this missing?" is a question someone asks six months later
# and an empty dict entry doesn't answer.
#
# Recorded rather than worked around. Detroit's 404 is a decision on their side about
# who may fetch that path; sending a browser user-agent to get past it would be evading
# a stated preference, which is a different act from reading a public page. Two of
# sixty-four sources isn't worth that, and neither club is among the four that actually
# curate an archive.
UNAVAILABLE: dict[tuple[str, str], str] = {
    ("DAL", "transactions"): (
        "no such page — the club's Team nav lists roster, depth chart, coaches, "
        "executives, stats, injury report and standings, and no transactions section"
    ),
    ("DET", "injury_index"): (
        "HTTP 404 for our user-agent, with and without the trailing slash. The same "
        "user-agent fetches det-transactions on the same domain successfully, so this "
        "is a per-path rule rather than a block on the client"
    ),
}


def is_available(abbr: str, kind: str) -> bool:
    return (abbr, kind) not in UNAVAILABLE


def unavailable_reason(abbr: str, kind: str) -> str | None:
    return UNAVAILABLE.get((abbr, kind))


@dataclass(frozen=True)
class SourceSeed:
    name: str
    kind: str
    url: str
    team: str
    cadence_seconds: int


def injury_index_seeds() -> list[SourceSeed]:
    """One index source per club, minus the ones known not to serve us."""
    return [
        SourceSeed(
            name=f"{club.abbr.lower()}-injury-index",
            kind="injury_index",
            url=club.url(INJURY_INDEX_PATH),
            team=club.abbr,
            cadence_seconds=INDEX_CADENCE_SECONDS,
        )
        for club in CLUBS
        if is_available(club.abbr, "injury_index")
    ]


def transaction_seeds() -> list[SourceSeed]:
    return [
        SourceSeed(
            name=f"{club.abbr.lower()}-transactions",
            kind="transactions",
            url=club.url(TRANSACTIONS_PATH),
            team=club.abbr,
            cadence_seconds=INDEX_CADENCE_SECONDS,
        )
        for club in CLUBS
        if is_available(club.abbr, "transactions")
    ]


def unavailable_source_names() -> dict[str, str]:
    """Source name -> reason, for sources that should exist but can't be fetched.

    Seeding skips them; `run seed` uses this to disable any that were registered before
    the exception was known. Left in the table rather than deleted so `/health` can say
    "known unavailable" instead of the source silently vanishing — a source that
    disappears looks like it was never wanted.
    """
    suffix = {"injury_index": "injury-index", "transactions": "transactions"}
    return {
        f"{abbr.lower()}-{suffix[kind]}": reason
        for (abbr, kind), reason in UNAVAILABLE.items()
        if kind in suffix
    }


def all_seeds() -> list[SourceSeed]:
    return injury_index_seeds() + transaction_seeds()


CLUBS_BY_ABBR = {club.abbr: club for club in CLUBS}
