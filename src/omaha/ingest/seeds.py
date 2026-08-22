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


@dataclass(frozen=True)
class SourceSeed:
    name: str
    kind: str
    url: str
    team: str
    cadence_seconds: int


def injury_index_seeds() -> list[SourceSeed]:
    """One index source per club."""
    return [
        SourceSeed(
            name=f"{club.abbr.lower()}-injury-index",
            kind="injury_index",
            url=club.url(INJURY_INDEX_PATH),
            team=club.abbr,
            cadence_seconds=INDEX_CADENCE_SECONDS,
        )
        for club in CLUBS
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
    ]


def all_seeds() -> list[SourceSeed]:
    return injury_index_seeds() + transaction_seeds()


CLUBS_BY_ABBR = {club.abbr: club for club in CLUBS}
