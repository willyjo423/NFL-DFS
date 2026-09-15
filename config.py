"""Every tunable, with the reasoning attached.

This build answers one question per slate: given who is playing, what they
cost, and how far each of them might go, which single lineup gives the best
chance of the outcome this contest pays for.

Two things follow from that sentence and shape everything else.

**"Might go", not "will score".** DraftKings pays a three-point bonus at 100
receiving yards. That is a step function, so two players with the same mean can
have materially different value - the one with the wider distribution crosses
the threshold more often. A mean projection cannot see that. So every player is
projected as a distribution and the slate is simulated, not summed.

**"The outcome this contest pays for".** A double-up pays a flat prize to
roughly the top half; a tournament pays almost everything to the top fraction
of a percent. Those are different objectives and they produce different
lineups. Maximising expected points is correct for neither.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
SLATES = DATA / "slates"          # salaries as they were, archived per slate
PROJECTIONS = DATA / "projections"
LINEUPS = DATA / "lineups"        # every lineup built, before it was scored
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

for _p in (DATA, CACHE, SLATES, PROJECTIONS, LINEUPS, MODELS, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Sources ---------------------------------------------------------------
# nflverse renamed its assets partway through: the deep history sits under
# `player_stats_{year}`, recent seasons under `stats_player_week_{year}`. The
# first probe pinned one and got a 404 for 2025. Every pattern is tried in turn
# and the one that answered is logged, so a rename shows up as a log line
# rather than as a season of missing data nobody noticed.
NFLVERSE_PATTERNS = [
    ("stats_player/stats_player_week_{season}.csv",
     "https://github.com/nflverse/nflverse-data/releases/download/"
     "stats_player/stats_player_week_{season}.csv"),
    ("player_stats/stats_player_week_{season}.csv",
     "https://github.com/nflverse/nflverse-data/releases/download/"
     "player_stats/stats_player_week_{season}.csv"),
    ("player_stats/player_stats_{season}.csv",
     "https://github.com/nflverse/nflverse-data/releases/download/"
     "player_stats/player_stats_{season}.csv"),
]

DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=NFL"
DK_DRAFTABLES = ("https://api.draftkings.com/draftgroups/v1/draftgroups/"
                 "{dg}/draftables")

# The team model next door. It publishes a market margin and total per game,
# which is where each side's implied points come from.
# Historical market lines. The reason this matters more than it looks: the
# implied team total is the single most informative piece of context a player
# projection can have, and the version already wired up (`team_context`) only
# publishes the CURRENT week - so there was nothing to train on and the feature
# could never be learned, only bolted on afterwards. This file carries the
# lines back to 1999, which turns it into a real feature.
NFLVERSE_SCHEDULES = ("https://github.com/nflverse/nflverse-data/releases/"
                      "download/schedules/games.csv")

TEAM_MODEL_JSON = "https://willyjo423.github.io/nfl-forecast/predictions.json"

# rotoguru's historical salaries, for an optional backfill. The form's own
# fields turned out to be `game` and `gameyr`, spelled as one token - `dk2021`,
# not `game=dk&year=2021`, which is why three guesses failed. Its newest option
# is 2021, so this covers roughly 2014-2021 and nothing since. Useful for
# validating the optimiser's mechanics on real prices; useless for measuring
# edge today, because salary-setting has moved on. Live capture is the real
# archive.
ROTOGURU = "http://rotoguru1.com/cgi-bin/fyday.pl?week={week}&game={site}{season}&scsv=1"
ROTOGURU_LAST_SEASON = 2021

# --- Training --------------------------------------------------------------
# Far enough back for volume patterns to be stable, recent enough that the
# passing-game era resembles the one being predicted. Extending this is cheap
# if the walk-forward says older seasons still help.
TRAIN_START_SEASON = int(os.environ.get("TRAIN_START_SEASON", 2017))

# A player needs this many prior games before his own history means anything.
# Below it the projection leans on his position and his team's context.
MIN_PRIOR_GAMES = 3

# Volume decays in relevance. Four games is roughly where a role change stops
# being noise and starts being the new normal.
USAGE_HALFLIFE_GAMES = 4.0

# --- The projection --------------------------------------------------------
# Quantiles, not a mean. The bonuses are step functions and tournaments pay for
# ceilings, so the shape is the product and the average is a by-product.
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.97]

# Positions that get a projection at all. Kickers are excluded from DraftKings
# Classic; defences score on a different set of events entirely and get their
# own treatment.
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

# --- Simulation ------------------------------------------------------------
# How many times the slate is played out. Ten thousand is enough to resolve a
# cash line; tournament tails need more, because the quantity of interest is
# the top fraction of a percent and it is estimated from the sims that land
# there.
SIMS_CASH = int(os.environ.get("SIMS_CASH", 10_000))
SIMS_GPP = int(os.environ.get("SIMS_GPP", 50_000))

# Correlations, as starting values fitted from historical game data rather than
# assumed. They are listed here so the assumptions are visible, and every one
# is re-estimated by the build rather than trusted.
CORRELATION_PRIORS = {
    ("QB", "WR", "same_team"): 0.35,
    ("QB", "TE", "same_team"): 0.25,
    ("QB", "RB", "same_team"): 0.05,
    ("WR", "WR", "same_team"): -0.10,     # competing for the same targets
    ("QB", "WR", "opponent"): 0.15,       # shootouts lift both sides
    ("RB", "RB", "same_team"): -0.35,     # one back eats the other's carries
}

# --- Contests --------------------------------------------------------------
# Roster rules, per site and game type. These are hard constraints, not
# preferences, and getting one wrong produces a lineup that cannot be entered.
ROSTERS = {
    ("dk", "Classic"): {
        "slots": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
        "flex_positions": ["RB", "WR", "TE"],
        "salary_cap": 50_000,
        "max_per_team": 8,
    },
    ("dk", "Showdown Captain Mode"): {
        "slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"],
        "flex_positions": ["QB", "RB", "WR", "TE", "DST", "K"],
        "salary_cap": 50_000,
        "captain_multiplier": 1.5,      # 1.5x points AND 1.5x salary
        "max_per_team": 5,
    },
    ("fd", "Classic"): {
        "slots": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
        "flex_positions": ["RB", "WR", "TE"],
        "salary_cap": 60_000,
        "max_per_team": 4,
    },
}

# What fraction of a field a cash game pays. Double-ups and 50/50s pay about
# half after the rake, which is the number the lineup has to beat.
CASH_PAYOUT_FRACTION = 0.5
# And the slice a tournament lineup is actually aiming at. Optimising a GPP for
# expected points targets the middle of a distribution that pays nothing.
GPP_TARGET_FRACTION = 0.001

# --- Runtime ---------------------------------------------------------------
RANDOM_SEED = 1729
REQUEST_TIMEOUT = 45
MAX_RETRIES = 3
