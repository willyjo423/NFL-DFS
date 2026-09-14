"""DraftKings and FanDuel scoring, from raw stat lines.

This has to be exactly right, because everything downstream is trained against
it. A projection model fitted to slightly wrong point totals learns a slightly
wrong game and nothing about it is loud.

So it is not taken on trust. nflverse publishes its own `fantasy_points` and
`fantasy_points_ppr` columns using standard scoring, and the two rule sets
differ in ways that are exactly enumerable:

    nflverse standard   interception -2, fumble lost -2, no bonuses, 0 PPR
    DraftKings          interception -1, fumble lost -1, three bonuses, 1 PPR
    FanDuel             interception -1, fumble lost -2, no bonuses, 0.5 PPR

Which means DK points minus nflverse PPR points must equal, to the decimal,
the interception difference plus the fumble difference plus the bonuses. That
is a test with teeth: it catches a wrong yardage rate, a missing two-point
conversion, a dropped return touchdown - anything at all - because the residual
would stop reconciling. `test_scoring.py` asserts it on every player-week in
three seasons rather than on a handful of examples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- DraftKings NFL Classic -------------------------------------------------
DK = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -1.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receptions": 1.0,            # full PPR
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "fumbles_lost": -1.0,
    "two_point_conversions": 2.0,
    "return_tds": 6.0,
}
# The bonuses are what make DraftKings a ceiling game rather than a floor game,
# and they are why a mean projection is the wrong object: three extra points
# for crossing 100 yards is a step function, so the SHAPE of a player's
# distribution changes his value, not just its middle.
DK_BONUSES = [
    ("passing_yards", 300.0, 3.0),
    ("rushing_yards", 100.0, 3.0),
    ("receiving_yards", 100.0, 3.0),
]

# --- FanDuel ----------------------------------------------------------------
FD = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -1.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receptions": 0.5,            # half PPR
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
    "two_point_conversions": 2.0,
    "return_tds": 6.0,
}
FD_BONUSES: list = []             # FanDuel has none

SITES = {"dk": (DK, DK_BONUSES), "fd": (FD, FD_BONUSES)}

# What nflverse itself uses, so the reconciliation below is exact rather than
# approximate.
NFLVERSE_STANDARD = {
    "passing_yards": 0.04, "passing_tds": 4.0, "interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receiving_yards": 0.1, "receiving_tds": 6.0,
    "fumbles_lost": -2.0, "two_point_conversions": 2.0,
}

# nflverse spells a few of these differently depending on the release. Each
# target maps to the first column that exists, and a missing one contributes
# zero rather than a NaN that would silently void a whole player's score.
ALIASES = {
    "passing_yards": ["passing_yards", "pass_yards", "passing_yds"],
    "passing_tds": ["passing_tds", "pass_tds", "pass_td"],
    "interceptions": ["passing_interceptions", "interceptions", "pass_int"],
    "rushing_yards": ["rushing_yards", "rush_yards", "rushing_yds"],
    "rushing_tds": ["rushing_tds", "rush_tds", "rush_td"],
    "receptions": ["receptions", "rec"],
    "receiving_yards": ["receiving_yards", "rec_yards", "receiving_yds"],
    "receiving_tds": ["receiving_tds", "rec_tds", "rec_td"],
    "fumbles_lost": ["sack_fumbles_lost", "rushing_fumbles_lost",
                     "receiving_fumbles_lost", "fumbles_lost"],
    "two_point_conversions": ["passing_2pt_conversions",
                              "rushing_2pt_conversions",
                              "receiving_2pt_conversions",
                              "two_point_conversions"],
    "return_tds": ["special_teams_tds", "return_tds", "kick_return_tds",
                   "punt_return_tds"],
}
# Fumbles and two-point conversions arrive split across several columns, so
# these are summed rather than first-match.
SUMMED = {"fumbles_lost", "two_point_conversions", "return_tds"}


def stat(df: pd.DataFrame, name: str) -> pd.Series:
    """One scoring input, however this release happens to spell it."""
    cols = [c for c in ALIASES.get(name, [name]) if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    if name in SUMMED:
        out = pd.Series(0.0, index=df.index)
        for c in cols:
            out = out + pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return out
    return pd.to_numeric(df[cols[0]], errors="coerce").fillna(0.0)


def score(df: pd.DataFrame, site: str = "dk") -> pd.Series:
    """Fantasy points for every row, under one site's rules."""
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}; expected one of {list(SITES)}")
    rules, bonuses = SITES[site]
    total = pd.Series(0.0, index=df.index)
    for name, weight in rules.items():
        total = total + stat(df, name) * weight
    for name, threshold, award in bonuses:
        total = total + (stat(df, name) >= threshold) * award
    return total.round(2)


def nflverse_standard(df: pd.DataFrame) -> pd.Series:
    """Recompute nflverse's own standard score, for the reconciliation test."""
    total = pd.Series(0.0, index=df.index)
    for name, weight in NFLVERSE_STANDARD.items():
        total = total + stat(df, name) * weight
    return total.round(2)


def reconcile(df: pd.DataFrame, site: str = "dk") -> pd.Series:
    """What the gap between this site and nflverse PPR *should* be.

    Used by the tests. If the measured gap ever stops matching this, something
    in the scoring is wrong - and because it is an identity rather than an
    approximation, it localises the error rather than merely flagging one.
    """
    rules, bonuses = SITES[site]
    gap = pd.Series(0.0, index=df.index)

    # Receptions: nflverse PPR pays 1.0, the site pays whatever it pays.
    gap = gap + stat(df, "receptions") * (rules["receptions"] - 1.0)
    # Interceptions and fumbles, where the penalties differ.
    for name in ("interceptions", "fumbles_lost"):
        gap = gap + stat(df, name) * (rules[name] - NFLVERSE_STANDARD[name])
    # Return touchdowns, which nflverse's standard score leaves out entirely.
    gap = gap + stat(df, "return_tds") * rules["return_tds"]
    for name, threshold, award in bonuses:
        gap = gap + (stat(df, name) >= threshold) * award
    return gap.round(2)


def cash_line(field_scores: np.ndarray, payout_fraction: float = 0.5) -> float:
    """The score a double-up lineup has to beat.

    Cash games pay a flat prize to roughly the top half, so the objective is
    not "score the most" but "clear this number" - which is a different
    optimisation and produces different lineups. Passing the realised field
    rather than assuming a fixed threshold keeps it honest week to week.
    """
    if field_scores is None or len(field_scores) == 0:
        return float("nan")
    return float(np.quantile(field_scores, 1.0 - payout_fraction))
