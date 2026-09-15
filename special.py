"""Defences, projected from the market rather than from usage.

Why this exists
---------------
A classic lineup requires exactly one defence. The player model fits skill
positions from usage history, so every defence was filtered out of the pool -
and the integer program then reported, correctly, that no legal lineup could be
built. The main slate type was unbuildable, and the tell was quiet: ownership
summed to 800% against nine roster slots instead of 900%, because the hundred
points belonging to the defence had nowhere to go.

A defence cannot be projected from usage because nflverse's weekly player file
has no defence rows at all. But it can be projected from the market, and the
market is unusually well suited to it: most of what a defence scores is decided
by how many points the other team puts up, and that is precisely what an
implied team total is.

The model
---------
Two independent pieces, simulated rather than averaged:

* **Points allowed.** Drawn around the opponent's implied total. NFL team
  scores have a standard deviation near ten, which matters enormously here
  because DraftKings pays defences on a STEP function - ten points for a
  shutout, minus four for conceding thirty-five. A defence facing a team
  implied for twenty is not worth the average of the tiers around twenty; it is
  worth the probability-weighted mix of them, and only a distribution can say
  what that is.

* **Splash plays.** Sacks, interceptions, fumble recoveries and the occasional
  return touchdown, drawn from Poisson counts at roughly league rates. These do
  not depend much on the opponent's total and are what give a defence its
  ceiling - a defensive touchdown is six points that no projection of points
  allowed will ever see coming.

What this deliberately does not do
----------------------------------
It does not try to rate defences against each other. A good defence facing a
bad offence and a bad defence facing the same offence get the same projection
here, because the only inputs are the market's view of the game. That is a real
limitation and it is the right one to accept for now: the market already prices
defensive quality into the total, so the double-count risk of adding a defensive
rating on top is larger than the signal.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

# DraftKings pays points allowed in tiers. This is a step function, which is
# exactly why the projection has to be simulated: the expected payout at an
# implied total of 20.4 is nothing like the payout at exactly 20.
DK_POINTS_ALLOWED = [
    (0, 0, 10.0), (1, 6, 7.0), (7, 13, 4.0), (14, 20, 1.0),
    (21, 27, 0.0), (28, 34, -1.0), (35, 999, -4.0),
]
FD_POINTS_ALLOWED = [
    (0, 0, 10.0), (1, 6, 7.0), (7, 13, 4.0), (14, 20, 1.0),
    (21, 27, 0.0), (28, 34, -1.0), (35, 999, -4.0),
]

# Per-game league rates, and what each is worth.
RATES = {"sack": (2.30, 1.0), "interception": (0.78, 2.0),
         "fumble": (0.55, 2.0), "safety": (0.05, 2.0),
         "touchdown": (0.11, 6.0)}

# The spread of an NFL team's score around its implied total. Close to ten in
# every season anyone has measured it; the step function makes this number
# matter more than the mean it is spread around.
SCORE_SD = 9.6


def _tier(points_allowed: np.ndarray, table) -> np.ndarray:
    out = np.zeros_like(points_allowed, dtype=float)
    for lo, hi, value in table:
        out[(points_allowed >= lo) & (points_allowed <= hi)] = value
    return out


def simulate_defence(opponent_total: float, site: str = "dk",
                     n: int = 20000,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """One defence's fantasy score, n times."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    table = DK_POINTS_ALLOWED if site == "dk" else FD_POINTS_ALLOWED

    conceded = rng.normal(float(opponent_total), SCORE_SD, n)
    # A team cannot score negative points, and scores land on the football
    # lattice rather than the real line. Rounding matters because the tiers
    # break at integers: 20.6 and 21.0 are a point apart in payout.
    conceded = np.clip(np.rint(conceded), 0, None)

    total = _tier(conceded, table)
    for name, (rate, value) in RATES.items():
        total = total + value * rng.poisson(rate, n)
    return total


def project(pool: pd.DataFrame, lines: pd.DataFrame, site: str = "dk",
            quantiles: list[float] | None = None) -> pd.DataFrame:
    """Fitted quantiles for every defence in the pool.

    `lines` is the upcoming-week market table: one row per team with the
    opponent's implied total. A defence with no line gets no projection rather
    than a guess - an unmatched defence is better left out of a lineup than
    rostered on a number nobody computed.
    """
    quantiles = quantiles or config.QUANTILES
    dst = pool[pool["position"].astype(str).str.upper().isin(("DST", "D", "DEF"))]
    if dst.empty:
        return pd.DataFrame()

    by_team = {}
    if lines is not None and len(lines):
        for r in lines.itertuples(index=False):
            by_team[str(r.team)] = float(r.opponent_implied)

    rng = np.random.default_rng(config.RANDOM_SEED + 7)
    rows, missing = [], []
    for r in dst.itertuples(index=False):
        team = str(r.team)
        opp_total = by_team.get(team)
        if opp_total is None or not np.isfinite(opp_total):
            missing.append(team)
            continue
        draws = simulate_defence(opp_total, site, rng=rng)
        q = np.quantile(draws, quantiles)
        row = {"name": r.name, "position": "DST", "team": team,
               "opponent_implied": round(opp_total, 2)}
        for value, qq in zip(q, quantiles):
            row[f"q{int(qq * 100)}"] = round(float(value), 2)
        row["median"] = round(float(np.median(draws)), 2)
        row["mean"] = round(float(draws.mean()), 2)
        row["ceiling"] = round(float(np.quantile(draws, quantiles[-1])), 2)
        row["spread"] = round(float(np.quantile(draws, 0.9)
                                    - np.quantile(draws, 0.1)), 2)
        rows.append(row)

    if missing:
        log.warning("no market line for %d defences (%s) - they are left "
                    "unprojected rather than guessed",
                    len(missing), ", ".join(missing[:8]))
    out = pd.DataFrame(rows)
    if len(out):
        log.info("projected %d defences from the market, opponent totals "
                 "%.1f to %.1f", len(out), out["opponent_implied"].min(),
                 out["opponent_implied"].max())
    return out


def upcoming_lines(schedules: pd.DataFrame) -> pd.DataFrame:
    """Each team's next unplayed game, with the OPPONENT's implied total.

    The opponent's number is the one that matters for a defence, and getting
    this the wrong way round would hand every defence its own offence's
    expectation - a sign error that produces plausible-looking numbers and is
    wrong in every single row.
    """
    df = schedules.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    season = df["season"].max()
    df = df[df["season"] == season]

    # The opponent's implied total is this team's game total minus its own.
    df["opponent_implied"] = df["game_total"] - df["implied_total"]
    df = df.dropna(subset=["opponent_implied"])
    if df.empty:
        return df

    # "Next" is the earliest week not already covered by results. Without a
    # result column, the latest week carrying a line is the best available
    # proxy and is right during a live week.
    latest = df["week"].max()
    # Every market column travels, not just the ones the defence model needs.
    # Returning a subset meant the projection refresh DROPPED team_spread and
    # is_home instead of updating them - two features silently deleted from
    # the frame at prediction time, which is worse than leaving them stale.
    want = ["season", "week", "team", "opponent", "implied_total",
            "opponent_implied", "game_total", "team_spread", "is_home"]
    out = (df[df["week"] == latest]
           [[c for c in want if c in df.columns]]
           .drop_duplicates("team")
           .reset_index(drop=True))
    out.attrs["week"] = int(latest) if pd.notna(latest) else 0
    return out
