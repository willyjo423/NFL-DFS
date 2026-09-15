"""What a player is likely to do, built only from what was known beforehand.

The one rule
------------
Every feature for week W is computed from weeks strictly before W. Not "mostly
before" - strictly. A season-long target share silently includes the week being
predicted, and a model trained on it looks superb in a backtest and collapses
live, because in the real Sunday the number it leans on does not exist yet.

`test_features.py` asserts this by deleting the future and checking the
features are bit-identical. That test is the reason to trust anything here.

Why volume and not efficiency
-----------------------------
Touchdowns and yards-per-catch bounce around week to week; targets and carries
do not. A receiver who saw nine targets last week will probably see something
like nine again, and the points follow from that. So the features are
overwhelmingly about opportunity - how much of his team's work a player is
getting - and only lightly about how well he converted it.

The weighting reflects the same thing. Usage is exponentially weighted with a
short half-life because a role change matters more than a season average: a
back who took over three weeks ago is a different player from his own
season-to-date line.

Team context
------------
A player's ceiling is mostly his team's. Ideally that means the implied team
total from a betting line, which is what the live run will eventually use. In
training it cannot: there is no historical line archive here yet. So team
context is built from the team's own recent offensive volume and the
opponent's recent volume allowed, both of which are in the same data and both
of which are available for every week back to 1999.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
import scoring

log = logging.getLogger(__name__)

# The raw volume a projection is built on. Everything else is derived.
USAGE = ["targets", "carries", "receptions", "attempts",
         "passing_yards", "rushing_yards", "receiving_yards"]
SHARES = ["target_share", "air_yards_share", "wopr"]

FEATURES = (
    [f"ewm_{c}" for c in USAGE]
    + [f"ewm_{c}" for c in SHARES]
    + ["ewm_points", "sd_points", "games_played", "ewm_touches",
       "team_ewm_pass_yards", "team_ewm_rush_yards", "team_ewm_points",
       "opp_ewm_points_allowed", "opp_ewm_pass_yards_allowed",
       "opp_ewm_rush_yards_allowed", "share_of_team_touches",
       # The market's view of the game, which is not leakage: a closing line
       # exists before kickoff, so it is available at the moment a projection
       # is made. It is also the only forward-looking input in the whole set -
       # every other feature describes what a player has already done, and so
       # cannot know about a new starting quarterback, a blowout script, or a
       # game expected to be played in the rain.
       "implied_total", "game_total", "team_spread",
       "is_home"]
)


def _ewm(s: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted mean of everything before this row.

    `shift(1)` is the whole point: without it the current week is inside its
    own feature. It is one character and it is the difference between a model
    and a leak.
    """
    return s.shift(1).ewm(halflife=halflife, min_periods=1).mean()


def attach_market(df: pd.DataFrame, lines: pd.DataFrame | None) -> pd.DataFrame:
    """Join each player-week to his game's market line.

    Kept separate from the rest of the build so it can be skipped entirely -
    the offline tests run on a synthetic league with no betting market, and a
    build that REQUIRED lines could not be tested without inventing them.
    Missing lines become missing values rather than zeros: a game total of
    zero is a claim that nobody will score, which is not what "we do not know"
    means, and the model would learn from it.
    """
    market = ["implied_total", "game_total", "team_spread", "is_home"]
    if lines is None or lines.empty:
        for c in market:
            if c not in df.columns:
                df[c] = np.nan
        return df

    keep = ["season", "week", "team"] + market
    lines = lines[[c for c in keep if c in lines.columns]].drop_duplicates(
        ["season", "week", "team"])
    for frame in (df, lines):
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
        frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
        frame["team"] = frame["team"].astype(str)

    before = len(df)
    # Any market column already present is dropped first, so a second call
    # cannot produce implied_total_x and implied_total_y and quietly leave the
    # real one out of the feature frame.
    df = df.drop(columns=[c for c in market if c in df.columns], errors="ignore")
    df = df.merge(lines, on=["season", "week", "team"], how="left")
    assert len(df) == before, "the market join duplicated player-weeks"

    hit = float(df["implied_total"].notna().mean())
    log.info("market lines joined to %.0f%% of player-weeks", 100 * hit)
    if hit < 0.5:
        log.warning("fewer than half the player-weeks found a market line - "
                    "check that team abbreviations agree between the schedule "
                    "and the player file")
    return df


def build(weeks: pd.DataFrame, site: str = "dk",
          lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player-week, with the target and every feature."""
    df = weeks.copy()
    df = attach_market(df, lines)
    for c in USAGE + SHARES:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["points"] = scoring.score(df, site)
    df["touches"] = df["targets"] + df["carries"] + df["attempts"]
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    g = df.groupby("player_id", sort=False)
    hl = config.USAGE_HALFLIFE_GAMES
    for c in USAGE + SHARES:
        df[f"ewm_{c}"] = g[c].transform(lambda s: _ewm(s, hl))
    df["ewm_points"] = g["points"].transform(lambda s: _ewm(s, hl))
    df["ewm_touches"] = g["touches"].transform(lambda s: _ewm(s, hl))
    # Dispersion, because a boom-or-bust player and a metronome with the same
    # average are worth different amounts once bonuses and tournaments are in
    # play. Expanding rather than rolling, so early weeks are not thrown away.
    df["sd_points"] = g["points"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std())
    df["games_played"] = g.cumcount()

    df = _team_context(df)
    df["is_home"] = _home_flag(df)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _team_context(df: pd.DataFrame) -> pd.DataFrame:
    """The team's recent offence, and what the opponent has been giving up.

    Built by summing the players, because team-level rows are a separate
    download and this is the same information. Both sides are lagged the same
    way the player features are - the opponent's defensive record must not
    include the game being predicted either, which is the leak people miss
    because they are watching the offensive side.
    """
    if "team" not in df.columns:
        df["team_ewm_pass_yards"] = np.nan
        df["team_ewm_rush_yards"] = np.nan
        df["team_ewm_points"] = np.nan
        df["opp_ewm_points_allowed"] = np.nan
        df["opp_ewm_pass_yards_allowed"] = np.nan
        df["opp_ewm_rush_yards_allowed"] = np.nan
        df["share_of_team_touches"] = np.nan
        return df

    team = (df.groupby(["team", "season", "week"], as_index=False)
              .agg(pass_yards=("passing_yards", "sum"),
                   rush_yards=("rushing_yards", "sum"),
                   points=("points", "sum"),
                   touches=("touches", "sum"))
              .sort_values(["team", "season", "week"]))
    hl = config.USAGE_HALFLIFE_GAMES
    tg = team.groupby("team", sort=False)
    for src, dest in (("pass_yards", "team_ewm_pass_yards"),
                      ("rush_yards", "team_ewm_rush_yards"),
                      ("points", "team_ewm_points")):
        team[dest] = tg[src].transform(lambda s: _ewm(s, hl))
    team["team_ewm_touches"] = tg["touches"].transform(lambda s: _ewm(s, hl))

    out = df.merge(
        team[["team", "season", "week", "team_ewm_pass_yards",
              "team_ewm_rush_yards", "team_ewm_points", "team_ewm_touches"]],
        on=["team", "season", "week"], how="left")

    # The defensive side is the same table read from the other direction: what
    # each team's opponents have managed against it.
    if "opponent" in out.columns:
        allowed = (df.groupby(["opponent", "season", "week"], as_index=False)
                     .agg(points=("points", "sum"),
                          pass_yards=("passing_yards", "sum"),
                          rush_yards=("rushing_yards", "sum"))
                     .rename(columns={"opponent": "team"})
                     .sort_values(["team", "season", "week"]))
        ag = allowed.groupby("team", sort=False)
        for src, dest in (("points", "opp_ewm_points_allowed"),
                          ("pass_yards", "opp_ewm_pass_yards_allowed"),
                          ("rush_yards", "opp_ewm_rush_yards_allowed")):
            allowed[dest] = ag[src].transform(lambda s: _ewm(s, hl))
        out = out.merge(
            allowed[["team", "season", "week", "opp_ewm_points_allowed",
                     "opp_ewm_pass_yards_allowed",
                     "opp_ewm_rush_yards_allowed"]]
            .rename(columns={"team": "opponent"}),
            on=["opponent", "season", "week"], how="left")
    else:
        for c in ("opp_ewm_points_allowed", "opp_ewm_pass_yards_allowed",
                  "opp_ewm_rush_yards_allowed"):
            out[c] = np.nan

    # How much of his team's work a player is getting. This is the single most
    # informative feature in the set: it separates a back who is the offence
    # from one who happens to play for a good one.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["share_of_team_touches"] = np.where(
            out["team_ewm_touches"] > 0,
            out["ewm_touches"] / out["team_ewm_touches"], np.nan)
    return out


def _home_flag(df: pd.DataFrame) -> pd.Series:
    """Home or away, from whichever column carries it.

    This read the column as a STRING and tested membership in ("1", "true",
    "home"). Once the schedule join began supplying a numeric flag, float 1.0
    stringified to "1.0" - which is not "1" - so every row failed the test and
    came out 0. Not missing: zero. The feature became the constant claim that
    no team is ever at home, and the model would have learned from it, because
    a column of zeros looks exactly like real data and nothing raises.

    So numbers are read as numbers, text as text, and anything unrecognised
    becomes missing rather than false. "We do not know" and "no" are different
    answers, and only one of them is safe to guess at.
    """
    for c in ("is_home", "home_away", "location"):
        if c not in df.columns:
            continue
        col = df[c]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().any():
            return num.where(num.isna(), (num > 0).astype(float)).astype(float)
        s = col.astype(str).str.strip().str.lower()
        home = s.isin(("true", "home", "h"))
        away = s.isin(("false", "away", "a", "neutral"))
        if (home | away).any():
            return pd.Series(np.where(home, 1.0, np.where(away, 0.0, np.nan)),
                             index=df.index, dtype=float)
    return pd.Series(np.nan, index=df.index)


def trainable(df: pd.DataFrame, positions: list[str] | None = None
              ) -> pd.DataFrame:
    """Rows with enough history behind them to be worth fitting on.

    A player's first two games carry no usable usage signal, and including
    them teaches the model that the features are noise. They are excluded from
    training and still projected at prediction time, from position and team
    context alone - which is the honest answer for a rookie's debut.
    """
    positions = positions or config.SKILL_POSITIONS
    out = df[df["position"].isin(positions)]
    out = out[out["games_played"] >= config.MIN_PRIOR_GAMES]
    return out.dropna(subset=["points"]).reset_index(drop=True)
