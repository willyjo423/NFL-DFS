"""Where the numbers come from, and how the names are made to agree.

Three sources, all free, all confirmed live by `dfs_probe.py`:

* **nflverse** - per-player weekly volume and stat lines, 1999 onward, under
  `stats_player_week_{season}`. This is what the projection trains on.
* **DraftKings** - the contest list (which slate is which) and the draftables
  endpoint (who is in it and what they cost). Live only; there is no history.
* **The team model** - `predictions.json` from the NFL forecast page, carrying
  a market margin and total per game, from which each side's implied team total
  falls out.

The asset names matter
----------------------
nflverse renamed these partway through: the deep history is under
`player_stats_{year}` and recent seasons are under `stats_player_week_{year}`.
The first probe pinned one name, got a 404 for 2025, and would have quietly
trained on stale data if the failure had been less visible. So every pattern is
tried in turn and the one that answered is logged.

Names are the real risk
-----------------------
Joining DraftKings' player names to nflverse's is where a build like this
fails quietly. A 96% match looks fine and is not: the 4% that fails is
disproportionately the players who changed teams, carry a suffix, or were
signed last week - exactly the population a projection is most needed for. So
the join is measured every run, the unmatched list is kept, and a rate below
the floor raises rather than warns.
"""
from __future__ import annotations

import io
import logging
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)"}


class DataUnavailable(RuntimeError):
    """A source this build cannot proceed without did not answer."""


def _get(url: str, as_json: bool = False):
    last = None
    for attempt in range(config.MAX_RETRIES):
        try:
            r = requests.get(url, headers=UA, timeout=config.REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json() if as_json else r.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise DataUnavailable(f"{url}: {last}")


# ------------------------------------------------------------------ nflverse
def player_weeks(seasons: list[int]) -> pd.DataFrame:
    """Per-player weekly stat lines for the seasons given."""
    frames, sources = [], {}
    for season in seasons:
        for label, pattern in config.NFLVERSE_PATTERNS:
            try:
                raw = _get(pattern.format(season=season))
            except DataUnavailable:
                continue
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            if df.empty:
                continue
            df["season"] = season
            frames.append(df)
            sources[season] = label
            break
        else:
            log.error("no nflverse pattern answered for %s", season)

    if not frames:
        raise DataUnavailable(
            f"no player stats for any of {seasons}. The asset names have "
            f"probably changed again - run dfs_probe.py, which reports which "
            f"patterns answered.")
    log.info("player weeks: %s", ", ".join(f"{k}={v.split('/')[-1]}"
                                           for k, v in sources.items()))
    out = pd.concat(frames, ignore_index=True)
    return _tidy_players(out)


def _tidy_players(df: pd.DataFrame) -> pd.DataFrame:
    """One consistent set of identity columns, whatever the release called them."""
    out = df.copy()
    rename = {}
    for target, options in (
            ("player_id", ["player_id", "gsis_id", "pfr_player_id"]),
            ("name", ["player_display_name", "player_name", "full_name"]),
            ("team", ["team", "recent_team", "posteam", "club_code"]),
            ("opponent", ["opponent_team", "opponent", "defteam"]),
            ("position", ["position", "pos"]),
    ):
        for o in options:
            if o in out.columns:
                rename[o] = target
                break
    out = out.rename(columns=rename)

    missing = [c for c in ("player_id", "name", "position", "season", "week")
               if c not in out.columns]
    if missing:
        raise DataUnavailable(f"player stats lack {missing}; got "
                              f"{sorted(out.columns)[:30]}")
    out["norm"] = out["name"].map(normalise_name)
    return out


# -------------------------------------------------------------------- names
_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalise_name(name) -> str:
    """A name reduced to something two sources can agree on.

    Handles the four things that actually differ between DraftKings and
    nflverse: "Last, First" ordering, generational suffixes, punctuation in
    names like D'Andre and Amon-Ra, and casing. Deliberately not fuzzy - an
    edit-distance match would quietly pair two different players with similar
    names, which is worse than failing to match at all.
    """
    s = str(name).lower().strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    # A period becomes a SPACE, not nothing. DraftKings writes "St.Brown"
    # with no space and nflverse writes "St. Brown" with one; deleting the
    # period turns those into "stbrown" and "st brown", which do not match.
    # Amon-Ra St. Brown is a top-thirty fantasy receiver, so that single
    # character would have silently dropped him from every slate he appeared
    # in - the exact shape of failure this normalisation exists to prevent.
    s = s.replace("-", " ").replace(".", " ").replace("'", "").replace("`", "")
    s = _SUFFIXES.sub(" ", s)
    return " ".join(s.split())


def join_report(left: pd.Series, right: pd.Series) -> dict:
    """How well two name columns match, and what failed.

    Returned rather than logged, because the caller has to be able to fail on
    it. A silent join rate is the failure this whole module is arranged around.
    """
    a = {normalise_name(x) for x in pd.Series(left).dropna().unique()}
    b = {normalise_name(x) for x in pd.Series(right).dropna().unique()}
    if not b:
        return {"rate": 0.0, "matched": 0, "total": 0, "unmatched": []}
    hit = a & b
    return {
        "rate": len(hit) / len(b),
        "matched": len(hit),
        "total": len(b),
        "unmatched": sorted(b - a),
    }


_DOTNET_DATE = re.compile(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/")


def start_time(raw) -> pd.Timestamp:
    """DraftKings' contest start time, in whichever shape it arrives.

    The lobby sends a .NET JSON date - "/Date(1757894400000)/" - and the first
    version of this passed that straight to `pd.to_datetime(..., unit="ms")`,
    which cannot read it. With `errors="coerce"` every start time became NaT,
    silently: nothing raised, the column existed, and the ownership probe's
    "has this contest started" filter therefore rejected all 3,705 archived
    contests and reported that none had started. A whole capability sat blocked
    on a date parse, and the error message pointed at the wrong thing.

    Four shapes are accepted because all four have been seen from this lobby:
    the .NET wrapper with and without a timezone suffix, a raw epoch in
    milliseconds, the same as a string of digits, and an ISO timestamp.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return pd.NaT
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return pd.to_datetime(int(raw), unit="ms", utc=True, errors="coerce")

    s = str(raw).strip()
    m = _DOTNET_DATE.search(s)
    if m:
        return pd.to_datetime(int(m.group(1)), unit="ms", utc=True,
                              errors="coerce")
    if s.lstrip("-").isdigit():
        return pd.to_datetime(int(s), unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


# ----------------------------------------------------------- draftkings live
def contests() -> pd.DataFrame:
    """Every listed NFL contest, one row each, with its slate key."""
    payload = _get(config.DK_CONTESTS, as_json=True)
    rows = payload.get("Contests") or []
    if not rows:
        raise DataUnavailable("DraftKings listed no contests")
    df = pd.DataFrame([{
        "contest_id": c.get("id"),
        "name": c.get("n"),
        "draft_group": c.get("dg"),
        "game_type": c.get("gameType"),
        "entry_fee": c.get("a"),
        "entries": c.get("nt"),
        "max_entries": c.get("m"),
        "max_per_user": c.get("mec"),
        "prize_pool": c.get("po"),
        "starts_utc": start_time(c.get("sd")),
        "starts_text": c.get("sdstring"),
    } for c in rows])

    # Madden Stream slates are simulated video-game contests that sit in the
    # NFL lobby all week. They are not football and must never reach a model
    # trained on football.
    # A column of NaT is what the old parser produced, and nothing noticed for
    # a day. Saying it out loud turns the next occurrence into one line of log
    # rather than a capability that quietly does nothing.
    if len(df) and df["starts_utc"].isna().all():
        log.error("every contest start time failed to parse - sample %r. "
                  "Anything that filters on whether a contest has started will "
                  "silently match nothing.",
                  (rows[0] or {}).get("sd"))

    simulated = df["name"].astype(str).str.contains("madden", case=False,
                                                    na=False)
    if simulated.any():
        log.info("dropping %d simulated (Madden) contests", int(simulated.sum()))
    return df[~simulated].reset_index(drop=True)


def slates(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per draft group - the thing a lineup is actually built for."""
    df = contests() if df is None else df
    g = df.groupby(["draft_group", "game_type"], dropna=False)
    out = g.agg(contests=("contest_id", "size"),
                entry_fee_min=("entry_fee", "min"),
                entry_fee_max=("entry_fee", "max"),
                biggest_prize=("prize_pool", "max"),
                starts=("starts_utc", "min"),
                example=("name", "first")).reset_index()
    return out.sort_values("contests", ascending=False).reset_index(drop=True)


# Statuses that mean the player will not take a snap. Matched as substrings of
# the lowercased status, so "OUT", "Out", "O" and "Inactive" all land here.
OUT_MARKERS = ("out", "inactive", " ir", "ir ", "injured reserve", "susp",
               "pup", "nfi", "dnp", "did not", "covid", "reserve")
# Genuinely in doubt. Excluded by default because a player who does not play
# scores zero, and a zero in a six-man showdown lineup is the whole entry.
DOUBTFUL_MARKERS = ("doubtful",)
QUESTIONABLE_MARKERS = ("questionable", "gtd", "game time")
# Statuses that mean nothing is wrong. DraftKings uses several spellings of
# "no news", and treating an unknown value as OUT would empty the board.
CLEAR_MARKERS = ("", "none", "null", "active", "probable", "available", "-")


def playing_status(status, disabled=False, attributes: str = "") -> str:
    """One of: out, doubtful, questionable, clear.

    DraftKings hands this over in the same payload as the salaries and the
    first version of this module captured it and then never read it - which is
    worse than not having it, because the cheapest player on a board is very
    often cheap precisely BECAUSE he is not playing. Troy Franklin came back as
    the best value on the slate at 2.87 points per $1,000 and was inactive; a
    value column that cannot see a status column will pick him every time, for
    both objectives, which is exactly what happened.

    Unknown values resolve to "clear" on purpose. Guessing OUT from a string
    nobody recognises would quietly delete half a slate, and the caller checks
    how much of the board this filter removes before trusting it.
    """
    if disabled:
        return "out"
    s = str(status or "").strip().lower()
    a = str(attributes or "").strip().lower()
    blob = f"{s} {a}"
    if any(m in blob for m in OUT_MARKERS):
        return "out"
    if any(m in blob for m in DOUBTFUL_MARKERS):
        return "doubtful"
    if any(m in blob for m in QUESTIONABLE_MARKERS):
        return "questionable"
    if s in CLEAR_MARKERS:
        return "clear"
    # A single letter is DraftKings' shorthand; O and D are the ones that bite.
    return {"o": "out", "d": "doubtful", "q": "questionable"}.get(s, "clear")


def draftables(draft_group: int) -> pd.DataFrame:
    """Who is in a slate and what they cost.

    DraftKings repeats a player once per roster slot he is eligible for, so a
    flex-eligible receiver appears twice at the same price. Collapsing to one
    row per player is essential: an optimiser fed the raw rows can select the
    same player twice and satisfy the salary cap while fielding an illegal
    lineup.

    Showdown is the exception, and the first version of this got it wrong. On a
    showdown slate the duplicate rows are NOT the same price: the captain slot
    costs 1.5x the flex slot and pays 1.5x the points. Collapsing on roster
    slot id kept the CAPTAIN row, so every salary on the board came out 1.5x
    too high while the projection attached to it was still a flex projection -
    a value column wrong in both directions at once, and a salary cap that
    would have let an optimiser field two-thirds of a legal lineup.

    So the collapse keeps the CHEAPEST row, which is the flex price, and the
    captain price rides alongside in its own column for the optimiser to use
    when it decides who wears the C.
    """
    payload = _get(config.DK_DRAFTABLES.format(dg=draft_group), as_json=True)
    rows = payload.get("draftables") or []
    if not rows:
        raise DataUnavailable(f"draft group {draft_group} returned no players")

    df = pd.DataFrame([{
        "dk_player_id": p.get("playerId"),
        "draftable_id": p.get("draftableId"),
        "name": p.get("displayName"),
        "position": p.get("position"),
        "team": p.get("teamAbbreviation"),
        "salary": p.get("salary"),
        "roster_slot": p.get("rosterSlotId"),
        "status": p.get("status"),
        "disabled": bool(p.get("isDisabled")),
        "attributes": "|".join(
            str(a.get("name") or a.get("id") or "")
            for a in (p.get("playerAttributes") or [])),
        "game": (p.get("competition") or {}).get("name"),
        "starts_utc": pd.to_datetime(
            (p.get("competition") or {}).get("startTime"), errors="coerce"),
    } for p in rows])

    df["norm"] = df["name"].map(normalise_name)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    slots = df.groupby("dk_player_id")["roster_slot"].nunique()
    top = df.groupby("dk_player_id")["salary"].max()
    base = df.groupby("dk_player_id")["salary"].min()

    df = (df.sort_values(["dk_player_id", "salary"])
            .drop_duplicates("dk_player_id", keep="first")
            .reset_index(drop=True))
    df["flex_eligible"] = df["dk_player_id"].map(slots).gt(1).astype(int)
    df["captain_salary"] = df["dk_player_id"].map(top)

    # A showdown board is the one where the two prices differ. Saying so out
    # loud is cheap and makes the 1.5x either visible or absent in the log
    # rather than something to infer from the salary range.
    priced_twice = int((top > base).sum())
    if priced_twice:
        ratio = float((top / base.replace(0, np.nan)).median())
        log.info("showdown pricing: %d players carry a captain price, "
                 "median %.2fx the flex price", priced_twice, ratio)

    df["playing"] = [playing_status(s, d, a) for s, d, a
                     in zip(df["status"], df["disabled"], df["attributes"])]
    counts = df["playing"].value_counts().to_dict()
    log.info("availability: %s",
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if set(df["playing"]) == {"clear"} and df["status"].notna().any():
        log.info("no player carries an injury designation on this board - "
                 "normal early in a week, suspicious an hour before kickoff")

    log.info("draft group %s: %d players, flex salary $%s-$%s",
             draft_group, len(df), int(df["salary"].min()),
             int(df["salary"].max()))
    return df


@lru_cache(maxsize=1)
def schedules() -> pd.DataFrame:
    """Every game with its closing market line, one row per TEAM.

    Two rows per game, so this joins straight onto player weeks by
    (season, week, team). The implied team total is the half of the market's
    total that the spread assigns to each side - the number that says how many
    points a team is actually expected to score, which is what a player's
    ceiling is mostly made of.

    This also supplies `is_home`, which nflverse's weekly player file does not
    carry. That column arriving empty is what killed the very first live fit:
    an all-NaN feature the histogram binner could not build a threshold from.
    """
    raw = _get(config.NFLVERSE_SCHEDULES)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    need = {"season", "week", "home_team", "away_team"}
    if not need <= set(df.columns):
        raise DataUnavailable(f"schedules lack {need - set(df.columns)}")

    spread = pd.to_numeric(df.get("spread_line"), errors="coerce")
    total = pd.to_numeric(df.get("total_line"), errors="coerce")
    # spread_line is from the HOME side, so the home implied total is the
    # bigger half when the home team is favoured. Getting this backwards would
    # hand every favourite's players their opponent's expectation, which is a
    # sign error that looks plausible in aggregate and is wrong every time.
    home_implied = (total + spread) / 2.0
    away_implied = total - home_implied

    home = pd.DataFrame({
        "season": df["season"], "week": df["week"], "team": df["home_team"],
        "opponent": df["away_team"], "is_home": 1.0,
        "game_total": total, "team_spread": spread,
        "implied_total": home_implied})
    away = pd.DataFrame({
        "season": df["season"], "week": df["week"], "team": df["away_team"],
        "opponent": df["home_team"], "is_home": 0.0,
        "game_total": total, "team_spread": -spread,
        "implied_total": away_implied})

    out = pd.concat([home, away], ignore_index=True)
    out["team"] = out["team"].astype(str)
    cover = float(out["implied_total"].notna().mean())
    log.info("schedules: %d team-games, %.0f%% carry a market line",
             len(out), 100 * cover)
    if cover < 0.5:
        log.warning("most games have no market line - the implied-total "
                    "features will be mostly missing and worth little")
    return out


# --------------------------------------------------------------- team model
def team_context() -> pd.DataFrame:
    """Implied team totals, from the NFL model's published forecast.

    A player's ceiling is mostly his team's. Splitting the market total by the
    market margin gives each side's implied points, which is the single most
    useful piece of context a player projection can have - and it is already
    being published by the model next door.
    """
    payload = _get(config.TEAM_MODEL_JSON, as_json=True)
    games = payload.get("games") or []
    rows = []
    for g in games:
        total = g.get("market_total")
        margin = g.get("market_margin")
        fc = g.get("forecast") or {}
        if total is None or margin is None:
            total = total if total is not None else fc.get("total")
            margin = margin if margin is not None else fc.get("margin")
        if total is None or margin is None:
            continue
        home = (float(total) + float(margin)) / 2.0
        rows.append({"game_id": g.get("game_id"), "week": g.get("week"),
                     "team": g.get("home_team"), "opponent": g.get("away_team"),
                     "home": 1, "implied_total": round(home, 2),
                     "game_total": float(total), "margin": float(margin)})
        rows.append({"game_id": g.get("game_id"), "week": g.get("week"),
                     "team": g.get("away_team"), "opponent": g.get("home_team"),
                     "home": 0, "implied_total": round(float(total) - home, 2),
                     "game_total": float(total), "margin": -float(margin)})
    if not rows:
        raise DataUnavailable("the team model published no usable games")
    return pd.DataFrame(rows)
