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
        "starts_utc": pd.to_datetime(c.get("sd"), unit="ms", errors="coerce"),
        "starts_text": c.get("sdstring"),
    } for c in rows])

    # Madden Stream slates are simulated video-game contests that sit in the
    # NFL lobby all week. They are not football and must never reach a model
    # trained on football.
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


def draftables(draft_group: int) -> pd.DataFrame:
    """Who is in a slate and what they cost.

    DraftKings repeats a player once per roster slot he is eligible for, so a
    flex-eligible receiver appears twice at the same price. Collapsing to one
    row per player is essential: an optimiser fed the raw rows can select the
    same player twice and satisfy the salary cap while fielding an illegal
    lineup.
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
        "game": (p.get("competition") or {}).get("name"),
        "starts_utc": pd.to_datetime(
            (p.get("competition") or {}).get("startTime"), errors="coerce"),
    } for p in rows])

    df["norm"] = df["name"].map(normalise_name)
    slots = df.groupby("dk_player_id")["roster_slot"].nunique()
    df = (df.sort_values("roster_slot")
            .drop_duplicates("dk_player_id", keep="first")
            .reset_index(drop=True))
    df["flex_eligible"] = df["dk_player_id"].map(slots).gt(1).astype(int)
    log.info("draft group %s: %d players, salary $%s-$%s",
             draft_group, len(df), df["salary"].min(), df["salary"].max())
    return df


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
