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


def name_keys(name) -> set[str]:
    """Every spelling two sources might agree on, not just one.

    `normalise_name` turns a period into a SPACE, which is exactly right for
    "St.Brown" against "St. Brown" and exactly wrong for initials: DraftKings
    writes "AJ Brown" and the injury report writes "A.J. Brown", which reduce
    to "aj brown" and "a j brown" and do not match. Both rules are correct and
    they contradict each other, so both spellings are carried and a match on
    either counts.

    This is the same fix the college model needed for Matthew/Matt and
    A.J./AJ. It lives here as well because an injury report that silently
    misses A.J. Brown is worse than one that misses nobody - it reports a
    healthy board while a starter is out.
    """
    raw = str(name).strip()
    if "," in raw:
        last, _, first = raw.partition(",")
        raw = f"{first.strip()} {last.strip()}"
    keys = set()
    for dot in (" ", ""):
        t = raw.lower().replace("-", " ").replace(".", dot)
        t = t.replace("'", "").replace("`", "")
        t = _SUFFIXES.sub(" ", t)
        t = " ".join(t.split())
        if t:
            keys.add(t)
    return keys


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


def _draft_rows(draft_group: int):
    """The slate's players, from whichever DraftKings endpoint still answers.

    The api.draftkings.com draftgroups endpoint returned 403 Access Denied to
    GitHub's runners on every URL shape tried - an IP-level block, not a wrong
    path, and it had been quietly breaking every scheduled run. The lobby's own
    player endpoint lives on www.draftkings.com, the same host as the contest
    list that never stopped working, and returns the full board.

    The old path is tried first so this reverts by itself if the block lifts,
    and which dialect answered is returned rather than inferred, because the
    two speak completely different field names.
    """
    try:
        payload = _get(config.DK_DRAFTABLES.format(dg=draft_group),
                       as_json=True)
        rows = payload.get("draftables") or []
        if rows:
            log.info("draft group %s: %d rows from the draftgroups API",
                     draft_group, len(rows))
            return rows, "api"
    except (DataUnavailable, ValueError) as exc:
        log.info("draftgroups API unavailable (%s); using the lobby endpoint",
                 str(exc)[:70])

    payload = _get(config.DK_PLAYERS.format(dg=draft_group), as_json=True)
    rows = (payload.get("playerList") or payload.get("draftables")
            or payload.get("players") or [])
    if not rows:
        raise DataUnavailable(
            f"draft group {draft_group} returned no players from either "
            f"endpoint (lobby top-level keys: {sorted(payload)[:10]})")
    log.info("draft group %s: %d rows from the lobby endpoint",
             draft_group, len(rows))
    return rows, "lobby"


def draftables(draft_group: int,
               captain_multiplier: float | None = None) -> pd.DataFrame:
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
    rows, dialect = _draft_rows(draft_group)

    if dialect == "lobby":
        # The abbreviated dialect, mapped from a row printed verbatim by the
        # college explorer. `pn` is the position; `pp` is an integer that is
        # zero for every player, and mapping position to it once produced a
        # whole board sharing a single position - which would have made every
        # positional constraint in the optimiser vacuous.
        built = []
        for p in rows:
            home, away = p.get("htabbr"), p.get("atabbr")
            tid, htid, atid = p.get("tid"), p.get("htid"), p.get("atid")
            team = home if tid == htid else away if tid == atid else None
            msgs = p.get("ExceptionalMessages") or []
            note = " ".join(
                str(m.get("message") or m.get("Message") or m)
                for m in msgs) if isinstance(msgs, list) else str(msgs)
            built.append({
                "dk_player_id": p.get("pid"),
                "draftable_id": p.get("did"),
                "name": " ".join(x for x in (p.get("fn"), p.get("ln")) if x),
                "position": p.get("pn"),
                "team": team,
                "salary": p.get("s"),
                "roster_slot": p.get("rosposid"),
                "status": note,
                "disabled": bool(p.get("IsDisabledFromDrafting")),
                "attributes": note,
                "game": f"{away} @ {home}" if home and away else None,
                "starts_utc": pd.to_datetime(p.get("dgst"), errors="coerce",
                                             utc=True),
            })
        df = pd.DataFrame(built)
    else:
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

    if df["position"].nunique() <= 1 and len(df) > 20:
        raise DataUnavailable(
            f"every player on draft group {draft_group} has position "
            f"{df['position'].iloc[0]!r} - the position field moved. Look at a "
            f"raw row before trusting anything downstream.")

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
    elif dialect == "lobby":
        # The lobby endpoint lists each player ONCE, so a showdown board
        # arrives with no captain row to read the multiplier from. The old
        # api.draftkings.com endpoint returned a second, dearer row per
        # player and that is where the 1.5x used to come from; losing it left
        # captain_salary equal to the flex price, which would let an
        # optimiser field a captain for free and build a lineup DraftKings
        # would reject on entry.
        #
        # So on this dialect the multiplier is applied from the roster rules
        # rather than read from the payload. It is stated rather than
        # inferred, and logged, because a silently wrong captain price is
        # wrong on the single most expensive slot in the lineup.
        if captain_multiplier:
            df["captain_salary"] = (df["salary"] * float(captain_multiplier))
            df["captain_salary"] = df["captain_salary"].round().astype("Int64")
            log.info("showdown: no captain row on the lobby endpoint, so the "
                     "captain price is the flex price x%.2f from the roster "
                     "rules ($%s-$%s)", captain_multiplier,
                     int(df["captain_salary"].min()),
                     int(df["captain_salary"].max()))
        else:
            log.info("no duplicate pricing on this board, and no captain "
                     "multiplier was passed - captain_salary equals the flex "
                     "price. Correct for Classic; WRONG for Showdown.")

    df["playing"] = [playing_status(s, d, a) for s, d, a
                     in zip(df["status"], df["disabled"], df["attributes"])]
    counts = df["playing"].value_counts().to_dict()
    log.info("availability: %s",
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if dialect == "lobby":
        # Availability on this dialect is UNVERIFIED. The old endpoint carried
        # an explicit `status` string and player attributes; this one carries
        # `IsDisabledFromDrafting` and a free-text message list whose contents
        # nobody here has yet seen populated. Troy Franklin was rostered while
        # inactive once already, so rather than assume this reads injuries
        # correctly, print what actually arrived and let the next run settle
        # it. An availability filter that cannot detect an inactive player is
        # worse than no filter, because it is trusted.
        flagged = int(df["disabled"].sum())
        noted = df.loc[df["attributes"].astype(str).str.strip().ne(""),
                       "attributes"]
        log.warning("availability from the lobby endpoint is UNVERIFIED: "
                    "%d players flagged disabled, %d carry a message",
                    flagged, len(noted))
        if len(noted):
            sample = noted.astype(str).value_counts().head(6).to_dict()
            log.warning("message values seen: %s", sample)
        else:
            log.warning("NO player carries any message text. Either nobody is "
                        "hurt, or injuries live in a field this mapping does "
                        "not read. Check against the real injury report "
                        "before trusting the board an hour before kickoff.")

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


# ------------------------------------------------------------------ injuries
# The official injury report, which is the thing the board cannot tell us.
#
# DraftKings' lobby endpoint carries no usable status - every player on a live
# 658-man board read as healthy - so the model has been unable to see who is
# out. That is not a small gap: an inactive player scores zero, and a zero in
# a lineup is the whole entry. nflverse republishes the league's own weekly
# report, free, and it is the correct fix.
#
# Several candidate paths because nflverse has renamed release assets before -
# the deep history and the recent seasons have lived under different tags, and
# a loader that knows one name silently trains on stale data when it moves.
INJURY_SOURCES = [
    ("https://github.com/nflverse/nflverse-data/releases/download/injuries/"
     "injuries_{season}.csv"),
    ("https://github.com/nflverse/nflverse-data/releases/download/injuries/"
     "injuries_{season}.csv.gz"),
    ("https://github.com/nflverse/nflverse-data/releases/download/"
     "weekly_injuries/injuries_{season}.csv"),
]

# What the report says versus what it means for a lineup. The league's own
# vocabulary, mapped onto the three states the optimiser acts on.
REPORT_OUT = ("out", "injured reserve", "ir", "pup", "nfi", "suspended",
              "did not report", "reserve")
REPORT_DOUBTFUL = ("doubtful",)
REPORT_QUESTIONABLE = ("questionable", "limited")


def injuries(season: int) -> pd.DataFrame:
    """The weekly injury report, or an empty frame and a loud complaint.

    Returns empty rather than raising: a missing injury feed must degrade the
    board to "we cannot see injuries", which is where it already was, not take
    the whole build down an hour before kickoff. But it says so at ERROR, and
    the caller reports the coverage it achieved, because silently having no
    injury data is exactly the failure this function exists to end.
    """
    for pattern in INJURY_SOURCES:
        url = pattern.format(season=season)
        try:
            raw = _get(url)
        except DataUnavailable:
            continue
        try:
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        except Exception as exc:                               # noqa: BLE001
            log.warning("injuries at %s did not parse: %s", url, str(exc)[:70])
            continue
        if df.empty:
            continue
        log.info("injury report: %d rows from %s", len(df), url.split("/")[-1])
        return _tidy_injuries(df, season)
    log.error("NO injury report could be loaded for %s. The board cannot see "
              "who is inactive, so an out player can reach a lineup. Tried: "
              "%s", season, [p.split("/")[-2] for p in INJURY_SOURCES])
    return pd.DataFrame()


def _tidy_injuries(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per player-week with a normalised name and a plain status."""
    out = df.copy()
    rename = {}
    for target, options in (
            ("name", ["full_name", "player_display_name", "player_name",
                      "gsis_name", "name"]),
            ("team", ["team", "club_code", "recent_team"]),
            ("week", ["week"]),
            ("season", ["season"]),
            ("report", ["report_status", "game_status", "status",
                        "injury_status"]),
            ("practice", ["practice_status", "practice_primary_injury"]),
    ):
        for o in options:
            if o in out.columns:
                rename[o] = target
                break
    out = out.rename(columns=rename)
    missing = [c for c in ("name", "week") if c not in out.columns]
    if missing:
        log.error("injury report lacks %s; columns were %s", missing,
                  sorted(df.columns)[:20])
        return pd.DataFrame()

    if "season" not in out.columns:
        out["season"] = season
    out["norm"] = out["name"].map(normalise_name)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    text = (out.get("report", pd.Series("", index=out.index)).astype(str)
            + " " + out.get("practice", pd.Series("", index=out.index))
            .astype(str)).str.lower()
    out["playing"] = [_injury_state(t) for t in text]
    keep = ["season", "week", "name", "norm", "team", "playing"]
    keep += [c for c in ("report", "practice") if c in out.columns]
    return out[keep].dropna(subset=["week"]).reset_index(drop=True)


def _injury_state(text: str) -> str:
    t = str(text).lower()
    if any(k in t for k in REPORT_OUT):
        return "out"
    if any(k in t for k in REPORT_DOUBTFUL):
        return "doubtful"
    if any(k in t for k in REPORT_QUESTIONABLE):
        return "questionable"
    return "clear"


def attach_injuries(pool: pd.DataFrame, report: pd.DataFrame,
                    season: int, week: int) -> pd.DataFrame:
    """Overwrite the board's availability with the official report.

    The report WINS wherever it has an opinion. DraftKings' own field has been
    empty on every live board we have pulled, so deferring to it would be
    deferring to nothing; the league's report is the only real evidence in the
    room. Players it says nothing about keep whatever the board said, which is
    normally "clear" and is the right default - most players are not on the
    report at all.

    The coverage is returned in `.attrs` so the caller can refuse to build on
    a report that matched almost nobody. A join that silently finds two
    players is indistinguishable from a healthy league, and that is precisely
    the confusion that let an inactive receiver into a lineup.
    """
    out = pool.copy()
    if report is None or report.empty:
        out.attrs["injury_matched"] = 0
        out.attrs["injury_note"] = "no report loaded"
        return out

    wk = report[(report["season"] == season) & (report["week"] == week)]
    if wk.empty:
        # Asking for a week the report does not cover used to apply NOTHING,
        # which is the worst of the three options: no filtering, dressed as
        # filtering. The report always carries the current week, so the latest
        # week it does have is a far better answer than none - and saying
        # which week was used makes an off-by-one visible instead of silent.
        have = sorted(int(w) for w in
                      report[report["season"] == season]["week"].dropna()
                      .unique())
        if not have:
            log.error("the injury report has no rows at all for %s. The board "
                      "cannot see who is inactive.", season)
            out.attrs["injury_matched"] = 0
            out.attrs["injury_note"] = f"no rows for season {season}"
            return out
        fallback = max(w for w in have if w <= week) if any(
            w <= week for w in have) else have[-1]
        log.error("the injury report has no rows for %s week %s (it covers "
                  "%s). FALLING BACK to week %s - check the week derivation, "
                  "because this is how a board goes out unfiltered.",
                  season, week, have[-5:], fallback)
        wk = report[(report["season"] == season)
                    & (report["week"] == fallback)]
        out.attrs["injury_week_used"] = fallback

    # Matched on the normalised name only, deliberately. Team codes disagree
    # between the two sources and a player who was traded on Tuesday is
    # exactly the player whose status matters most.
    # Built over BOTH spellings of every reported name, so "A.J. Brown" on
    # the report reaches "AJ Brown" on the board. Sorted so that when a name
    # appears twice the more severe status wins - "doubtful" sorts before
    # "out"... which is the wrong way round, so the order is stated
    # explicitly rather than left to the alphabet.
    severity = {"out": 0, "doubtful": 1, "questionable": 2, "clear": 3}
    state: dict[str, str] = {}
    for r in wk.itertuples(index=False):
        for key in name_keys(r.name):
            if key not in state or severity.get(r.playing, 9) < severity.get(
                    state[key], 9):
                state[key] = r.playing

    def look_up(name) -> str | float:
        for key in sorted(name_keys(name)):
            if key in state:
                return state[key]
        return np.nan

    hit = out["name"].map(look_up)
    matched = int(hit.notna().sum())
    out["injury_status"] = hit.fillna("")
    out["playing"] = np.where(hit.notna(), hit, out.get("playing", "clear"))

    counts = out["playing"].value_counts().to_dict()
    log.info("injury report matched %d of %d priced players; board now reads "
             "%s", matched, len(out),
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if matched == 0:
        log.error("the injury report matched NOBODY on this board. Either the "
                  "names do not join or the week is wrong - do not trust the "
                  "availability column.")
    out.attrs["injury_matched"] = matched
    out.attrs["injury_note"] = f"{matched}/{len(out)} matched"
    return out
