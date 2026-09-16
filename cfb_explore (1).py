"""College football, looked at rather than guessed at.

Why this exists
---------------
The sport probe answered the cheap question - can CFB be built at all - and
the answer was yes: 1,749 live DraftKings contests, and a CollegeFootballData
key that returns games, per-player game stats and betting lines, all HTTP 200.
That is all four inputs from one source, which no other sport has.

What it did NOT answer is the expensive question, and the expensive question is
always the same one: **do the names join?** CFBD returned 124 player-stat
records whose only top-level keys were `id` and `teams`, which means the actual
statistics are nested somewhere underneath and nobody has looked. DraftKings
prices roughly a hundred and thirty schools' worth of players; college rosters
turn over completely every four years, carry far more duplicate surnames than
the NFL, and DraftKings' team abbreviations are its own invention.

A 96% name match is not fine. The 4% that fails is not random - it is skewed
towards freshmen, transfers and the players who just became relevant, which is
to say exactly the players a DFS model needs to price. So this script does not
report a rate and move on. It prints the failures by name, because the failures
are the finding.

It also refuses to guess at the JSON shape. Every structure below is walked and
printed as it actually arrived, so the ingest that follows is written from a
transcript rather than from memory.

What this deliberately does not do
----------------------------------
It does not chase the sportsdataverse data repositories. The probe reported
that four of them publish no releases while nflverse publishes twenty-five
tags through the identical code path, so the listing logic is sound and the
repos genuinely are not serving what we hoped. That question is now moot:
CFBD covers college football end to end, MLB's statsapi and Baseball Savant
cover baseball, and NHL's api-web covers hockey. None of the three sports
worth building next needs those repos at all, so the right move is to stop
paying for the question.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
import requests

from data import normalise_name

UA = {"User-Agent": "Mozilla/5.0 (dfs research)", "Accept": "application/json"}
TIMEOUT = 40
CFBD = "https://api.collegefootballdata.com"
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=CFB"
DK_BASE = "https://api.draftkings.com/draftgroups/v1/draftgroups"


def head(t):
    print(f"\n{t}\n{'=' * 74}")


def sub(t):
    print(f"\n{t}\n{'-' * 74}")


def shape(obj, path="", depth=0, out=None, max_depth=9):
    """Walk a JSON blob and print its real structure, not a guess at it.

    A list is described by its FIRST element and its length; a dict by its
    keys. Leaf values are shown truncated, because the shape is the point and
    the values are the illustration.
    """
    out = out if out is not None else []
    pad = "  " * depth
    if isinstance(obj, dict):
        out.append(f"{pad}{path or '.'}  dict({len(obj)}) {sorted(obj)[:14]}")
        if depth < max_depth:
            for k in sorted(obj):
                v = obj[k]
                if isinstance(v, (dict, list)) and v:
                    shape(v, k, depth + 1, out, max_depth)
    elif isinstance(obj, list):
        out.append(f"{pad}{path}  list({len(obj)})")
        if obj and depth < max_depth:
            shape(obj[0], f"{path}[0]", depth + 1, out, max_depth)
    else:
        out.append(f"{pad}{path} = {str(obj)[:60]}")
    return out


def cfbd_get(path: str, key: str, **params):
    url = f"{CFBD}/{path}"
    hdr = dict(UA)
    hdr["Authorization"] = f"Bearer {key}"
    try:
        r = requests.get(url, headers=hdr, params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"  {path:<20} unreachable {str(exc)[:40]}")
        return None
    if r.status_code != 200:
        print(f"  {path:<20} HTTP {r.status_code}  {r.text[:120]}")
        return None
    try:
        j = r.json()
    except json.JSONDecodeError:
        print(f"  {path:<20} 200 but not JSON ({len(r.content):,}b)")
        return None
    n = len(j) if isinstance(j, (list, dict)) else 1
    print(f"  {path:<20} HTTP 200  {n} records  {params}")
    return j


def _start(game):
    for k in ("startDate", "start_date", "startTimeTbd", "start_time"):
        v = game.get(k)
        if isinstance(v, str) and len(v) >= 10:
            return pd.to_datetime(v, utc=True, errors="coerce")
    return pd.NaT


def find_live_week(key: str, season: int):
    """Which week DraftKings is actually pricing.

    The first version took the earliest week containing ANY unplayed game, and
    on this season that returned week 2 - which had three stragglers sitting
    unplayed while DraftKings was selling week 3 slates for Friday and
    Saturday. That is the worst class of bug in this whole project: it does not
    crash, it returns last week's betting lines for this week's board, and
    every implied total downstream is confidently wrong.

    So the week is chosen by the CLOCK, not by the completion flag: the live
    week is the one containing the next game that has not kicked off yet.
    """
    now = pd.Timestamp.now("UTC")
    table = []
    for week in range(1, 17):
        games = cfbd_get("games", key, year=season, week=week,
                         seasonType="regular")
        if not games:
            continue
        done = sum(1 for g in games if g.get("completed"))
        starts = pd.Series([_start(g) for g in games]).dropna()
        future = starts[starts > now]
        table.append({"week": week, "games": games, "n": len(games),
                      "done": done, "ahead": len(future),
                      "next": future.min() if len(future) else pd.NaT})
        # Stop once we are clearly past the live week: a week with nothing
        # played and nothing started yet is the future, not the present.
        if len(table) >= 3 and done == 0 and len(future) == len(games):
            break

    if not table:
        print("  -> the schedule returned nothing for any week")
        return None, []

    print(f"\n  {'week':<6}{'games':>6}{'completed':>11}{'still ahead':>13}"
          f"   next kickoff (UTC)")
    for r in table:
        nxt = "-" if pd.isna(r["next"]) else str(r["next"])[:16]
        print(f"  {r['week']:<6}{r['n']:>6}{r['done']:>11}{r['ahead']:>13}"
              f"   {nxt}")

    live = [r for r in table if pd.notna(r["next"])]
    if not live:
        print("\n  -> no game anywhere in the season has yet to start")
        return None, []
    pick = min(live, key=lambda r: r["next"])
    stragglers = [r for r in live if r is not pick and r["week"] < pick["week"]]
    print(f"\n  -> week {pick['week']} is live: it holds the next kickoff, "
          f"{str(pick['next'])[:16]}.")
    if stragglers:
        print(f"     Earlier weeks {[r['week'] for r in stragglers]} still show "
              f"unplayed games. Those are postponements, not the live week -")
        print(f"     picking by 'has an unplayed game' would have chosen week "
              f"{stragglers[0]['week']} and priced this board off stale lines.")
    return pick["week"], pick["games"]


# ------------------------------------------------------------------ draftkings
def dk_slates():
    sub("DRAFTKINGS: THE COLLEGE FOOTBALL BOARD")
    try:
        payload = requests.get(DK_CONTESTS, headers=UA, timeout=TIMEOUT).json()
    except Exception as exc:
        print(f"  lobby unreachable: {str(exc)[:70]}")
        return []
    contests = payload.get("Contests") or []
    if not contests:
        print("  lobby returned zero contests")
        return []

    rows = {}
    for c in contests:
        dg = c.get("dg")
        if not dg:
            continue
        r = rows.setdefault(dg, {"dg": dg, "n": 0, "type": c.get("gameType"),
                                 "example": c.get("n"),
                                 "starts": c.get("sdstring"),
                                 "prize": 0})
        r["n"] += 1
        r["prize"] = max(r["prize"], c.get("po") or 0)
    out = sorted(rows.values(), key=lambda r: -r["n"])
    print(f"  {len(contests)} contests across {len(out)} draft groups\n")
    print(f"  {'group':<10}{'contests':>9}  {'top prize':>11}  "
          f"{'type':<24} starts")
    for r in out[:12]:
        print(f"  {r['dg']:<10}{r['n']:>9}  {r['prize']:>11,}  "
              f"{str(r['type'])[:23]:<24} {r['starts']}")
    return out


def dk_board(dg: int):
    """One slate's players: positions, prices, and the roster rules implied.

    The first version of this printed "unreachable" when the response failed
    to parse as JSON, which was a lie: the server answered perfectly well and
    sent something that was not JSON. A diagnostic that names the wrong cause
    is worse than no diagnostic, because it sends the next hour of work in the
    wrong direction - so this now reports the status, the content type and the
    first bytes of whatever actually arrived, for every variant it tries.

    Two variants, because the football model's URL carries no query string and
    sends no Accept header, and this one added both. One of those is the
    difference; rather than reason about which, try them and print the answer.
    """
    sub(f"DRAFTKINGS DRAFT GROUP {dg}: WHO IS PRICED")
    plain = {"User-Agent": UA["User-Agent"]}
    variants = [
        ("no query, no Accept (the football model's exact call)",
         f"{DK_BASE}/{dg}/draftables", plain),
        ("?format=json, Accept: application/json",
         f"{DK_BASE}/{dg}/draftables?format=json", UA),
    ]
    raw = []
    for label, url, hdr in variants:
        try:
            r = requests.get(url, headers=hdr, timeout=TIMEOUT)
        except requests.RequestException as exc:
            print(f"  {label}\n     genuinely unreachable: {str(exc)[:60]}")
            continue
        ctype = r.headers.get("Content-Type", "")[:30]
        print(f"  {label}\n     HTTP {r.status_code}  {len(r.content):,}b  {ctype}")
        try:
            payload = r.json()
        except json.JSONDecodeError:
            first = r.content[:160].decode("utf-8", "replace").replace("\n", " ")
            print(f"     answered, but not JSON. First bytes: {first!r}")
            continue
        raw = payload.get("draftables") or []
        print(f"     parsed: {len(raw)} draftable rows")
        if raw:
            break
    if not raw:
        print("\n  No variant returned players. The lobby listed this group, so")
        print("  the slate exists - the draftables endpoint is the problem, and")
        print("  the bytes above say which kind of problem it is.")
        return pd.DataFrame()

    df = pd.DataFrame([{
        "name": p.get("displayName"),
        "position": p.get("position"),
        "team": p.get("teamAbbreviation"),
        "salary": pd.to_numeric(p.get("salary"), errors="coerce"),
        "slot": p.get("rosterSlotId"),
        "status": p.get("status"),
        "game": (p.get("competition") or {}).get("name"),
    } for p in raw])
    # Same collapse as football: keep the CHEAPEST row per player, because on
    # a showdown board the duplicate row is the 1.5x captain price, and keeping
    # it would make every salary on the board wrong in the same direction.
    df = (df.sort_values("salary")
            .drop_duplicates(["name", "team"], keep="first")
            .reset_index(drop=True))
    df["norm"] = df["name"].map(normalise_name)

    print(f"  {len(df)} distinct players, {df['team'].nunique()} teams, "
          f"{df['game'].nunique()} games")
    print(f"  salary ${df['salary'].min():,.0f} to ${df['salary'].max():,.0f}\n")
    print("  positions on the board:")
    for pos, n in df["position"].value_counts().items():
        s = df.loc[df["position"] == pos, "salary"]
        print(f"     {str(pos):<8}{n:>5}   ${s.min():>6,.0f} - ${s.max():>7,.0f}"
              f"   median ${s.median():>6,.0f}")
    print(f"\n  DraftKings team codes (these must map to CFBD school names):")
    print(f"     {sorted(df['team'].dropna().unique().tolist())}")
    print("\n  a few rows as they actually arrive:")
    for r in df.nlargest(6, "salary").itertuples(index=False):
        print(f"     {r.name[:26]:<28}{str(r.position):<5}{str(r.team):<6}"
              f"${r.salary:>7,.0f}   {str(r.game)[:34]}")
    return df


# ----------------------------------------------------------------------- cfbd
def player_stats_shape(key: str, season: int, week: int):
    sub("CFBD PLAYER GAME STATS: THE ACTUAL NESTING")
    print("  The probe saw top-level keys ['id', 'teams'] and stopped. The")
    print("  statistics are underneath, and an ingest written without looking")
    print("  is an ingest written from memory.\n")
    js = cfbd_get("games/players", key, year=season, week=week,
                  seasonType="regular")
    if not js:
        return pd.DataFrame()
    print()
    for line in shape(js[0], "game", 0, max_depth=9)[:44]:
        print("  " + line)

    # Flatten it, because the flattened frame is what the model will train on.
    rows = []
    for game in js:
        for team in game.get("teams") or []:
            school = team.get("school") or team.get("team")
            for cat in team.get("categories") or []:
                cname = cat.get("name")
                for typ in cat.get("types") or []:
                    tname = typ.get("name")
                    for ath in typ.get("athletes") or []:
                        rows.append({"game_id": game.get("id"),
                                     "school": school,
                                     "athlete_id": ath.get("id"),
                                     "name": ath.get("name"),
                                     "category": cname, "stat": tname,
                                     "value": ath.get("stat")})
    flat = pd.DataFrame(rows)
    if flat.empty:
        print("\n  FLATTEN PRODUCED NOTHING. The nesting above is not the")
        print("  nesting this code assumed - read the dump and fix the walk.")
        return flat
    print(f"\n  flattened to {len(flat):,} stat lines, "
          f"{flat['athlete_id'].nunique():,} distinct athletes, "
          f"{flat['school'].nunique()} schools")
    print(f"\n  categories: {sorted(flat['category'].dropna().unique())}")
    print(f"\n  stats available per category:")
    for cat, g in flat.groupby("category"):
        print(f"     {str(cat):<14} {sorted(g['stat'].dropna().unique())}")
    print(f"\n  sample rows:")
    print(flat.head(8).to_string(index=False))
    return flat


def lines_shape(key: str, season: int, week: int):
    sub("CFBD BETTING LINES: WHICH BOOKS, AND WHAT FIELDS")
    print("  Football's single best feature is the implied team total, and it")
    print("  is built from spread and total. Providers disagree, so which one")
    print("  is present matters as much as whether any are.\n")
    js = cfbd_get("lines", key, year=season, week=week, seasonType="regular")
    if not js:
        return
    print()
    for line in shape(js[0], "game", 0, max_depth=6)[:30]:
        print("  " + line)

    provider = Counter()
    complete = 0
    for g in js:
        got_any = False
        for ln in g.get("lines") or []:
            provider[ln.get("provider")] += 1
            if ln.get("spread") is not None and ln.get("overUnder") is not None:
                got_any = True
        complete += got_any
    print(f"\n  {complete}/{len(js)} games carry at least one line with BOTH "
          f"a spread and a total")
    print(f"  providers: {dict(provider)}")
    # Last run returned {'DraftKings': 120, 'Bovada': 87, 'Draft Kings': 120}.
    # Two spellings of one book, one row each per game. Averaging "all
    # providers" would therefore weight DraftKings twice and Bovada once,
    # which is not an average of the market, it is a made-up blend nobody
    # chose. The ingest must pick ONE provider by name, not aggregate.
    spellings = [p for p in provider if p and "draft" in str(p).lower()]
    if len(spellings) > 1:
        print(f"\n  NOTE: {spellings} are the same book under two spellings.")
        print("  Averaging across providers would silently double-weight it.")
        print("  The ingest picks one provider explicitly instead.")
    if complete < len(js):
        print(f"  -> {len(js) - complete} games have no usable line. Those "
              f"teams get no implied total, which is a missing feature and")
        print(f"     must be left missing rather than filled with a league "
              f"average - a made-up number is indistinguishable from a real")
        print(f"     one downstream, and that is how a model learns noise.")


def fbs_teams(key: str, season: int):
    """The ~136 schools DraftKings actually prices, and their alternate names.

    The roster endpoint returned 31,070 athletes across 306 schools - which
    includes FCS and below, and is why a stat line came back for a school
    called "Roosevelt". DraftKings prices FBS. Testing the name join against
    all 306 would flatter it, because the extra schools add names that can
    only match by coincidence; testing against FBS only is the honest test.

    This endpoint also carries `abbreviation` and `alternateNames`, which is
    the raw material for the DraftKings-code-to-school map.
    """
    sub("CFBD FBS TEAMS: THE MAPPING RAW MATERIAL")
    js = cfbd_get("teams/fbs", key, year=season)
    if not js:
        print("  teams/fbs returned nothing - falling back to the full roster,")
        print("  which will make the join look better than it is.")
        return pd.DataFrame()
    df = pd.DataFrame([{
        "school": t.get("school"), "abbreviation": t.get("abbreviation"),
        "mascot": t.get("mascot"), "conference": t.get("conference"),
        "alternates": "|".join(t.get("alternateNames") or []),
    } for t in js])
    print(f"  {len(df)} FBS schools")
    print(f"\n  {'school':<26}{'abbr':<8}{'conference':<20} alternates")
    for r in df.head(10).itertuples(index=False):
        print(f"  {str(r.school)[:25]:<26}{str(r.abbreviation):<8}"
              f"{str(r.conference)[:19]:<20} {r.alternates[:34]}")
    have_abbr = df["abbreviation"].notna().sum()
    print(f"\n  {have_abbr}/{len(df)} carry an abbreviation. DraftKings uses its")
    print("  own codes, so the map is built by matching those against school,")
    print("  abbreviation and alternates in that order - and anything left")
    print("  over is written by hand rather than guessed.")
    return df


def roster_names(key: str, season: int):
    """Every athlete CFBD knows about this season, for the join test."""
    js = cfbd_get("roster", key, year=season)
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "first": r.get("firstName"), "last": r.get("lastName"),
        "team": r.get("team"), "position": r.get("position"),
        "id": r.get("id"),
    } for r in js])
    df["name"] = (df["first"].fillna("") + " " + df["last"].fillna("")).str.strip()
    df["norm"] = df["name"].map(normalise_name)
    return df


def join_test(board: pd.DataFrame, roster: pd.DataFrame, stats: pd.DataFrame):
    sub("THE QUESTION THAT DECIDES WHETHER CFB IS BUILDABLE")
    print("  Every DraftKings player has to find his history. A player who")
    print("  does not join gets no projection, and a board with holes in it")
    print("  cannot be optimised over honestly.\n")
    if board.empty:
        print("  no DraftKings board to test against")
        return

    for label, right in (("season roster", roster), ("game stats", stats)):
        if right is None or right.empty or "norm" not in right:
            print(f"  {label:<16} nothing to join against")
            continue
        have = set(right["norm"])
        want = board.dropna(subset=["norm"])
        hit = want["norm"].isin(have)
        rate = hit.mean() if len(want) else 0.0
        print(f"  {label:<16} {hit.sum():>4}/{len(want):<4} matched "
              f"({rate:6.1%})")
        miss = want.loc[~hit]
        if len(miss):
            print(f"       the failures, which are the finding:")
            for r in miss.nlargest(min(14, len(miss)), "salary").itertuples(
                    index=False):
                print(f"         ${r.salary:>6,.0f}  {str(r.position):<4}"
                      f"{str(r.team):<6}{r.name}")
            by_pos = miss["position"].value_counts().to_dict()
            print(f"       unmatched by position: {by_pos}")
            print(f"       unmatched salary median ${miss['salary'].median():,.0f}"
                  f" vs matched ${want.loc[hit, 'salary'].median():,.0f}")
            print("       -> if the unmatched are systematically CHEAPER, the")
            print("          misses are deep-bench players and the model can")
            print("          drop them. If they are priced like starters, the")
            print("          join is broken and nothing downstream is safe.")

    # Team codes are the other half of the join, and they fail differently:
    # a wrong team code does not drop a player, it attaches him to the wrong
    # game and therefore the wrong opponent, the wrong total and the wrong
    # correlation group. That is worse than a missing row.
    if not roster.empty:
        dk_teams = set(board["team"].dropna().unique())
        cfbd_teams = set(roster["team"].dropna().unique())
        print(f"\n  team codes: DraftKings uses {len(dk_teams)} on this slate, "
              f"CFBD names {len(cfbd_teams)} schools")
        overlap = {t for t in dk_teams if t in cfbd_teams}
        print(f"  exact matches: {len(overlap)}/{len(dk_teams)}")
        if len(overlap) < len(dk_teams):
            print(f"  NOT matched: {sorted(dk_teams - overlap)}")
            print("  -> a hand-written code->school map is needed. A wrong map")
            print("     does not drop a player, it gives him the wrong")
            print("     opponent and the wrong implied total, which produces")
            print("     confident numbers that are wrong in every row.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--week", type=int, default=0, help="0 = find the live one")
    args = p.parse_args(argv)

    key = os.environ.get("CFBD_API_KEY")
    head("COLLEGE FOOTBALL: LOOKED AT, NOT GUESSED AT")
    if not key:
        print("  CFBD_API_KEY is not set. Everything below needs it.")
        return 1
    print(f"  key present ({len(key)} chars), season {args.season}")

    slates = dk_slates()
    board = pd.DataFrame()
    if slates:
        # The biggest CLASSIC group, not just the biggest group. Showdown
        # boards have twelve players on them and would make the join test
        # look like a coin flip either way.
        classic = [s for s in slates
                   if "classic" in str(s["type"]).lower()] or slates
        board = dk_board(classic[0]["dg"])

    sub("FINDING THE LIVE WEEK")
    week = args.week
    if week:
        print(f"  week {week} forced on the command line")
    else:
        week, _ = find_live_week(key, args.season)
    if not week:
        print("  falling back to week 1 so the shapes below still get dumped")
        week = 1

    stats = player_stats_shape(key, args.season, max(1, week - 1))
    lines_shape(key, args.season, week)
    fbs = fbs_teams(key, args.season)

    sub("CFBD SEASON ROSTER")
    roster = roster_names(key, args.season)
    if not roster.empty:
        print(f"  {len(roster):,} athletes across {roster['team'].nunique()} "
              f"schools")
        print(f"  positions: {sorted(roster['position'].dropna().unique())[:24]}")
        if not fbs.empty:
            keep = set(fbs["school"].dropna())
            before = len(roster)
            roster = roster[roster["team"].isin(keep)].reset_index(drop=True)
            print(f"  restricted to FBS: {len(roster):,} of {before:,} athletes, "
                  f"{roster['team'].nunique()} schools")
            print("  (the rest are FCS and below - DraftKings does not price")
            print("   them, and leaving them in would flatter the join below)")
            if not stats.empty and "school" in stats:
                s_before = stats["athlete_id"].nunique()
                stats = stats[stats["school"].isin(keep)].reset_index(drop=True)
                print(f"  game stats restricted to FBS: "
                      f"{stats['athlete_id'].nunique():,} of {s_before:,} athletes")

    join_test(board, roster, stats)

    head("WHAT TO BUILD")
    print("  Read the join numbers above before anything else. Everything")
    print("  else here is solvable; a broken name join is not, and it is the")
    print("  one failure that produces a full board of confident projections")
    print("  attached to the wrong players.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
