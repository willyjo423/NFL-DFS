"""What the DFS data actually contains. Run this before anything is built on it.

Six questions, in the order that decides whether the rest is worth writing.

1. **Player usage.** Does nflverse give per-player weekly volume - targets,
   carries, snaps - and how far back?
2. **Historical salaries.** Does rotoguru still serve DraftKings and FanDuel
   salaries with actual scored points beside them, and for how many seasons?
   Without this there is no backtest, only a projection tool nobody can check.
3. **Name matching.** Can those salary rows be joined to nflverse players? This
   is the question that decides whether the project works, and it is the one
   most likely to be waved through. A 96% join rate looks fine and is not - the
   4% that fails is disproportionately the players who changed teams, have a
   suffix, or are newly signed, which is exactly the population a projection
   model most needs.
4. **Live slates.** Does the DraftKings contest endpoint expose the contest
   list and draft groups, so a specific contest can actually be selected?
5. **Live salaries.** Does the draftables endpoint return this slate's players
   and prices?
6. **The team model.** Is the NFL forecast page serving JSON that can feed
   implied team totals into the player projections?

    python dfs_probe.py
    python dfs_probe.py --seasons 2022 2023 2024
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)"}
TIMEOUT = 60

# nflverse renamed these assets partway through. The old `player_stats_{year}`
# form covers the deep history; recent seasons live under `stats_player_week`.
# The probe tries every known pattern per season and reports which answered,
# rather than pinning one name that will rot - the same failure that froze the
# stock model's index membership at 2019 for seven years.
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
NFLVERSE_SNAPS = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "snap_counts/snap_counts_{season}.csv")
NFLVERSE_ROSTER = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{season}.csv")
NFLVERSE_INJURIES = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "injuries/injuries_{season}.csv")

# rotoguru publishes one page per week per site. The layout has changed over
# the years, so the probe tries the documented form and reports what it got
# rather than assuming.
# Several spellings, because the first run got a 40KB page back with no
# semicolon table in it - which means the site answered but not in the form
# expected. Guessing a second time without looking would repeat the mistake, so
# the probe now prints what it actually received.
ROTOGURU_FORMS = [
    "http://rotoguru1.com/cgi-bin/fyday.pl?week={week}&year={season}&game={game}&scsv=1",
    "https://rotoguru1.com/cgi-bin/fyday.pl?week={week}&year={season}&game={game}&scsv=1",
    "http://rotoguru1.com/cgi-bin/fyday.pl?game={game}&week={week}&year={season}&scsv=1",
    "http://rotoguru1.com/cgi-bin/fyday.pl?week={week}&year={season}&game={game}",
]
ROTOGURU_GAMES = {"dk": "dk", "fd": "fd"}

DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=NFL"
DK_DRAFTABLES = ("https://api.draftkings.com/draftgroups/v1/draftgroups/"
                 "{dg}/draftables")
# The NFL model publishes `predictions.json`, not `forecasts.json` - the first
# probe guessed and got a 404. It carries market_margin and market_total per
# game alongside the forecast, so an implied team total is directly derivable.
FD_MODEL = "https://willyjo423.github.io/nfl-forecast/predictions.json"


def head(t):
    print(f"\n{t}\n{'=' * 72}")


def sub(t):
    print(f"\n{t}\n{'-' * 72}")


def _get(url, as_json=False):
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json() if as_json else r.content


def _csv(url) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get(url)), low_memory=False)


# ---------------------------------------------------------------------- 1
def probe_usage(seasons: list[int]) -> pd.DataFrame:
    head("1. PLAYER USAGE FROM NFLVERSE")
    print("  Fantasy points follow volume far more reliably than they follow")
    print("  efficiency, so the question is whether per-player weekly volume")
    print("  is there, and whether it is there for enough seasons to train on.")
    frames = []
    for season in seasons:
        got = False
        for label, pattern in NFLVERSE_PATTERNS:
            try:
                df = _csv(pattern.format(season=season))
            except Exception as exc:  # noqa: BLE001
                print(f"  {season}: {label.split('/')[-1]:<32} no "
                      f"({str(exc)[:46]})")
                continue
            frames.append(df.assign(_season=season))
            idcol = next((c for c in ("player_id", "gsis_id", "pfr_id")
                          if c in df.columns), None)
            print(f"  {season}: {label.split('/')[-1]:<32} YES  "
                  f"{len(df):,} rows, "
                  f"{df[idcol].nunique() if idcol else '?'} players")
            got = True
            break
        if not got:
            print(f"  {season}: no pattern answered")

    if not frames:
        print("\n  No player stats at all. Nothing downstream can work.")
        return pd.DataFrame()

    stats = pd.concat(frames, ignore_index=True)
    cols = set(stats.columns)
    sub("The columns a projection needs")
    wanted = {
        "identity": ["player_id", "player_name", "player_display_name",
                     "position", "recent_team", "season", "week"],
        "volume": ["targets", "carries", "receptions", "attempts"],
        "depth": ["target_share", "air_yards_share", "wopr"],
        "scoring": ["fantasy_points", "fantasy_points_ppr",
                    "passing_tds", "rushing_tds", "receiving_tds"],
    }
    for group, names in wanted.items():
        have = [c for c in names if c in cols]
        miss = [c for c in names if c not in cols]
        print(f"  {group:<9} have: {', '.join(have) if have else 'NONE'}")
        if miss:
            print(f"  {'':<9} missing: {', '.join(miss)}")

    if "fantasy_points_ppr" in cols and "position" in cols:
        sub("Scoring by position, to sanity check the units")
        for pos in ("QB", "RB", "WR", "TE"):
            s = stats[stats["position"] == pos]["fantasy_points_ppr"].dropna()
            if len(s) > 50:
                print(f"  {pos}  median {s.median():5.1f}   "
                      f"90th {np.percentile(s, 90):5.1f}   "
                      f"max {s.max():5.1f}   (n={len(s):,})")
        print("\n  A QB median near 15 and a WR median near 8 is about right.")
        print("  Wildly different numbers mean the scoring column is not what")
        print("  its name says.")
    return stats


# ---------------------------------------------------------------------- 2
def probe_salaries(seasons: list[int], weeks: list[int]) -> pd.DataFrame:
    head("2. HISTORICAL SALARIES")
    print("  Without these there is no backtest. A DFS model that cannot be")
    print("  graded against what lineups actually scored is a spreadsheet with")
    print("  opinions in it.")
    rows = []
    _sample: list = []
    for season in seasons:
        for week in weeks:
            for site, game in ROTOGURU_GAMES.items():
                got = None
                for form in ROTOGURU_FORMS:
                    url = form.format(week=week, season=season, game=game)
                    try:
                        raw = _get(url).decode("utf-8", "replace")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {season} wk{week:>2} {site}: fetch failed "
                              f"- {str(exc)[:70]}")
                        continue
                    block = [ln for ln in raw.splitlines()
                             if ln.count(";") >= 5 and "<" not in ln]
                    if len(block) >= 5:
                        got = (block, raw, url)
                        break
                    if _sample is not None and not _sample:
                        _sample.append((url, raw))
                    time.sleep(0.3)

                if got is None:
                    print(f"  {season} wk{week:>2} {site}: no semicolon table")
                    continue
                block, raw, url = got
                try:
                    df = pd.read_csv(io.StringIO("\n".join(block)), sep=";")
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    rows.append(df.assign(_season=season, _week=week,
                                          _site=site))
                    print(f"  {season} wk{week:>2} {site}: {len(df):>4} rows "
                          f"via {url.split('?')[0][-24:]}?... "
                          f"cols {list(df.columns)[:5]}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  {season} wk{week:>2} {site}: table found but "
                          f"unparseable - {str(exc)[:70]}")
                time.sleep(0.4)

    if not rows:
        print("\n  Nothing came back in the expected form.")
        if _sample:
            url, raw = _sample[0]
            print(f"\n  What the server actually returned for {url}")
            print(f"  ({len(raw):,} bytes). First 1,200 characters, so the")
            print("  format can be read rather than guessed at a third time:")
            print("  " + "-" * 68)
            for line in raw[:1200].splitlines():
                print(f"  | {line[:100]}")
            print("  " + "-" * 68)
            marks = {m: raw.lower().count(m) for m in
                     ("<pre", "<table", "<td", "salary", "gid", ";")}
            print(f"  markers: {marks}")
        return pd.DataFrame()

    sal = pd.concat(rows, ignore_index=True)
    sub("What the salary rows contain")
    print(f"  total rows            {len(sal):,}")
    print(f"  columns               {', '.join(map(str, sal.columns))}")
    for c in sal.columns:
        if "salary" in str(c):
            v = pd.to_numeric(sal[c], errors="coerce").dropna()
            if len(v):
                print(f"  {c}: median ${v.median():,.0f}, "
                      f"range ${v.min():,.0f}-${v.max():,.0f}")
        if str(c).startswith("dk points") or str(c) in ("fd points", "points"):
            v = pd.to_numeric(sal[c], errors="coerce").dropna()
            if len(v):
                print(f"  {c}: median {v.median():.1f}, max {v.max():.1f}")
    return sal


# ---------------------------------------------------------------------- 3
def _norm(name: str) -> str:
    """A name reduced to something two sources might agree on."""
    s = str(name).lower().strip()
    if "," in s:                      # "Smith, Devonta" -> "devonta smith"
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    for junk in (" jr.", " jr", " sr.", " sr", " iii", " ii", " iv", "'", ".",
                 "-"):
        s = s.replace(junk, " " if junk in ("-",) else "")
    return " ".join(s.split())


def probe_join(stats: pd.DataFrame, sal: pd.DataFrame) -> None:
    head("3. DO THE NAMES JOIN")
    print("  This is the question that decides the project. A join that fails")
    print("  on a few percent is not a rounding error: the players who fail to")
    print("  match are the ones who changed teams, carry a suffix, or were")
    print("  signed last week - which is precisely the population a projection")
    print("  is most needed for. A silent 96% is worse than a loud 80%.")
    if stats.empty or sal.empty:
        print("\n  Cannot test - one of the two sources returned nothing.")
        return

    namecol = next((c for c in ("player_display_name", "player_name")
                    if c in stats.columns), None)
    salname = next((c for c in sal.columns if "name" in str(c)), None)
    if not namecol or not salname:
        print(f"\n  No comparable name column (stats={namecol}, "
              f"salaries={salname})")
        return

    left = {_norm(n) for n in stats[namecol].dropna().unique()}
    right_raw = sal[salname].dropna().unique()
    right = {_norm(n) for n in right_raw}

    hit = right & left
    miss = sorted(right - left)
    print(f"\n  distinct salary names   {len(right):,}")
    print(f"  matched into nflverse   {len(hit):,} "
          f"({100 * len(hit) / max(len(right), 1):.1f}%)")
    print(f"  unmatched               {len(miss):,}")
    if miss:
        print("\n  A sample of what did not match - read these, they tell you")
        print("  what rule is missing:")
        for m in miss[:20]:
            print(f"    {m}")

    rate = len(hit) / max(len(right), 1)
    print()
    if rate >= 0.98:
        print("  Good enough to build on, with the unmatched list logged every")
        print("  run so it cannot drift unnoticed.")
    elif rate >= 0.90:
        print("  Workable but not yet. The sample above will usually show one")
        print("  or two systematic rules - defences, suffixes, punctuation -")
        print("  that lift this most of the way to 100%.")
    else:
        print("  Too low to proceed. Something structural is wrong, most")
        print("  likely a different name convention rather than many small")
        print("  mismatches.")


# ---------------------------------------------------------------------- 4/5
def probe_live() -> None:
    head("4. LIVE CONTESTS AND SLATES")
    try:
        payload = _get(DK_CONTESTS, as_json=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  contest list FAILED - {str(exc)[:120]}")
        return

    contests = payload.get("Contests") or []
    print(f"  contests listed       {len(contests):,}")
    groups: dict = {}
    for c in contests:
        dg = c.get("dg")
        if dg is None:
            continue
        g = groups.setdefault(dg, {"type": c.get("gameType"), "n": 0,
                                   "example": c.get("n"), "fees": set()})
        g["n"] += 1
        g["fees"].add(c.get("a"))

    print(f"  distinct draft groups {len(groups)}\n")
    print(f"  {'draft group':>12}  {'contests':>8}  {'type':<26} example")
    for dg, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"])[:10]:
        print(f"  {dg:>12}  {g['n']:>8}  {str(g['type'])[:26]:<26} "
              f"{str(g['example'])[:38]}")

    fields = sorted({k for c in contests[:50] for k in c})
    print(f"\n  contest fields available: {', '.join(fields)}")
    print("\n  `dg` is the slate key - every contest sharing one is the same")
    print("  player pool and the same salaries, which is what makes 'optimise")
    print("  for this specific contest' a real operation rather than a label.")

    head("5. LIVE SALARIES FOR A SLATE")
    if not groups:
        print("  no draft group to test")
        return
    dg = max(groups.items(), key=lambda kv: kv[1]["n"])[0]
    try:
        d = _get(DK_DRAFTABLES.format(dg=dg), as_json=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  draftables FAILED for {dg} - {str(exc)[:120]}")
        return

    players = d.get("draftables") or []
    print(f"  draft group {dg}: {len(players):,} draftable rows")
    if not players:
        return
    sample = players[0]
    print(f"  fields: {', '.join(sorted(sample.keys()))}")
    sal = [p.get("salary") for p in players if p.get("salary")]
    pos = {}
    for p in players:
        pos[p.get("position")] = pos.get(p.get("position"), 0) + 1
    print(f"  salaries: ${min(sal):,} to ${max(sal):,}, median "
          f"${int(np.median(sal)):,}")
    print(f"  positions: {dict(sorted(pos.items(), key=lambda kv: -kv[1]))}")
    # Rows repeat per roster slot on showdown slates; the distinct player count
    # is what matters for an optimiser.
    ids = {p.get("playerId") for p in players}
    print(f"  distinct players: {len(ids):,} across {len(players):,} rows"
          f"{'  (rows repeat per roster slot)' if len(players) > len(ids) else ''}")


# ---------------------------------------------------------------------- 6
def probe_team_model() -> None:
    head("6. THE TEAM MODEL AS AN INPUT")
    print("  A player's ceiling is mostly his team's. The implied team total -")
    print("  from the margin and total your NFL model already forecasts - is")
    print("  one of the strongest single inputs to a player projection, and it")
    print("  is sitting on a page that is already published.")
    try:
        d = _get(FD_MODEL, as_json=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  could not read the forecast JSON - {str(exc)[:120]}")
        print("  Not fatal: the player model can stand alone, it just loses")
        print("  its best context feature.")
        return
    games = d.get("games") or d.get("forecasts") or d.get("tickers") or []
    print(f"\n  as of      {d.get('asof') or d.get('week')}")
    print(f"  entries    {len(games)}")
    if games:
        print(f"  fields     {', '.join(sorted(games[0].keys()))[:300]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", nargs="*", type=int,
                   default=[2023, 2024, 2025])
    p.add_argument("--weeks", nargs="*", type=int, default=[3, 10])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    print("DFS DATA PROBE")
    print(f"run at {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC")
    print("Nothing here is fatal on its own. Sections 2 and 3 are the ones")
    print("that decide whether this is buildable at all.")

    stats = probe_usage(args.seasons)
    sal = probe_salaries(args.seasons[-2:], args.weeks)
    probe_join(stats, sal)
    probe_live()
    probe_team_model()

    print("\n" + "=" * 72)
    print("Send all of it back. Section 3 first - if names do not join, the")
    print("rest does not matter yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
