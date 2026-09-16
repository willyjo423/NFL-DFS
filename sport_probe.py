"""Which sports can actually be built, and from what.

Run this before writing a line of NBA, MLB, NHL or college football code.

The question it answers
----------------------
Adding a sport is not one job, it is four, and three of them can fail
independently:

1. **A live slate.** DraftKings has to list contests for the sport, and the
   draftables endpoint has to return players with salaries. Without this there
   is nothing to optimise for.
2. **History to train on.** A per-player, per-game box score going back several
   seasons. Without it there is no projection, only a guess dressed as one.
3. **A market line.** The single most informative feature in the football model
   is the implied team total, and it is the only forward-looking input in the
   whole set. A sport without one is a materially weaker model.
4. **Names that join.** A 96% match looks fine and is not: the 4% that fails is
   disproportionately the players who changed teams or were signed last week.

This prints what each source actually returned rather than whether it "worked",
because the failure that matters is the one that returns 200 and garbage. Every
endpoint here is a guess at a public URL - the point is to find out which
guesses are right, cheaply, before building on any of them.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys

import pandas as pd
import requests

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)",
      "Accept": "application/json,text/csv,*/*"}
TIMEOUT = 30

# DraftKings uses one lobby endpoint with a sport code. These are the codes
# worth trying; an unknown code returns an empty contest list rather than an
# error, which is why the probe counts contests instead of trusting a 200.
DK_SPORTS = ["NFL", "NBA", "MLB", "NHL", "CFB", "GOLF", "SOC", "TEN", "MMA"]

# Candidate history sources, most promising first. Several of these are
# educated guesses at public endpoints; that is the point of a probe.
HISTORY = {
    "nba": [
        ("nba_api leaguegamelog",
         "https://stats.nba.com/stats/leaguegamelog?Season=2024-25&"
         "SeasonType=Regular+Season&PlayerOrTeam=P&Counter=0&Direction=DESC&"
         "Sorter=DATE&LeagueID=00"),
        ("data.nba scoreboard",
         "https://data.nba.net/prod/v1/20250110/scoreboard.json"),
        ("hoopR play-by-play releases",
         "https://github.com/sportsdataverse/hoopR-data/releases/download/"
         "nba_player_box/player_box_2025.csv"),
        ("hoopR team box",
         "https://github.com/sportsdataverse/hoopR-data/releases/download/"
         "nba_team_box/team_box_2025.csv"),
    ],
    "mlb": [
        ("MLB statsapi schedule",
         "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2025-07-04"),
        ("baseballr releases",
         "https://github.com/sportsdataverse/baseballr-data/releases/download/"
         "mlb_batter_game_logs/batter_game_logs_2025.csv"),
        ("statcast via savant",
         "https://baseballsavant.mlb.com/statcast_search/csv?all=true&"
         "game_date_gt=2025-07-04&game_date_lt=2025-07-04&type=details"),
    ],
    "nhl": [
        ("NHL api-web schedule",
         "https://api-web.nhle.com/v1/schedule/2025-10-10"),
        ("fastRhockey releases",
         "https://github.com/sportsdataverse/fastRhockey-data/releases/"
         "download/nhl_player_box/player_box_2025.csv"),
        ("NHL club stats",
         "https://api-web.nhle.com/v1/club-stats/TOR/20242025/2"),
    ],
    "cfb": [
        ("cfbfastR player box",
         "https://github.com/sportsdataverse/cfbfastR-data/releases/download/"
         "player_box/player_box_2025.csv"),
        ("CFBD games (needs a key)",
         "https://api.collegefootballdata.com/games?year=2025&week=1"),
        ("cfbfastR team box",
         "https://github.com/sportsdataverse/cfbfastR-data/releases/download/"
         "team_box/team_box_2025.csv"),
    ],
}

# Market lines. nflverse's schedules file is the model for what good looks
# like: closing spread and total, per game, going back decades.
LINES = {
    "nba": [("hoopR schedules",
             "https://github.com/sportsdataverse/hoopR-data/releases/download/"
             "nba_schedules/nba_schedule_2025.csv")],
    "mlb": [("baseballr schedules",
             "https://github.com/sportsdataverse/baseballr-data/releases/"
             "download/mlb_schedule/mlb_schedule_2025.csv")],
    "nhl": [("fastRhockey schedules",
             "https://github.com/sportsdataverse/fastRhockey-data/releases/"
             "download/nhl_schedule/nhl_schedule_2025.csv")],
    "cfb": [("cfbfastR schedules",
             "https://github.com/sportsdataverse/cfbfastR-data/releases/"
             "download/schedules/schedules_2025.csv")],
}


# Which GitHub data repo belongs to which sport. nflverse is in here as the
# control: football already works off it, so if its listing comes back empty
# the problem is the listing code, not the other four repos.
REPOS = {
    "nba": "sportsdataverse/hoopR-data",
    "cfb": "sportsdataverse/cfbfastR-data",
    "nhl": "sportsdataverse/fastRhockey-data",
    "mlb": "sportsdataverse/baseballr-data",
    "nfl": "nflverse/nflverse-data",
}


def head(t):
    print(f"\n{t}\n{'=' * 74}")


def sub(t):
    print(f"\n{t}\n{'-' * 74}")


def describe(label: str, url: str) -> dict:
    """Fetch, and say what came back - not merely whether it came back."""
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"  {label:<34} unreachable  {str(exc)[:34]}")
        return {"label": label, "ok": False, "why": "unreachable"}

    ctype = r.headers.get("Content-Type", "")[:28]
    size = len(r.content)
    print(f"  {label:<34} HTTP {r.status_code:<4}{size:>10,}b  {ctype}")

    if r.status_code != 200 or not size:
        return {"label": label, "ok": False, "why": f"HTTP {r.status_code}"}

    # A login wall is 200 and HTML, which is the failure most easily mistaken
    # for success.
    body = r.content[:300].decode("utf-8", "replace").lower()
    if "<html" in body and "csv" not in ctype:
        print("       -> HTML, not data. A wall or a redirect, not a source.")
        return {"label": label, "ok": False, "why": "html"}

    # Sniff the CONTENT, do not trust the header. Baseball Savant returns
    # three megabytes of perfectly good CSV under Content-Type
    # "application/download", and the first version of this probe dismissed it
    # as "not JSON and not CSV" - throwing away a working source on the
    # strength of a header nobody guarantees.
    head_txt = r.content[:2000].decode("utf-8", "replace")
    looks_csv = ("csv" in ctype or url.endswith(".csv")
                 or (head_txt.count(",") > 8 and "\n" in head_txt
                     and not head_txt.lstrip().startswith(("{", "["))))
    if looks_csv:
        try:
            df = pd.read_csv(io.BytesIO(r.content), low_memory=False, nrows=4000)
        except Exception as exc:
            print(f"       -> unparseable CSV: {str(exc)[:50]}")
            return {"label": label, "ok": False, "why": "bad csv"}
        cols = list(df.columns)
        print(f"       -> {len(df):,} rows sampled, {len(cols)} columns")
        print(f"       -> {cols[:12]}")
        hits = {k: [c for c in cols if k in c.lower()]
                for k in ("name", "player", "team", "date", "min", "point",
                          "spread", "total")}
        useful = {k: v[:3] for k, v in hits.items() if v}
        print(f"       -> looks like: {useful}")
        return {"label": label, "ok": True, "rows": len(df), "columns": cols}

    try:
        j = r.json()
    except json.JSONDecodeError:
        print("       -> not JSON and not CSV")
        return {"label": label, "ok": False, "why": "unknown format"}
    keys = sorted(j)[:12] if isinstance(j, dict) else f"list of {len(j)}"
    print(f"       -> JSON: {keys}")
    return {"label": label, "ok": True, "json": True}


def release_assets(repo: str) -> list[tuple[str, str, str]]:
    """List what a GitHub data repo ACTUALLY publishes.

    Guessing release URLs produced four 404s in the first run, which says
    nothing except that the guesses were wrong. The releases API says what the
    tags and asset names really are, which turns guessing into looking - the
    same move that fixed the stock universe build when a Wikipedia table
    vanished and the code insisted the data was missing rather than moved.

    Returns (tag, asset_name, download_url) so the caller can probe the real
    files in the SAME run instead of coming back tomorrow with better guesses.
    """
    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    hdr = dict(UA)
    # Actions hands every run a token. Unauthenticated the releases API allows
    # sixty calls an hour shared across the whole runner IP, which is the kind
    # of limit that makes a probe report "no releases" when the truth is "not
    # right now" - a false negative that would send us building the wrong sport.
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, headers=hdr, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"  {repo:<38} unreachable {str(exc)[:30]}")
        return []
    if r.status_code != 200:
        note = "rate limited, not empty" if r.status_code in (403, 429) else ""
        print(f"  {repo:<38} HTTP {r.status_code} - no releases listing {note}")
        return []
    rel = r.json()
    if not rel:
        print(f"  {repo:<38} no releases published")
        return []

    out = []
    print(f"  {repo}  ({len(rel)} release tags)")
    for entry in rel:
        tag = entry.get("tag_name", "?")
        assets = entry.get("assets") or []
        for a in assets:
            out.append((tag, a.get("name", ""), a.get("browser_download_url", "")))
    for entry in rel[:16]:
        tag = entry.get("tag_name", "?")
        names = [a.get("name", "") for a in (entry.get("assets") or [])]
        recent = [n for n in names if "2026" in n or "2025" in n][:3] or names[:3]
        print(f"     {tag:<30} {len(names):>4} assets  {recent}")
    if len(rel) > 16:
        print(f"     ... and {len(rel) - 16} more tags")
    return out


# What each sport needs out of a release listing: a per-player, per-game box
# score, and a schedule carrying a line. The keywords are deliberately loose -
# the whole point is that we do not know the naming convention yet.
WANTED = {
    "player box": ("player_box", "player-box", "playerbox", "game_logs",
                   "gamelogs", "box_score", "boxscore"),
    "schedule": ("schedule", "games"),
}


def resolve(assets: list[tuple[str, str, str]], season_hint=("2025", "2024")):
    """Pick the real URLs worth probing out of a release listing.

    This is the step that collapses two round trips into one. Without it the
    probe reports "your guessed URL 404s" and the next run guesses again; with
    it the probe finds the actual file and reads its columns immediately.
    """
    picks = []
    for kind, keys in WANTED.items():
        cands = [(t, n, u) for (t, n, u) in assets
                 if any(k in (t + "/" + n).lower() for k in keys)
                 and n.lower().endswith((".csv", ".csv.gz", ".parquet", ".rds"))]
        # Prefer a recent season, and CSV over parquet - pandas reads parquet
        # only if pyarrow is installed, and a probe that fails on a missing
        # dependency reports a data problem that is really an install problem.
        def rank(item):
            _, name, _ = item
            season = next((i for i, s in enumerate(season_hint) if s in name), 9)
            fmt = 0 if name.endswith(".csv") else 1 if name.endswith(".csv.gz") else 2
            return (fmt, season, name)
        cands.sort(key=rank)
        for tag, name, url in cands[:2]:
            # Label from the ASSET name, not the tag. Truncating "tag/name"
            # cut off the season and made two different files print under one
            # identical label - a report where two rows cannot be told apart
            # is a report that cannot be acted on.
            picks.append((f"{kind}: {name}"[:34], url))
    return picks


def cfbd(key: str | None) -> None:
    """College football, which is the one sport already unblocked by a key."""
    sub("COLLEGE FOOTBALL: WITH THE KEY")
    if not key:
        print("  No CFBD_API_KEY secret set. The endpoint answered 401 rather")
        print("  than refusing the connection, which means it is reachable and")
        print("  only wants authorisation - add the secret and rerun.")
        return
    hdr = dict(UA)
    hdr["Authorization"] = f"Bearer {key}"
    for label, url in [
        ("games", "https://api.collegefootballdata.com/games?year=2025&week=3"),
        ("player game stats",
         "https://api.collegefootballdata.com/games/players?year=2025&week=3"),
        ("betting lines",
         "https://api.collegefootballdata.com/lines?year=2025&week=3"),
    ]:
        try:
            r = requests.get(url, headers=hdr, timeout=TIMEOUT)
        except requests.RequestException as exc:
            print(f"  {label:<22} unreachable {str(exc)[:34]}")
            continue
        note = ""
        if r.status_code == 200:
            try:
                j = r.json()
                note = f"{len(j)} records"
                if isinstance(j, list) and j:
                    note += f", keys {sorted(j[0])[:8]}"
            except Exception:
                note = f"{len(r.content):,} bytes"
        print(f"  {label:<22} HTTP {r.status_code:<4} {note}")


def draftkings() -> dict:
    sub("DRAFTKINGS: WHICH SPORTS HAVE LIVE SLATES")
    print("  A sport with no listed contests cannot be optimised for, however")
    print("  good the projection is. An unknown sport code returns an empty")
    print("  list rather than an error, so contests are counted, not assumed.\n")
    out = {}
    for sport in DK_SPORTS:
        url = f"https://www.draftkings.com/lobby/getcontests?sport={sport}"
        try:
            r = requests.get(url, headers=UA, timeout=TIMEOUT)
            payload = r.json() if r.status_code == 200 else {}
        except Exception as exc:
            print(f"  {sport:<6} error {str(exc)[:50]}")
            out[sport] = 0
            continue
        contests = payload.get("Contests") or []
        groups = {c.get("dg") for c in contests if c.get("dg")}
        types = sorted({str(c.get("gameType")) for c in contests})[:4]
        out[sport] = len(contests)
        print(f"  {sport:<6}{len(contests):>6} contests  {len(groups):>3} slates"
              f"  {types}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sports", nargs="*",
                   default=["nba", "mlb", "nhl", "cfb"])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    head("CAN THESE SPORTS BE BUILT")
    print("  Four things have to be true, and three can fail independently:")
    print("  a live slate, a per-player history, a market line, and names")
    print("  that join. This finds out which, cheaply, before anything is")
    print("  built on top of a source that turns out not to exist.")

    live = draftkings()

    sub("WHAT THESE DATA REPOS ACTUALLY PUBLISH")
    print("  Four guessed URLs returned 404 in the first run, which says only")
    print("  that the guesses were wrong. This asks the releases API what the")
    print("  tags and asset names really are - and then reads the real files,")
    print("  in this run, rather than coming back tomorrow with better guesses.\n")
    discovered = {}
    for sport, repo in REPOS.items():
        assets = release_assets(repo)
        discovered[sport] = resolve(assets)
        print()

    cfbd(os.environ.get("CFBD_API_KEY"))

    found = {}
    for sport in args.sports:
        sub(f"{sport.upper()}: HISTORY TO TRAIN ON")
        found[sport] = {"history": [describe(l, u)
                                    for l, u in HISTORY.get(sport, [])]}

        real = discovered.get(sport) or []
        if real:
            sub(f"{sport.upper()}: THE FILES THAT ACTUALLY EXIST")
            print("  Found by listing, not by guessing.\n")
            for label, url in real:
                got = describe(label, url)
                key = "lines" if label.startswith("schedule") else "history"
                found[sport].setdefault(key, []).append(got)

        sub(f"{sport.upper()}: MARKET LINES")
        found[sport].setdefault("lines", [])
        found[sport]["lines"] += [describe(l, u)
                                  for l, u in LINES.get(sport, [])]

    head("WHAT THIS MEANS")
    for sport in args.sports:
        hist = [d for d in found[sport]["history"] if d.get("ok")]
        line = [d for d in found[sport]["lines"] if d.get("ok")]
        slate = live.get(sport.upper(), 0)
        verdict = ("buildable" if hist and slate else
                   "no history" if slate else
                   "no live slate" if hist else "blocked on both")
        print(f"\n  {sport.upper():<5} {verdict}")
        print(f"        live contests : {slate}")
        print(f"        history       : "
              f"{hist[0]['label'] if hist else 'NONE FOUND'}")
        print(f"        market lines  : "
              f"{line[0]['label'] if line else 'none - a weaker model'}")

    print("\n  A sport missing only the market line is still worth building;")
    print("  football's implied total is the best single feature it has, but")
    print("  usage history carries most of the weight. A sport missing the")
    print("  HISTORY is not worth building at all - there would be nothing to")
    print("  fit, and a projection with nothing behind it is worse than none.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
