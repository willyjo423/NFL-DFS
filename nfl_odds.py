"""Live market lines, in the shape this model already fits on.

What this is NOT
----------------
It is not a new idea. The model already fits `implied_total`, `game_total`
and `team_spread` as features, `refresh_market` already puts the upcoming
week's line on the projection row, and the whole stale-line trap is already
documented there. The odds therefore ALREADY influence both the projections
and, through them, the lineups.

What this adds is FRESHNESS. Those numbers come from nflverse's schedule
file, which carries whatever line was last published to it. A total that
moved from 44 to 48 on Friday is exactly the information nothing else in the
model has, and it is worth nothing if the model is reading Tuesday's number.
This reads the market live and hands back the identical columns, so nothing
downstream changes.

The sign, which is the only thing here that can be badly wrong
-------------------------------------------------------------
`data.schedules()` computes `home_implied = (total + spread_line) / 2`, so
its `spread_line` is POSITIVE WHEN THE HOME TEAM IS FAVOURED. The odds feed
uses the opposite convention: a home favourite is quoted at a NEGATIVE point,
because that is what you lay. So `spread_line = -point_home`.

Get that backwards and every favourite's players are handed their opponent's
expectation - a sign error that looks entirely plausible in aggregate and is
wrong in every single row. `test_nfl_odds.py` checks it against a worked
example rather than against my confidence.

    python nfl_odds.py --probe
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import statistics

import pandas as pd
import requests

log = logging.getLogger("nfl_odds")

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
TIMEOUT = 30
REGIONS = "us"
MARKETS = "spreads,totals"

# Each club plays once a week, so "the upcoming week" is the earliest game
# per team inside this horizon. Deliberately not a week-number lookup: the
# feed has no week number, and deriving one is the bug that made the NHL
# build read a different night's slate.
HORIZON_DAYS = 9


class OddsUnavailable(RuntimeError):
    """The market could not be read. The caller decides whether that is fatal."""


# The feed names clubs in full. Everything here speaks nflverse's codes, and
# the codes are VERIFIED against the schedule frame rather than trusted -
# nflverse writes LA for the Rams, WAS for Washington and JAX for
# Jacksonville, and a code this file invents would silently drop a team.
TEAM_CODES = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL",
    "baltimore ravens": "BAL", "buffalo bills": "BUF",
    "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE",
    "dallas cowboys": "DAL", "denver broncos": "DEN",
    "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX", "kansas city chiefs": "KC",
    "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LA", "miami dolphins": "MIA",
    "minnesota vikings": "MIN", "new england patriots": "NE",
    "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT", "san francisco 49ers": "SF",
    "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
}
# Spellings nflverse has used for the same club, tried if the first code is
# not among the ones the schedule frame actually contains.
CODE_ALTS = {"LA": ["LAR"], "LAR": ["LA"], "WAS": ["WSH"], "WSH": ["WAS"],
             "JAX": ["JAC"], "JAC": ["JAX"], "LV": ["OAK"], "NO": ["NOR"],
             "GB": ["GNB"], "KC": ["KAN"], "NE": ["NWE"], "SF": ["SFO"],
             "TB": ["TAM"]}


def code_for(name: str, valid: set | None = None) -> str | None:
    """A club's nflverse code, checked against the codes really in use."""
    key = re.sub(r"[^a-z0-9 ]+", " ", str(name or "").lower())
    key = re.sub(r"\s+", " ", key).strip()
    code = TEAM_CODES.get(key)
    if code is None:
        return None
    if not valid or code in valid:
        return code
    for alt in CODE_ALTS.get(code, []):
        if alt in valid:
            log.info("%s is spelled %s in this schedule, not %s", name, alt, code)
            return alt
    return code


# ------------------------------------------------------------------ fetching
def fetch(key: str | None = None, markets: str = MARKETS) -> list:
    key = key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise OddsUnavailable(
            "ODDS_API_KEY is not set. In Actions add it under Settings -> "
            "Secrets and variables -> Actions, and pass it into the job's env.")
    try:
        r = requests.get(API, timeout=TIMEOUT, params={
            "apiKey": key, "regions": REGIONS, "markets": markets,
            "oddsFormat": "american", "dateFormat": "iso"})
    except Exception as exc:                                   # noqa: BLE001
        raise OddsUnavailable(f"{type(exc).__name__}: {str(exc)[:120]}") from exc
    left = r.headers.get("x-requests-remaining")
    if left is not None:
        log.info("odds api quota: %s used, %s remaining this month",
                 r.headers.get("x-requests-used"), left)
    if r.status_code != 200:
        raise OddsUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not isinstance(data, list):
        raise OddsUnavailable(f"expected a list of games, got {type(data).__name__}")
    return data


# ------------------------------------------------------------------ parsing
def _median_across_books(game: dict, market: str, picker):
    vals = []
    for bk in game.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") != market:
                continue
            v = picker(mk.get("outcomes") or [])
            if v is not None:
                vals.append(float(v))
    return statistics.median(vals) if vals else None


def _start(game: dict):
    try:
        t = dt.datetime.fromisoformat(
            str(game.get("commence_time", "")).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def upcoming_lines(games: list, valid: set | None = None,
                   horizon_days: int = HORIZON_DAYS,
                   now: dt.datetime | None = None) -> pd.DataFrame:
    """One row per team for the upcoming week, in `refresh_market`'s shape."""
    now = now or dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(days=horizon_days)

    rows, unmapped = [], set()
    for g in games:
        t = _start(g)
        if t is None or t > horizon:
            continue
        home, away = g.get("home_team"), g.get("away_team")
        hc, ac = code_for(home, valid), code_for(away, valid)
        if not hc or not ac:
            unmapped.add(str(home) if not hc else str(away))
            continue

        total = _median_across_books(
            g, "totals",
            lambda o: next((x.get("point") for x in o
                            if str(x.get("name", "")).lower() == "over"), None))
        point_home = _median_across_books(
            g, "spreads",
            lambda o: next((x.get("point") for x in o
                            if x.get("name") == home), None))
        if total is None or point_home is None:
            log.warning("%s @ %s has no %s; skipped", ac, hc,
                        "total" if total is None else "spread")
            continue

        # THE SIGN. The feed quotes a home favourite at a negative point;
        # `data.schedules()` wants a spread that is positive when the home
        # side is favoured, because it adds it to the total. See the module
        # docstring - this one line is the whole risk in this file.
        spread_line = -float(point_home)
        home_implied = (float(total) + spread_line) / 2.0
        away_implied = float(total) - home_implied

        rows.append({"team": hc, "opponent": ac, "is_home": 1.0,
                     "game_total": float(total), "team_spread": spread_line,
                     "implied_total": home_implied,
                     "opponent_implied": away_implied, "start": t})
        rows.append({"team": ac, "opponent": hc, "is_home": 0.0,
                     "game_total": float(total), "team_spread": -spread_line,
                     "implied_total": away_implied,
                     "opponent_implied": home_implied, "start": t})

    if unmapped:
        log.error("these club names are not in TEAM_CODES and were SKIPPED, "
                  "so those games carry no live line: %s",
                  ", ".join(sorted(unmapped)))
    if not rows:
        raise OddsUnavailable("no game inside the horizon produced a line")

    out = pd.DataFrame(rows)
    # A club plays once a week, so the earliest game inside the horizon IS
    # next week's. Anything beyond that is the week after, and averaging two
    # weeks together is a quiet way to be wrong about both.
    dup = int(out["team"].duplicated().sum())
    if dup:
        log.info("%d team(s) appear more than once inside %d days; keeping "
                 "each club's earliest game", dup, horizon_days)
    out = out.sort_values("start").drop_duplicates("team", keep="first")

    mean = float(out["implied_total"].mean())
    if not 14.0 <= mean <= 30.0:
        log.error("implied team totals average %.1f points, which is not a "
                  "football number - check the parse before trusting it", mean)
    if len(out) > 32:
        log.error("%d teams, which is more than the league has", len(out))
    log.info("live market: %d teams, implied totals %.1f to %.1f (mean %.1f)",
             len(out), out["implied_total"].min(), out["implied_total"].max(),
             mean)
    return out.drop(columns=["start"]).reset_index(drop=True)


def load(valid: set | None = None, key: str | None = None) -> pd.DataFrame:
    return upcoming_lines(fetch(key), valid=valid)


# --------------------------------------------------------------------- probe
def probe(key: str | None = None) -> None:
    data = fetch(key)
    print(f"games returned: {len(data)}")
    if not data:
        print("No games. Out of season, or no lines posted yet.")
        return
    g = data[0]
    print("\nTOP-LEVEL KEYS:", sorted(g.keys()))
    for k in ("id", "commence_time", "home_team", "away_team"):
        print(f"  {k:16s} {g.get(k)!r}")
    for bk in (g.get("bookmakers") or [])[:2]:
        print(f"\n--- {bk.get('key')} ---")
        for mk in bk.get("markets") or []:
            print(f"  market {mk.get('key')!r}")
            for o in (mk.get("outcomes") or [])[:3]:
                print(f"    {o}")
    print("\n--- WHAT THIS PARSES TO ---")
    try:
        df = upcoming_lines(data)
        print(df.sort_values("implied_total", ascending=False).to_string(index=False))
    except Exception as exc:                                   # noqa: BLE001
        print(f"PARSE FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        probe()
    else:
        print(load().to_string(index=False))
