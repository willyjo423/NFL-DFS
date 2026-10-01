"""Checks for the live market lines. No network.

The sign is the whole risk in this module, so it is checked against a worked
example rather than against anyone's confidence: a home side favoured by 7.5
in a 47-point game must come out at 27.25, and the road side at 19.75. Get it
backwards and every favourite's players are handed their opponent's
expectation - a sign error that is wrong in every row and looks fine in
aggregate.

    python test_nfl_odds.py
"""
from __future__ import annotations

import datetime as dt
import sys

import nfl_odds as O

FAIL = []


def ok(cond, msg):
    print(("  ok  " if cond else "FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def near(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


NOW = dt.datetime(2026, 10, 1, 12, 0, tzinfo=dt.timezone.utc)


def game(home, away, total, point_home, when="2026-10-04T17:00:00Z", books=2):
    bks = []
    for i, name in enumerate(("draftkings", "fanduel")[:books]):
        bks.append({"key": name, "title": name.title(), "markets": [
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": -110, "point": total + i * 0.5},
                {"name": "Under", "price": -110, "point": total + i * 0.5}]},
            {"key": "spreads", "outcomes": [
                {"name": home, "price": -110, "point": point_home},
                {"name": away, "price": -110, "point": -point_home}]}]})
    return {"id": f"{away}@{home}", "commence_time": when,
            "home_team": home, "away_team": away, "bookmakers": bks}


def test_the_sign():
    """A worked example, computed by hand.

    Kansas City at home, favoured by 7.5, in a 47-point game.
      home implied = (47 + 7.5) / 2 = 27.25
      away implied = 47 - 27.25    = 19.75
    The favourite scores MORE. That is the whole check.
    """
    df = O.upcoming_lines([game("Kansas City Chiefs", "Denver Broncos",
                                47.0, -7.5, books=1)], now=NOW)
    kc = df[df["team"] == "KC"].iloc[0]
    den = df[df["team"] == "DEN"].iloc[0]
    ok(near(kc["implied_total"], 27.25), f"the home favourite gets 27.25 "
                                         f"({kc['implied_total']})")
    ok(near(den["implied_total"], 19.75), f"the road dog gets 19.75 "
                                          f"({den['implied_total']})")
    ok(kc["implied_total"] > den["implied_total"], "the favourite scores more")
    ok(near(kc["implied_total"] + den["implied_total"], 47.0),
       "the two halves add back to the game total")
    # team_spread must be positive for the favourite, matching data.schedules()
    ok(near(kc["team_spread"], 7.5), f"the favourite's spread is +7.5 "
                                     f"({kc['team_spread']})")
    ok(near(den["team_spread"], -7.5), "and the dog's is -7.5")
    ok(near(kc["opponent_implied"], den["implied_total"]),
       "each side carries the OTHER's implied total for the defence model")
    ok(kc["is_home"] == 1.0 and den["is_home"] == 0.0, "home flag is right")


def test_road_favourite():
    """The other direction, which a sign error passes half the time."""
    df = O.upcoming_lines([game("New York Giants", "Philadelphia Eagles",
                                44.0, 6.5, books=1)], now=NOW)
    nyg = df[df["team"] == "NYG"].iloc[0]
    phi = df[df["team"] == "PHI"].iloc[0]
    ok(near(phi["implied_total"], 25.25),
       f"the ROAD favourite gets the bigger number ({phi['implied_total']})")
    ok(near(nyg["implied_total"], 18.75), "and the home dog the smaller")
    ok(phi["team_spread"] > 0 > nyg["team_spread"],
       "the spread is positive for whoever is favoured, home or away")


def test_pick_em():
    df = O.upcoming_lines([game("Chicago Bears", "Detroit Lions",
                                49.0, 0.0, books=1)], now=NOW)
    ok(near(df["implied_total"].iloc[0], 24.5)
       and near(df["implied_total"].iloc[1], 24.5),
       "a pick'em splits exactly in half")


def test_median_across_books():
    df = O.upcoming_lines([game("Buffalo Bills", "Miami Dolphins",
                                48.0, -3.0, books=2)], now=NOW)
    buf = df[df["team"] == "BUF"].iloc[0]
    ok(near(buf["game_total"], 48.25),
       f"the total is the median across books ({buf['game_total']})")


def test_team_codes():
    ok(O.code_for("Los Angeles Rams") == "LA", "nflverse writes the Rams LA")
    ok(O.code_for("Los Angeles Rams", {"LAR", "KC"}) == "LAR",
       "but LAR is used when that is what the schedule contains")
    ok(O.code_for("Washington Commanders") == "WAS", "Washington is WAS")
    ok(O.code_for("Washington Commanders", {"WSH"}) == "WSH",
       "unless the schedule says WSH")
    ok(O.code_for("Jacksonville Jaguars") == "JAX", "Jacksonville is JAX")
    ok(O.code_for("San Francisco 49ers") == "SF", "digits in a name are fine")
    ok(O.code_for("Not A Team") is None, "an unknown club is None, not a guess")
    ok(len(set(O.TEAM_CODES.values())) == 32,
       f"all 32 clubs covered ({len(set(O.TEAM_CODES.values()))})")


def test_horizon_and_dedup():
    """A club plays once a week, so the earliest game inside the horizon is
    next week's. Averaging two weeks is a quiet way to be wrong about both."""
    # One book, so the total is exactly the number written here rather than
    # the median of two.
    g1 = game("Kansas City Chiefs", "Denver Broncos", 47.0, -7.5,
              when="2026-10-04T17:00:00Z", books=1)
    g2 = game("Kansas City Chiefs", "Las Vegas Raiders", 41.0, -9.5,
              when="2026-10-08T17:00:00Z", books=1)   # still inside 9 days
    g3 = game("Buffalo Bills", "Miami Dolphins", 48.0, -3.0,
              when="2026-10-25T17:00:00Z", books=1)   # beyond the horizon
    df = O.upcoming_lines([g1, g2, g3], now=NOW)
    ok(set(df["team"]) == {"KC", "DEN", "LV"},
       f"only games inside the horizon ({sorted(df['team'])})")
    ok(len(df[df["team"] == "KC"]) == 1, "Kansas City appears once")
    kc = df[df["team"] == "KC"].iloc[0]
    ok(near(kc["game_total"], 47.0),
       f"and it is the EARLIER game that is kept ({kc['game_total']})")
    ok("BUF" not in set(df["team"]), "a game three weeks out is excluded")


def test_shape_matches_refresh_market():
    """`project.refresh_market` drops and re-merges exactly these columns. A
    missing one is silently DELETED from the feature frame at prediction
    time, which is worse than leaving it stale."""
    df = O.upcoming_lines([game("Kansas City Chiefs", "Denver Broncos",
                                47.0, -7.5)], now=NOW)
    for c in ["team", "implied_total", "game_total", "team_spread", "is_home"]:
        ok(c in df.columns, f"carries '{c}', which refresh_market merges on")
    ok("opponent_implied" in df.columns,
       "and opponent_implied, which the defence model needs")
    ok(df["implied_total"].notna().all(), "no missing implied totals")


def test_damage():
    g = game("Kansas City Chiefs", "Denver Broncos", 47.0, -7.5)
    for bk in g["bookmakers"]:
        bk["markets"] = [m for m in bk["markets"] if m["key"] != "spreads"]
    try:
        O.upcoming_lines([g], now=NOW)
        ok(False, "a game with no spread is skipped and the call raises")
    except O.OddsUnavailable:
        ok(True, "a slate with no usable game raises rather than returning junk")

    good = game("Buffalo Bills", "Miami Dolphins", 48.0, -3.0)
    df = O.upcoming_lines([g, good], now=NOW)
    ok(set(df["team"]) == {"BUF", "MIA"},
       "one broken game does not take the good ones down")

    bad = game("Kansas City Chefs", "Denver Broncos", 47.0, -7.5)  # misspelt
    df = O.upcoming_lines([bad, good], now=NOW)
    ok(set(df["team"]) == {"BUF", "MIA"},
       "an unmappable club is dropped, not guessed at")


def test_key_required():
    import os
    saved = os.environ.pop("ODDS_API_KEY", None)
    try:
        O.fetch()
        ok(False, "a missing key raises")
    except O.OddsUnavailable as exc:
        ok("ODDS_API_KEY" in str(exc), "a missing key names the variable")
    except Exception:                                          # noqa: BLE001
        ok(False, "raises OddsUnavailable, not something else")
    finally:
        if saved is not None:
            os.environ["ODDS_API_KEY"] = saved


if __name__ == "__main__":
    for fn in (test_the_sign, test_road_favourite, test_pick_em,
               test_median_across_books, test_team_codes,
               test_horizon_and_dedup, test_shape_matches_refresh_market,
               test_damage, test_key_required):
        print("\n" + fn.__name__)
        fn()
    print("\n" + (f"{len(FAIL)} FAILURES" if FAIL else "all checks passed"))
    sys.exit(1 if FAIL else 0)
