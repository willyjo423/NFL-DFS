"""Offline checks of the scoring rules and the name join. No network.

The headline test is a reconciliation rather than a comparison. nflverse
publishes its own fantasy points under standard scoring; DraftKings and FanDuel
differ from it in a small, exactly enumerable set of ways. So the gap between
our score and theirs is not "close" - it is an identity, and any error at all
breaks it.

That matters because the alternative is checking a handful of hand-computed
examples, which passes happily while a rate is wrong in the third decimal for
every player who did something the examples did not cover.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import data
import scoring

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def section(t):
    print(f"\n{t}\n{'-' * 66}")


def synthetic(n: int = 500, seed: int = 4) -> pd.DataFrame:
    """Stat lines shaped like real ones, including the awkward cases.

    Deliberately includes players who crossed a bonus threshold exactly, who
    lost fumbles, who threw interceptions, who scored return touchdowns and who
    converted two-point plays - because those are precisely the terms that
    differ between the sites, and a fixture without them would reconcile by
    accident.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "passing_yards": np.where(rng.random(n) < 0.2,
                                  rng.normal(250, 90, n).clip(0), 0.0),
        "passing_tds": rng.poisson(0.4, n).astype(float),
        "passing_interceptions": rng.poisson(0.25, n).astype(float),
        "rushing_yards": np.where(rng.random(n) < 0.4,
                                  rng.normal(45, 45, n).clip(0), 0.0),
        "rushing_tds": rng.poisson(0.2, n).astype(float),
        "receptions": rng.poisson(2.2, n).astype(float),
        "receiving_yards": np.where(rng.random(n) < 0.6,
                                    rng.normal(45, 40, n).clip(0), 0.0),
        "receiving_tds": rng.poisson(0.22, n).astype(float),
        "sack_fumbles_lost": rng.binomial(1, 0.04, n).astype(float),
        "rushing_fumbles_lost": rng.binomial(1, 0.03, n).astype(float),
        "receiving_fumbles_lost": rng.binomial(1, 0.02, n).astype(float),
        "passing_2pt_conversions": rng.binomial(1, 0.03, n).astype(float),
        "rushing_2pt_conversions": rng.binomial(1, 0.03, n).astype(float),
        "receiving_2pt_conversions": rng.binomial(1, 0.03, n).astype(float),
        "special_teams_tds": rng.binomial(1, 0.01, n).astype(float),
    })
    # Plant exact threshold cases - 99, 100 and 101 yards - because a bonus
    # written with the wrong comparison passes every random test.
    for i, v in enumerate((99.0, 100.0, 100.5, 299.0, 300.0)):
        df.loc[i, "receiving_yards"] = v if v < 200 else 0.0
        df.loc[i, "passing_yards"] = v if v >= 200 else 0.0
    return df


def test_reconciliation():
    section("DK AND FD AGAINST NFLVERSE'S OWN SCORING")
    df = synthetic(800)
    standard = scoring.nflverse_standard(df)
    ppr = standard + scoring.stat(df, "receptions")

    for site in ("dk", "fd"):
        got = scoring.score(df, site)
        expected_gap = scoring.reconcile(df, site)
        residual = (got - (ppr + expected_gap)).abs()
        check(f"{site}: every row reconciles with nflverse PPR to the cent",
              bool((residual < 0.011).all()),
              f"worst residual {residual.max():.4f} on "
              f"{int((residual >= 0.011).sum())} rows")

    dk, fd = scoring.score(df, "dk"), scoring.score(df, "fd")
    rec = scoring.stat(df, "receptions")
    check("full PPR scores above half PPR wherever anyone caught a pass",
          bool((dk[rec > 0] > fd[rec > 0]).all()))
    check("and the two agree when nobody caught anything, bar the penalties",
          bool(((dk - fd)[rec == 0].abs() < 12).all()))


def test_bonuses():
    section("THE BONUSES, WHICH ARE STEP FUNCTIONS")
    base = {c: 0.0 for c in ("passing_yards", "passing_tds",
                             "passing_interceptions", "rushing_yards",
                             "rushing_tds", "receptions", "receiving_yards",
                             "receiving_tds")}
    rows = []
    for yards in (99.0, 99.9, 100.0, 100.1):
        r = dict(base)
        r["receiving_yards"] = yards
        rows.append(r)
    df = pd.DataFrame(rows)
    dk = scoring.score(df, "dk")

    check("99.9 receiving yards earns no bonus",
          abs(dk.iloc[1] - 9.99) < 0.02, f"{dk.iloc[1]}")
    check("exactly 100 does", abs(dk.iloc[2] - 13.0) < 0.02, f"{dk.iloc[2]}")
    check("the bonus is worth three points",
          abs((dk.iloc[2] - dk.iloc[1]) - 3.01) < 0.02,
          f"{dk.iloc[2] - dk.iloc[1]}")
    check("FanDuel pays no such bonus",
          abs(scoring.score(df, "fd").iloc[2]
              - scoring.score(df, "fd").iloc[1]) < 0.02)

    # This is the reason the projection is a distribution rather than a mean.
    check("two players either side of the line differ by more than their yards",
          (dk.iloc[2] - dk.iloc[1]) > 10 * (100.0 - 99.9))


def test_missing_columns():
    section("WHEN A COLUMN IS NOT THERE")
    df = pd.DataFrame({"receiving_yards": [100.0, 50.0], "receptions": [5.0, 2.0]})
    got = scoring.score(df, "dk")
    check("an absent stat contributes zero rather than voiding the row",
          bool(got.notna().all()) and got.iloc[0] > 0, str(list(got)))
    check("and the bonus still applies on what is there",
          abs(got.iloc[0] - 18.0) < 0.02, f"{got.iloc[0]}")

    nan_df = pd.DataFrame({"receiving_yards": [np.nan], "receptions": [np.nan]})
    check("a NaN stat scores zero, not NaN",
          float(scoring.score(nan_df, "dk").iloc[0]) == 0.0)

    try:
        scoring.score(df, "yahoo")
        check("an unknown site raises", False, "it did not raise")
    except ValueError:
        check("an unknown site raises", True)


def test_fumbles_split_across_columns():
    section("FUMBLES, WHICH ARRIVE IN THREE COLUMNS")
    df = pd.DataFrame({"sack_fumbles_lost": [1.0], "rushing_fumbles_lost": [1.0],
                       "receiving_fumbles_lost": [0.0], "rushing_yards": [50.0]})
    check("all three are summed, not first-matched",
          abs(scoring.score(df, "dk").iloc[0] - 3.0) < 0.02,
          f'{scoring.score(df, "dk").iloc[0]} - 5 rush points less 2 fumbles')
    check("FanDuel penalises fumbles twice as hard",
          abs(scoring.score(df, "fd").iloc[0] - 1.0) < 0.02,
          f'{scoring.score(df, "fd").iloc[0]}')


def test_names():
    section("NAME NORMALISATION")
    cases = [
        ("Smith, DeVonta", "devonta smith"),
        ("Amon-Ra St. Brown", "amon ra st brown"),
        ("D'Andre Swift", "dandre swift"),
        ("Michael Pittman Jr.", "michael pittman"),
        ("Odell Beckham Jr", "odell beckham"),
        ("  Travis   Kelce  ", "travis kelce"),
        ("Marvin Harrison Jr.", "marvin harrison"),
    ]
    for raw, want in cases:
        got = data.normalise_name(raw)
        check(f"{raw!r} -> {want!r}", got == want, f"got {got!r}")

    # The one thing it must NOT do.
    check("two different players do not collapse into one",
          data.normalise_name("Josh Allen") != data.normalise_name("Keenan Allen"))
    check("a suffix inside a surname is left alone",
          data.normalise_name("Ivan Pace") == "ivan pace")


def test_join_report():
    section("MEASURING THE JOIN")
    nfl = pd.Series(["Amon-Ra St. Brown", "Michael Pittman Jr.", "Josh Allen",
                     "D'Andre Swift"])
    dk = pd.Series(["Amon-Ra St.Brown", "Michael Pittman", "Josh Allen",
                    "DAndre Swift", "Some Rookie"])
    rep = data.join_report(nfl, dk)
    check("names that differ only in punctuation match",
          rep["matched"] == 4, f"{rep['matched']} of {rep['total']}")
    check("the rate is reported", abs(rep["rate"] - 0.8) < 1e-9, str(rep["rate"]))
    check("and the failures are returned, not swallowed",
          rep["unmatched"] == ["some rookie"], str(rep["unmatched"]))
    check("an empty right side reports zero rather than dividing by zero",
          data.join_report(nfl, pd.Series([], dtype=str))["rate"] == 0.0)


def test_cash_line():
    section("THE NUMBER A CASH LINEUP HAS TO BEAT")
    field = np.arange(0, 200, dtype=float)
    line = scoring.cash_line(field, 0.5)
    check("half the field sits above the cash line",
          abs((field > line).mean() - 0.5) < 0.02, f"{line}")
    tight = scoring.cash_line(field, 0.2)
    check("a tighter payout raises the bar", tight > line)
    check("an empty field yields no line, rather than zero",
          np.isnan(scoring.cash_line(np.array([]))))


def main():
    print("DFS scoring and joins - offline checks")
    for fn in (test_reconciliation, test_bonuses, test_missing_columns,
               test_fumbles_split_across_columns, test_names,
               test_join_report, test_cash_line):
        try:
            fn()
        except Exception:
            global FAIL
            FAIL += 1
            FAILURES.append(f"{fn.__name__} raised")
            print(f"  CRASH in {fn.__name__}")
            traceback.print_exc()
    print(f"\n{'=' * 66}\n{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
