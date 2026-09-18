"""Tests for the ownership model.

Runs as a script, in the house style: `python test_ownership.py`, prose names,
non-zero exit on failure. No pytest, because the workflow has none.

The important one is the CALIBRATION section. Every other test here checks an
invariant that was always true; that one checks a claim about reality against a
real DraftKings contest, and it can fail. It also PRINTS the three scores on
every run, which is the point: the publish log then says out loud whether the
calibrated module is the one that built the board. Three rounds were spent
inferring that from the outside because nothing in the build would say it.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import ownership as OWN


SHOWDOWN = {"slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"],
            "flex_positions": []}
CLASSIC = {"slots": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
           "flex_positions": ["RB", "WR", "TE"]}


def board(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """A synthetic pool of any size, with every position represented.

    The position cycle repeats rather than truncating, because an earlier
    version carried a fixed 20-entry list and silently produced a ragged
    frame - "All arrays must be of the same length" - the moment a test asked
    for 24 players.
    """
    rng = np.random.default_rng(seed)
    cycle = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST"]
    return pd.DataFrame({
        "name": [f"p{i}" for i in range(n)],
        "position": [cycle[i % len(cycle)] for i in range(n)],
        "salary": rng.integers(2000, 11000, n),
        "median": rng.uniform(1.0, 25.0, n),
        "ceiling": rng.uniform(5.0, 45.0, n),
    })


def near(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Calibration. These can fail.
# ---------------------------------------------------------------------------

def beats_the_old_model_on_the_real_board():
    """Measured against DK contest 195775024, not against an opinion."""
    got = OWN.check_calibration()
    assert got["n"] == 29, f"expected 29 calibration players, got {got['n']}"
    assert got["kl"] < OWN.BASELINE["kl"], (
        f"KL divergence got WORSE: {got['kl']:.3f} vs "
        f"{OWN.BASELINE['kl']:.3f} - has a value term come back?")
    assert got["mae"] < OWN.BASELINE["mae"], (
        f"mean error got WORSE: {100 * got['mae']:.1f}pp vs "
        f"{100 * OWN.BASELINE['mae']:.1f}pp")
    assert got["rho"] > OWN.BASELINE["rho"] + 0.20, (
        f"rank correlation got WORSE: {got['rho']:.2f} vs "
        f"{OWN.BASELINE['rho']:.2f}")


def the_calibration_check_can_fail():
    """A check that cannot fail proves nothing.

    Without this, a `check_calibration` that returned constants would pass the
    test above forever.
    """
    good = OWN.check_calibration()
    saved = OWN.SALARY_WEIGHT, OWN.POINTS_WEIGHT
    try:
        OWN.SALARY_WEIGHT, OWN.POINTS_WEIGHT = -2.0, -1.0
        bad = OWN.check_calibration()
    finally:
        OWN.SALARY_WEIGHT, OWN.POINTS_WEIGHT = saved
    assert bad["rho"] < 0 < good["rho"]
    assert bad["kl"] > OWN.BASELINE["kl"]


def value_is_not_a_positive_driver():
    """The specific regression this rewrite exists to prevent.

    Points per dollar correlated -0.42 with real ownership, so two players
    with the same projection must not be ranked by cheapness.
    """
    pool = pd.DataFrame({
        "name": ["dear", "cheap"], "position": ["WR", "WR"],
        "salary": [9000, 4000], "median": [12.0, 12.0],
        "ceiling": [24.0, 24.0]})
    s = OWN.appeal(pool)
    assert s.iloc[0] > s.iloc[1], (
        "the dearer of two equal projections must be the more owned - the "
        "field does not hunt value")


# ---------------------------------------------------------------------------
# Invariants that were always true and must stay true.
# ---------------------------------------------------------------------------

def classic_sums_to_the_roster():
    """Exactly one quarterback per entry, so QB ownership sums to exactly 1.0.

    Arithmetic, not opinion. An earlier cap fix broke this and produced 128%.
    """
    pool = board(20)
    own = OWN.project(pool, CLASSIC)
    demand = OWN.slot_demand(CLASSIC)
    assert near(own[pool["position"] == "QB"].sum(), demand["QB"])
    assert near(own.sum(), len(CLASSIC["slots"]))


def showdown_sums_to_six():
    own = OWN.project(board(20), SHOWDOWN)
    assert near(own.sum(), 6.0)


def the_cap_holds_and_the_excess_is_not_lost():
    pool = board(12)
    pool.loc[0, "median"] = 500.0
    pool.loc[0, "salary"] = 11000
    own = OWN.project(pool, SHOWDOWN)
    assert own.max() <= OWN.MAX_OWNERSHIP + 1e-9
    assert near(own.sum(), 6.0)


def a_tiny_pool_stays_feasible():
    """Six slots across eight players forces everyone above the cap.

    The old cap logic pushed the excess round the board fifty times and
    returned 135% ownership for three of them.
    """
    own = OWN.project(board(8, seed=1), SHOWDOWN)
    assert near(own.sum(), 6.0)
    assert own.max() <= 6.0 / 8 + 1e-9
    assert (own >= 0).all()


def an_out_player_is_owned_by_nobody():
    """The Buxton rule. Ruled out is not rostered at any price."""
    pool = board(12)
    pool["playing"] = "clear"
    pool.loc[3, ["playing", "median", "salary"]] = ["out", 30.0, 2000]
    own = OWN.project(pool, SHOWDOWN)
    assert own.iloc[3] < 1e-6
    assert near(own.sum(), 6.0)


def an_unknown_status_is_not_treated_as_out():
    """Midweek, before the injury report lands, every player reads "unknown".

    If that were treated as OUT the whole board would be zeroed and no lineup
    could be built at all.
    """
    pool = board(12)
    pool["playing"] = "unknown"
    own = OWN.project(pool, SHOWDOWN)
    assert near(own.sum(), 6.0)
    assert own.max() > 0.01


def the_punt_gate_excludes_the_unprojected():
    """The long-snapper guard, which failed once with five of them at 65%."""
    pool = board(24, seed=3)
    pool.loc[0, ["name", "salary", "median", "ceiling"]] = [
        "snapper", 200, 0.4, 1.0]
    pool.loc[1, ["name", "salary", "median", "ceiling"]] = [
        "real punt", 400, 14.0, 30.0]
    own = OWN.project(pool, SHOWDOWN)
    assert own.iloc[0] < own.iloc[1]
    assert own.iloc[0] < own.median(), (
        "a player who will not take a snap belongs in the bottom half of the "
        "board however cheap he is")


def missing_columns_do_not_raise():
    own = OWN.project(board(10).drop(columns=["ceiling"]), SHOWDOWN)
    assert own.notna().all()
    assert near(own.sum(), 6.0)


def duplication_scales_with_the_field():
    assert near(OWN.duplication(np.array([0.3] * 6), 200_000),
                200_000 * 0.3 ** 6, tol=1e-3)
    assert OWN.duplication(np.array([0.08] * 6), 200_000) < 1.0


def leverage_is_a_rank_gap():
    pool = board(10)
    assert OWN.leverage(pool, OWN.project(pool, SHOWDOWN)).abs().max() <= 1.0


SUITES = [
    ("OWNERSHIP, MEASURED AGAINST A REAL CONTEST FOR THE FIRST TIME", [
        ("beats the model that shipped before it",
         beats_the_old_model_on_the_real_board),
        ("and the comparison can actually fail", the_calibration_check_can_fail),
        ("points per dollar is not a positive driver of ownership",
         value_is_not_a_positive_driver),
    ]),
    ("OWNERSHIP IS MODELLED, BUT IT MUST STILL ADD UP", [
        ("every position sums to exactly what a lineup demands",
         classic_sums_to_the_roster),
        ("showdown sums to six, since it has no position requirements",
         showdown_sums_to_six),
        ("nobody is owned by the entire field, and the excess is not lost",
         the_cap_holds_and_the_excess_is_not_lost),
        ("a pool smaller than the roster does not exceed 100%",
         a_tiny_pool_stays_feasible),
    ]),
    ("WHO CAN BE OWNED AT ALL", [
        ("a player ruled out attracts no ownership",
         an_out_player_is_owned_by_nobody),
        ("but an UNKNOWN status does not empty the board",
         an_unknown_status_is_not_treated_as_out),
        ("a minimum-salary non-factor is not a punt",
         the_punt_gate_excludes_the_unprojected),
        ("a missing column degrades instead of raising",
         missing_columns_do_not_raise),
    ]),
    ("WHAT OWNERSHIP IS FOR", [
        ("a chalk lineup is fielded by many other entries",
         duplication_scales_with_the_field),
        ("leverage is a rank gap, not a subtraction", leverage_is_a_rank_gap),
    ]),
]


def main() -> int:
    # First, and loudly: is this even the calibrated module? Three publish
    # rounds were spent asking that question from the outside, so it gets
    # answered on line one instead of inferred from a board.
    if not hasattr(OWN, "check_calibration"):
        print("\n" + "=" * 66)
        print("THE OLD ownership.py IS DEPLOYED.")
        print("=" * 66)
        print("This ownership module has no check_calibration, which means it")
        print("predates the calibration against DK contest 195775024 - the")
        print("one that removed the points-per-dollar term. Whatever board")
        print("this build produces will carry the OLD ownership numbers.")
        print("\nReplace ownership.py in the repo root and re-run.")
        return 1

    passed = failed = 0
    for title, cases in SUITES:
        print(f"\n{title}")
        print("-" * 66)
        for name, fn in cases:
            try:
                fn()
                print(f"  ok    {name}")
                passed += 1
            except Exception as exc:                           # noqa: BLE001
                print(f"  FAIL  {name}\n        {exc}")
                failed += 1

    # Printed on every run, pass or fail. This is the line that tells a
    # publish log which ownership model actually built the board.
    got = OWN.check_calibration()
    print("\nCALIBRATION against DK contest 195775024 (DET @ BUF, 784 entries)")
    print("-" * 66)
    print(f"  {'':18}{'now':>9}{'before':>9}")
    print(f"  {'KL divergence':18}{got['kl']:9.3f}{OWN.BASELINE['kl']:9.3f}")
    print(f"  {'mean abs error':18}{100 * got['mae']:8.1f}%"
          f"{100 * OWN.BASELINE['mae']:8.1f}%")
    print(f"  {'spearman rho':18}{got['rho']:9.2f}{OWN.BASELINE['rho']:9.2f}")
    print(f"  {'players joined':18}{int(got['n']):9d}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
