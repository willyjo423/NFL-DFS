"""Tests for the ownership model.

The important one is `test_beats_the_old_model_on_the_real_board`. Every other
test here checks an invariant that was always true; that one checks a claim
about reality, against a real DraftKings contest, and it can fail. If someone
reintroduces a points-per-dollar term, or retunes CONCENTRATION on a hunch,
this is what stops it reaching a live slate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ownership as OWN


SHOWDOWN = {"slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"],
            "flex_positions": []}
CLASSIC = {"slots": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
           "flex_positions": ["RB", "WR", "TE"]}


def board(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """A synthetic pool of any size, with every position represented.

    The position cycle repeats rather than truncating, because an earlier
    version of this helper carried a fixed 20-entry list and silently produced
    a ragged frame - "All arrays must be of the same length" - the moment a
    test asked for 24 players.
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


# ---------------------------------------------------------------------------
# The calibration test. This one can fail.
# ---------------------------------------------------------------------------

def test_beats_the_old_model_on_the_real_board():
    """Measured against DK contest 195775024, not against an opinion.

    The baseline numbers are what the shipped model actually scored on that
    board before the rewrite. Beating them is the whole justification for the
    coefficients in the module, so it is asserted rather than described.
    """
    got = OWN.check_calibration()
    assert got["n"] == 29
    assert got["kl"] < OWN.BASELINE["kl"], (
        f"KL divergence got worse: {got['kl']:.3f} vs "
        f"{OWN.BASELINE['kl']:.3f}")
    assert got["mae"] < OWN.BASELINE["mae"], (
        f"mean error got worse: {100*got['mae']:.1f}pp vs "
        f"{100*OWN.BASELINE['mae']:.1f}pp")
    assert got["rho"] > OWN.BASELINE["rho"] + 0.20, (
        f"rank correlation got worse: {got['rho']:.2f} vs "
        f"{OWN.BASELINE['rho']:.2f}")


def test_the_calibration_test_can_fail():
    """A check that cannot fail proves nothing.

    Deliberately breaking the ranking has to break the assertion. Without this,
    a `check_calibration` that silently returned constants would pass the test
    above forever.
    """
    good = OWN.check_calibration()
    saved = OWN.SALARY_WEIGHT, OWN.POINTS_WEIGHT
    try:
        # Invert the two terms that carry the ranking. If the calibration is
        # measuring anything, this must score worse than the baseline.
        OWN.SALARY_WEIGHT = -2.0
        OWN.POINTS_WEIGHT = -1.0
        bad = OWN.check_calibration()
    finally:
        OWN.SALARY_WEIGHT, OWN.POINTS_WEIGHT = saved
    assert bad["rho"] < 0 < good["rho"]
    assert bad["kl"] > OWN.BASELINE["kl"]


def test_value_is_not_a_positive_driver():
    """The specific regression this rewrite exists to prevent.

    Points per dollar correlated -0.42 with real ownership. Two players with
    the same projection must NOT be ranked by cheapness; the cheaper one is
    the LESS owned of the two, unless he clears the punt gate.
    """
    pool = pd.DataFrame({
        "name": ["dear", "cheap"],
        "position": ["WR", "WR"],
        "salary": [9000, 4000],
        "median": [12.0, 12.0],
        "ceiling": [24.0, 24.0],
    })
    score = OWN.appeal(pool)
    assert score.iloc[0] > score.iloc[1], (
        "the expensive player must be the more owned of two equal "
        "projections - the field does not hunt value")


# ---------------------------------------------------------------------------
# Invariants that were always true and must stay true.
# ---------------------------------------------------------------------------

def test_classic_sums_to_the_roster():
    """Exactly one quarterback per entry, so QB ownership sums to exactly 1.0.

    Arithmetic, not opinion. This is the check that makes the model more than
    a guess, and an earlier version of the cap logic broke it - quarterback
    ownership came out at 128%.
    """
    pool = board(20)
    own = OWN.project(pool, CLASSIC)
    demand = OWN.slot_demand(CLASSIC)
    qb = own[pool["position"] == "QB"].sum()
    assert qb == pytest.approx(demand["QB"], abs=1e-6)
    assert own.sum() == pytest.approx(len(CLASSIC["slots"]), abs=1e-6)


def test_showdown_sums_to_six():
    pool = board(20)
    own = OWN.project(pool, SHOWDOWN)
    assert own.sum() == pytest.approx(6.0, abs=1e-6)


def test_cap_is_respected_and_the_excess_is_not_lost():
    """One dominant player must not break the sum.

    Clipping at 65% removes ownership the roster constraint says exists, so it
    has to land on somebody else rather than vanish.
    """
    pool = board(12)
    pool.loc[0, "median"] = 500.0
    pool.loc[0, "salary"] = 11000
    own = OWN.project(pool, SHOWDOWN)
    assert own.max() <= OWN.MAX_OWNERSHIP + 1e-9
    assert own.sum() == pytest.approx(6.0, abs=1e-6)


def test_out_players_are_owned_by_nobody():
    """The Buxton rule. A player ruled out is not rostered at any price."""
    pool = board(12)
    pool["playing"] = "clear"
    pool.loc[3, "playing"] = "out"
    pool.loc[3, "median"] = 30.0
    pool.loc[3, "salary"] = 2000
    own = OWN.project(pool, SHOWDOWN)
    assert own.iloc[3] < 1e-6
    assert own.sum() == pytest.approx(6.0, abs=1e-6)


def test_punt_gate_excludes_the_unprojected():
    """A $200 player with no projection gets no punt bonus.

    This is the long-snapper guard. It failed once, publicly, with five
    minimum-salary special-teamers modelled at 65% ownership.
    """
    pool = board(24, seed=3)
    pool.loc[0, ["name", "salary", "median", "ceiling"]] = ["snapper", 200, 0.4, 1.0]
    pool.loc[1, ["name", "salary", "median", "ceiling"]] = ["real punt", 400, 14.0, 30.0]
    own = OWN.project(pool, SHOWDOWN)
    assert own.iloc[0] < own.iloc[1], (
        "an unprojected minimum-salary player must not out-own a cheap "
        "player with a real projection")
    assert own.iloc[0] < own.median(), (
        "a player who will not take a snap belongs in the bottom half of the "
        "board however cheap he is")


def test_cap_stays_feasible_on_a_tiny_pool():
    """A pool smaller than the roster must not return ownership above the cap.

    Six showdown slots across eight players forces everyone above 65% by
    arithmetic. The old cap logic pushed the excess round the board fifty
    times and returned 135% ownership for three of them.
    """
    pool = board(8, seed=1)
    own = OWN.project(pool, SHOWDOWN)
    assert own.sum() == pytest.approx(6.0, abs=1e-6)
    assert own.max() <= 6.0 / 8 + 1e-9
    assert (own >= 0).all()


def test_missing_columns_do_not_raise():
    """No ceiling column, no p_play column - still has to produce a board."""
    pool = board(10).drop(columns=["ceiling"])
    own = OWN.project(pool, SHOWDOWN)
    assert own.notna().all()
    assert own.sum() == pytest.approx(6.0, abs=1e-6)


def test_duplication_scales_with_field():
    assert OWN.duplication(np.array([0.3] * 6), 200_000) == pytest.approx(
        200_000 * 0.3 ** 6)
    assert OWN.duplication(np.array([0.08] * 6), 200_000) < 1.0


def test_leverage_is_a_rank_gap():
    pool = board(10)
    own = OWN.project(pool, SHOWDOWN)
    lev = OWN.leverage(pool, own)
    assert lev.abs().max() <= 1.0
