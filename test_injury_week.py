"""The 2026-09-18 publish failure, pinned down as tests.

What happened: nflverse carried injury weeks [1, 2], the board was for week 3,
`attach_injuries` fell back to week 2, none of the 53 players on a NYG @ LAR
showdown appeared on week 2's report, the zero-match guard raised SystemExit,
SystemExit walked through publish.py's `except Exception`, and a run in which
fourteen slates had already built correctly committed nothing. The page served
the previous day's data and the ownership rewrite looked like it had not taken.

Three separate defects in that sentence, one test each.
"""
from __future__ import annotations

import pandas as pd

import data
import project as P


def report(weeks, names=("Real Player",), season=2026) -> pd.DataFrame:
    rows = []
    for w in weeks:
        for n in names:
            rows.append({"season": season, "week": w, "name": n,
                         "norm": data.normalise(n) if hasattr(data, "normalise")
                         else n.lower(), "team": "BUF", "playing": "out"})
    return pd.DataFrame(rows)


def pool(names) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names),
                         "salary": [5000] * len(names),
                         "playing": ["clear"] * len(names)})


# ---------------------------------------------------------------------------
# Defect 1: last week's report is a different fact, not a stale copy.
# ---------------------------------------------------------------------------

def test_a_future_week_does_not_fall_back_to_last_week():
    """Week 3 asked for, weeks [1, 2] available: apply NOTHING.

    The old code applied week 2, which marks a player who has since been
    cleared as OUT and a player hurt on Sunday as CLEAR - wrong in both
    directions on the same board.
    """
    out = data.attach_injuries(pool(["Real Player", "Someone Else"]),
                               report([1, 2]), 2026, 3)
    assert out.attrs["injury_reason"] == "not-published"
    assert out.attrs["injury_week_used"] is None
    # Real Player is "out" in week 2's report and must NOT be marked out here.
    assert set(out["playing"]) == {"unknown"}


def test_missing_data_reads_unknown_not_clear():
    """"clear" means confirmed healthy everywhere downstream. It is a claim.

    With no report for the week there is no such claim to make, and writing
    "clear" is how a board goes out looking filtered when it is not.
    """
    out = data.attach_injuries(pool(["A", "B", "C"]), report([1, 2]), 2026, 3)
    assert "clear" not in set(out["playing"])


def test_a_present_week_is_still_applied_normally():
    """The fix must not break the ordinary case."""
    out = data.attach_injuries(pool(["Real Player", "Healthy Guy"]),
                               report([1, 2, 3]), 2026, 3)
    assert out.attrs["injury_reason"] == "ok"
    assert out.attrs["injury_matched"] == 1
    assert out.loc[0, "playing"] == "out"
    assert out.loc[1, "playing"] == "clear"


def test_a_hole_inside_the_range_is_not_the_same_as_being_early():
    """Weeks [1, 3] with week 2 asked for means the file is damaged."""
    out = data.attach_injuries(pool(["A"]), report([1, 3]), 2026, 2)
    assert out.attrs["injury_reason"] == "gap"


# ---------------------------------------------------------------------------
# Defect 2: a genuine join failure must still stop the board.
# ---------------------------------------------------------------------------

def test_a_real_join_failure_is_still_a_join_failure():
    """Week 3 present, nobody matched: that is the dangerous case, kept.

    This is the Pacheco case - the report exists, and not one name on it
    reached the board, so the board reads as a league where nobody is hurt.
    """
    out = data.attach_injuries(pool(["Nobody Here", "Nor Here"]),
                               report([3]), 2026, 3)
    assert out.attrs["injury_matched"] == 0
    assert out.attrs["injury_reason"] == "no-match"


# ---------------------------------------------------------------------------
# Defect 3: the refusal must not be able to kill the whole publish.
# ---------------------------------------------------------------------------

def test_the_refusal_is_catchable_by_publish():
    """publish.py builds each slate inside `except Exception: continue`.

    SystemExit inherits from BaseException, so it went straight through that
    handler and took fourteen good slates down with one bad one. This asserts
    the promise publish.py's comment makes.
    """
    assert issubclass(P.BoardUnfiltered, Exception)
    assert not issubclass(P.BoardUnfiltered, SystemExit)

    caught = False
    try:
        try:
            raise P.BoardUnfiltered("board cannot see injuries")
        except Exception:                                      # noqa: BLE001
            caught = True
    except BaseException:                                      # noqa: BLE001
        caught = False
    assert caught, "publish.py's per-slate handler must catch this"


def test_systemexit_would_not_have_been_catchable():
    """The control: prove the old behaviour really did escape.

    Without this the test above passes trivially and proves nothing about
    what was actually wrong.
    """
    escaped = False
    try:
        try:
            raise SystemExit("the old guard")
        except Exception:                                      # noqa: BLE001
            pass
    except SystemExit:
        escaped = True
    assert escaped
