"""The 2026-09-18 publish failure, pinned down as tests.

Runs as a script, in the house style: `python test_injury_week.py`.

What happened: nflverse carried injury weeks [1, 2], the board was for week 3,
`attach_injuries` fell back to week 2, none of the 53 players on a NYG @ LAR
showdown appeared on week 2's report, the zero-match guard raised SystemExit,
SystemExit walked through publish.py's `except Exception`, and a run in which
fourteen slates had already built correctly committed nothing. The page served
the previous day's data while the run looked fine at a glance.

Three separate defects in that sentence. One section each.
"""
from __future__ import annotations

import sys

import pandas as pd

import data
import project as P


def report(weeks, names=("Real Player",), season=2026) -> pd.DataFrame:
    rows = []
    for w in weeks:
        for n in names:
            rows.append({"season": season, "week": w, "name": n,
                         "team": "BUF", "playing": "out"})
    return pd.DataFrame(rows)


def pool(names) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names),
                         "salary": [5000] * len(names),
                         "playing": ["clear"] * len(names)})


# ---------------------------------------------------------------------------
# Defect 1: last week's report is a different fact, not a stale copy.
# ---------------------------------------------------------------------------

def a_future_week_does_not_fall_back():
    """Week 3 asked for, weeks [1, 2] available: apply NOTHING.

    The old code applied week 2, which marks a player since cleared as OUT
    and a player hurt on Sunday as CLEAR - wrong in both directions on the
    same board.
    """
    out = data.attach_injuries(pool(["Real Player", "Someone Else"]),
                               report([1, 2]), 2026, 3)
    assert out.attrs["injury_reason"] == "not-published", out.attrs
    assert out.attrs["injury_week_used"] is None
    assert set(out["playing"]) == {"unknown"}, (
        "Real Player is OUT in week 2's report and must not be marked out on "
        "a week 3 board")


def missing_data_reads_unknown_not_clear():
    """"clear" is a claim - confirmed healthy - and there is nothing to claim."""
    out = data.attach_injuries(pool(["A", "B", "C"]), report([1, 2]), 2026, 3)
    assert "clear" not in set(out["playing"])


def a_present_week_is_still_applied():
    """The fix must not break the ordinary case."""
    out = data.attach_injuries(pool(["Real Player", "Healthy Guy"]),
                               report([1, 2, 3]), 2026, 3)
    assert out.attrs["injury_reason"] == "ok"
    assert out.attrs["injury_matched"] == 1
    assert out.loc[0, "playing"] == "out"
    assert out.loc[1, "playing"] == "clear"


def a_hole_is_not_the_same_as_being_early():
    """Weeks [1, 3] with week 2 asked for means the file is damaged."""
    out = data.attach_injuries(pool(["A"]), report([1, 3]), 2026, 2)
    assert out.attrs["injury_reason"] == "gap"


# ---------------------------------------------------------------------------
# Defect 2: a genuine join failure must still stop the board.
# ---------------------------------------------------------------------------

def a_real_join_failure_still_stops_the_board():
    """Week 3 present, nobody matched - the Pacheco case, hard stop kept."""
    out = data.attach_injuries(pool(["Nobody Here", "Nor Here"]),
                               report([3]), 2026, 3)
    assert out.attrs["injury_matched"] == 0
    assert out.attrs["injury_reason"] == "no-match"


# ---------------------------------------------------------------------------
# Defect 3: the refusal must not be able to kill the whole publish.
# ---------------------------------------------------------------------------

def the_refusal_is_catchable_by_publish():
    """publish.py builds each slate inside `except Exception: continue`.

    SystemExit inherits from BaseException, so it went through that handler
    and took fourteen good slates down with one bad one.
    """
    assert issubclass(P.BoardUnfiltered, Exception)
    assert not issubclass(P.BoardUnfiltered, SystemExit)
    caught = False
    try:
        raise P.BoardUnfiltered("board cannot see injuries")
    except Exception:                                          # noqa: BLE001
        caught = True
    assert caught, "publish.py's per-slate handler must catch this"


def systemexit_would_not_have_been_catchable():
    """The control. Without it the test above proves nothing."""
    escaped = False
    try:
        try:
            raise SystemExit("the old guard")
        except Exception:                                      # noqa: BLE001
            pass
    except SystemExit:
        escaped = True
    assert escaped


SUITES = [
    ("LAST WEEK'S INJURY REPORT IS A DIFFERENT FACT, NOT A STALE COPY", [
        ("a week the report does not reach yet applies nothing",
         a_future_week_does_not_fall_back),
        ("and reads UNKNOWN rather than clear",
         missing_data_reads_unknown_not_clear),
        ("a week the report does have is applied normally",
         a_present_week_is_still_applied),
        ("a hole inside the covered range is a damaged file, not an early one",
         a_hole_is_not_the_same_as_being_early),
    ]),
    ("A SILENT FILTER IS STILL WORSE THAN NO FILTER", [
        ("this week's report matching nobody still stops the board",
         a_real_join_failure_still_stops_the_board),
    ]),
    ("ONE DEAD SLATE MUST NOT TAKE THE WHOLE PUBLISH DOWN", [
        ("the refusal is an Exception, so publish.py can catch it",
         the_refusal_is_catchable_by_publish),
        ("and SystemExit provably could not be caught",
         systemexit_would_not_have_been_catchable),
    ]),
]


def main() -> int:
    if not hasattr(P, "BoardUnfiltered"):
        print("\n" + "=" * 66)
        print("THE OLD project.py IS DEPLOYED.")
        print("=" * 66)
        print("There is no BoardUnfiltered, so the injury guard still raises")
        print("SystemExit - which publish.py cannot catch. One slate with a")
        print("bad injury join will discard every other slate in the run.")
        print("\nReplace project.py and data.py in the repo root and re-run.")
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
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
