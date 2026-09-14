"""Archive today's slates before they disappear.

This is the most time-critical file in the project and the least interesting,
which is a bad combination.

DraftKings publishes salaries for a slate, the slate locks, and the data is
gone - there is no historical endpoint. Rotoguru, the only public archive,
stopped at 2021. So for 2022 onward the only salary history that will ever
exist is the one captured live, week by week, and **every week that passes
uncaptured is permanently lost**. No amount of later work recovers it.

That is why this runs before any of the modelling. The projection model can be
built at leisure from twenty-five years of nflverse data. The optimiser it
feeds cannot be backtested at all without prices, and prices only exist while
the slate is on the board.

What gets kept, and why each piece
----------------------------------
**Player pool and salaries** - the obvious part. Who was available, at what
price, in which game.

**Contest metadata** - entry fee, field size, prize pool, maximum entries. Not
decoration: a tournament's payout curve is what defines the objective the
optimiser is solving for, and reconstructing it after the fact is impossible.
A lineup graded against the wrong payout structure is graded against the wrong
question.

**Every capture, not just the last one.** Salaries do not move, but player pools
do - a player ruled out on Saturday is removed from the pool, and the Sunday
capture cannot tell you he was ever in it. Keeping each day's snapshot
preserves that, which is what makes a late-swap model possible later.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import config
import data

log = logging.getLogger(__name__)

# Slate types worth keeping. In-game and single-stat contests are different
# games with different rules; Madden is not football at all and is filtered
# upstream in `data.contests`.
KEEP_TYPES = ("Classic", "Showdown Captain Mode")


def capture(keep_types: tuple = KEEP_TYPES, all_types: bool = False) -> dict:
    """Fetch every live slate and write it down."""
    stamp = datetime.now(timezone.utc)
    day = stamp.strftime("%Y-%m-%d")
    folder = config.SLATES / day
    folder.mkdir(parents=True, exist_ok=True)

    contests = data.contests()
    slates = data.slates(contests)
    if not all_types:
        slates = slates[slates["game_type"].isin(keep_types)]

    # The contest list is written whole, because the payout structure cannot be
    # reconstructed later and it is what the GPP objective is defined against.
    contests_path = folder / f"contests_{stamp:%H%M}.csv"
    contests.to_csv(contests_path, index=False)

    written, failed = [], {}
    for row in slates.itertuples(index=False):
        dg = int(row.draft_group)
        try:
            players = data.draftables(dg)
        except data.DataUnavailable as exc:
            failed[dg] = str(exc)[:160]
            log.warning("draft group %s: %s", dg, exc)
            continue

        players["draft_group"] = dg
        players["game_type"] = row.game_type
        players["captured_at"] = stamp.isoformat(timespec="seconds")
        path = folder / f"dg{dg}_{stamp:%H%M}.csv"
        players.to_csv(path, index=False)
        written.append({
            "draft_group": dg,
            "game_type": row.game_type,
            "players": int(len(players)),
            "games": int(players["game"].nunique()),
            "salary_min": int(players["salary"].min()),
            "salary_max": int(players["salary"].max()),
            "starts": str(row.starts),
            "contests": int(row.contests),
            "file": path.name,
        })

    summary = {
        "captured_at": stamp.isoformat(timespec="seconds"),
        "day": day,
        "slates": written,
        "failed": failed,
        "contests_file": contests_path.name,
        "contests_kept": int(len(contests)),
    }
    (folder / f"summary_{stamp:%H%M}.json").write_text(
        json.dumps(summary, indent=2))
    _write_latest(summary)
    return summary


def _write_latest(summary: dict) -> None:
    """A pointer to the newest capture, so the rest of the build has one path."""
    (config.DATA / "latest_capture.json").write_text(
        json.dumps(summary, indent=2))


def load_slate(day: str, draft_group: int,
               which: str = "last") -> pd.DataFrame:
    """One archived slate.

    `which="last"` takes the capture closest to lock, which is the pool that
    was actually playable. `which="first"` takes the earliest, which is the one
    that still contains the players who were later ruled out - the difference
    between the two is the injury news, and it is recoverable only because
    every capture is kept.
    """
    folder = config.SLATES / day
    files = sorted(folder.glob(f"dg{draft_group}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no capture of {draft_group} on {day}")
    path = files[-1] if which == "last" else files[0]
    return pd.read_csv(path)


def coverage() -> pd.DataFrame:
    """What has been archived so far. Printed after every run.

    A capture job that silently stopped working looks exactly like a quiet
    week, which is how a season of missing prices happens. This is the line
    that makes it visible.
    """
    rows = []
    for folder in sorted(config.SLATES.glob("*")):
        if not folder.is_dir():
            continue
        files = list(folder.glob("dg*.csv"))
        if not files:
            continue
        groups = {f.name.split("_")[0] for f in files}
        rows.append({"day": folder.name, "captures": len(files),
                     "draft_groups": len(groups)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all-types", action="store_true",
                   help="also keep in-game and single-stat slates")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summary = capture(all_types=args.all_types)

    print(f"captured {len(summary['slates'])} slates at "
          f"{summary['captured_at']}")
    for s in summary["slates"]:
        print(f"  {s['draft_group']}  {s['game_type']:<24} "
              f"{s['players']:>4} players  {s['games']:>2} games  "
              f"${s['salary_min']:,}-${s['salary_max']:,}  "
              f"{s['contests']:,} contests")
    if summary["failed"]:
        print(f"\n  {len(summary['failed'])} draft groups failed:")
        for dg, err in list(summary["failed"].items())[:5]:
            print(f"    {dg}: {err}")

    cov = coverage()
    if not cov.empty:
        print(f"\narchive: {len(cov)} days, "
              f"{int(cov['captures'].sum())} slate captures, "
              f"{cov['day'].min()} to {cov['day'].max()}")
        print("This archive is the only salary history that will exist for "
              "these weeks.")
    if not summary["slates"]:
        print("\nNOTHING CAPTURED. If this repeats, the endpoint has changed "
              "and\nevery week it stays broken is permanently lost data.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
