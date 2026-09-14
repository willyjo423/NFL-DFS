"""Projections for one live slate.

    python project.py --draft-group 153086
    python project.py --slate showdown        # picks the busiest showdown
    python project.py --slate classic

Loads the seasons, builds leak-free features, fits the quantile models, pulls
the slate's player pool from DraftKings, joins on name, and prints a
distribution for every player who can be matched.

The join rate is printed and enforced. A silent 90% would drop precisely the
players who changed teams or were signed last week, and those are the ones a
projection is most needed for - so below the floor it stops rather than
quietly handing over a slate with holes in it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import data
import features as F
import model as M

log = logging.getLogger(__name__)

# Below this share of the slate matched, something structural is wrong and the
# output should not be used.
MIN_JOIN_RATE = 0.85


def load_seasons(seasons: list[int]) -> pd.DataFrame:
    """Player weeks, tolerating a season that has not been published yet."""
    got, missing = [], []
    for s in seasons:
        try:
            got.append(data.player_weeks([s]))
        except data.DataUnavailable:
            missing.append(s)
    if not got:
        raise SystemExit(f"no player data for any of {seasons}")
    if missing:
        log.warning("no nflverse file yet for %s - normal within a day or two "
                    "of a week being played", missing)
    return pd.concat(got, ignore_index=True)


def pick_slate(kind: str) -> tuple[int, str]:
    """The busiest live slate of the kind asked for."""
    want = ("Showdown Captain Mode" if kind.lower().startswith("show")
            else "Classic")
    slates = data.slates()
    match = slates[slates["game_type"] == want]
    if match.empty:
        raise SystemExit(f"no live {want} slate. Available:\n"
                         f"{slates[['draft_group', 'game_type', 'contests']]}")
    row = match.iloc[0]
    return int(row["draft_group"]), str(row["example"])


def run(draft_group: int, site: str = "dk",
        seasons: list[int] | None = None) -> dict:
    seasons = seasons or list(range(config.TRAIN_START_SEASON,
                                    datetime.now(timezone.utc).year + 1))
    log.info("loading seasons %s-%s", seasons[0], seasons[-1])
    weeks = load_seasons(seasons)
    log.info("%d player-weeks, through %s week %s", len(weeks),
             int(weeks["season"].max()),
             int(weeks[weeks["season"] == weeks["season"].max()]["week"].max()))

    built = F.build(weeks, site=site)
    proj = M.Projections().fit(built)
    latest = M.latest_rows(built)

    pool = data.draftables(draft_group)
    report = data.join_report(latest["norm"], pool["norm"])
    log.info("join: %d of %d slate players matched (%.1f%%)",
             report["matched"], report["total"], 100 * report["rate"])

    merged = pool.merge(
        latest[["norm", "player_id", "position", "team", "games_played"]
               + [c for c in F.FEATURES if c in latest.columns]],
        on="norm", how="left", suffixes=("", "_hist"))

    known = merged[merged["player_id"].notna()].copy()
    if known.empty:
        raise SystemExit("nothing in this slate matched the history")

    q = proj.predict(known)
    for c in q.columns:
        known[c] = q[c].to_numpy()

    known["value"] = (known["median"] / (known["salary"] / 1000.0)).round(2)
    known["ceiling_value"] = (known["ceiling"]
                              / (known["salary"] / 1000.0)).round(2)
    known = known.sort_values("median", ascending=False).reset_index(drop=True)

    unmatched = merged[merged["player_id"].isna()]["name"].tolist()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "draft_group": draft_group,
        "site": site,
        "trained_rows": proj.trained_rows,
        "join_rate": round(report["rate"], 4),
        "matched": int(len(known)),
        "slate_size": int(len(pool)),
        "unmatched": unmatched[:40],
        "players": known,
    }


def report(out: dict, top: int = 30) -> str:
    df = out["players"]
    lines = [
        f"PROJECTIONS  draft group {out['draft_group']}  ({out['site'].upper()})",
        "=" * 78,
        f"trained on {out['trained_rows']:,} player-weeks",
        f"matched {out['matched']} of {out['slate_size']} slate players "
        f"({out['join_rate'] * 100:.1f}%)",
        "",
        f"{'player':<22}{'pos':<5}{'team':<5}{'salary':>8}"
        f"{'median':>8}{'ceiling':>9}{'value':>7}",
        "-" * 78,
    ]
    for r in df.head(top).itertuples(index=False):
        lines.append(
            f"{str(r.name)[:21]:<22}{str(r.position):<5}{str(r.team):<5}"
            f"{int(r.salary):>8,}{r.median:>8.1f}{r.ceiling:>9.1f}"
            f"{r.value:>7.2f}")

    lines += ["", "Best value per $1,000 of salary, among players over 8 points:"]
    worth = df[df["median"] >= 8].nlargest(10, "value")
    for r in worth.itertuples(index=False):
        lines.append(f"  {str(r.name)[:24]:<25} ${int(r.salary):>6,}  "
                     f"{r.median:5.1f} pts  {r.value:5.2f} per $1k")

    lines += ["", "Highest ceilings, which is what a tournament pays for:"]
    for r in df.nlargest(10, "ceiling").itertuples(index=False):
        lines.append(f"  {str(r.name)[:24]:<25} ${int(r.salary):>6,}  "
                     f"{r.median:5.1f} median -> {r.ceiling:5.1f} ceiling")

    if out["unmatched"]:
        lines += ["", f"{len(out['unmatched'])} slate players had no history "
                      f"and were skipped:",
                  "  " + ", ".join(out["unmatched"][:12])]
    lines += ["", "-" * 78,
              "These are projections, not a lineup. No correlation, no field,",
              "no contest objective yet - all of that is what turns a list of",
              "numbers into a decision, and none of it has been graded."]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int)
    p.add_argument("--slate", default="showdown")
    p.add_argument("--site", default="dk", choices=["dk", "fd"])
    p.add_argument("--out", default=str(config.PROJECTIONS / "latest.csv"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.draft_group:
        dg, label = args.draft_group, "(given)"
    else:
        dg, label = pick_slate(args.slate)
    print(f"slate: {label}\n")

    out = run(dg, site=args.site)
    print(report(out))

    if out["join_rate"] < MIN_JOIN_RATE:
        print(f"\nSTOPPING: only {out['join_rate'] * 100:.0f}% of the slate "
              f"matched.\nThe players who fail to match are the ones who "
              f"changed teams or were\nsigned recently - exactly who a "
              f"projection is most needed for. Fix the\njoin before using "
              f"any of this.")
        return 1

    df = out["players"]
    keep = [c for c in ("name", "position", "team", "salary", "q10", "q25",
                        "q50", "q75", "q90", "q97", "median", "mean",
                        "ceiling", "spread", "value", "ceiling_value")
            if c in df.columns]
    df[keep].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
