"""Projections for one live slate.

    python project.py --draft-group 153086
    python project.py --slate showdown        # picks the busiest showdown
    python project.py --slate classic

Loads the seasons, builds leak-free features, fits the quantile models, pulls
the slate's player pool from DraftKings, joins on name, and prints a
distribution for every player who can be matched.

About the join rate
-------------------
The first version of this gated on "matched / everyone on the slate", which is
the wrong denominator and said so loudly: a DraftKings showdown pool carries a
kicker, two defences and a tail of minimum-salary special-teamers who have no
skill-position history and never will. Counting them as misses turns a healthy
join into a red light.

So three numbers are reported instead of one, and the gate sits on the middle
of them:

  overall     - every name in the pool, for context
  projectable - skill positions only, which is what this model can even fit
  by salary   - the share of the pool's total salary that matched, which is the
                number that actually says whether anyone who matters is missing

A miss at $200 is a long snapper. A miss at $9,000 is a starting receiver and
the run should stop. The salary weighting is what tells those apart, and every
unmatched player is printed WITH his price so a failure diagnoses itself rather
than leaving the next person to guess.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import config
import data
import features as F
import model as M

log = logging.getLogger(__name__)

# Below this share of the PROJECTABLE slate matched, something structural is
# wrong and the output should not be used.
MIN_JOIN_RATE = 0.85

# And below this share of slate salary, a starter is missing even if the count
# looks fine - one unmatched $11,000 quarterback is worse than thirty
# unmatched $200 linemen.
MIN_SALARY_COVERAGE = 0.90


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


def history_columns(latest: pd.DataFrame) -> list[str]:
    """The columns to carry across from history, each exactly once.

    `games_played` is both an identifying column worth keeping and a member of
    FEATURES. Listing it in both places produced a frame with two columns of
    that name, which pandas tolerated and sklearn did not - the fit died on
    `Expected unique column names` with the slate already loaded. Order is
    preserved so the frame stays readable; `dict.fromkeys` is the deduplication.
    """
    wanted = ["norm", "player_id", "position", "team", "games_played"]
    wanted += list(F.FEATURES)
    return [c for c in dict.fromkeys(wanted) if c in latest.columns]


def coverage(pool: pd.DataFrame, merged: pd.DataFrame) -> dict:
    """Three views of the join, because one of them is misleading on its own."""
    matched = merged["player_id"].notna()
    skill = merged["position"].isin(config.SKILL_POSITIONS)

    salary = pd.to_numeric(merged.get("salary"), errors="coerce").fillna(0.0)
    total_salary = float(salary.sum())

    missed = merged[~matched][["name", "position", "salary"]].copy()
    missed = missed.sort_values("salary", ascending=False)

    return {
        "overall": float(matched.mean()) if len(merged) else 0.0,
        "projectable": (float(matched[skill].mean()) if int(skill.sum())
                        else 0.0),
        "projectable_n": int(skill.sum()),
        "projectable_matched": int((matched & skill).sum()),
        "by_salary": (float(salary[matched].sum() / total_salary)
                      if total_salary else 0.0),
        "total": int(len(merged)),
        "matched": int(matched.sum()),
        "misses": missed.to_dict("records"),
    }


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

    merged = pool.merge(latest[history_columns(latest)],
                        on="norm", how="left", suffixes=("", "_hist"))
    dupes = merged.columns[merged.columns.duplicated()].tolist()
    if dupes:                       # belt and braces; the guard above is the fix
        raise SystemExit(f"duplicated columns after merge: {dupes}")

    cov = coverage(pool, merged)
    log.info("join: %d of %d matched overall; %d of %d projectable (%.1f%%); "
             "%.1f%% of slate salary",
             cov["matched"], cov["total"], cov["projectable_matched"],
             cov["projectable_n"], 100 * cov["projectable"],
             100 * cov["by_salary"])

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

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "draft_group": draft_group,
        "site": site,
        "trained_rows": proj.trained_rows,
        "coverage": cov,
        "matched": int(len(known)),
        "slate_size": int(len(pool)),
        "players": known,
    }


def report(out: dict, top: int = 40) -> str:
    df = out["players"]
    cov = out["coverage"]
    lines = [
        f"PROJECTIONS  draft group {out['draft_group']}  ({out['site'].upper()})",
        "=" * 78,
        f"trained on {out['trained_rows']:,} player-weeks",
        f"join: {cov['projectable_matched']}/{cov['projectable_n']} projectable "
        f"({cov['projectable'] * 100:.1f}%)   "
        f"{cov['by_salary'] * 100:.1f}% of slate salary   "
        f"{cov['matched']}/{cov['total']} overall",
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

    if cov["misses"]:
        lines += ["", f"{len(cov['misses'])} slate players had no history, "
                      f"most expensive first:"]
        for m in cov["misses"][:20]:
            sal = m.get("salary")
            sal = f"${int(sal):>6,}" if pd.notna(sal) else "     ?"
            lines.append(f"  {str(m['name'])[:24]:<25} {str(m['position']):<5}"
                         f"{sal}")
        lines.append("  (kickers and defences are expected here - this model "
                     "fits skill positions only)")

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
    p.add_argument("--force", action="store_true",
                   help="write the file even if the join gate fails")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.draft_group:
        dg, label = args.draft_group, "(given)"
    else:
        dg, label = pick_slate(args.slate)
    print(f"slate: {label}\n")

    out = run(dg, site=args.site)
    print(report(out))

    cov = out["coverage"]
    bad = (cov["projectable"] < MIN_JOIN_RATE
           or cov["by_salary"] < MIN_SALARY_COVERAGE)
    if bad and not args.force:
        print(f"\nSTOPPING: {cov['projectable'] * 100:.0f}% of projectable "
              f"players matched (floor {MIN_JOIN_RATE * 100:.0f}%), covering "
              f"{cov['by_salary'] * 100:.0f}% of slate salary "
              f"(floor {MIN_SALARY_COVERAGE * 100:.0f}%).\nThe unmatched list "
              f"above is sorted by price. If the expensive names on it are\n"
              f"kickers and defences, pass --force. If any of them is a "
              f"starter, the join\nis broken and nothing here should be used.")
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
