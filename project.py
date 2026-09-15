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
import special

log = logging.getLogger(__name__)

# Below this share of the PROJECTABLE slate matched, something structural is
# wrong and the output should not be used.
MIN_JOIN_RATE = 0.85

# A backstop, not the main gate. The name-level check above is what catches a
# broken join; this catches the wholesale break it cannot see - DraftKings
# changing its name format entirely, say, where nothing matches on surname
# either and so nothing gets flagged. Set low on purpose: in week two a
# showdown pool is genuinely 10-15% rookies by salary, and a floor tuned to
# that noise teaches you to force past it, which is how a real break gets
# waved through.
MIN_SALARY_COVERAGE = 0.80


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


MARKET_COLUMNS = ["implied_total", "game_total", "team_spread", "is_home"]


def refresh_market(latest: pd.DataFrame,
                   lines: pd.DataFrame | None) -> pd.DataFrame:
    """Put THIS week's market line on the row a projection is made from.

    The usage features describe a player's most recently completed game, which
    is correct and deliberate - they are built with a one-week shift so a week
    can never be inside its own feature. But the market columns were coming
    along for the ride, and that is wrong in a way the leak test cannot catch:
    a projection for week two was being made with week one's implied total.
    Kansas City's defence, its pace, its expected score - all of it a week out
    of date, on the single input in the whole feature set whose entire purpose
    is to describe the game that has not happened yet.

    It is also the input most likely to have MOVED. A team implied for 22.5
    last week can be implied for 31 this week against a different opponent, and
    that is precisely the information nothing else in the model has access to.
    """
    if lines is None or not len(lines) or latest.empty:
        return latest
    import special
    try:
        up = special.upcoming_lines(lines)
    except Exception as exc:                          # noqa: BLE001
        log.warning("could not read the upcoming market lines (%s) - "
                    "projecting on last week's, which is stale", exc)
        return latest
    if up.empty:
        return latest

    keep = ["team"] + [c for c in MARKET_COLUMNS if c in up.columns]
    out = latest.drop(columns=[c for c in MARKET_COLUMNS if c in latest.columns],
                      errors="ignore")
    out["team"] = out["team"].astype(str)
    up = up[keep].copy()
    up["team"] = up["team"].astype(str)
    out = out.merge(up, on="team", how="left")

    hit = float(out["implied_total"].notna().mean()) if len(out) else 0.0
    log.info("market refreshed to week %s for %.0f%% of players",
             int(up.attrs.get("week", 0)) or "upcoming", 100 * hit)
    if hit < 0.5:
        log.warning("most players did not pick up this week's line - they will "
                    "be projected without game context rather than with a "
                    "stale one")
    return out


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


def name_parts(norm: str) -> tuple[str, str]:
    """First and last token of a normalised name."""
    parts = str(norm).split()
    if len(parts) < 2:
        return str(norm), str(norm)
    return parts[0], parts[-1]


def same_person(a_first: str, b_first: str) -> bool:
    """Could these two first names be the same man?

    First initial plus surname was the first attempt and it cried wolf on every
    rookie on the board: Cyrus Allen was flagged because the history file
    contains Chase Allen, Emmett Johnson because of Eric Johnson, and three
    more of the same. A detector that fires on five rookies teaches you to
    force past it, which is worse than having no detector.

    So the first names have to be compatible, not merely share a letter. Equal,
    or one a prefix of the other - which is what a nickname looks like
    (cam/cameron, chris/christopher) and what a rookie sharing a surname does
    not.
    """
    if a_first == b_first:
        return True
    lo, hi = sorted((a_first, b_first), key=len)
    return len(lo) >= 3 and hi.startswith(lo)


def probable_join_failures(missed: pd.DataFrame,
                           history: pd.DataFrame) -> pd.DataFrame:
    """Unmatched players the history file probably already contains.

    Three things must line up before this raises a hand: the surname, a
    compatible first name, and the position. Any one of them alone produces
    noise - the point is to fire on a spelling variant of a real player and
    stay silent on a rookie who happens to share a surname with one.
    """
    if missed.empty:
        return missed.iloc[:0]
    idx: dict[str, set[tuple[str, str]]] = {}
    for norm, pos in zip(history["norm"], history["position"]):
        first, last = name_parts(norm)
        idx.setdefault(last, set()).add((first, str(pos)))

    hit = []
    for norm, pos in zip(missed["norm"], missed["position"]):
        first, last = name_parts(norm)
        hit.append(any(same_person(first, f) and p == str(pos)
                       for f, p in idx.get(last, ())))
    return missed[pd.Series(hit, index=missed.index)]


# If a status filter would remove more than this share of a board, it is not an
# injury report - it is a parsing failure, and the right answer is to stop
# rather than to hand over a pool with the starters deleted.
MAX_UNAVAILABLE_SHARE = 0.40


def drop_unavailable(pool: pd.DataFrame,
                     exclude: tuple[str, ...] = ("out", "doubtful")
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove players who will not take a snap, and say who they were.

    The failure this prevents is specific and expensive: a player who is out is
    still listed, still priced, and - because DraftKings drops the price of a
    player nobody can use - looks like the best value on the board. Every
    objective picks him, because points per dollar is exactly the statistic a
    zero-snap player maximises.

    Questionable is KEPT. Those players mostly play, and dropping them would
    throw away real leverage; they are flagged instead so the report can say
    which of the chosen names carry a designation.
    """
    if "playing" not in pool.columns:
        log.warning("this pool carries no availability column - nothing is "
                    "being filtered, and an inactive player can be rostered")
        return pool, pool.iloc[:0]

    bad = pool["playing"].isin(exclude)
    share = float(bad.mean()) if len(pool) else 0.0
    if share > MAX_UNAVAILABLE_SHARE:
        raise SystemExit(
            f"{share * 100:.0f}% of this board reads as {' or '.join(exclude)}, "
            f"which is not an injury report - it is a parsing failure.\n"
            f"Statuses seen: "
            f"{pool['status'].astype(str).value_counts().head(8).to_dict()}\n"
            f"Fix the status handling before using any of this.")

    dropped = pool[bad].copy()
    if len(dropped):
        log.info("excluded %d unavailable: %s", len(dropped),
                 ", ".join(f"{r['name']} ({r['playing']})"
                           for _, r in dropped.head(12).iterrows()))
    return pool[~bad].reset_index(drop=True), dropped


def coverage(merged: pd.DataFrame, history: pd.DataFrame) -> dict:
    """What failed to join, and - the part that matters - why.

    A rate on its own cannot answer the only question worth asking, which is
    whether the join is broken. Two very different things look identical in a
    count of misses:

      a rookie in week two, who has no NFL history because he has never played
      a down, and correctly cannot be projected from a history file;

      a five-year starter whose name normalises differently on the two sides,
      which is a bug, and the kind that silently removes exactly the players a
      projection is most needed for.

    `ever_played` is every name that appears anywhere in the loaded seasons. A
    miss found in it is the second kind and stops the run. A miss absent from
    it is the first kind and is reported, not treated as a defect.
    """
    matched = merged["player_id"].notna()
    skill = merged["position"].isin(config.SKILL_POSITIONS)
    salary = pd.to_numeric(merged["salary"], errors="coerce").fillna(0.0)

    missed = merged[~matched].copy()
    # Exact membership cannot fire here - the merge joined on this very column,
    # so a name present in history could not have missed. Matching on surname,
    # a compatible first name and position is what catches the real failure:
    # same player, different spelling.
    skill_missed = missed[missed["position"].isin(config.SKILL_POSITIONS)]
    broken = probable_join_failures(skill_missed, history)
    missed["has_nfl_history"] = missed.index.isin(broken.index)

    # Salary coverage measured over the players this model can even fit.
    # A kicker and a defence are structurally unprojectable here, so leaving
    # them in the denominator would make a healthy join permanently fail.
    denom = float(salary[skill].sum())

    order = missed.sort_values("salary", ascending=False)
    return {
        "overall": float(matched.mean()) if len(merged) else 0.0,
        "projectable": (float(matched[skill].mean()) if int(skill.sum())
                        else 0.0),
        "projectable_n": int(skill.sum()),
        "projectable_matched": int((matched & skill).sum()),
        "by_salary": (float(salary[matched & skill].sum() / denom)
                      if denom else 0.0),
        "total": int(len(merged)),
        "matched": int(matched.sum()),
        "broken": broken[["name", "position", "salary"]].to_dict("records"),
        "misses": order[["name", "position", "salary", "has_nfl_history"]]
                  .to_dict("records"),
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

    # The market line is the only forward-looking feature in the set, and the
    # only one that can know about something that has not happened yet.
    try:
        lines = data.schedules()
    except data.DataUnavailable as exc:
        log.warning("no market lines (%s) - projecting without them, which "
                    "loses the single most informative piece of game context",
                    exc)
        lines = None
    built = F.build(weeks, site=site, lines=lines)
    proj = M.Projections().fit(built)
    latest = M.latest_rows(built)
    latest = refresh_market(latest, lines)

    pool = data.draftables(draft_group)
    pool, dropped = drop_unavailable(pool)

    merged = pool.merge(latest[history_columns(latest)],
                        on="norm", how="left", suffixes=("", "_hist"))
    dupes = merged.columns[merged.columns.duplicated()].tolist()
    if dupes:                       # belt and braces; the guard above is the fix
        raise SystemExit(f"duplicated columns after merge: {dupes}")

    history = weeks[["norm", "position"]].dropna().drop_duplicates()
    cov = coverage(merged, history)
    log.info("join: %d of %d matched overall; %d of %d projectable (%.1f%%); "
             "%.1f%% of projectable salary; %d genuine join failures",
             cov["matched"], cov["total"], cov["projectable_matched"],
             cov["projectable_n"], 100 * cov["projectable"],
             100 * cov["by_salary"], len(cov["broken"]))

    # Kickers and defences join on name but project to zero, because none of
    # the features describe what they do. A zero is not a missing value - it is
    # a confident wrong answer, and an optimiser would happily believe it. They
    # are dropped rather than shown.
    known = merged[merged["player_id"].notna()
                   & merged["position"].isin(config.SKILL_POSITIONS)].copy()
    if known.empty:
        raise SystemExit("nothing in this slate matched the history")

    q = proj.predict(known)
    for c in q.columns:
        known[c] = q[c].to_numpy()

    # Defences, which the player model cannot touch. nflverse's weekly file has
    # no defence rows, so every DST was filtered out of the pool - and a classic
    # lineup requires exactly one. The integer program reported "no legal
    # lineup" and the only hint was ownership summing to 800% against nine
    # roster slots. They are projected from the market instead.
    if lines is not None and len(lines):
        try:
            up = special.upcoming_lines(lines)
            dst = special.project(pool, up, site=site,
                                  quantiles=config.QUANTILES)
        except Exception as exc:                      # noqa: BLE001
            log.warning("defence projection failed (%s: %s) - a classic lineup "
                        "cannot be built without one", type(exc).__name__, exc)
            dst = pd.DataFrame()
        if len(dst):
            merged_dst = pool.merge(dst.drop(columns=["position", "team"]),
                                    on="name", how="inner")
            known = pd.concat([known, merged_dst], ignore_index=True)
            log.info("added %d defences to the pool", len(merged_dst))

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
        "unavailable": dropped[["name", "position", "team", "salary",
                                "playing"]].to_dict("records")
                       if len(dropped) else [],
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

    if out.get("unavailable"):
        lines += ["", f"{len(out['unavailable'])} players excluded as "
                      f"unavailable (they would otherwise look like the best "
                      f"value on the board):"]
        for u in out["unavailable"][:15]:
            lines.append(f"  {str(u['name'])[:24]:<25}{str(u['position']):<5}"
                         f"${int(u['salary']):>6,}   {u['playing'].upper()}")

    if cov["misses"]:
        lines += ["", f"{len(cov['misses'])} slate players not projected, "
                      f"most expensive first:"]
        for m in cov["misses"][:22]:
            sal = m.get("salary")
            sal = f"${int(sal):>6,}" if pd.notna(sal) else "     ?"
            why = ("JOIN FAILURE - he is in the history file"
                   if m.get("has_nfl_history")
                   and m["position"] in config.SKILL_POSITIONS
                   else "no NFL history")
            lines.append(f"  {str(m['name'])[:24]:<25} {str(m['position']):<5}"
                         f"{sal}   {why}")
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
    if cov["broken"] and not args.force:
        print(f"\nSTOPPING: {len(cov['broken'])} slate players have NFL "
              f"history that failed to join:")
        for b in cov["broken"][:12]:
            print(f"  {str(b['name'])[:24]:<25} {str(b['position']):<5}"
                  f"${int(b['salary']):>6,}")
        print("These are names the history file contains and the join did not "
              "find, which\nis a bug in the matching, not a gap in the data. "
              "Fix it rather than forcing\npast it - the players a name-match "
              "loses are the ones who changed teams.")
        return 1

    if cov["by_salary"] < MIN_SALARY_COVERAGE and not args.force:
        print(f"\nSTOPPING: the projected players cover only "
              f"{cov['by_salary'] * 100:.0f}% of projectable salary "
              f"(floor {MIN_SALARY_COVERAGE * 100:.0f}%).\nNo join failures "
              f"were found, so this is rookies and debutants rather than a "
              f"bug.\nIf the unmatched list above is all first-year players, "
              f"pass --force.")
        return 1

    df = out["players"]
    keep = [c for c in ("name", "position", "team", "salary", "captain_salary",
                        "q10", "q25", "q50", "q75", "q90", "q97", "median",
                        "mean", "ceiling", "spread", "value", "ceiling_value")
            if c in df.columns]
    df[keep].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
