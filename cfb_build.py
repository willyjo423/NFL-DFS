"""Run the college football pipeline end to end and show its working.

This is the step between "the data exists" and "the model trains on it". It
takes a live DraftKings slate, solves the team map from the fixture list,
pulls the settled weeks, scores them, attaches the market, and then reports
what actually joined - by name, with prices, for anything that did not.

Everything it prints is something that could be wrong in a way no exception
would catch:

* a team map that is confident and wrong points a whole roster at the wrong
  opponent and the wrong implied total;
* a name join that quietly drops starters leaves holes the optimiser cannot
  see;
* a scoring rule that mis-parses one stat shifts every projection in the same
  direction, which looks like a calibration problem rather than a bug.

So the output is arranged to be READ, not scanned for a traceback.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

import cfb

VERSION = "v1"


def head(t):
    print(f"\n{t}\n{'=' * 74}")


def sub(t):
    print(f"\n{t}\n{'-' * 74}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--week", type=int, default=0)
    p.add_argument("--slate", type=int, default=0,
                   help="draft group id; 0 = the biggest Classic slate")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="  %(levelname)-7s %(message)s")

    key = os.environ.get("CFBD_API_KEY")
    head(f"COLLEGE FOOTBALL PIPELINE  ({VERSION})")
    if not key:
        print("  CFBD_API_KEY is not set; nothing below can run.")
        return 1

    # ---------------------------------------------------------------- the slate
    sub("THE SLATE")
    board_groups = cfb.slates()
    print(board_groups.head(8).to_string(index=False))
    if args.slate:
        dg = args.slate
    else:
        classic = board_groups[board_groups["game_type"]
                               .astype(str).str.contains("Classic", na=False)]
        if classic.empty:
            print("\n  No Classic slate is on sale. Showdown needs its own "
                  "roster rules, so this stops rather than guessing them.")
            return 1
        dg = int(classic.iloc[0]["draft_group"])
    print(f"\n  building for draft group {dg}")

    board = cfb.board(dg)
    sub("THE BOARD")
    print(f"  {len(board)} players, {board['team'].nunique()} teams, "
          f"{board['game'].nunique()} games")
    print(f"\n  {'position':<10}{'players':>8}{'min':>9}{'median':>9}{'max':>9}")
    for pos, g in board.groupby("position"):
        print(f"  {str(pos):<10}{len(g):>8}{g['salary'].min():>9,.0f}"
              f"{g['salary'].median():>9,.0f}{g['salary'].max():>9,.0f}")
    disabled = int(board["disabled"].sum())
    if disabled:
        print(f"\n  {disabled} players are flagged unavailable by DraftKings "
              f"and are dropped before anything else sees them")
        board = board[~board["disabled"]].reset_index(drop=True)

    # ----------------------------------------------------------------- the week
    sub("THE LIVE WEEK")
    week = args.week or cfb.live_week(key, args.season)
    games = cfb.cfbd("games", key, year=args.season, week=week,
                     seasonType="regular")
    print(f"  week {week}: {len(games)} games on the college schedule, "
          f"{board['game'].nunique()} of them priced by DraftKings")

    # ------------------------------------------------------------- the team map
    sub("THE TEAM MAP, SOLVED FROM THE FIXTURE LIST")
    print("  Each DraftKings fixture must be exactly one college fixture, and")
    print("  BOTH sides have to agree. That is what settles codes no string")
    print("  comparison can - UL is Louisiana because Louisiana is the team")
    print("  playing this opponent this week.\n")
    teams = cfb.cfbd("teams/fbs", key, year=args.season)
    mapping, report = cfb.team_map(board, games, teams)
    print(f"  {'draftkings':<16} {'resolved to':<38} {'score':>6}  evidence")
    for dk_fixture, resolved, score, evidence in report:
        print(f"  {dk_fixture:<16} {str(resolved)[:37]:<38} {score:>6}  "
              f"{evidence[:30]}")
    unresolved = sorted(set(board["team"]) - set(mapping))
    print(f"\n  {len(mapping)} codes resolved, {len(unresolved)} unresolved"
          + (f": {unresolved}" if unresolved else ""))
    if unresolved:
        print("  Players on unresolved teams are dropped. A guessed school")
        print("  would give them the wrong opponent and the wrong total.")

    board["school"] = board["team"].map(mapping)
    board["opponent_school"] = board["opponent"].map(mapping)
    before = len(board)
    board = board.dropna(subset=["school"]).reset_index(drop=True)
    if len(board) < before:
        print(f"  dropped {before - len(board)} players on unresolved teams")

    # -------------------------------------------------------------- the history
    sub("THE HISTORY")
    hist = cfb.history(key, args.season, week)
    per_player = (hist.groupby(["norm", "school"])
                  .agg(games=("dk_points", "size"),
                       mean=("dk_points", "mean"),
                       best=("dk_points", "max"))
                  .reset_index())
    print(f"  {len(hist):,} scored player-games over weeks 1-{week - 1}")
    print(f"\n  the best single games so far, as a scoring sanity check:")
    top = hist.nlargest(8, "dk_points")
    for r in top.itertuples(index=False):
        print(f"     {r.dk_points:>6.1f}  {r.name[:24]:<26}{str(r.school)[:20]:<22}"
              f"wk{r.week}")

    # --------------------------------------------------------------- the market
    sub("THE MARKET")
    lines = cfb.market(key, args.season, week)
    if len(lines):
        hot = lines.nlargest(6, "implied_total")
        print(f"\n  highest implied totals this week:")
        for r in hot.itertuples(index=False):
            print(f"     {r.implied_total:>5.1f}  {str(r.school)[:24]:<26}"
                  f"vs {str(r.opponent_school)[:20]:<22}total {r.game_total}")

    # ------------------------------------------------------------------ the join
    sub("WHAT ACTUALLY JOINED")
    merged = board.merge(per_player, on=["norm", "school"], how="left")
    merged = merged.merge(lines[["school", "implied_total", "opponent_implied",
                                 "game_total", "team_spread"]],
                          on="school", how="left")
    has_hist = merged["games"].notna()
    has_line = merged["implied_total"].notna()
    print(f"  history : {has_hist.sum():>4}/{len(merged)} "
          f"({has_hist.mean():.1%})")
    print(f"  market  : {has_line.sum():>4}/{len(merged)} "
          f"({has_line.mean():.1%})")

    missing = merged[~has_hist]
    if len(missing):
        print(f"\n  priced but with no scored game this season "
              f"({len(missing)}), most expensive first:")
        for r in missing.nlargest(12, "salary").itertuples(index=False):
            print(f"     ${r.salary:>6,.0f}  {str(r.position):<4}"
                  f"{str(r.team):<6}{str(r.name)[:26]}")
        cheap = missing["salary"].median()
        rich = merged.loc[has_hist, "salary"].median()
        print(f"\n  unmatched median ${cheap:,.0f} vs matched ${rich:,.0f}")
        if cheap <= rich:
            print("  -> the misses are at the bottom of the board, which is "
                  "where a join is allowed to fail: nobody rosters them.")
        else:
            print("  -> the misses are priced ABOVE the matched players. That "
                  "is not a bench problem, it is a broken join, and nothing")
            print("     downstream should be trusted until it is fixed.")

    sub("THE BOARD, AS THE MODEL WILL SEE IT")
    ready = merged[has_hist & has_line].copy()
    ready["per_1k"] = (ready["mean"] / (ready["salary"] / 1000)).round(2)
    print(f"  {len(ready)} players carry both a history and a market line\n")
    cols = ["name", "position", "team", "salary", "games", "mean", "best",
            "implied_total", "per_1k"]
    show = ready.nlargest(15, "mean")[cols]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    head("WHERE THIS LEAVES US")
    print(f"  {len(ready)} of {len(merged)} priced players are model-ready.")
    print("  Points per $1,000 is printed above as a SANITY CHECK only. It is")
    print("  not the objective and must never become one: it is maximised by")
    print("  the cheapest player who ever scored, which is how a lineup ends")
    print("  up full of minimum-salary players who happened to score once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
