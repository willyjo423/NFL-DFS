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

VERSION = "v4"


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
    print("  comparison can. Worth knowing: I predicted in advance that UL")
    print("  would resolve to Louisiana. The fixture list said Louisville,")
    print("  and the fixture list is the one with evidence.\n")
    teams = cfb.cfbd("teams/fbs", key, year=args.season)
    mapping, report = cfb.team_map(board, games, teams)
    print(f"  {'draftkings':<16} {'resolved to':<38} {'score':>6}  evidence")
    for dk_fixture, resolved, score, evidence in report:
        print(f"  {dk_fixture:<16} {str(resolved)[:37]:<38} {score:>6}  "
              f"{evidence[:30]}")
    warnings = cfb.map_warnings(mapping, teams, report)
    if warnings:
        print("\n  bindings worth a second look:")
        for w in warnings:
            print(f"     ! {w}")
    else:
        print("\n  no binding was contested: every code had one candidate.")

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
    merged = cfb.attach_history(board, hist)
    merged = merged.merge(lines[["school", "implied_total", "opponent_implied",
                                 "game_total", "team_spread"]],
                          on="school", how="left")
    has_hist = merged["games"].notna()
    has_line = merged["implied_total"].notna()
    print(f"  history : {has_hist.sum():>4}/{len(merged)} "
          f"({has_hist.mean():.1%})")
    print(f"  market  : {has_line.sum():>4}/{len(merged)} "
          f"({has_line.mean():.1%})")

    # ------------------------------------------------------------------------
    # DraftKings publishes its OWN points-per-game for every player it prices.
    # That is ground truth sitting in the payload, and it settles two
    # questions no amount of staring at prices could.
    #
    # First, it grades the scoring rules: if the points computed here from
    # CFBD box scores agree with DraftKings' own average, the whole scoring
    # implementation is right. Nothing else available checks that.
    #
    # Second, it separates a broken join from an idle player. An unmatched
    # player whose DraftKings average is ZERO has not played - there is no
    # history to find and nothing is wrong. An unmatched player whose average
    # is above zero DID score, and his absence is a genuine failure. Price
    # cannot tell those apart; this can. It is what I should have measured
    # instead of guessing from salary twice in a row.
    sub("GRADING THE SCORING AGAINST DRAFTKINGS' OWN NUMBERS")
    gradeable = merged[has_hist & merged["dk_points_per_game"].notna()].copy()
    if len(gradeable):
        # Compared on PER-TEAM-GAME, which is DraftKings' own definition.
        # Comparing our per-game-played average instead produced a tidy 0.65
        # mean error that looked like a small calibration wobble and was
        # actually a divisor: every disagreement was exactly 2.00x, because
        # those players had appeared in one of their team's two games.
        gradeable["gap"] = (gradeable["per_team_game"]
                            - gradeable["dk_points_per_game"])
        mae = gradeable["gap"].abs().mean()
        close = (gradeable["gap"].abs() <= 0.5).mean()
        print(f"  {len(gradeable)} players carry both our computed average and")
        print(f"  DraftKings' published one.\n")
        print(f"  mean absolute difference : {mae:.2f} points")
        print(f"  within half a point      : {close:.1%}")
        worst = gradeable.reindex(
            gradeable["gap"].abs().sort_values(ascending=False).index).head(8)
        print(f"\n  the biggest disagreements, which are either a scoring rule")
        print(f"  we have wrong or a player matched to the wrong history:")
        for r in worst.itertuples(index=False):
            ratio = ("-" if not r.dk_points_per_game
                     else f"{r.per_team_game / r.dk_points_per_game:.2f}x")
            print(f"     ours {r.per_team_game:>6.1f}  theirs "
                  f"{r.dk_points_per_game:>6.1f}  {ratio:>6}  "
                  f"{str(r.name)[:22]:<24}{str(r.position):<4}{r.team}")
        # A player we credit with points that DraftKings scores at zero is not
        # a rounding difference. It is the signature of a match to the wrong
        # person - the failure that does real damage, because it is confident.
        phantom = gradeable[(gradeable["dk_points_per_game"] == 0)
                            & (gradeable["per_team_game"] > 3)]
        if len(phantom):
            print(f"\n  {len(phantom)} players have points here and ZERO at "
                  f"DraftKings. That is the")
            print("  signature of a match to the wrong person, not a rounding "
                  "gap:")
            for r in phantom.nlargest(6, "per_team_game").itertuples(
                    index=False):
                alts = cfb.near_misses(r.name, r.school, hist)
                print(f"     {str(r.name)[:24]:<26}{str(r.team):<6}"
                      f"ours {r.per_team_game:.1f}   others named similarly "
                      f"at this school: {alts}")
        if mae <= 1.0:
            print(f"\n  -> the scoring rules agree with DraftKings. Yardage,")
            print(f"     touchdowns, receptions and the bonus thresholds are")
            print(f"     all being read correctly.")
        else:
            print(f"\n  -> a systematic gap of {mae:.1f} points means a scoring")
            print(f"     rule is wrong. Every projection is shifted by it, which")
            print(f"     looks like miscalibration rather than a bug.")

    sub("WHAT THE MISSES ACTUALLY ARE")
    missing = merged[~has_hist].copy()
    if len(missing):
        played = missing["dk_points_per_game"].fillna(0) > 0
        print(f"  {len(missing)} priced players have no scored game here.")
        print(f"  DraftKings says {int(played.sum())} of them HAVE scored this")
        print(f"  season and {int((~played).sum())} have not.\n")
        if played.any():
            print(f"  REAL JOIN FAILURES ({int(played.sum())}) - DraftKings has")
            print(f"  points for these players and we found none:")
            print("  Each line shows the closest names in that school's own")
            print("  box scores, so the cause is readable rather than guessed:")
            for r in missing[played].nlargest(15, "salary").itertuples(
                    index=False):
                alts = cfb.near_misses(r.name, r.school, hist)
                print(f"     ${r.salary:>6,.0f} {str(r.position):<4}"
                      f"{str(r.team):<6}{str(r.name)[:24]:<26}"
                      f"DK {r.dk_points_per_game:>5.1f}   -> {alts}")
        else:
            print("  Not one unmatched player has scored a DraftKings point.")
            print("  The join is not broken - DraftKings prices a whole depth")
            print("  chart (135 quarterbacks across 24 teams) and most of them")
            print("  never take a snap. I called this a broken join last run on")
            print("  the strength of price alone; the published averages say")
            print("  otherwise, and they are evidence where price was a guess.")

    # The other half of the same question: is the presumed STARTER at every
    # team and position matched? A join that catches every first-choice player
    # and misses backups is healthy; one that drops starters is not, and this
    # is the test that can tell, because it does not depend on price levels.
    sub("IS EVERY LIKELY STARTER MATCHED?")
    ranked = merged.copy()
    ranked["depth"] = (ranked.groupby(["team", "position"])["salary"]
                       .rank(ascending=False, method="first"))
    print(f"  {'depth':<8}{'players':>9}{'with history':>15}{'rate':>9}")
    for depth in (1, 2, 3, 4):
        g = ranked[ranked["depth"] == depth]
        if not len(g):
            continue
        hit = g["games"].notna()
        print(f"  {'#' + str(int(depth)):<8}{len(g):>9}{int(hit.sum()):>15}"
              f"{hit.mean():>9.1%}")
    deep = ranked[ranked["depth"] > 4]
    if len(deep):
        hit = deep["games"].notna()
        print(f"  {'#5+':<8}{len(deep):>9}{int(hit.sum()):>15}{hit.mean():>9.1%}")
    starters = ranked[ranked["depth"] == 1]
    gaps = starters[starters["games"].isna()]
    if len(gaps):
        print(f"\n  {len(gaps)} team-position leaders have no history:")
        for r in gaps.nlargest(10, "salary").itertuples(index=False):
            ppg = 0.0 if pd.isna(r.dk_points_per_game) else r.dk_points_per_game
            print(f"     ${r.salary:>6,.0f}  {str(r.position):<4}{str(r.team):<6}"
                  f"{str(r.name)[:24]:<26}DK avg {ppg:.1f}")

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
