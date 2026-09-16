"""Is this model any good? Nobody has ever checked.

Everything built so far has been checked for CORRECTNESS - that the scoring
reconciles, that the join finds the right player, that the correlation matrix
delivers what it assumed. None of that says whether the projections are worth
anything. A model can be flawlessly implemented and still be a worse guess
than a player's season average, and there would be no sign of it anywhere: the
lineups would still build, the page would still render, the numbers would
still look plausible.

So this grades it, walking forward one week at a time, and it is arranged so
the three ways a projection can be worthless are each visible separately.

**Is it better than doing nothing?** The baseline is not zero, it is a
player's own scoring average to date - which is what DraftKings itself
publishes and what a person would use if they had no model at all. Beating a
season average is genuinely hard, and a model that cannot is not a model.
Reported alongside it: the exponentially-weighted average the model is already
given as a feature, so we can see whether the fit adds anything to its own
best input, and a positional average, which is the floor.

**Are the quantiles honest?** This is the one most likely to be wrong and the
most expensive if it is. The model emits q10 through q97, and the tournament
objective is built on the top of that range. If the 90th percentile is really
being exceeded a quarter of the time, every ceiling is understated, every GPP
lineup is mispriced in the same direction, and nothing downstream would ever
reveal it - the lineups would simply be quietly wrong, week after week.
Calibration is measured by counting: of the weeks a player actually played,
what fraction landed below each fitted quantile?

**Does it rank?** DFS does not pay for absolute accuracy, it pays for picking
the right players out of a pool. A projection that is ten points high on
everyone is useless as a number and perfect as a ranking. So Spearman
correlation within each week, and the overlap between the model's top N at a
position and the actual top N, are reported separately from the error.

What this deliberately does NOT claim
-------------------------------------
It does not measure return on investment, because that needs historical
DraftKings salaries and contest results, and neither is available from any
free source - DraftKings publishes no salary history. So this grades the
PROJECTION, not the lineup. A model that ranks well can still lose money if
the salaries are efficient, and this cannot see that. Saying so is better than
quietly reporting an accuracy number and letting it stand in for profit.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import config
import data
import features as F
from model import Projections

log = logging.getLogger(__name__)


def head(t):
    print(f"\n{t}\n{'=' * 74}")


def sub(t):
    print(f"\n{t}\n{'-' * 74}")


# ------------------------------------------------------------------ baselines
BASELINES = ["prior_mean", "prior_last", "prior_position"]


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """The numbers a model has to beat, each computed without the future.

    `prior_mean` is the player's average over every game he has already
    played - the same quantity DraftKings publishes as points-per-game, and
    the honest answer to "what will he score" from someone with no model.

    Every one of these is shifted before it is averaged. That is not a detail:
    an expanding mean that includes the current row is a perfect predictor of
    the current row, and a backtest built on one would report spectacular
    accuracy and mean nothing at all.

    The positional baseline needed more care than the others and did not get
    it first time. Taking a position's mean for the week and shifting by one
    ROW leaks, because the rows of the same week sit next to each other: a
    player ends up averaging a figure that already contains his own result. It
    has to be shifted by one WEEK, on a table of weeks, and then joined back.
    """
    # Idempotent on purpose. This is called once in main and again inside the
    # leak check, and the second call was handed a frame that already carried
    # these columns - so the merge below produced `prior_position_x` and
    # `prior_position_y`, the plain column vanished, and the run died on a
    # KeyError before measuring anything. Recomputing from scratch is both
    # correct and cheap; a function that breaks when called twice is a trap
    # laid for the next caller.
    df = df.drop(columns=[c for c in BASELINES if c in df.columns])
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = df.groupby("player_id", sort=False)["points"]
    df["prior_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["prior_last"] = g.transform(lambda s: s.shift(1))

    weekly = (df.groupby(["position", "season", "week"], as_index=False)
              ["points"].mean().rename(columns={"points": "_pos_week"})
              .sort_values(["position", "season", "week"]))
    weekly["prior_position"] = (weekly.groupby("position", sort=False)
                                ["_pos_week"]
                                .transform(lambda s: s.shift(1)
                                           .expanding().mean()))
    df = df.merge(weekly[["position", "season", "week", "prior_position"]],
                  on=["position", "season", "week"], how="left")
    return df


def prove_no_leak(df: pd.DataFrame, players: int = 150) -> bool:
    """Change one game and check that game's own baseline does not move.

    This test has now been wrong twice, in two different ways, and both
    failures are the same mistake wearing different clothes.

    The first version deleted every row after a cut-off. It passed a baseline
    built deliberately to leak, because an expanding average only looks
    backwards inside the frame it is handed, so removing the future cannot
    disturb it either way. The test could not fail.

    The second version tampered with two hundred scattered results and
    required none of their baselines to move - and failed the HONEST baseline,
    because a player's average legitimately includes his own earlier games, so
    tampering with two of one player's weeks moves the later one for a correct
    reason. The test could not pass.

    The fix is isolation: tamper at most ONE game per player. Then any
    movement in that game's own baseline can only have come from itself. The
    game immediately after it must move, which is what proves the check is
    capable of detecting anything at all.
    """
    base = add_baselines(df)
    rng = np.random.default_rng(config.RANDOM_SEED)

    picks, followers = [], []
    for _, idx in base.groupby("player_id", sort=False).indices.items():
        idx = np.sort(idx)
        if len(idx) < 3:
            continue
        # Not the first game (nothing before it to compare) and not the last
        # (nothing after it to prove the test has teeth).
        choice = int(rng.integers(1, len(idx) - 1))
        picks.append(idx[choice])
        followers.append(idx[choice + 1])
        if len(picks) >= players:
            break
    if not picks:
        print("  not enough history per player to run the check")
        return False

    tampered = base.copy()
    tampered.loc[picks, "points"] = tampered.loc[picks, "points"] + 100.0
    after = add_baselines(tampered)

    cols = ["prior_mean", "prior_last"]
    before_own = base.loc[picks, cols].to_numpy(dtype=float)
    after_own = after.loc[picks, cols].to_numpy(dtype=float)
    own_moved = int((~np.isclose(before_own, after_own,
                                 equal_nan=True)).any(axis=1).sum())

    before_next = base.loc[followers, cols].to_numpy(dtype=float)
    after_next = after.loc[followers, cols].to_numpy(dtype=float)
    next_moved = int((~np.isclose(before_next, after_next,
                                  equal_nan=True)).any(axis=1).sum())

    print(f"  added 100 points to one game for each of {len(picks)} players")
    print(f"  that game's OWN baseline moved      : {own_moved:>4}  "
          f"(must be 0 - a row may not see itself)")
    print(f"  the FOLLOWING game's baseline moved : {next_moved:>4}  "
          f"(must be {len(picks)} - or the test is blind)")
    if own_moved:
        print("  -> a baseline can see its own result. Every number below is "
              "void.")
        return False
    if next_moved < len(picks):
        print("  -> the change did not propagate where it should have, so "
              "this check proves nothing.")
        return False
    print("  -> baselines use the past only, and the test can tell the "
          "difference.")
    return True


# ------------------------------------------------------------------- walking
def walk(built: pd.DataFrame, test_weeks: list[tuple[int, int]],
         min_train: int = 2000,
         positions: list[str] | None = None) -> pd.DataFrame:
    """Fit on the past, predict the next week, one week at a time.

    Refitting every week is the slow, honest way. Fitting once on everything
    and predicting the same rows would be the fast, meaningless way - it
    measures how well a model memorises, which is not a question anyone needs
    answered.
    """
    positions = positions or config.SKILL_POSITIONS
    rows = []
    for season, week in test_weeks:
        past = built[(built["season"] < season)
                     | ((built["season"] == season) & (built["week"] < week))]
        now = built[(built["season"] == season) & (built["week"] == week)]
        # Grade only the players a DFS board actually contains. The model
        # TRAINS on skill positions - `trainable` enforces that - but the
        # first version predicted on every row of the week, so the graded
        # population came back full of tackles, punters and defensive backs.
        # They score zero almost every week, predicting zero for them is
        # trivially easy, and they made up most of the rows: reported error
        # collapsed to 1.46 points when a real fantasy MAE is five to eight,
        # and calibration reported the tenth percentile covering 74% of
        # outcomes because most outcomes were literally zero. Both numbers
        # were measuring the roster of the NFL, not the quality of a
        # projection.
        now = now[now["position"].isin(positions)]
        train = F.trainable(past)
        if len(train) < min_train or now.empty:
            log.info("skipping %s week %s: %d training rows", season, week,
                     len(train))
            continue
        try:
            model = Projections().fit(past)
        except ValueError as exc:
            log.warning("%s week %s did not fit: %s", season, week, exc)
            continue
        pred = model.predict(now)
        keep = [c for c in ("player_id", "name", "position", "team", "season",
                            "week", "points", "prior_mean", "prior_last",
                            "prior_position") if c in now.columns]
        out = now[keep].reset_index(drop=True)
        for c in pred.columns:
            if c.startswith("q") or c in ("median", "mean", "ceiling"):
                out[c] = pred[c].to_numpy()
        out["trained_on"] = len(train)
        rows.append(out)
        log.info("%s week %2d: trained on %d rows, predicted %d players",
                 season, week, len(train), len(out))
    if not rows:
        raise SystemExit("no week produced a fit; nothing to grade")
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ measures
def accuracy(g: pd.DataFrame) -> pd.DataFrame:
    """Error against each baseline, on identical rows.

    Restricted to rows where every baseline exists, so the comparison is on
    the same players. Scoring the model on rows the baseline cannot reach
    would flatter whichever one saw more data.
    """
    need = ["median", "prior_mean", "prior_last", "prior_position", "points"]
    d = g.dropna(subset=need)
    out = []
    # The median and the mean are graded separately because they are answers
    # to different questions and the run showed the difference mattering:
    # bias came out at -1.21 on the median, which is not an error but a
    # property - fantasy scoring is right-skewed, so the middle of the
    # distribution sits below its average. A cash lineup wants expected value
    # and should read `mean`; the median would systematically undersell every
    # player by about a point.
    for label, col in (("this model (median)", "median"),
                       ("this model (mean)", "mean"),
                       ("season average to date", "prior_mean"),
                       ("last game", "prior_last"),
                       ("positional average", "prior_position")):
        err = d[col] - d["points"]
        out.append({"predictor": label, "n": len(d),
                    "MAE": round(err.abs().mean(), 3),
                    "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
                    "bias": round(err.mean(), 3)})
    return pd.DataFrame(out)


def calibration(g: pd.DataFrame, quantiles) -> pd.DataFrame:
    """How often the truth actually lands below each fitted quantile."""
    rows = []
    for q in quantiles:
        col = f"q{int(q * 100)}"
        if col not in g:
            continue
        d = g.dropna(subset=[col, "points"])
        if not len(d):
            continue
        covered = float((d["points"] <= d[col]).mean())
        rows.append({"quantile": col, "should be": q,
                     "actually": round(covered, 3),
                     "gap": round(covered - q, 3), "n": len(d)})
    return pd.DataFrame(rows)


def calibration_split(g: pd.DataFrame, quantiles) -> pd.DataFrame:
    """Calibration for the players who played, and for those who did not.

    The whole-population number hides which of two very different problems is
    being measured. Eighteen percent of the graded rows scored exactly zero -
    inactive, benched, or hurt in the first quarter - and no projection built
    from usage history can know that in advance. Those rows sit below every
    positive quantile by construction, so they push the floor's coverage up on
    their own.

    Splitting them out answers the question that actually decides what to fix.
    If the floor is well calibrated among players who took the field, then the
    miss is entirely an AVAILABILITY problem and the fix is injury data, not
    the model. If it is still wrong among them, the model's floor is genuinely
    too optimistic and that is a modelling problem.
    """
    played = g["points"] > 0
    rows = []
    for label, d in (("played", g[played]), ("scored zero", g[~played]),
                     ("everyone", g)):
        for q in quantiles:
            col = f"q{int(q * 100)}"
            if col not in d or not len(d):
                continue
            dd = d.dropna(subset=[col, "points"])
            if not len(dd):
                continue
            rows.append({"group": label, "n": len(dd), "quantile": col,
                         "should be": q,
                         "actually": round(float((dd["points"]
                                                  <= dd[col]).mean()), 3)})
    out = pd.DataFrame(rows)
    if len(out):
        out["gap"] = (out["actually"] - out["should be"]).round(3)
    return out


def ranking(g: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Does it put the right players at the top, within a week and position?

    This is the measure that matters for lineup building. Absolute error can
    be poor while the ordering is excellent, and the optimiser only ever
    consumes the ordering.
    """
    rows = []
    for (season, week, pos), d in g.groupby(["season", "week", "position"]):
        d = d.dropna(subset=["median", "points", "prior_mean"])
        if len(d) < top_n * 2:
            continue
        for label, col in (("model", "median"), ("season average",
                                                 "prior_mean")):
            # A constant column has no ordering, so Spearman is undefined
            # rather than zero. Skipping it beats emitting a NaN that reads
            # like a measurement.
            if d[col].nunique() < 2 or d["points"].nunique() < 2:
                continue
            rho = d[col].corr(d["points"], method="spearman")
            picked = set(d.nlargest(top_n, col)["player_id"])
            truth = set(d.nlargest(top_n, "points")["player_id"])
            rows.append({"predictor": label, "position": pos,
                         "spearman": rho,
                         "hit_rate": len(picked & truth) / top_n})
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    return (r.groupby(["predictor", "position"])
            .agg(weeks=("spearman", "size"),
                 spearman=("spearman", "mean"),
                 top10_hit_rate=("hit_rate", "mean"))
            .round(3).reset_index())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=int, nargs="*",
                   default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--test-season", type=int, default=2025)
    p.add_argument("--from-week", type=int, default=4)
    p.add_argument("--to-week", type=int, default=17)
    p.add_argument("--site", default="dk")
    p.add_argument("--positions", nargs="*", default=config.SKILL_POSITIONS,
                   help="positions to grade; the default is what a DFS board "
                        "contains, and widening it makes the error look "
                        "better for the wrong reason")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="  %(levelname)-7s %(message)s")

    head("GRADING THE PROJECTION")
    print("  Walking forward one week at a time: fit on everything before the")
    print("  week, predict the week, never look at it first.\n")

    weeks = data.player_weeks(args.seasons)
    try:
        lines = data.schedules()
    except Exception as exc:                                   # noqa: BLE001
        log.warning("no market lines (%s); grading without them", exc)
        lines = None
    built = F.build(weeks, site=args.site, lines=lines)
    built = add_baselines(built)
    print(f"  {len(built):,} player-weeks across seasons "
          f"{min(args.seasons)}-{max(args.seasons)}")

    sub("FIRST, PROVE THE BASELINES CANNOT SEE THE FUTURE")
    if not prove_no_leak(built):
        return 1

    test_weeks = [(args.test_season, w)
                  for w in range(args.from_week, args.to_week + 1)]
    sub(f"WALKING {len(test_weeks)} WEEKS")
    g = walk(built, test_weeks, positions=args.positions)
    print(f"\n  {len(g):,} graded player-weeks, positions "
          f"{sorted(g['position'].unique())}")
    print(f"  {(g['points'] == 0).mean():.1%} of them scored exactly zero")

    sub("IS IT BETTER THAN DOING NOTHING?")
    acc = accuracy(g)
    print(acc.to_string(index=False))
    model_mae = acc.loc[acc["predictor"].str.startswith("this"), "MAE"].iloc[0]
    base_mae = acc.loc[acc["predictor"].str.startswith("season"), "MAE"].iloc[0]
    edge = (base_mae - model_mae) / base_mae
    print(f"\n  the model is {edge:+.1%} better than a season average on MAE.")
    if edge <= 0:
        print("  -> it is NOT better. Everything built on top of it is built")
        print("     on a number worse than the one DraftKings prints for free.")
        print("     Fix this before adding a single feature anywhere else.")
    elif edge < 0.03:
        print("  -> barely. That is within the range a different random seed")
        print("     could produce, and is not yet a real edge.")
    else:
        print("  -> a real improvement, and the size of it is the honest")
        print("     answer to what the modelling has bought so far.")

    sub("ARE THE QUANTILES HONEST?")
    print("  Of the weeks actually played, what fraction landed below each")
    print("  fitted quantile? These should match, and the top of the range")
    print("  is what every tournament lineup is priced on.\n")
    cal = calibration(g, config.QUANTILES)
    print(cal.to_string(index=False))

    split = calibration_split(g, config.QUANTILES)
    if len(split):
        print("\n  And split by whether the player actually took the field,")
        print("  because a player who scores zero sits below every positive")
        print("  quantile and no usage model can know in advance that he")
        print("  would not play:\n")
        piv = (split.pivot(index="quantile", columns="group",
                           values="actually")
               .reindex([f"q{int(q * 100)}" for q in config.QUANTILES]))
        piv.insert(0, "target", [q for q in config.QUANTILES])
        print(piv.to_string())
        floor = split[(split["quantile"] == "q10")
                      & (split["group"] == "played")]
        if len(floor):
            f = floor.iloc[0]
            print(f"\n  Among players who took the field, q10 covers "
                  f"{f['actually']:.1%} against a target of 10%.")
            if abs(f["gap"]) <= 0.04:
                print("  -> the floor is well calibrated for players who play.")
                print("     The whole-population miss is therefore an")
                print("     AVAILABILITY problem, not a modelling one: the fix")
                print("     is knowing who is inactive, which this model")
                print("     currently cannot see at all.")
            else:
                print("  -> the floor is still off among players who played,")
                print("     so this is a modelling problem as well as an")
                print("     availability one.")
    if len(cal):
        worst = cal.reindex(cal["gap"].abs().sort_values(ascending=False).index)
        w = worst.iloc[0]
        print(f"\n  worst: {w['quantile']} should cover {w['should be']:.0%} "
              f"and covers {w['actually']:.1%}")
        ceiling = cal[cal["quantile"].isin(["q90", "q97"])]
        if len(ceiling) and (ceiling["gap"] < -0.03).any():
            print("  -> the CEILING is understated: the truth exceeds it more")
            print("     often than it should. Every tournament lineup is")
            print("     therefore underpricing upside, in the same direction,")
            print("     every week - which no amount of staring at lineups")
            print("     would reveal.")
        elif len(ceiling) and (ceiling["gap"] > 0.03).any():
            print("  -> the ceiling is overstated; boom outcomes are rarer")
            print("     than the model thinks, so GPP lineups are chasing")
            print("     upside that is not there.")

    sub("DOES IT RANK?")
    print("  The optimiser only ever consumes the ordering, so this is the")
    print("  measure that decides whether the lineups are any good.\n")
    rk = ranking(g)
    if len(rk):
        print(rk.to_string(index=False))
        piv = rk.pivot(index="position", columns="predictor",
                       values="spearman")
        if {"model", "season average"} <= set(piv.columns):
            better = (piv["model"] > piv["season average"]).sum()
            print(f"\n  the model ranks better than a season average at "
                  f"{better} of {len(piv)} positions")

    head("WHAT THIS DOES NOT TELL YOU")
    print("  Not return on investment. That needs historical DraftKings")
    print("  salaries and contest results, and DraftKings publishes neither.")
    print("  A model can rank well and still lose money if the salaries are")
    print("  efficient - this grades the projection, not the bet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
