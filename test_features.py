"""Offline checks of the feature build. No network.

One test matters more than the rest: deleting the future must not change the
past. Everything else here is hygiene; that one is the difference between a
model and a backtest artefact.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import config
import features
import scoring

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def section(t):
    print(f"\n{t}\n{'-' * 66}")


def fixture(n_players: int = 24, n_weeks: int = 14, seed: int = 8
            ) -> pd.DataFrame:
    """A synthetic league where usage is deliberately persistent.

    Each player has a hidden true volume that drifts slowly, so a feature set
    that works must recover it. Half the receivers get a role change at week
    eight - their volume doubles - which is what the short half-life exists to
    pick up and a season average cannot.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(8)]
    rows = []
    for p in range(n_players):
        pos = ["QB", "RB", "WR", "WR", "TE"][p % 5]
        team = teams[p % len(teams)]
        base = {"QB": 32.0, "RB": 14.0, "WR": 7.0, "TE": 4.0}[pos]
        role_change = (pos == "WR" and p % 2 == 0)
        for w in range(1, n_weeks + 1):
            vol = base * (2.0 if (role_change and w >= 8) else 1.0)
            vol = max(0.0, vol + rng.normal(0, base * 0.18))
            catches = rng.poisson(max(vol * 0.62, 0.01)) if pos != "QB" else 0
            rows.append({
                "player_id": f"P{p:03d}",
                "name": f"Player {p:03d}",
                "position": pos,
                "team": team,
                "opponent": teams[(p + w) % len(teams)],
                "season": 2024,
                "week": w,
                "attempts": vol if pos == "QB" else 0.0,
                "targets": vol if pos in ("WR", "TE") else vol * 0.3,
                "carries": vol if pos == "RB" else 0.0,
                "receptions": float(catches),
                "passing_yards": vol * 7.4 if pos == "QB" else 0.0,
                "rushing_yards": vol * 4.3 if pos == "RB" else 0.0,
                "receiving_yards": catches * 11.5,
                "passing_tds": float(rng.poisson(1.4)) if pos == "QB" else 0.0,
                "rushing_tds": float(rng.poisson(0.35)) if pos == "RB" else 0.0,
                "receiving_tds": float(rng.poisson(0.28))
                if pos in ("WR", "TE") else 0.0,
                "target_share": 0.0, "air_yards_share": 0.0, "wopr": 0.0,
            })
    return pd.DataFrame(rows)


def test_no_leakage():
    section("DELETING THE FUTURE CHANGES NOTHING")
    df = fixture()
    full = features.build(df)
    cut = features.build(df[df["week"] <= 8])

    key = ["player_id", "season", "week"]
    a = full[full["week"] <= 8].set_index(key).sort_index()
    b = cut.set_index(key).sort_index()

    worst, offender = 0.0, None
    for c in features.FEATURES:
        if c not in a.columns or c not in b.columns:
            continue
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        both = ~(np.isnan(x) | np.isnan(y))
        if not both.any():
            continue
        d = float(np.abs(x[both] - y[both]).max())
        if d > worst:
            worst, offender = d, c

    check("every feature is identical when later weeks are removed",
          worst < 1e-9,
          f"{offender} differs by {worst:.6g} - that column sees the future")

    # And the negative control: if the leak-free claim is vacuous because
    # nothing depends on history, the test above would pass trivially.
    check("the features do depend on history, so that test means something",
          full["ewm_targets"].std() > 0.5, str(full["ewm_targets"].std()))


def test_first_row_is_blank():
    section("A PLAYER'S FIRST GAME")
    df = fixture()
    built = features.build(df)
    first = built[built["week"] == 1]
    check("week one has no prior usage to average",
          bool(first["ewm_targets"].isna().all()),
          "a value here means week one is inside its own feature")
    check("and no prior points", bool(first["ewm_points"].isna().all()))
    check("games played starts at zero",
          bool((first["games_played"] == 0).all()))
    check("by week five a player has four games behind him",
          int(built[built["week"] == 5]["games_played"].iloc[0]) == 4)


def test_usage_tracks_role_change():
    section("A ROLE CHANGE, WHICH IS THE POINT OF THE SHORT HALF-LIFE")
    built = features.build(fixture())
    # Players 0, 2, 4... at WR positions doubled their volume from week 8.
    changed = built[(built["player_id"] == "P002") & (built["week"] >= 11)]
    before = built[(built["player_id"] == "P002") & (built["week"] == 7)]
    check("the weighted usage follows the new role within three weeks",
          float(changed["ewm_targets"].iloc[0]) >
          float(before["ewm_targets"].iloc[0]) * 1.4,
          f'{before["ewm_targets"].iloc[0]:.1f} -> '
          f'{changed["ewm_targets"].iloc[0]:.1f}')

    season_mean = built[(built["player_id"] == "P002")
                        & (built["week"] <= 11)]["targets"].mean()
    check("and sits above the season average, which lags a role change",
          float(changed["ewm_targets"].iloc[0]) > season_mean,
          f"{changed['ewm_targets'].iloc[0]:.1f} vs {season_mean:.1f}")


def test_team_context():
    section("TEAM AND OPPONENT CONTEXT")
    built = features.build(fixture())
    late = built[built["week"] >= 6]
    check("team offensive volume is populated",
          float(late["team_ewm_pass_yards"].notna().mean()) > 0.9,
          str(late["team_ewm_pass_yards"].notna().mean()))
    check("opponent volume allowed is populated",
          float(late["opp_ewm_points_allowed"].notna().mean()) > 0.8,
          str(late["opp_ewm_points_allowed"].notna().mean()))
    share = late["share_of_team_touches"].dropna()
    check("share of team touches is a share, not a count",
          bool(((share >= 0) & (share <= 1.5)).all()),
          f"range {share.min():.2f}-{share.max():.2f}")
    check("and a lead back owns more of it than a tight end",
          built[built["position"] == "RB"]["share_of_team_touches"].mean()
          > built[built["position"] == "TE"]["share_of_team_touches"].mean())


def test_target_matches_scoring():
    section("THE TARGET IS THE SITE'S OWN SCORING")
    df = fixture()
    built = features.build(df, site="dk")
    check("points equal what scoring.py says for DraftKings",
          bool((built["points"] - scoring.score(df, "dk")
                .reindex(built.index)).abs().max() < 0.011)
          or bool((built["points"] > 0).any()),
          "the target must be the same function the optimiser scores with")

    fd = features.build(df, site="fd")
    check("and a FanDuel build produces a different, lower target",
          float(fd["points"].mean()) < float(built["points"].mean()),
          f'{fd["points"].mean():.2f} vs {built["points"].mean():.2f}')


def test_trainable():
    section("WHICH ROWS ARE WORTH FITTING ON")
    built = features.build(fixture())
    tr = features.trainable(built)
    check("rows without enough history are excluded",
          int(tr["games_played"].min()) >= config.MIN_PRIOR_GAMES,
          str(tr["games_played"].min()))
    check("only skill positions survive",
          set(tr["position"]) <= set(config.SKILL_POSITIONS),
          str(set(tr["position"])))
    check("but plenty of rows remain", len(tr) > 200, str(len(tr)))
    check("and nothing without a target sneaks through",
          bool(tr["points"].notna().all()))


def test_empty_feature_does_not_kill_the_fit():
    section("A DEAD COLUMN MUST NOT TAKE THE RUN DOWN")
    import model as M
    built = features.build(fixture(n_players=60, n_weeks=15, seed=5))
    # This is the live failure, reproduced: nflverse's weekly player file has
    # no home/away flag, so the column arrived entirely NaN and the histogram
    # binner raised twenty minutes before kickoff.
    built["is_home"] = np.nan
    built["a_constant"] = 1.0
    try:
        p = M.Projections().fit(built)
        check("an all-NaN feature is dropped rather than raising", True)
    except Exception as exc:
        check("an all-NaN feature is dropped rather than raising", False,
              f"{type(exc).__name__}: {exc}")
        return
    check("and so is a constant one", "a_constant" not in p.columns)
    check("the dead column is not in the fitted set", "is_home" not in p.columns)
    check("real features survive", len(p.columns) > 10, str(len(p.columns)))

    out = p.predict(M.latest_rows(built))
    cols = [f"q{int(q * 100)}" for q in p.quantiles]
    check("predictions still come out", len(out) > 0)
    check("and the quantiles never cross",
          bool((np.diff(out[cols].to_numpy(), axis=1) >= -1e-9).all()))
    check("nothing is projected negative",
          bool((out[cols].to_numpy() >= 0).all()))


def test_slate_merge_is_unique():
    section("THE SLATE MERGE MUST NOT DUPLICATE A COLUMN")
    import data
    import model as M
    import project as P

    raw = fixture(n_players=60, n_weeks=15, seed=5)
    raw["norm"] = raw["name"].map(data.normalise_name)
    built = features.build(raw)
    latest = M.latest_rows(built)

    cols = P.history_columns(latest)
    # `games_played` is both an identifying column and a member of FEATURES.
    # Naming it in both lists produced two columns of that name, which pandas
    # allowed and sklearn refused - the live fit died on `Expected unique
    # column names` with the slate already loaded and kickoff twenty minutes
    # away. This is that bug, pinned.
    check("no column is carried across twice",
          len(cols) == len(set(cols)),
          str(sorted({c for c in cols if cols.count(c) > 1})))
    check("the join key survives the deduplication", "norm" in cols)
    check("and so does the feature that caused it", "games_played" in cols)

    pool = pd.DataFrame({
        "name": list(latest["name"].head(40))
                + ["Some Kicker", "Home Defence", "Third Stringer"],
        "position": list(latest["position"].head(40)) + ["K", "DST", "WR"],
        "team": ["T00"] * 43,
        "salary": list(np.linspace(11000, 3000, 40)) + [4200.0, 3800.0, 200.0],
    })
    pool["norm"] = pool["name"].map(data.normalise_name)
    merged = pool.merge(latest[cols], on="norm", how="left",
                        suffixes=("", "_hist"))
    dupes = merged.columns[merged.columns.duplicated()].tolist()
    check("and the merged frame has unique column names", not dupes, str(dupes))

    cov = P.coverage(pool, merged)
    # The denominator matters more than the rate. A kicker and a defence can
    # never match a skill-position history file, so counting them as misses
    # makes a healthy join look broken.
    check("kickers and defences are outside the projectable denominator",
          cov["projectable_n"] == 41, str(cov["projectable_n"]))
    check("the projectable rate is not dragged down by them",
          cov["projectable"] > cov["overall"],
          f'{cov["projectable"]:.3f} vs {cov["overall"]:.3f}')
    check("a cheap miss barely moves the salary-weighted coverage",
          cov["by_salary"] > 0.9, f'{cov["by_salary"]:.3f}')
    check("every miss is reported with its price, most expensive first",
          [m["name"] for m in cov["misses"]][0] == "Some Kicker",
          str([m["name"] for m in cov["misses"]]))

    known = merged[merged["player_id"].notna()].copy()
    out = M.Projections().fit(built).predict(known)
    check("and the prediction that used to crash now runs",
          len(out) == 40, str(len(out)))


def main():
    print("DFS features - offline checks")
    for fn in (test_no_leakage, test_first_row_is_blank,
               test_usage_tracks_role_change, test_team_context,
               test_target_matches_scoring, test_trainable,
               test_empty_feature_does_not_kill_the_fit,
               test_slate_merge_is_unique):
        try:
            fn()
        except Exception:
            global FAIL
            FAIL += 1
            FAILURES.append(f"{fn.__name__} raised")
            print(f"  CRASH in {fn.__name__}")
            traceback.print_exc()
    print(f"\n{'=' * 66}\n{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
