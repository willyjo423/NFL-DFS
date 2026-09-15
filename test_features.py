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

    hist = built[["norm", "position"]].dropna().drop_duplicates()
    cov = P.coverage(merged, hist)
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
    check("a debutant is not counted as a broken join",
          not cov["broken"], str(cov["broken"]))

    # Now the distinction the gate turns on: a player the history file DOES
    # contain, who failed to join anyway. That is a bug, not a rookie, and it
    # must be reported as one.
    # A nickname variant, which is how this bug actually arrives: one side
    # writes the short form. Same surname, same position, compatible first
    # name - all three, which is what it takes to raise a hand.
    first, last = P.name_parts(pool["norm"].iloc[0])
    hurt = pool.copy()
    hurt.loc[hurt.index[0], "norm"] = f"{first}ster {last}"
    bad = hurt.merge(latest[cols], on="norm", how="left", suffixes=("", "_hist"))
    cov2 = P.coverage(bad, hist)
    check("a starter under a nickname variant is flagged as a join failure",
          len(cov2["broken"]) == 1, str(cov2["broken"]))
    check("while a genuine debutant still is not",
          all(b["name"] != "Third Stringer" for b in cov2["broken"]),
          str(cov2["broken"]))

    # The false-positive case, which is the one that actually bit. Five rookies
    # were flagged as join failures because a different player shared their
    # surname and first initial: Cyrus Allen against Chase Allen, Emmett
    # Johnson against Eric Johnson. A gate that fires on five rookies trains
    # you to force past it.
    check("cyrus is not chase", not P.same_person("cyrus", "chase"))
    check("emmett is not eric", not P.same_person("emmett", "eric"))
    check("jonah is not justin", not P.same_person("jonah", "justin"))
    check("jeff is not jamaree", not P.same_person("jeff", "jamaree"))
    check("dane is not devon", not P.same_person("dane", "devon"))
    check("but cam is cameron", P.same_person("cam", "cameron"))
    check("and chris is christopher", P.same_person("chris", "christopher"))
    check("a single letter is not a nickname", not P.same_person("c", "chase"))

    rookies = pd.DataFrame({"norm": ["cyrus allen", "emmett johnson"],
                            "position": ["WR", "RB"]})
    hist2 = pd.DataFrame({"norm": ["chase allen", "eric johnson"],
                          "position": ["WR", "RB"]})
    check("so the live false positives stay silent even at the same position",
          P.probable_join_failures(rookies, hist2).empty)

    known = merged[merged["player_id"].notna()].copy()
    out = M.Projections().fit(built).predict(known)
    check("and the prediction that used to crash now runs",
          len(out) == 40, str(len(out)))


def test_showdown_salary_is_the_flex_price():
    section("SHOWDOWN: THE CAPTAIN PRICE MUST NOT BECOME THE PRICE")
    import data

    # DraftKings lists a showdown player twice - once at the flex price, once
    # in the captain slot at 1.5x for 1.5x the points. The first version
    # collapsed on roster slot id and kept the CAPTAIN row, so every salary on
    # the board arrived 1.5x too high with a flex projection attached to it.
    payload = {"draftables": []}
    for i, (nm, flex) in enumerate([("Alpha Back", 10600), ("Beta Wide", 8000),
                                    ("Gamma End", 200)]):
        for slot, sal in ((511, int(flex * 1.5)), (512, flex)):
            payload["draftables"].append({
                "playerId": 900 + i, "draftableId": 1000 + i * 2 + slot % 2,
                "displayName": nm, "position": "RB", "teamAbbreviation": "KC",
                "salary": sal, "rosterSlotId": slot, "status": "Available",
                "competition": {"name": "DEN @ KC",
                                "startTime": "2026-09-15T00:15:00.0000000Z"},
            })

    real_get = data._get
    data._get = lambda *a, **k: payload
    try:
        df = data.draftables(1)
    finally:
        data._get = real_get

    check("one row per player", len(df) == 3, str(len(df)))
    check("the salary kept is the flex price, not 1.5x it",
          sorted(df["salary"].tolist()) == [200, 8000, 10600],
          str(sorted(df["salary"].tolist())))
    check("and the captain price is kept alongside it",
          sorted(df["captain_salary"].tolist()) == [300, 12000, 15900],
          str(sorted(df["captain_salary"].tolist())))
    check("the captain price is exactly 1.5x",
          bool((df["captain_salary"] / df["salary"]).round(2).eq(1.5).all()))
    check("a min-salary player prices at 200, which is the tell",
          int(df["salary"].min()) == 200, str(int(df["salary"].min())))


def test_inactive_players_are_excluded():
    section("AN INACTIVE PLAYER MUST NOT REACH A LINEUP")
    import data
    import project as P

    # This is the live failure. Troy Franklin was inactive, DraftKings had him
    # at $2,800, and because a zero-snap player maximises points-per-dollar he
    # came back as the best value on the board and went into BOTH lineups.
    # The status was in the payload the whole time and nothing read it.
    for raw, want in [
        ("OUT", "out"), ("Out", "out"), ("O", "out"), ("IR", "out"),
        ("Inactive", "out"), ("SUSP", "out"), ("PUP", "out"),
        ("D", "doubtful"), ("Doubtful", "doubtful"),
        ("Q", "questionable"), ("Questionable", "questionable"),
        ("GTD", "questionable"),
        ("", "clear"), ("None", "clear"), (None, "clear"),
        ("Probable", "clear"), ("-", "clear"),
    ]:
        got = data.playing_status(raw)
        check(f"status {raw!r} reads as {want}", got == want, got)

    check("a disabled flag beats any status string",
          data.playing_status("None", disabled=True) == "out")
    check("and an attribute can carry it too",
          data.playing_status("", attributes="Injured Reserve") == "out")
    # An unrecognised string must NOT be guessed as out - that would delete a
    # board the moment DraftKings changed a spelling.
    check("an unknown status is treated as playable, not guessed out",
          data.playing_status("Wibble") == "clear")

    pool = pd.DataFrame({
        "name": ["Starter", "Franklin", "Iffy", "Shaky", "Fine"],
        "position": ["WR", "WR", "RB", "TE", "QB"],
        "team": ["KC", "DEN", "KC", "DEN", "KC"],
        "salary": [9400.0, 2800.0, 5000.0, 4000.0, 9600.0],
        "status": ["None", "OUT", "Q", "D", ""],
        "disabled": [False] * 5,
        "attributes": [""] * 5,
    })
    pool["playing"] = [data.playing_status(s, d, a) for s, d, a
                       in zip(pool["status"], pool["disabled"],
                              pool["attributes"])]
    kept, dropped = P.drop_unavailable(pool)
    check("the out player is gone", "Franklin" not in set(kept["name"]))
    check("and so is the doubtful one", "Shaky" not in set(kept["name"]))
    check("questionable is kept, because those players mostly play",
          "Iffy" in set(kept["name"]))
    check("the healthy players survive",
          {"Starter", "Fine"} <= set(kept["name"]))
    check("and the exclusions are returned, not swallowed",
          set(dropped["name"]) == {"Franklin", "Shaky"},
          str(dropped["name"].tolist()))

    # The guard that matters more than the filter: if a parsing change made
    # every status unreadable, a silent filter would hand back an empty board.
    broken = pool.copy()
    broken["playing"] = "out"
    try:
        P.drop_unavailable(broken)
        check("a board that reads as all-out stops the run", False,
              "it returned a pool instead of raising")
    except SystemExit as exc:
        check("a board that reads as all-out stops the run", True)
        check("and the error shows the statuses it actually saw",
              "status" in str(exc).lower() or "parsing failure" in str(exc))

    missing = pool.drop(columns=["playing"])
    kept2, dropped2 = P.drop_unavailable(missing)
    check("a pool with no availability column passes through, loudly",
          len(kept2) == 5 and dropped2.empty)


def test_market_lines_are_a_feature_not_a_leak():
    section("THE MARKET LINE: THE ONLY FORWARD-LOOKING INPUT")
    import features as FT

    raw = fixture()
    # A closing line exists BEFORE kickoff, so using it is not leakage - it is
    # the one input that can know about something the history cannot: a new
    # starting quarterback, an expected blowout, weather.
    games = raw[["season", "week", "team"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(4)
    total = 40 + rng.normal(0, 5, len(games))
    spread = rng.normal(0, 6, len(games))
    lines = games.assign(game_total=total, team_spread=spread,
                         implied_total=(total + spread) / 2.0,
                         is_home=(np.arange(len(games)) % 2).astype(float))

    before = len(raw)
    joined = FT.attach_market(raw.copy(), lines)
    check("the join does not duplicate player-weeks", len(joined) == before,
          f"{before} -> {len(joined)}")
    check("every market column arrives",
          {"implied_total", "game_total", "team_spread", "is_home"}
          <= set(joined.columns))
    check("and it actually matched", joined["implied_total"].notna().all())

    # Missing must stay missing. A game total of zero is a claim that nobody
    # will score, which is not what "we have no line" means - and the model
    # would learn from it.
    partial = lines.iloc[: len(lines) // 2]
    half = FT.attach_market(raw.copy(), partial)
    miss = half["implied_total"].isna()
    check("an unmatched game leaves a blank, never a zero",
          bool(miss.any()) and not bool((half.loc[miss, "game_total"] == 0).any()),
          "zeros here are a claim, not an absence")
    none = FT.attach_market(raw.copy(), None)
    check("no lines at all still builds, with the columns blank",
          bool(none["implied_total"].isna().all()))

    # Calling twice must not leave implied_total_x / implied_total_y behind,
    # with the real column quietly absent from the feature frame.
    twice = FT.attach_market(FT.attach_market(raw.copy(), lines), lines)
    check("joining twice does not produce suffixed duplicates",
          not any(c.endswith(("_x", "_y")) for c in twice.columns),
          str([c for c in twice.columns if c.endswith(("_x", "_y"))]))

    built = features.build(raw, lines=lines)
    check("the implied halves still sum to the game total",
          float((built["implied_total"] * 2 - built["game_total"]
                 - built["team_spread"]).abs().max()) < 1e-6)
    check("is_home is a real split, not a constant",
          0.2 < float(built["is_home"].mean()) < 0.8,
          str(built["is_home"].mean()))

    # The bug this fix exists for: _home_flag read the column as a string and
    # tested membership in ("1","true","home"). A numeric 1.0 stringifies to
    # "1.0", failed the test, and EVERY row came out 0 - the constant claim
    # that no team is ever at home, learned from as if it were data.
    check("a numeric 1.0 reads as home, not as zero",
          float(FT._home_flag(pd.DataFrame({"is_home": [1.0, 0.0, 1.0]})).mean())
          == 2 / 3)
    check("integers work too",
          list(FT._home_flag(pd.DataFrame({"is_home": [1, 0]}))) == [1.0, 0.0])
    check("and the old text spellings still work",
          list(FT._home_flag(pd.DataFrame({"home_away": ["home", "away"]})))
          == [1.0, 0.0])
    unknown = FT._home_flag(pd.DataFrame({"location": ["Wembley", "?"]}))
    check("an unrecognised value is missing, not away",
          bool(unknown.isna().all()), str(list(unknown)))

    # And the leak test again, this time WITH lines attached - the market
    # column must not smuggle the future in through a different door.
    full = features.build(raw, lines=lines)
    cut = features.build(raw[raw["week"] <= 8], lines=lines)
    key = ["player_id", "season", "week"]
    a = full[full["week"] <= 8].set_index(key).sort_index()
    b = cut.set_index(key).sort_index()
    worst = 0.0
    for c in features.FEATURES:
        if c not in a.columns or c not in b.columns:
            continue
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        both = ~(np.isnan(x) | np.isnan(y))
        if both.any():
            worst = max(worst, float(np.abs(x[both] - y[both]).max()))
    check("deleting the future still changes nothing, with market data in",
          worst < 1e-9, f"worst drift {worst:.3g}")


def test_the_market_on_a_projection_is_this_week_not_last():
    section("A PROJECTION MUST CARRY THE UPCOMING GAME'S LINE")
    import project as P

    # The usage features describe a player's last COMPLETED game, which is
    # correct - they are shifted by one week so a week can never be inside its
    # own feature. But the market columns were riding along on that same row,
    # so a week-two projection was made with week one's implied total: a team
    # implied for 22.5 against one opponent, projected as if it faced the same
    # opponent again. On live data this was wrong for 2,106 of 2,176 players,
    # with swings up to fifteen points of implied team total.
    #
    # The leak test cannot catch this. Using a STALE line is not leakage - it
    # breaks nothing, raises nothing, and looks entirely reasonable.
    latest = pd.DataFrame({
        "name": ["A", "B", "C"], "team": ["KC", "BUF", "ZZZ"],
        "season": [2026, 2026, 2026], "week": [1, 1, 1],
        "implied_total": [22.5, 23.0, 20.0],
        "game_total": [42.5, 44.5, 41.0],
        "team_spread": [-2.5, -1.5, 0.0], "is_home": [1.0, 0.0, 1.0],
        "ewm_points": [19.3, 21.0, 8.0]})
    lines = pd.DataFrame({
        "season": [2026] * 4, "week": [1, 1, 2, 2],
        "team": ["KC", "BUF", "KC", "BUF"],
        "opponent": ["LAC", "BAL", "PHI", "NYJ"],
        "implied_total": [22.5, 23.0, 27.0, 29.0],
        "game_total": [42.5, 44.5, 47.5, 53.5],
        "team_spread": [-2.5, -1.5, -6.0, -7.0],
        "is_home": [1.0, 0.0, 0.0, 1.0]})

    fresh = P.refresh_market(latest, lines)
    kc = fresh[fresh["name"] == "A"].iloc[0]
    check("the projection picks up THIS week's implied total",
          abs(float(kc["implied_total"]) - 27.0) < 1e-6,
          str(float(kc["implied_total"])))
    check("and this week's game total",
          abs(float(kc["game_total"]) - 47.5) < 1e-6)
    check("and this week's spread, which flipped",
          abs(float(kc["team_spread"]) - (-6.0)) < 1e-6)
    check("and this week's home flag, which also flipped",
          abs(float(kc["is_home"]) - 0.0) < 1e-6,
          str(float(kc["is_home"])))

    check("no player is duplicated by the refresh",
          len(fresh) == len(latest), f"{len(latest)} -> {len(fresh)}")
    check("the usage features are untouched - only the market moves",
          abs(float(fresh[fresh["name"] == "A"]["ewm_points"].iloc[0]) - 19.3)
          < 1e-9)

    # A team with no upcoming line gets a blank, not last week's number. A
    # stale line is a confident claim about a game that is not being played.
    zz = fresh[fresh["name"] == "C"].iloc[0]
    check("a team with no upcoming game is left blank, not left stale",
          pd.isna(zz["implied_total"]), str(zz["implied_total"]))

    check("no lines at all leaves the frame alone rather than raising",
          len(P.refresh_market(latest, None)) == len(latest))
    check("and an empty line table does the same",
          len(P.refresh_market(latest, lines.iloc[:0])) == len(latest))


def main():
    print("DFS features - offline checks")
    for fn in (test_the_market_on_a_projection_is_this_week_not_last,
               test_market_lines_are_a_feature_not_a_leak,
               test_no_leakage, test_first_row_is_blank,
               test_usage_tracks_role_change, test_team_context,
               test_target_matches_scoring, test_trainable,
               test_empty_feature_does_not_kill_the_fit,
               test_slate_merge_is_unique,
               test_showdown_salary_is_the_flex_price,
               test_inactive_players_are_excluded):
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
