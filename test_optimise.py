"""Offline checks of the simulator and the optimisers. No network.

Two of these matter more than the rest.

The first is that the correlation asked for comes back out of the simulations.
A copula that quietly delivers zero correlation is indistinguishable from an
independent simulator, and an independent simulator gets stacking exactly
backwards while looking like it works.

The second is that the cash and tournament objectives actually disagree. If
they return the same lineup, the whole two-stage architecture is decoration
over a points-maximiser and should be deleted rather than believed.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import config
import simulate as S

try:
    import optimise as O
    HAVE_SOLVER = True
except ImportError as exc:                      # pulp absent
    O, HAVE_SOLVER = None, False
    print(f"NOTE: solver checks skipped - {exc}")

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


QUANTILES = config.QUANTILES


def slate(showdown: bool = True, seed: int = 3) -> pd.DataFrame:
    """A plausible player pool with fitted quantiles already attached."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = ["DEN", "KC"] if showdown else [f"T{i}" for i in range(6)]
    spec = ([("QB", 2), ("RB", 4), ("WR", 6), ("TE", 3), ("K", 1), ("DST", 1)]
            if showdown else
            [("QB", 4), ("RB", 8), ("WR", 12), ("TE", 6), ("DST", 6)])
    for team in teams:
        for pos, count in spec:
            for k in range(count):
                base = {"QB": 18.0, "RB": 11.0, "WR": 9.0, "TE": 6.0,
                        "K": 7.0, "DST": 7.0}[pos] * (1.0 - 0.14 * k)
                base = max(base, 1.0)
                sd = base * 0.55
                qs = {f"q{int(q * 100)}":
                      max(0.0, base + sd * _z(q)) for q in QUANTILES}
                rows.append({
                    "name": f"{team} {pos}{k}", "position": pos, "team": team,
                    "game": "DEN@KC" if showdown else f"G{teams.index(team)//2}",
                    "salary": float(max(200, round(base * 620 + rng.normal(0, 400), -2))),
                    **qs})
    return pd.DataFrame(rows)


def _z(q: float) -> float:
    """Standard normal quantile, good enough to shape a fixture."""
    from math import sqrt
    try:
        from scipy.stats import norm
        return float(norm.ppf(q))
    except ImportError:                         # pragma: no cover
        return float(sqrt(2) * (2 * q - 1))


# ---------------------------------------------------------------- simulator
def test_correlation_survives_the_copula():
    section("THE CORRELATION ASSUMED MUST COME BACK OUT, NOT TWO-THIRDS OF IT")
    import factors
    pool = slate()
    draws = S.simulate(pool, QUANTILES, 8000)

    corr, info = factors.build(pool, "nfl")
    # The whole point of the factor model: a sum of positive semi-definite
    # pieces is positive semi-definite, so nothing is ever repaired and nothing
    # is ever silently shrunk. The pairwise table it replaced was infeasible -
    # six receivers at +0.35 to one quarterback cannot be below -0.053 with
    # each other, the prior said -0.10, and the repair charged every other pair
    # a third of its size to fix it.
    check("the matrix is valid by construction, with no repair",
          info["min_eigenvalue"] >= -1e-9, str(info["min_eigenvalue"]))
    check("so a Cholesky succeeds on it directly",
          np.linalg.cholesky(corr).shape == (len(pool), len(pool)))

    rep = S.correlation_report(pool, draws)
    worst = rep.reindex(rep["gap"].abs().sort_values(ascending=False).index)
    check("every pair is delivered as assumed, within sampling noise",
          bool((worst["gap"].abs() < 0.05).all()),
          str(worst.head(3).to_dict("records")))

    def pair(a, b, rel):
        r = rep[(rep["pair"] == f"{a}-{b}") & (rep["relationship"] == rel)]
        return float(r["delivered"].iloc[0]) if len(r) else float("nan")

    qbwr = pair("QB", "WR", "same_team")
    check("a quarterback and his own receivers move together, at full size",
          qbwr > 0.30, f"{qbwr:.3f}")
    check("an opposing receiver is correlated, but less than a team-mate",
          0.0 < pair("QB", "WR", "opponent") < qbwr,
          f'{pair("QB", "WR", "opponent"):.3f} vs {qbwr:.3f}')

    # Same-position team-mates compete for one pool of touches. Getting these
    # signs wrong is what lets an optimiser roster both of a team's running
    # backs, or a starting and a backup quarterback, as if they were
    # independent bets.
    check("two backs on one team are negatively correlated",
          pair("RB", "RB", "same_team") < 0,
          f'{pair("RB", "RB", "same_team"):.3f}')
    check("and two quarterbacks more so, since only one plays",
          pair("QB", "QB", "same_team") < pair("RB", "RB", "same_team"),
          f'{pair("QB", "QB", "same_team"):.3f}')

    # A shootout is good for passers and bad for both defences. A model with
    # this sign wrong would stack a quarterback with the defence facing him.
    check("a defence moves against the quarterback it is facing",
          pair("DST", "QB", "opponent") < 0,
          f'{pair("DST", "QB", "opponent"):.3f}')
    check("and with its own offence",
          pair("DST", "RB", "same_team") > 0,
          f'{pair("DST", "RB", "same_team"):.3f}')

    # The honest consequence, asserted so nobody later "fixes" it: a shared
    # cause cannot produce anticorrelated effects. Receivers tied to one
    # quarterback come out positive with each other, and the old prior that
    # said otherwise was not achievable.
    check("same-team receivers come out positive, as the arithmetic forces",
          pair("WR", "WR", "same_team") > 0,
          f'{pair("WR", "WR", "same_team"):.3f}')

    # The negative control. Without it, a model that correlated EVERYTHING
    # would pass every check above.
    ind = S.simulate(pool, QUANTILES, 8000, priors={})
    flat = S.realised_correlation(ind, pool, "QB", "WR", "same_team")
    check("with the factors switched off the same players are independent",
          abs(flat) < 0.05, f"{flat:.3f}")
    print(rep.head(12).to_string(index=False))


def test_marginals_are_not_distorted():
    section("CORRELATION MUST NOT RESHAPE THE INDIVIDUAL PLAYERS")
    pool = slate()
    draws = S.simulate(pool, QUANTILES, 20000)
    # This is the reason for a copula rather than correlated normals: each
    # player has to keep the skewed distribution the quantile model fitted.
    worst, who = 0.0, None
    for i in range(len(pool)):
        for q in (0.25, 0.50, 0.90):
            want = float(pool[f"q{int(q * 100)}"].iloc[i])
            got = float(np.quantile(draws[i], q))
            scale = max(1.0, want)
            if abs(got - want) / scale > worst:
                worst, who = abs(got - want) / scale, (pool["name"].iloc[i], q)
    check("every fitted percentile is reproduced by the sims",
          worst < 0.08, f"worst {worst:.3f} at {who}")


def test_psd_repair():
    section("AN IMPOSSIBLE CORRELATION MATRIX IS REPAIRED, NOT RAISED")
    bad = np.array([[1.0, 0.9, -0.9],
                    [0.9, 1.0, 0.9],
                    [-0.9, 0.9, 1.0]])
    check("the fixture really is not positive semi-definite",
          np.linalg.eigvalsh(bad).min() < 0)
    fixed = S.nearest_psd(bad)
    check("the repaired matrix is usable",
          np.linalg.eigvalsh(fixed).min() >= -1e-9,
          str(np.linalg.eigvalsh(fixed).min()))
    check("its diagonal is still one",
          bool(np.allclose(np.diag(fixed), 1.0)))
    check("and it is still symmetric", bool(np.allclose(fixed, fixed.T)))
    check("a Cholesky now succeeds, which is what the draw needs",
          np.linalg.cholesky(fixed).shape == (3, 3))


# ---------------------------------------------------------------- optimiser
def test_showdown_lineups_are_legal():
    section("EVERY SHOWDOWN LINEUP MUST BE ENTERABLE")
    roster = config.ROSTERS[("dk", "Showdown Captain Mode")]
    pool = slate()
    draws = S.simulate(pool, QUANTILES, 3000)
    cands = O.candidates(pool, roster, draws, n_candidates=25)
    check("the solver produced alternatives, not one lineup",
          len(cands) >= 10, str(len(cands)))

    mult = roster["captain_multiplier"]
    bad_size = bad_cap = bad_dupe = bad_team = no_captain = 0
    for c in cands:
        rows = c["rows"]
        if len(rows) != 6:
            bad_size += 1
        if len(set(rows)) != len(rows):
            bad_dupe += 1
        if c["captain"] is None:
            no_captain += 1
        charged = sum(float(pool["salary"].iloc[i])
                      * (mult if i == c["captain"] else 1.0) for i in rows)
        if charged > roster["salary_cap"] + 1e-6:
            bad_cap += 1
        counts = pool.iloc[rows]["team"].value_counts()
        if counts.max() > roster["max_per_team"]:
            bad_team += 1

    check("every lineup has six players", bad_size == 0, str(bad_size))
    check("nobody appears twice", bad_dupe == 0, str(bad_dupe))
    check("every lineup names exactly one captain", no_captain == 0,
          str(no_captain))
    check("the captain's 1.5x salary is inside the cap, not applied after",
          bad_cap == 0, f"{bad_cap} lineups over the cap")
    check("no more than five from one team", bad_team == 0, str(bad_team))


def test_classic_lineups_are_legal():
    section("AND SO MUST A CLASSIC LINEUP")
    roster = config.ROSTERS[("dk", "Classic")]
    pool = slate(showdown=False)
    draws = S.simulate(pool, QUANTILES, 2000)
    cands = O.candidates(pool, roster, draws, n_candidates=15)

    bad = []
    for c in cands:
        got = pool.iloc[c["rows"]]["position"].value_counts().to_dict()
        if len(c["rows"]) != 9:
            bad.append(("size", got))
        if got.get("QB", 0) != 1 or got.get("DST", 0) != 1:
            bad.append(("qb/dst", got))
        if not 2 <= got.get("RB", 0) <= 3:
            bad.append(("rb", got))
        if not 3 <= got.get("WR", 0) <= 4:
            bad.append(("wr", got))
        if not 1 <= got.get("TE", 0) <= 2:
            bad.append(("te", got))
        if sum(pool["salary"].iloc[i] for i in c["rows"]) > 50_000:
            bad.append(("salary", got))
    check("position counts and the cap all hold", not bad, str(bad[:3]))
    check("a kicker never sneaks into a classic lineup",
          all("K" not in pool.iloc[c["rows"]]["position"].tolist()
              for c in cands))


def test_cash_and_gpp_disagree():
    section("THE TWO OBJECTIVES MUST NOT RETURN THE SAME LINEUP")
    roster = config.ROSTERS[("dk", "Showdown Captain Mode")]
    pool = slate()
    draws = S.simulate(pool, QUANTILES, 8000)

    cash = O.build(pool, roster, draws, objective="cash", n_candidates=60)
    gpp = O.build(pool, roster, draws, objective="gpp", n_candidates=60)

    cash_names = set(cash["name"])
    gpp_names = set(gpp["name"])
    check("each build returns a full lineup",
          len(cash) == 6 and len(gpp) == 6, f"{len(cash)} / {len(gpp)}")
    check("the tournament lineup is not the cash lineup",
          cash_names != gpp_names,
          "identical lineups mean the objective is not being used")

    # And the direction of the disagreement has to be right: the tournament
    # build should reach higher and risk more, not merely differ.
    check("the tournament build has the higher ceiling",
          float(gpp["ceiling"].iloc[0]) >= float(cash["ceiling"].iloc[0]),
          f'{gpp["ceiling"].iloc[0]:.1f} vs {cash["ceiling"].iloc[0]:.1f}')
    check("and the cash build has the higher floor",
          float(cash["floor"].iloc[0]) >= float(gpp["floor"].iloc[0]),
          f'{cash["floor"].iloc[0]:.1f} vs {gpp["floor"].iloc[0]:.1f}')
    check("the cash build beats the cash line more often",
          float(cash["p_cash"].iloc[0]) >= float(gpp["p_cash"].iloc[0]),
          f'{cash["p_cash"].iloc[0]:.3f} vs {gpp["p_cash"].iloc[0]:.3f}')
    check("and the tournament build reaches the tournament bar more often",
          float(gpp["p_gpp"].iloc[0]) >= float(cash["p_gpp"].iloc[0]),
          f'{gpp["p_gpp"].iloc[0]:.4f} vs {cash["p_gpp"].iloc[0]:.4f}')


def test_multi_entry_is_actually_diverse():
    section("FIVE ENTRIES MUST BE FIVE BETS, NOT ONE BET FIVE TIMES")
    roster = config.ROSTERS[("dk", "Showdown Captain Mode")]
    pool = slate()
    draws = S.simulate(pool, QUANTILES, 5000)
    out = O.build(pool, roster, draws, objective="gpp", entries=5,
                  n_candidates=120)

    entries = [set(g["name"]) for _, g in out.groupby("entry")]
    check("five entries came back", len(entries) == 5, str(len(entries)))
    worst = max((len(a & b) for i, a in enumerate(entries)
                 for b in entries[i + 1:]), default=0)
    check("no two entries share more than three of six players",
          worst <= 3, f"worst overlap {worst}")
    check("and they are not duplicates of each other",
          len({frozenset(e) for e in entries}) == 5)


def test_lineup_scoring_uses_shared_draws():
    section("TWO LINEUPS MUST BE COMPARED ON THE SAME SIMULATIONS")
    pool = slate()
    draws = S.simulate(pool, QUANTILES, 1000)
    rows = [0, 1, 2, 3, 4, 5]
    a = S.lineup_scores(draws, rows)
    b = S.lineup_scores(draws, rows)
    check("scoring the same lineup twice gives the same answer",
          bool(np.array_equal(a, b)),
          "re-simulating per lineup makes the comparison partly noise")
    capt = S.lineup_scores(draws, rows, [1.5, 1, 1, 1, 1, 1])
    check("and the captain multiplier raises the total",
          bool((capt >= a - 1e-9).all()) and float(capt.mean()) > float(a.mean()))


def test_ownership_has_to_add_up():
    section("OWNERSHIP IS MODELLED, BUT IT IS NOT FREE TO BE ANYTHING")
    import ownership as OWN

    # The constraint that stops this being a guess. Every classic entry fields
    # exactly one quarterback, so across the slate quarterback ownership sums
    # to exactly 1.0. Not approximately. That is arithmetic, and it pins the
    # scale of the whole distribution - only the concentration is guessed.
    classic = config.ROSTERS[("dk", "Classic")]
    pool = slate(showdown=False)
    pool["median"] = pool["q50"]
    pool["ceiling"] = pool["q97"]
    own = OWN.project(pool, classic)
    demand = OWN.slot_demand(classic)

    check("the demands sum to the roster size",
          abs(sum(demand.values()) - len(classic["slots"])) < 1e-9,
          str(demand))
    worst = max(abs(float(own[pool["position"] == p].sum()) - w)
                for p, w in demand.items())
    check("every position sums to exactly what a lineup demands",
          worst < 1e-6, f"worst error {worst:.4f}")
    check("and the whole board sums to the roster size",
          abs(float(own.sum()) - len(classic["slots"])) < 1e-6,
          str(float(own.sum())))

    showdown = config.ROSTERS[("dk", "Showdown Captain Mode")]
    sp = slate()
    sp["median"] = sp["q50"]
    sp["ceiling"] = sp["q97"]
    so = OWN.project(sp, showdown)
    check("showdown sums to six, since it has no position requirements",
          abs(float(so.sum()) - 6.0) < 1e-6, str(float(so.sum())))
    check("nobody is owned by the entire field",
          float(so.max()) <= OWN.MAX_OWNERSHIP + 1e-9, str(float(so.max())))
    check("and nobody is owned by nobody",
          float(so.min()) > 0, str(float(so.min())))

    # The failure that made the first version useless: points-per-dollar is
    # degenerate at the bottom of a board. A $200 player projected for 2.3
    # points scores 11.4 per $1,000 against a $9,600 quarterback's 1.82, so
    # leading with value predicted the field would roster long snappers at 65%.
    # A real board, not a toy one. Six roster slots drawn from five players
    # forces everybody to the ownership cap and tests nothing - the pool has to
    # be bigger than the lineup for any of this to mean anything.
    real = pd.DataFrame(
        [("Mahomes", "QB", "KC", 9600.0, 17.5, 33.1),
         ("Nix", "QB", "DEN", 9800.0, 17.3, 33.7),
         ("Rice", "WR", "KC", 9400.0, 14.7, 35.7),
         ("Walker", "RB", "KC", 10600.0, 14.5, 32.9),
         ("Waddle", "WR", "DEN", 9000.0, 11.1, 31.2),
         ("Kelce", "TE", "KC", 7000.0, 9.8, 24.7),
         ("Dobbins", "RB", "DEN", 6400.0, 9.6, 30.0),
         ("Franklin", "WR", "DEN", 2800.0, 8.0, 25.4),
         ("Mims", "WR", "DEN", 3000.0, 5.7, 23.6),
         ("Engram", "TE", "DEN", 3400.0, 3.6, 18.3),
         ("Gray", "TE", "KC", 2000.0, 2.2, 13.6),
         ("LongSnapper", "TE", "KC", 200.0, 1.7, 9.0),
         ("Punter", "TE", "DEN", 200.0, 1.0, 11.4)],
        columns=["name", "position", "team", "salary", "median", "ceiling"])
    ro = OWN.project(real, showdown)
    ranked = real.assign(own=ro).sort_values("own", ascending=False)
    check("the best player on the board is the most owned",
          ranked["name"].iloc[0] == "Mahomes", str(ranked["name"].tolist()))
    check("a minimum-salary non-factor is not top-two owned",
          "LongSnapper" not in set(ranked["name"].head(2)),
          str(ranked["name"].tolist()))
    check("and he is owned less than the cheap player who can actually score",
          float(ro[real["name"] == "LongSnapper"].iloc[0])
          < float(ro[real["name"] == "Franklin"].iloc[0]))

    # A player who will not take a snap is owned by nobody, whatever he costs.
    hurt = real.assign(playing=np.where(real["name"] == "Franklin",
                                        "out", "clear"))
    ho = OWN.project(hurt, showdown)
    check("an inactive player attracts no ownership",
          float(ho[hurt["name"] == "Franklin"].iloc[0]) < 0.01,
          str(float(ho[hurt["name"] == "Franklin"].iloc[0])))

    # Duplication is the number a tournament is actually played for.
    chalk = np.array([0.60, 0.55, 0.50, 0.45, 0.40, 0.35])
    contrarian = np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
    d_chalk = OWN.duplication(chalk, 200_000)
    d_lev = OWN.duplication(contrarian, 200_000)
    check("a chalk lineup is fielded by many other entries",
          d_chalk > 100, f"{d_chalk:,.0f}")
    check("a contrarian one by almost none",
          d_lev < 1.0, f"{d_lev:.4f}")
    check("which is a difference of orders of magnitude, not a rounding",
          d_chalk / max(d_lev, 1e-9) > 1000, f"{d_chalk / max(d_lev, 1e-9):,.0f}x")

    lev = OWN.leverage(real, ro)
    check("leverage is positive where the model likes a player more than "
          "the field will", bool((lev != 0).any()), str(list(lev)))


def main():
    print("DFS simulator and optimisers - offline checks")
    maths = (test_ownership_has_to_add_up,
             test_correlation_survives_the_copula,
             test_marginals_are_not_distorted,
             test_psd_repair,
             test_lineup_scoring_uses_shared_draws)
    solver = (test_showdown_lineups_are_legal,
              test_classic_lineups_are_legal,
              test_cash_and_gpp_disagree,
              test_multi_entry_is_actually_diverse)
    if not HAVE_SOLVER:
        print("\nSOLVER CHECKS SKIPPED - pulp is not installed here.\n"
              "They run in CI, where it is. Do not read a pass below as a\n"
              "pass on the optimiser.")
    for fn in maths + (solver if HAVE_SOLVER else ()):
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
