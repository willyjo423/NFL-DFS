"""Play the slate out fifty thousand times, with the players correlated.

Why correlation is the whole point
----------------------------------
Sampling each player independently from his own fitted distribution is easy and
badly wrong. Football outcomes are not independent: when a quarterback throws
for 400 yards, his receivers caught them. An independent simulator will report
that a QB-WR stack has the same ceiling as two unrelated players with the same
projections, because in its world the two events never coincide. They coincide
constantly, and that coincidence is the entire reason tournaments are won with
stacks.

So the simulator draws correlated uniforms first and pushes them through each
player's own inverse CDF. This is a Gaussian copula, and the reason to use one
here rather than simply drawing correlated normals is that it keeps the
marginal distributions exactly as the quantile model fitted them - a skewed,
step-function-aware receiver stays skewed after correlation is imposed. Draw
correlated normals directly and every player silently becomes symmetric, which
throws away the reason for fitting quantiles at all.

The correlation matrix
----------------------
Built from the priors in config, by position pair and by relationship (same
team, opposing team, same game). A matrix assembled this way is not guaranteed
to be positive semi-definite - it is a table of pairwise opinions, not a
covariance estimated from data - so it is repaired by clipping negative
eigenvalues before use. Skipping that repair does not raise; it silently
produces a Cholesky failure or, worse, nonsense draws.

What comes out
--------------
A matrix of shape (players, sims). Every downstream question - what does this
lineup's distribution look like, how often does it clear the cash line, how
often does it reach the top tenth of a percent - is answered by indexing into
that matrix and summing, never by re-simulating.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
import factors

log = logging.getLogger(__name__)


def relationship(team_a: str, team_b: str, game_a: str, game_b: str) -> str:
    """How two players on a slate are related."""
    if team_a == team_b:
        return "same_team"
    if game_a and game_b and game_a == game_b:
        return "opponent"
    return "unrelated"


def correlation_matrix(players: pd.DataFrame,
                       priors: dict | None = None,
                       sport: str = "nfl") -> np.ndarray:
    """Correlation from the factor model.

    `priors` is kept only so the independence control still works - passing an
    empty dict returns the identity, which is what a test that wants
    uncorrelated draws needs. Everything else comes from `factors`, where the
    matrix is positive semi-definite by construction and so never gets shrunk
    on the way through.

    The pairwise table this replaced could not be satisfied: six receivers at
    +0.35 to one quarterback force those receivers above -0.053 with each
    other, the prior asked for -0.10, and the repair paid for that by pulling
    every correlation in the matrix down about a third.
    """
    if priors is not None and not priors:
        return np.eye(len(players))
    corr, info = factors.build(players, sport)
    if info["min_eigenvalue"] < -1e-9:        # should be impossible
        log.warning("factor matrix came out indefinite (%.2e) - repairing",
                    info["min_eigenvalue"])
        return nearest_psd(corr)
    return corr


def nearest_psd(corr: np.ndarray) -> np.ndarray:
    """Repair a correlation matrix that is not positive semi-definite.

    A table of pairwise opinions need not be a valid joint distribution. Two
    perfectly reasonable priors - a QB correlates with both his receivers, the
    receivers compete with each other - can imply something impossible when
    taken together. Clipping the negative eigenvalues finds the nearest matrix
    that is possible, and rescaling restores the unit diagonal.

    Without this the failure is a LinAlgError deep inside the draw, or silently
    unusable samples, and neither says what actually went wrong.
    """
    sym = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= -1e-10:
        return sym
    log.info("correlation matrix was not positive semi-definite "
             "(smallest eigenvalue %.4f); repairing", vals.min())
    vals = np.clip(vals, 1e-8, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def correlation_report(players: pd.DataFrame, draws: np.ndarray,
                       priors: dict | None = None,
                       sport: str = "nfl") -> pd.DataFrame:
    """What the factor model assumes, and what the sims delivered.

    Two columns that should now agree. Under the old pairwise table they did
    not and could not: the table was infeasible, the repair shrank it, and the
    delivered column ran about two-thirds of the asked column with no
    indication on the page that this was happening. Any gap here now is
    sampling noise, and a large one is a bug.
    """
    corr = correlation_matrix(players, priors, sport)
    implied = factors.implied_pairs(players, corr)
    got = []
    for r in implied.itertuples(index=False):
        a, _, b = r.pair.partition("-")
        got.append(realised_correlation(draws, players, a, b, r.relationship))
    out = implied.copy()
    out["delivered"] = [round(v, 3) if v == v else float("nan") for v in got]
    out["gap"] = (out["delivered"] - out["correlation"]).round(3)
    return out.rename(columns={"correlation": "assumed"})


def simulate(players: pd.DataFrame, quantiles: list[float], n: int,
             priors: dict | None = None,
             rng: np.random.Generator | None = None,
             sport: str = "nfl") -> np.ndarray:
    """Correlated fantasy-point draws: one row per player, one column per sim.

    The copula step in three lines: draw correlated standard normals, convert
    each to a uniform through the normal CDF - which preserves the correlation
    structure while making every margin uniform - then read each player's own
    fitted quantile curve at that uniform. What comes back has the model's
    marginals and the priors' dependence.
    """
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    cols = [f"q{int(q * 100)}" for q in quantiles]
    missing = [c for c in cols if c not in players.columns]
    if missing:
        raise ValueError(f"players lack fitted quantiles {missing}")

    grid = players[cols].to_numpy(dtype=float)
    zq = _z_of(np.asarray(quantiles, dtype=float))

    corr = correlation_matrix(players, priors, sport)
    chol = np.linalg.cholesky(corr)
    z = chol @ rng.standard_normal((len(players), n))

    out = np.empty_like(z)
    for i in range(len(players)):
        out[i] = _quantile_curve(zq, grid[i], z[i])
    np.clip(out, 0.0, None, out=out)
    return out


def _z_of(q: np.ndarray) -> np.ndarray:
    """Standard normal quantile function."""
    from scipy.stats import norm
    return norm.ppf(q)


def _quantile_curve(z_grid: np.ndarray, values: np.ndarray,
                    z: np.ndarray) -> np.ndarray:
    """Read a player's fitted curve at a latent normal draw.

    Interpolating in z-space rather than in uniform space matters more than it
    sounds. The fitted quantiles stop at the 10th and the 97th, and np.interp
    CLAMPS outside its range - so a straight uniform lookup hands roughly one
    draw in ten the exact same q10 value and one in thirty the exact q97. Those
    two flat shelves are 13% of every simulation frozen at a constant, and a
    constant correlates with nothing: measured QB-WR correlation came out at
    0.24 against a 0.35 prior, and the tournament tail - the whole reason the
    97th percentile is fitted at all - was a step instead of a tail.

    Interpolating against z and extending the outer segments linearly gives
    each player a real tail and makes the transform close to affine, which is
    what lets the imposed correlation survive it.
    """
    out = np.interp(z, z_grid, values)
    lo = z < z_grid[0]
    if lo.any():
        slope = (values[1] - values[0]) / (z_grid[1] - z_grid[0])
        out[lo] = values[0] + slope * (z[lo] - z_grid[0])
    hi = z > z_grid[-1]
    if hi.any():
        slope = (values[-1] - values[-2]) / (z_grid[-1] - z_grid[-2])
        out[hi] = values[-1] + slope * (z[hi] - z_grid[-1])
    return out


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised error function. numpy has no erf; scipy does."""
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:          # pragma: no cover - scipy is a dependency
        # Abramowitz & Stegun 7.1.26, good to ~1.5e-7 - fine for a uniform.
        sign = np.sign(x)
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741)
                    * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
        return sign * y


def realised_correlation(draws: np.ndarray, players: pd.DataFrame,
                         pos_a: str, pos_b: str, rel: str) -> float:
    """Measure back out of the sims what was asked for going in.

    A simulator that claims a 0.35 QB-WR correlation and delivers 0.02 is worse
    than one that claims nothing, and the only way to know which you have is to
    measure it. Used by the tests and printed by the report.
    """
    pos = players["position"].astype(str).to_numpy()
    team = players["team"].astype(str).to_numpy()
    game = players.get("game", pd.Series([""] * len(players))).astype(str) \
                  .to_numpy()
    got = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            if {pos[i], pos[j]} != {pos_a, pos_b}:
                continue
            if relationship(team[i], team[j], game[i], game[j]) != rel:
                continue
            a, b = draws[i], draws[j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            got.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(got)) if got else float("nan")


def lineup_scores(draws: np.ndarray, rows: list[int],
                  multipliers: list[float] | None = None) -> np.ndarray:
    """What one lineup scored in every simulation.

    Multipliers carry the showdown captain: 1.5x on that row and 1.0 elsewhere.
    Summing the same draws the rest of the slate was scored from is what keeps
    two lineups comparable - re-simulating per lineup would make the difference
    between them partly noise.
    """
    m = np.ones(len(rows)) if multipliers is None \
        else np.asarray(multipliers, dtype=float)
    return (draws[rows, :] * m[:, None]).sum(axis=0)


def cash_line(draws: np.ndarray, roster: dict,
              fraction: float | None = None,
              trials: int = 2000,
              rng: np.random.Generator | None = None) -> float:
    """The score a cash lineup has to beat, estimated from the slate itself.

    Rather than assume a number, build a crowd of legal-ish random lineups,
    score each in its own simulation, and take the quantile the payout implies.
    It is a rough field - it does not know that nobody rosters a $200 long
    snapper - so it understates the real line, and the report says so rather
    than dressing it up.
    """
    rng = rng or np.random.default_rng(config.RANDOM_SEED + 1)
    fraction = config.CASH_PAYOUT_FRACTION if fraction is None else fraction
    size = len([s for s in roster["slots"]])
    n_players, n_sims = draws.shape
    if n_players < size:
        raise ValueError("fewer players than roster slots")

    picks = np.array([rng.choice(n_players, size=size, replace=False)
                      for _ in range(trials)])
    sims = rng.integers(0, n_sims, size=trials)
    totals = draws[picks, sims[:, None]].sum(axis=1)
    return float(np.quantile(totals, 1.0 - fraction))
