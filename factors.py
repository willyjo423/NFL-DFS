"""Correlation built from factors, not from a table of opinions.

Why this replaces the pairwise priors
-------------------------------------
A table of pairwise correlations is not a distribution. It is a list of
separate beliefs, and separate beliefs can contradict each other. The previous
build asked for a quarterback correlating +0.35 with each of his receivers and
those receivers correlating -0.10 with each other, and that combination does
not exist: if six receivers each move with the same quarterback, they must move
with each other too. The exact floor is -0.053, so -0.10 missed feasibility by
about double - and because the repair pulls the WHOLE matrix toward the nearest
possible one, that small infeasibility cost every other pair a third of its
size. QB-WR was delivered at 0.244 against a 0.35 prior.

A factor model cannot contradict itself. Every player is a weighted sum of a
few shared shocks plus his own noise:

    z_i = g_i * Game + t_i * Team + (competition within team and position) + e_i

Correlation is then a CONSEQUENCE of the loadings rather than an assertion
about pairs, and the covariance is a sum of positive semi-definite pieces - so
it is always a real distribution and the repair step never runs.

What the structure says about football
--------------------------------------
* **Game** is shared by both teams on the field: a shootout lifts every
  passing game in it, which is why an opposing receiver is worth something in a
  stack and why a defence loads NEGATIVELY on it.
* **Team** is shared by one side: touchdowns, time of possession, a game script
  that keeps the offence on the field.
* **Competition** is within one team and one position, and it is the only
  negative term: two backs splitting carries, receivers splitting targets.

An honest consequence, stated rather than hidden
------------------------------------------------
Under this structure two receivers on the same team come out mildly POSITIVE,
not negative. That is not a bug being papered over - it is what the arithmetic
forces once you accept a strong quarterback link, and it matches what the game
actually does: in a pass-heavy script every receiver eats. The competition term
pulls it down but cannot pull it below zero, because a shared cause cannot
produce anticorrelated effects. The old prior asserted otherwise and was
infeasible.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Loadings are per position, per sport. Each is the share of a player's
# variance that moves with that factor, as a standard deviation - so a loading
# of 0.4 on the game factor and 0.45 on the team factor leaves
# 1 - 0.16 - 0.2025 of his variance to himself.
#
# A defence's NEGATIVE game loading is the important sign in here: a shootout
# is good for every passer on the field and bad for both defences, and a model
# that gets that sign wrong will happily build a lineup stacking a quarterback
# with the defence trying to stop him.
LOADINGS = {
    "nfl": {
        "game":    {"QB": 0.39, "WR": 0.39, "TE": 0.34, "RB": 0.20,
                    "K": 0.28, "DST": -0.34},
        "team":    {"QB": 0.45, "WR": 0.45, "TE": 0.42, "RB": 0.38,
                    "K": 0.42, "DST": 0.46},
        # Competition is the only negative term, and it is bounded: a player
        # cannot spend more than all of his variance on it. Two backs or two
        # quarterbacks are near that bound on purpose - only one of each plays
        # a meaningful snap count, so their outcomes really are close to
        # mutually exclusive, and a model that left them positively correlated
        # would happily roster both.
        "compete": {"QB": 0.98, "WR": 0.86, "TE": 0.94, "RB": 0.92,
                    "K": 0.00, "DST": 0.00},
    },
    # Placeholders, deliberately conservative, so a sport that has not been
    # studied cannot silently inherit football's structure. Each needs its own
    # study before it is trusted; until then the factors are weak and the
    # simulation behaves close to independent, which understates stacking
    # rather than inventing it.
    "nba": {
        "game":    {"PG": 0.22, "SG": 0.22, "SF": 0.22, "PF": 0.22, "C": 0.22},
        "team":    {"PG": 0.30, "SG": 0.30, "SF": 0.30, "PF": 0.30, "C": 0.30},
        "compete": {"PG": 0.45, "SG": 0.45, "SF": 0.45, "PF": 0.45, "C": 0.45},
    },
}

DEFAULT = {"game": 0.20, "team": 0.28, "compete": 0.40}

# Every player keeps at least this much of his variance to himself.
#
# Not a safety margin - a modelling claim. A player whose outcome is entirely
# explained by his game, his team and his team-mates has no individual
# variation left: no dropped pass, no broken tackle, no goal-line vulture. It
# is also what keeps the matrix invertible. Capping competition at exactly the
# available room left idiosyncratic variance at zero, which made a classic
# slate's correlation matrix singular - eigenvalue -0.00000 and a Cholesky
# that failed, from a model whose whole promise is that it cannot produce one.
MIN_IDIOSYNCRATIC = 0.06


def _loading(table: dict, kind: str, pos: str) -> float:
    return float(table.get(kind, {}).get(pos, DEFAULT[kind]))


def build(players: pd.DataFrame, sport: str = "nfl") -> tuple[np.ndarray, dict]:
    """A correlation matrix, and the loadings that produced it.

    `players` needs `position`, `team` and `game`. Returns a matrix that is
    positive semi-definite by construction - it is a sum of PSD pieces - so
    nothing downstream has to repair it, and nothing gets silently shrunk.
    """
    table = LOADINGS.get(sport, {})
    pos = players["position"].astype(str).to_numpy()
    team = players["team"].astype(str).to_numpy()
    game = (players["game"].astype(str).to_numpy()
            if "game" in players.columns
            else np.array([""] * len(players)))
    n = len(players)

    g = np.array([_loading(table, "game", p) for p in pos])
    t = np.array([_loading(table, "team", p) for p in pos])
    c = np.array([_loading(table, "compete", p) for p in pos])

    # Group sizes are needed BEFORE the competition loading can be trusted,
    # because the variance competition consumes grows with the size of the
    # group: c^2 (1 - 1/k). The loadings were tuned on a showdown board where
    # a position group is two to four players, and on a classic slate with
    # twelve receivers on a team the same numbers pushed shared variance past
    # 1.0 - a player explaining more than all of his own variation, which is
    # not a distribution. The first version noticed and rescaled the finished
    # covariance, which destroyed the positive semi-definiteness that is the
    # entire reason for building it this way, and the Cholesky failed.
    #
    # So the cap is applied to the INPUT instead, per player, from the room
    # actually left: c is at most sqrt((1 - g^2 - t^2) / (1 - 1/k)). Nothing
    # downstream has to repair anything, because nothing invalid is ever built.
    counts: dict[tuple[str, str], int] = {}
    for i in range(n):
        key = (team[i], pos[i])
        counts[key] = counts.get(key, 0) + 1
    sizes = np.array([counts[(team[i], pos[i])] for i in range(n)], dtype=float)

    room = np.clip(1.0 - g ** 2 - t ** 2 - MIN_IDIOSYNCRATIC, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ceiling = np.where(sizes > 1,
                           np.sqrt(room / np.maximum(1.0 - 1.0 / sizes, 1e-9)),
                           np.inf)
    capped = np.minimum(c, ceiling)
    if (capped < c - 1e-9).any():
        hit = sorted({pos[i] for i in range(n) if capped[i] < c[i] - 1e-9})
        log.info("competition loading capped for %s - a group of that size "
                 "cannot support the full value without exceeding unit "
                 "variance", ", ".join(hit))
    c = capped

    same_game = (game[:, None] == game[None, :]) & (game[:, None] != "")
    same_team = team[:, None] == team[None, :]

    cov = np.zeros((n, n))
    # Two rank-one pieces, each PSD, masked to the players that share the shock.
    cov += np.outer(g, g) * same_game
    cov += np.outer(t, t) * same_team

    # Competition: within each team-and-position group, the centred projection
    # c^2 (I - J/k). PSD for the same reason a projection is, and the only
    # source of negative correlation in the model.
    groups: dict[tuple[str, str], list[int]] = {}
    for i in range(n):
        groups.setdefault((team[i], pos[i]), []).append(i)
    group_size = np.ones(n)
    for (_, _), idx in groups.items():
        k = len(idx)
        group_size[idx] = k
        if k < 2:
            continue
        block = np.array(idx)
        cc = c[block]
        # off-diagonal -c_i c_j / k, diagonal c_i^2 (1 - 1/k)
        sub = -np.outer(cc, cc) / k
        np.fill_diagonal(sub, cc ** 2 * (1.0 - 1.0 / k))
        cov[np.ix_(block, block)] += sub

    # Idiosyncratic variance, chosen so every player has unit total variance -
    # which makes the covariance a correlation matrix directly.
    shared = g ** 2 + t ** 2 + np.where(group_size > 1,
                                        c ** 2 * (1.0 - 1.0 / group_size), 0.0)
    idio = 1.0 - shared
    assert (idio >= -1e-9).all(), "competition cap failed to bound the variance"
    cov[np.diag_indices(n)] = 1.0

    info = {"sport": sport, "loadings": table,
            "min_eigenvalue": float(np.linalg.eigvalsh(cov).min())}
    return cov, info


def implied_pairs(players: pd.DataFrame, corr: np.ndarray) -> pd.DataFrame:
    """What the factor structure actually says about each kind of pair.

    The point of reporting this is that the loadings are the inputs now, and
    nobody thinks in loadings. This translates them back into the language the
    old priors were written in, so the assumptions stay arguable.
    """
    pos = players["position"].astype(str).to_numpy()
    team = players["team"].astype(str).to_numpy()
    game = (players["game"].astype(str).to_numpy()
            if "game" in players.columns
            else np.array([""] * len(players)))

    buckets: dict[tuple, list[float]] = {}
    n = len(players)
    for i in range(n):
        for j in range(i + 1, n):
            if team[i] == team[j]:
                rel = "same_team"
            elif game[i] and game[i] == game[j]:
                rel = "opponent"
            else:
                continue
            key = (tuple(sorted((pos[i], pos[j]))), rel)
            buckets.setdefault(key, []).append(float(corr[i, j]))

    rows = [{"pair": f"{a}-{b}", "relationship": rel,
             "correlation": round(float(np.mean(v)), 3), "pairs": len(v)}
            for (a, b), rel in [(k[0], k[1]) for k in buckets]
            for v in [buckets[((a, b), rel)]]]
    return (pd.DataFrame(rows)
            .sort_values("correlation", ascending=False)
            .reset_index(drop=True))
