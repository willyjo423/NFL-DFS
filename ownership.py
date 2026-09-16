"""How much of the field will own each player, when nobody will tell you.

The measurement is closed
------------------------
DraftKings' standings endpoint returns a login page - identically 139,263 bytes
for a 182,000-entry contest and a 52,000-entry one, which is how you know it is
one generic wall rather than data. So ownership cannot be counted. It has to be
modelled, and everything built on it is an estimate of an estimate.

What stops this being a free guess
----------------------------------
Ownership is not an arbitrary set of percentages. It has to add up. Every
entry in a classic contest fields exactly one quarterback, so across all the
quarterbacks on a slate, ownership sums to 100% - not approximately, exactly.
Across running backs it sums to 200% plus whatever share of the flex they take.
That constraint is arithmetic, not opinion, and it pins the SCALE of the whole
distribution.

What is left to guess is only the SHAPE: how sharply the field concentrates on
its favourites. That is one parameter, it is stated in the open below, and it
is the only thing here that a real ownership file would change much.

How the shape is chosen
-----------------------
Projected points first, then value, then a bump at each end of the price range:
the field over-owns the cheapest usable players (a punt frees up salary) and
the most expensive (the obvious stud), so appeal is not monotone in salary.
Each effect is one coefficient rather than a fitted curve, because there is
nothing to fit against.

Ordering those two matters more than their weights. Leading with value gets a
board catastrophically wrong, and the failure is instructive: points per dollar
is degenerate at the bottom of a slate. On the real DEN @ KC board a $200
running back projected for 2.3 points scored 11.4 per $1,000 against Patrick
Mahomes' 1.82, so a z-score of value stood five minimum-salary special-teamers
five standard deviations clear of everyone and predicted the field would roster
long snappers at 65%. Value is real, but only among players somebody would
consider; it cannot rescue a player who is not projected to score.

Why it matters even as an estimate
----------------------------------
A tournament is not won by a good lineup; it is won by a good lineup nobody
else had. Two entries with identical projections are worth very different
amounts if one is 40% owned and the other 4%, because the first splits its
prize forty thousand ways. Even a roughly-right ownership estimate separates
those two cases, and having none at all cannot.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# How sharply the field concentrates. Higher means the chalk is chalkier.
# This is the one genuinely free parameter in the module and the first thing to
# calibrate if a real ownership file ever arrives. Set by eye against a real
# board - last night's DEN @ KC showdown - so that the top play lands near 55%,
# a strong value play near 45%, and the median player in single figures, which
# is roughly the shape a showdown actually prints.
CONCENTRATION = 0.75

# The field's preferences, as four coefficients.
#
# POINTS leads, and that ordering is the correction that matters. The first
# version led with points-per-dollar, which is DEGENERATE at the bottom of a
# board: a long snapper projected for 1.7 points at $200 scores 11.4 per
# $1,000 against Patrick Mahomes' 1.82, so a z-score of value put five
# minimum-salary special-teamers five standard deviations clear of the field
# and predicted they would each be 65% owned. Nobody has ever rostered a long
# snapper. Value is a real driver, but only among players somebody would
# actually consider - it cannot rescue a player who is not projected to score.
POINTS_WEIGHT = 1.00     # absolute projected points - what gets a player noticed
VALUE_WEIGHT = 0.80      # then points per $1,000, among players worth a look
CEILING_WEIGHT = 0.30    # tournament players chase upside, not medians
PUNT_BONUS = 0.35        # the cheapest USABLE players get played for salary relief
STUD_BONUS = 0.25        # and the dearest get played because they are obvious

# Value is winsorised before it is scored. Without a ceiling, the cheapest man
# on the board defines the scale: on the real MNF board the value column ran
# from 1.8 (Mahomes) to 11.4 (a $200 running back), so every player anybody
# would actually roster sat in the bottom sixth of the range. Capping at the
# 60th percentile puts the minimum-salary tail level with the best real value
# play instead of six times clear of it.
VALUE_CAP_PCT = 0.60


def slot_demand(roster: dict) -> dict[str, float]:
    """How many of each position a single entry must field.

    This is where the scale comes from. A classic lineup holds exactly one
    quarterback, so quarterback ownership across the slate sums to exactly 1.0
    - and any model that produces something else is wrong in a way that can be
    checked rather than argued about.
    """
    slots = roster["slots"]
    flex_positions = list(roster.get("flex_positions") or [])
    demand: dict[str, float] = {}
    for s in slots:
        if s in ("FLEX", "CPT", "UTIL"):
            continue
        demand[s] = demand.get(s, 0.0) + 1.0

    spare = sum(1 for s in slots if s in ("FLEX", "UTIL"))
    if spare and flex_positions:
        # A flex is shared. Split it by how much of the flex-eligible pool each
        # position represents, which is closer to how a field actually fills it
        # than splitting evenly would be.
        base = {p: demand.get(p, 0.0) for p in flex_positions}
        total = sum(base.values()) or float(len(flex_positions))
        for p in flex_positions:
            share = (base.get(p, 0.0) / total) if total else 1.0 / len(flex_positions)
            demand[p] = demand.get(p, 0.0) + spare * share
    return demand


def appeal(pool: pd.DataFrame) -> pd.Series:
    """How attractive each player looks to somebody building a lineup fast.

    Deliberately built from what a person actually looks at on the screen -
    value, ceiling, and price extremes - rather than from the projection the
    model believes. The field is not running this model, and an ownership
    estimate that assumes it is will predict that the field plays exactly what
    we like, which is the one thing that would destroy the leverage number.
    """
    salary = pd.to_numeric(pool["salary"], errors="coerce")
    per_k = salary / 1000.0
    # The AVAILABILITY-ADJUSTED projection, not the conditional one. The field
    # does not roster a player who is not playing, however good he is when he
    # does - so ownership has to be driven by the unconditional expectation.
    # Reading the conditional median here would have made a doubtful star look
    # like the chalk of the slate.
    def column(name: str) -> pd.Series:
        """A numeric Series aligned to the pool, whether or not the column
        exists. `pool.get(name)` returns None for a missing column and
        `pd.to_numeric(None)` returns a bare scalar, not a Series - which is
        how the first version of this turned a missing column into an
        AttributeError two calls later instead of an empty column here."""
        if name not in pool.columns:
            return pd.Series(np.nan, index=pool.index, dtype=float)
        return pd.to_numeric(pool[name], errors="coerce")

    # The AVAILABILITY-ADJUSTED projection, not the conditional one. The field
    # does not roster a player who is not playing, however good he is when he
    # does - so ownership has to be driven by the unconditional expectation.
    # Reading the conditional median here would have made a doubtful star look
    # like the chalk of the slate.
    median = column("mean")
    if median.isna().all():
        median = column("median")
    ceiling = column("ceiling")
    if "p_play" in pool.columns:
        ceiling = ceiling * column("p_play").fillna(1.0)
    if ceiling.isna().all():
        ceiling = median

    value = (median / per_k.replace(0, np.nan)).fillna(0.0)
    ceil_value = (ceiling / per_k.replace(0, np.nan)).fillna(0.0)
    # Winsorise, so the cheapest man on the board cannot define the scale for
    # everyone else. Without this the value column spans 1.8 to 11.4 and every
    # real player is squeezed into the bottom tenth of it.
    for s in (value, ceil_value):
        cap = float(s.quantile(VALUE_CAP_PCT))
        if np.isfinite(cap) and cap > 0:
            s.clip(upper=cap, inplace=True)

    def z(s: pd.Series) -> pd.Series:
        sd = float(s.std(ddof=0))
        return (s - float(s.mean())) / sd if sd > 1e-9 else s * 0.0

    score = (POINTS_WEIGHT * z(median.fillna(0.0))
             + VALUE_WEIGHT * z(value)
             + CEILING_WEIGHT * z(ceil_value))

    # Price extremes, measured within the slate rather than in dollars, so the
    # same code works on a $200-$11,000 showdown and on a classic board.
    #
    # The punt bonus applies only to players with a REAL projection. A cheap
    # player is played to free up salary, but only if he might score - and
    # without this condition the bonus lands hardest on exactly the players who
    # will not play a snap, which is how the first version arrived at a board
    # led by long snappers.
    rank = salary.rank(pct=True)
    usable = median >= float(median.quantile(0.35))
    score = score + PUNT_BONUS * ((rank <= 0.15) & usable).astype(float)
    score = score + STUD_BONUS * (rank >= 0.94).astype(float)

    # A player nobody can use is owned by nobody, whatever his price implies.
    if "playing" in pool.columns:
        score = score.mask(pool["playing"].isin(("out", "doubtful")), -1e9)
    return score


# No single player is ever owned by two-thirds of a field. Even the most
# obvious play on the most obvious slate tops out somewhere near here, and a
# model that puts a player at 100% is saying every entry in the contest made
# the same choice, which has never happened.
MAX_OWNERSHIP = 0.65


def _allocate(scores: np.ndarray, want: float,
              concentration: float) -> np.ndarray:
    """Shape from the softmax, scale from the roster, cap applied honestly.

    The cap is the awkward part. Clipping a player at 65% removes ownership
    that the roster constraint says must exist somewhere - every entry still
    fields that slot - so the excess is pushed onto the players who are not at
    the cap, repeatedly until it settles. An earlier version redistributed
    across the WHOLE board instead of within the constrained group, which broke
    the very sums that make this model more than a guess: quarterback ownership
    came out at 128% when exactly one quarterback is in every lineup.
    """
    n = len(scores)
    if n == 0 or want <= 0:
        return np.zeros(n)
    out = _softmax(scores, concentration) * want
    cap = min(MAX_OWNERSHIP, want / n if want < n * 1e-9 else MAX_OWNERSHIP)

    for _ in range(50):
        over = out > cap
        if not over.any():
            break
        excess = float((out[over] - cap).sum())
        out[over] = cap
        room = ~over
        if not room.any() or out[room].sum() <= 0:
            # Everyone is at the cap; the demand cannot be met without
            # exceeding it, which means the slate is smaller than the roster.
            break
        out[room] += excess * (out[room] / out[room].sum())
    return out


def project(pool: pd.DataFrame, roster: dict,
            concentration: float = CONCENTRATION) -> pd.Series:
    """Projected ownership per player, as a fraction of the field.

    Softmax of appeal within each position, scaled so the position sums to
    what a lineup actually demands. The softmax supplies the shape; the roster
    supplies the scale; only the temperature is a guess.
    """
    score = appeal(pool)
    out = pd.Series(0.0, index=pool.index, dtype=float)

    if "CPT" in roster["slots"]:
        # Showdown has no position requirements - six slots drawn from the
        # whole board - so the board is one pool and the demand is six.
        out.loc[:] = _allocate(score.to_numpy(dtype=float),
                               float(len(roster["slots"])), concentration)
    else:
        demand = slot_demand(roster)
        for position, idx in pool.groupby(
                pool["position"].astype(str)).groups.items():
            idx = list(idx)
            out.loc[idx] = _allocate(score.loc[idx].to_numpy(dtype=float),
                                     float(demand.get(position, 0.0)),
                                     concentration)

    log.info("ownership: %d players, total %.0f%% against %d roster slots",
             len(out), 100 * out.sum(), len(roster["slots"]))
    return out


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = np.where(np.isfinite(x), x, -1e9)
    z = x * temperature
    z = z - z.max()
    e = np.exp(z)
    total = e.sum()
    return e / total if total > 0 else np.full_like(e, 1.0 / len(e))


def duplication(own: np.ndarray, field_size: int) -> float:
    """Roughly how many other entries will field this exact lineup.

    The number that turns a good lineup into a good BET. A lineup whose six
    players are each owned by 30% is played by about 0.3^6 of the field - which
    in a 200,000-entry tournament is 146 other people splitting first place
    with you. The same score from six 8%-owned players is yours alone.

    Approximate on purpose: it ignores that roster rules make some combinations
    impossible, which overstates duplication slightly and uniformly. Since it
    is used to RANK lineups against each other, a uniform bias is harmless in a
    way that a wrong shape would not be.
    """
    p = np.clip(np.asarray(own, dtype=float), 1e-9, 1.0)
    return float(field_size * np.prod(p))


def leverage(pool: pd.DataFrame, own: pd.Series) -> pd.Series:
    """Where the model likes a player more than the field will.

    Expressed as the gap between two ranks rather than two raw numbers,
    because ownership and projected points are not in the same units and
    subtracting them directly would mean nothing.
    """
    median = pd.to_numeric(pool.get("median"), errors="coerce").fillna(0.0)
    return (median.rank(pct=True) - own.rank(pct=True)).round(3)
