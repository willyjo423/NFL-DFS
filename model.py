"""Each player as a distribution, not a number.

Why quantiles rather than a mean
--------------------------------
DraftKings pays three points at exactly 100 receiving yards. That is a step
function, so two players with identical averages are worth different amounts -
the wider one crosses the line more often. And a tournament pays almost nothing
for the middle of a distribution; it pays for the top of it. A mean projection
cannot express either fact.

So the model fits the 10th, 25th, 50th, 75th, 90th and 97th percentiles
directly with pinball loss. The 97th is in there deliberately: it is the part
of a player that wins a tournament, and it is the part a mean-squared-error fit
smooths away.

One model per quantile, not per position
----------------------------------------
Position enters as a feature instead. Splitting by position sounds tidier but
quarters the data behind each fit, and the thing being learned - that volume
predicts points, that a role change matters more than a season average - is
shared across positions. The model can still separate them; it just is not
forced to relearn the same relationship four times from a quarter of the
evidence each time.

Quantile crossing
-----------------
Six independently fitted models can produce a 75th below a 25th on an odd row.
Each row is sorted afterwards, so a projection can never read backwards.

Two questions, not one
----------------------
The first version of this fitted the quantiles on every row, including the
nineteen percent where a player scored exactly zero because he did not play.
That makes one model answer two unrelated questions at once - WILL he play,
and HOW WELL - and report the blend as a single distribution. Grading it
proved the damage: among players who actually took the field, every fitted
quantile undershot. The tenth percentile covered 3.1% of outcomes instead of
10%, the median covered 42% instead of 50%. The whole distribution had been
dragged down by games nobody played in.

So the two questions are now asked separately. The quantiles are fitted ONLY
on games a player played, which makes them an honest answer to "what does he
do when he plays". A separate classifier estimates the probability he plays at
all. The simulator combines them - a Bernoulli draw for availability, then the
conditional curve - which is where a mixture belongs, because that is the only
place the two can be recombined without losing the shape of either.

The side benefit is the one that matters most in practice: availability stops
being smeared invisibly through every percentile and becomes a number you can
look at, argue with, and override when the injury report says something the
model cannot see.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import config
import features as F

log = logging.getLogger(__name__)

POSITIONS = ["QB", "RB", "WR", "TE"]


class Projections:
    """Fitted quantile models, and the prediction they produce."""

    def __init__(self, quantiles: list[float] | None = None):
        self.quantiles = quantiles or config.QUANTILES
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.availability: HistGradientBoostingClassifier | None = None
        self.base_rate = 1.0
        self.columns: list[str] = []
        self.trained_rows = 0
        self.played_rows = 0

    # ------------------------------------------------------------------ fit
    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[[c for c in F.FEATURES if c in df.columns]].copy()
        for pos in POSITIONS:
            X[f"is_{pos}"] = (df["position"] == pos).astype(float)
        return X

    @staticmethod
    def _usable(X: pd.DataFrame) -> list[str]:
        """Columns with something in them to learn from.

        A feature that is entirely missing, or that holds a single repeated
        value, carries no information - and the histogram binner cannot even
        build a threshold from one distinct value, so it raises rather than
        ignoring it. That is what killed the first live fit: nflverse's weekly
        player file has no home/away flag, so `is_home` arrived as a column of
        NaN and took the whole run down twenty minutes before kickoff.

        Dropping them here rather than pruning the feature list keeps the
        build tolerant of a source that adds or removes a column, which is a
        thing these sources demonstrably do.
        """
        keep = []
        for c in X.columns:
            col = X[c]
            if col.notna().sum() < 2:
                continue
            if col.nunique(dropna=True) < 2:
                continue
            keep.append(c)
        return keep

    def fit(self, df: pd.DataFrame) -> "Projections":
        train = F.trainable(df)
        if len(train) < 500:
            raise ValueError(
                f"only {len(train)} usable rows; a quantile fit on that little "
                f"evidence is noise wearing a model's clothes")
        X = self._design(train)
        dropped = [c for c in X.columns if c not in self._usable(X)]
        if dropped:
            log.warning("dropping %d empty or constant features: %s",
                        len(dropped), ", ".join(dropped))
        X = X[self._usable(X)]
        if X.empty or not len(X.columns):
            raise ValueError("no usable features survived")
        self.columns = list(X.columns)
        self.trained_rows = len(train)

        # --- part one: will he play at all -----------------------------------
        # Fitted on EVERY trainable row, because the question is precisely
        # about the rows where nothing happened. A zero here means "did not
        # take the field", which in this data is the same thing as scoring
        # nothing at all.
        played = (train["points"].to_numpy(dtype=float) > 0).astype(int)
        self.base_rate = float(played.mean())
        if played.min() == played.max():
            # One class only. A classifier cannot be fitted and does not need
            # to be; saying so beats a silent constant nobody can see.
            self.availability = None
            log.warning("every training row has the same play/no-play "
                        "outcome (%d%%); availability will be a constant",
                        int(self.base_rate * 100))
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_depth=6,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=config.RANDOM_SEED)
            clf.fit(X, played)
            self.availability = clf

        # --- part two: how well, GIVEN he plays ------------------------------
        # Only the games he played in. Fitting these on the zeros too is what
        # dragged every percentile downward: among players who took the field,
        # the tenth percentile covered 3.1% of outcomes instead of 10%.
        active = played.astype(bool)
        self.played_rows = int(active.sum())
        if self.played_rows < 400:
            raise ValueError(
                f"only {self.played_rows} games were actually played in the "
                f"training set; a conditional quantile fit on that little "
                f"evidence is noise wearing a model's clothes")
        Xa = X[active]
        ya = train["points"].to_numpy(dtype=float)[active]

        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q,
                max_iter=300, learning_rate=0.06, max_depth=6,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=config.RANDOM_SEED)
            m.fit(Xa, ya)
            self.models[q] = m
        log.info("fitted %d conditional quantiles on %d played games "
                 "(of %d rows; %.1f%% played)", len(self.models),
                 self.played_rows, self.trained_rows, self.base_rate * 100)
        return self

    def play_probability(self, X: pd.DataFrame) -> np.ndarray:
        """How likely each player is to take the field, as a number."""
        if self.availability is None:
            return np.full(len(X), self.base_rate, dtype=float)
        return self.availability.predict_proba(X)[:, 1]

    # -------------------------------------------------------------- predict
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise ValueError("not fitted")
        X = self._design(df)
        for c in self.columns:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.columns]

        out = pd.DataFrame(index=df.index)
        for q in self.quantiles:
            out[f"q{int(q * 100)}"] = self.models[q].predict(X)

        # Sort each row, so a 75th can never sit below a 25th.
        cols = [f"q{int(q * 100)}" for q in self.quantiles]
        vals = np.sort(out[cols].to_numpy(), axis=1)
        out[cols] = vals
        out[cols] = out[cols].clip(lower=0.0)   # nobody scores negative often

        # Everything from here is CONDITIONAL on the player taking the field,
        # because that is what the quantiles were fitted on.
        out["p_play"] = np.clip(self.play_probability(X), 0.0, 1.0)
        out["median"] = out["q50"]
        # The mean of a skewed distribution is not its median. Approximated
        # from the quantiles rather than fitted separately, because what the
        # optimiser needs is the shape and this keeps the two consistent.
        out["cond_mean"] = (0.1 * out["q10"] + 0.2 * out["q25"]
                            + 0.4 * out["q50"] + 0.2 * out["q75"]
                            + 0.1 * out["q90"])
        # And this is the UNCONDITIONAL expectation - what he is worth before
        # you know whether he suits up. It is the number a cash lineup should
        # be built on, and the number the field prices him at. Keeping the two
        # apart under different names is the whole point of the split: a
        # doubtful star has a huge `cond_mean` and a modest `mean`, and
        # collapsing them back into one column would rebuild the exact bug
        # this model was restructured to remove.
        out["mean"] = out["cond_mean"] * out["p_play"]
        out["ceiling"] = out[f"q{int(self.quantiles[-1] * 100)}"]
        out["spread"] = out["q90"] - out["q10"]
        return out

    # ------------------------------------------------------------- sampling
    def sample(self, projected: pd.DataFrame, n: int,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw outcomes from each player's own fitted distribution.

        Inverse-CDF sampling through the quantiles: pick a uniform, find where
        it falls between two fitted percentiles, interpolate. This is what lets
        the simulator work with the SHAPE the model produced rather than
        flattening every player back to a mean and a standard deviation, which
        would throw away the reason for fitting quantiles in the first place.
        """
        rng = rng or np.random.default_rng(config.RANDOM_SEED)
        qs = np.array(self.quantiles, dtype=float)
        cols = [f"q{int(q * 100)}" for q in self.quantiles]
        grid = projected[cols].to_numpy(dtype=float)

        u = rng.random((len(projected), n))
        out = np.empty_like(u)
        for i in range(len(projected)):
            out[i] = np.interp(u[i], qs, grid[i])

        # The availability gate. The curve above says what he scores when he
        # plays; this says whether he played. Multiplying is what turns two
        # honest halves back into the mixture a lineup actually faces, and
        # doing it HERE rather than inside the fitted quantiles is what keeps
        # each half interpretable on its own.
        if "p_play" in projected.columns:
            p = projected["p_play"].to_numpy(dtype=float)[:, None]
            out = out * (rng.random(out.shape) < p)
        return out


def latest_rows(built: pd.DataFrame) -> pd.DataFrame:
    """The most recent feature row per player - what a projection is made from.

    Deliberately the last row available rather than a row for the upcoming
    week: the features describe what a player has done, and the upcoming week
    has not happened. Using the last completed week is the only honest way to
    project the next one.
    """
    return (built.sort_values(["player_id", "season", "week"])
                 .groupby("player_id", as_index=False).tail(1)
                 .reset_index(drop=True))
