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
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config
import features as F

log = logging.getLogger(__name__)

POSITIONS = ["QB", "RB", "WR", "TE"]


class Projections:
    """Fitted quantile models, and the prediction they produce."""

    def __init__(self, quantiles: list[float] | None = None):
        self.quantiles = quantiles or config.QUANTILES
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.columns: list[str] = []
        self.trained_rows = 0

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
        y = train["points"].to_numpy(dtype=float)
        self.columns = list(X.columns)
        self.trained_rows = len(train)

        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q,
                max_iter=300, learning_rate=0.06, max_depth=6,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=config.RANDOM_SEED)
            m.fit(X, y)
            self.models[q] = m
        log.info("fitted %d quantiles on %d player-weeks",
                 len(self.models), self.trained_rows)
        return self

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

        out["median"] = out["q50"]
        # The mean of a skewed distribution is not its median. Approximated
        # from the quantiles rather than fitted separately, because what the
        # optimiser needs is the shape and this keeps the two consistent.
        out["mean"] = (0.1 * out["q10"] + 0.2 * out["q25"] + 0.4 * out["q50"]
                       + 0.2 * out["q75"] + 0.1 * out["q90"])
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
