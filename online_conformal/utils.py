#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
import numpy as np
import pandas as pd


def quantile(arr, q, weights=None):
    q = np.clip(q, 0, 1)
    if len(arr) == 0:
        return np.zeros(len(q)) if hasattr(q, "__len__") else 0
    if weights is None:
        return np.quantile(arr, q, method="inverted_cdf")
    assert len(weights) == len(arr)
    idx = np.argsort(arr)
    weights = np.cumsum(weights[idx])
    q_idx = np.searchsorted(weights / weights[-1], q)
    return np.asarray(arr)[idx[q_idx]]


def pinball_loss(y, yhat, q: float):
    return np.maximum(q * (y - yhat), (1 - q) * (yhat - y))


def pinball_loss_grad(y, yhat, q: float):
    return -q * (y > yhat) + (1 - q) * (y < yhat)


def interval_miscoverage(y: pd.Series, yhat: pd.Series, lb: pd.Series, ub: pd.Series, window: int, cov: float):
    interval_cov = ((lb <= y) & (y <= ub)).rolling(window).mean().dropna()
    return np.abs(interval_cov - cov).values.max()


def interval_regret(y: pd.Series, yhat: pd.Series, lb: pd.Series, ub: pd.Series, window: int, cov: float):
    resid = np.abs(y - yhat)
    interval_losses = pinball_loss(resid, (ub - lb) / 2, cov).rolling(window).mean().dropna()
    opts = resid.rolling(window).quantile(cov).dropna()
    opt_losses = [pinball_loss(resid.values[i : i + window], opt, cov).mean() for i, opt in enumerate(opts)]
    return max(interval_losses.values - np.asarray(opt_losses)) if len(opt_losses) == len(interval_losses) else np.nan


def coverage(y: pd.Series, yhat: pd.Series, lb: pd.Series, ub: pd.Series):
    return ((lb <= y) & (y <= ub)).mean()


def mae(y: pd.Series, yhat: pd.Series, lb: pd.Series, ub: pd.Series):
    return np.abs(y - yhat).mean()


def err_std(y: pd.Series, yhat: pd.Series, lb: pd.Series, ub: pd.Series):
    return np.abs(y - yhat).std()


def width(y: pd.Series, yhat: pd.Series, lb: pd.Series, ub: pd.Series):
    return (ub - lb).median() / 2


class Residuals:
    def __init__(self, horizon):
        assert isinstance(horizon, int) and horizon > 0
        self.horizon = horizon
        self.horizon2residuals = {h + 1: [] for h in range(self.horizon)}

    def __len__(self):
        return max(len(r) for r in self.horizon2residuals.values())

    def get(self, horizon):
        assert isinstance(horizon, int) and 1 <= horizon <= self.horizon, f"Got {horizon}, self.horizon={self.horizon}"
        max_h = max(h for h, v in self.horizon2residuals.items() if 3 * len(v) >= len(self))
        return self.horizon2residuals[min(horizon, max_h)]

    def extend(self, horizon, vals):
        assert isinstance(horizon, int) and 1 <= horizon <= self.horizon, f"Got {horizon}, self.horizon={self.horizon}"
        self.horizon2residuals[horizon] += [v for v in (vals if isinstance(vals, list) else [vals]) if not np.isnan(v)]

    def remove_outliers(self):
        for h, resid in self.horizon2residuals.items():
            resid = np.asarray(resid)
            resid = resid[~np.isnan(resid)]
            z_score = np.abs(resid - np.mean(resid)) / np.std(resid)
            self.horizon2residuals[h] = resid[z_score < 5].tolist()
