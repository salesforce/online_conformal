#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from online_conformal.base import BasePredictor
from online_conformal.enbpi import EnbMixIn
from online_conformal.utils import pinball_loss, pinball_loss_grad, Residuals


class FACI(BasePredictor):
    """
    Fully Adaptive Conformal Inference, which is the algorithm proposed by Gibbs & Candes, 2022,
    "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts." https://arxiv.org/abs/2208.08401
    """

    def __init__(self, *args, horizon=1, coverage=0.9, **kwargs):
        self.gammas = np.asarray([0.001 * 2**k for k in range(8)])
        self.alphas = np.full((horizon, self.k), 1 - coverage)
        self.log_w = np.zeros((horizon, self.k))
        super().__init__(*args, horizon=horizon, coverage=coverage, **kwargs)

    @property
    def I(self):
        return 100

    @property
    def k(self):
        return len(self.gammas)

    @property
    def sigma(self):
        return 1 / (2 * self.I)

    @property
    def eta(self):
        alpha = 1 - self.coverage
        denom = ((1 - alpha) ** 2 * alpha**3 + alpha**2 * (1 - alpha) ** 3) / 3
        return np.sqrt(3 / self.I) * np.sqrt((np.log(self.I * self.k) + 2) / denom)

    def predict(self, horizon) -> Tuple[float, float]:
        log_w = self.log_w[horizon - 1]
        alpha = np.dot(np.exp(log_w - logsumexp(log_w)), self.alphas[horizon - 1])
        delta = self.quantile(np.abs(self.residuals.get(horizon)), 1 - alpha)
        return -delta, delta

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon: int):
        h = horizon - 1
        residuals = self.residuals.horizon2residuals[horizon]
        for s in np.abs(forecast - ground_truth).values:
            if len(residuals) > math.floor(1 / (1 - self.coverage)):
                # Compute pinball losses incurred by the current residual
                beta = np.mean(residuals >= s)
                losses = pinball_loss(beta, self.alphas[h], 1 - self.coverage)

                # Update weights
                wbar = self.log_w[h] - self.eta * losses
                self.log_w[h] = logsumexp(
                    [wbar, np.full(self.k, logsumexp(wbar))], b=[[1 - self.sigma], [self.sigma / self.k]], axis=0
                )
                self.log_w[h] = self.log_w[h] - logsumexp(self.log_w[h])

                # Compute coverage errors & update alphas
                err = self.alphas[h] > beta
                self.alphas[h] = np.clip(self.alphas[h] + self.gammas * ((1 - self.coverage) - err), 0, 1)
            residuals.append(s)


class FACI_S(BasePredictor):
    """FACI on radius, instead of quantiles."""

    def __init__(self, *args, horizon=1, coverage=0.9, max_scale=None, **kwargs):
        self.gammas = np.asarray([0.001 * 2**k for k in range(8)])
        self.s_hats = np.zeros((horizon, self.k))
        self.log_w = np.zeros((horizon, self.k))
        if max_scale is None:
            self.scale = {}
        else:
            self.scale = {h + 1: float(max_scale) for h in range(horizon)}
        self.prev_loss_sq = {h + 1: [] for h in range(horizon)}
        super().__init__(*args, horizon=horizon, coverage=coverage, **kwargs)

        # Use calibration to initialize learning rate & estimates for deltas
        residuals = self.residuals
        self.residuals = Residuals(self.horizon)
        for h in range(1, self.horizon + 1):
            r = residuals.horizon2residuals[h]
            self.scale[h] = 1 if len(r) == 0 else np.max(np.abs(r)) * np.sqrt(3)
            self.update(pd.Series(r, dtype=float), pd.Series(np.zeros(len(r))), h)

    @property
    def I(self):
        return 100

    @property
    def k(self):
        return len(self.gammas)

    @property
    def sigma(self):
        return 1 / (2 * self.I)

    def eta(self, horizon):
        loss_sq = self.prev_loss_sq[horizon][-self.I :]
        if len(loss_sq) == 0:
            loss_sq_sum = self.I * (self.scale[horizon] * self.coverage) ** 2
        else:
            loss_sq_sum = np.sum(loss_sq) * (self.I / len(loss_sq))
        return np.sqrt((np.log(self.k * self.I) + 2) / loss_sq_sum)

    def predict(self, horizon) -> Tuple[float, float]:
        log_w = self.log_w[horizon - 1]
        s_hat = np.dot(np.exp(log_w - logsumexp(log_w)), self.s_hats[horizon - 1])
        return -s_hat, s_hat

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon: int):
        residuals = np.abs(ground_truth - forecast).values
        self.residuals.extend(horizon, residuals.tolist())
        if horizon not in self.scale:
            return
        h = horizon - 1
        for s in np.abs(forecast - ground_truth).values:
            if horizon in self.scale:
                # Compute loss
                w = np.exp(self.log_w[h] - logsumexp(self.log_w[h]))
                losses = pinball_loss(s, self.s_hats[h], self.coverage)

                # Update weights
                wbar = self.log_w[h] - self.eta(horizon) * losses
                self.log_w[h] = logsumexp(
                    [wbar, np.full(self.k, logsumexp(wbar))], b=[[1 - self.sigma], [self.sigma / self.k]], axis=0
                )
                self.log_w[h] = self.log_w[h] - logsumexp(self.log_w[h])

                # Add previous expected loss squared to the list
                self.prev_loss_sq[horizon].append(np.dot(w, losses**2))

                # Update s_hat's
                grad = pinball_loss_grad(s, self.s_hats[h], self.coverage)
                self.s_hats[h] = self.s_hats[h] - self.gammas * self.scale[horizon] * grad


class EnbFACI(EnbMixIn, FACI):
    pass
