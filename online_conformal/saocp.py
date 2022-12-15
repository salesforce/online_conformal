#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from online_conformal.base import BasePredictor
from online_conformal.enbpi import EnbMixIn
from online_conformal.utils import pinball_loss, pinball_loss_grad, Residuals


class _OGD:
    def __init__(self, t, scale, alpha, yhat_0, g=8):
        """
        Instantiates an online gradient descent learner which starts at time t and has a finite lifetime. The lifetime
        is given by the data streaming intervals described in Hazan & Seshadhri, 2007, "Adaptive Algorithms for Online
        Decision Problems." (Appendix B) https://www.cs.princeton.edu/techreports/2007/798.pdf. The underlying algorithm
        is Scale-Free Online Mirror Descent with regularizer ||y - y_0||^2 / 2 (https://arxiv.org/abs/1601.01974).
        """
        # Scale-free online gradient descent parameters
        self.scale = scale
        self.base_lr = scale / np.sqrt(3)
        self.alpha = alpha
        self.yhat = yhat_0
        self.grad_norm = 0

        # Meta-algorithm parameters
        u = 0
        while t % 2 == 0:
            t /= 2
            u += 1
        self.lifetime = g * 2**u
        self.z = 0  # sum of differences between losses & meta-losses
        self.wz = 0  # weighted sum of differences between losses & meta-losses
        self.s_t = 0  # how long the predictor has been alive

    @property
    def expired(self):
        return self.s_t > self.lifetime

    def loss(self, y):
        return pinball_loss(y, self.yhat, 1 - self.alpha)

    @property
    def w(self):
        return 0 if self.s_t == 0 else self.z / self.s_t * (1 + self.wz)

    def update(self, y, meta_loss):
        # Update meta-algorithm weights
        w = self.w
        g = np.clip((meta_loss - self.loss(y)) / self.scale / max(self.alpha, 1 - self.alpha), -1 * (w > 0), 1)
        self.z += g
        self.wz += g * w
        self.s_t += 1

        # Update estimator
        grad = pinball_loss_grad(y, self.yhat, 1 - self.alpha)
        self.grad_norm += grad**2
        if self.grad_norm != 0:
            self.yhat = max(0, self.yhat - self.base_lr / np.sqrt(self.grad_norm) * grad)


class SAOCP(BasePredictor):
    """
    Strongly Adaptive Online Conformal Prediction (SAOCP). The main algorithm of the paper. This algorithm adapts
    Coin Betting for Changing Environment (CBCE) to learn conformal confidence intervals.
    From Jun et al., 2017, "Improved Strongly Adaptive Online Learning using Coin Betting".
    https://proceedings.mlr.press/v54/jun17a/jun17a-supp.pdf.
    """

    def __init__(self, *args, horizon=1, max_scale=None, lifetime=8, **kwargs):
        self.t = 1
        if max_scale is None:
            self.scale = {}
        else:
            self.scale = {h + 1: max_scale for h in range(horizon)}
        self.experts = {h + 1: {} for h in range(horizon)}
        self.lifetime = lifetime
        super().__init__(*args, horizon=horizon, **kwargs)

        residuals = self.residuals
        self.residuals = Residuals(self.horizon)
        for h in range(1, self.horizon + 1):
            r = residuals.horizon2residuals[h]
            if h not in self.scale:
                self.scale[h] = 1 if len(r) == 0 else np.max(np.abs(r)) * np.sqrt(3)
            self.update(pd.Series(r, dtype=float), pd.Series(np.zeros(len(r))), h)

    def get_p(self, horizon) -> Dict[int, float]:
        experts = self.experts[horizon]
        prior = {t: 1 / (t**2 * (1 + np.floor(np.log2(t)))) for t in experts.keys()}
        z = sum(prior.values())
        prior = {t: v / z for t, v in prior.items()}
        p = {t: prior[t] * max(0, expert.w) for t, expert in experts.items()}
        sum_p = sum(p.values())
        return {t: v / sum_p for t, v in p.items()} if sum_p > 0 else prior

    def predict(self, horizon) -> Tuple[float, float]:
        p = self.get_p(horizon)
        delta = sum(p[t] * expert.yhat for t, expert in self.experts[horizon].items())
        return -delta, delta

    def create_expert(self, horizon, s_hat):
        return _OGD(self.t, self.scale[horizon], 1 - self.coverage, g=self.lifetime, yhat_0=s_hat)

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon: int):
        residuals = np.abs(ground_truth - forecast).values
        self.residuals.extend(horizon, residuals.tolist())
        if horizon not in self.scale:
            return

        experts = self.experts[horizon]
        for s in residuals:
            # Remove expired experts & add new expert
            _, s_hat = self.predict(horizon)
            [experts.pop(t) for t in [k for k, v in experts.items() if v.expired]]
            experts[self.t] = self.create_expert(horizon, s_hat)

            # Update experts
            meta_loss = pinball_loss(s, self.predict(horizon)[1], self.coverage)
            [expert.update(s, meta_loss) for expert in experts.values()]
            self.t += 1


class EnbSAOCP(EnbMixIn, SAOCP):
    pass
