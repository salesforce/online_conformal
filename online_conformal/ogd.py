#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Tuple

from online_conformal.base import BasePredictor
from online_conformal.enbpi import EnbMixIn
from online_conformal.utils import pinball_loss_grad, Residuals


class ScaleFreeOGD(BasePredictor):
    """
    Scale-free online gradient descent to learn conformal confidence intervals via online quantile regression. We
    perform online gradient descent on the pinball loss to learn the relevant quantiles of the residuals.
    From Orabona & Pal, 2016, "Scale-Free Online Learning." https://arxiv.org/abs/1601.01974.
    """

    def __init__(self, *args, horizon=1, max_scale=None, **kwargs):
        self.scale = {}
        self.delta = defaultdict(float)
        self.grad_norm = defaultdict(float)
        if max_scale is None:
            self.scale = {}
        else:
            self.scale = {h + 1: float(max_scale) for h in range(horizon)}
        super().__init__(*args, horizon=horizon, **kwargs)

        # Use calibration to initialize learning rate & estimates for deltas
        residuals = self.residuals
        self.residuals = Residuals(self.horizon)
        for h in range(1, self.horizon + 1):
            r = residuals.horizon2residuals[h]
            if h not in self.scale:
                self.scale[h] = 1 if len(r) == 0 else np.max(np.abs(r)) * np.sqrt(3)
            self.update(pd.Series(r, dtype=float), pd.Series(np.zeros(len(r))), h)

    def predict(self, horizon) -> Tuple[float, float]:
        return -self.delta[horizon], self.delta[horizon]

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon):
        residuals = np.abs(ground_truth - forecast).values
        self.residuals.extend(horizon, residuals.tolist())
        if horizon not in self.scale:
            return
        for s in residuals:
            delta = self.delta[horizon]
            grad = pinball_loss_grad(np.abs(s), delta, self.coverage)
            self.grad_norm[horizon] += grad**2
            if self.grad_norm[horizon] != 0:
                self.delta[horizon] = max(0, delta - self.scale[horizon] / np.sqrt(3 * self.grad_norm[horizon]) * grad)


class EnbOGD(EnbMixIn, ScaleFreeOGD):
    pass
