#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
import pandas as pd

from online_conformal.base import BasePredictor


class SplitConformal(BasePredictor):
    """
    Split Conformal Prediction, adapted to time series.
    """

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon):
        self.residuals.extend(horizon, (ground_truth - forecast).values.tolist())
