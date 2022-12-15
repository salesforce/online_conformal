#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
import numpy as np

from online_conformal.enbpi import EnbMixIn
from online_conformal.split_conformal import SplitConformal
from online_conformal.utils import quantile


class NExConformal(SplitConformal):
    """
    Non-Exchangeable Split Conformal Prediction, one of the algorithms described in Barber et al., 2022,
    "Conformal Prediction Beyond Exchangeability." https://arxiv.org/abs/2202.13415.
    """

    @property
    def gamma(self):
        return self.coverage + 3 * (1 - self.coverage) / 4

    def quantile(self, arr, q):
        weights = np.exp(np.log(self.gamma) * np.arange(len(arr) - 1, -1, -1))
        return quantile(arr, q, weights=weights)


class EnbNEx(EnbMixIn, NExConformal):
    pass
