#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from datasets import load_dataset
import numpy as np
import pandas as pd
from ts_datasets.forecast import M4 as _M4


class MonashTSF:
    def __init__(self, dataset, freq, horizon):
        self.freq = freq
        self.horizon = horizon
        self.data = load_dataset("monash_tsf", dataset)["test"]

    def __getitem__(self, i):
        data = self.data[i]
        arr = np.asarray(data["target"]).T
        t0 = data["start"][0] if isinstance(data["start"], list) else data["start"]
        ts = pd.DataFrame(arr.reshape(len(arr), -1), index=pd.date_range(start=t0, periods=len(arr), freq=self.freq))
        ts = (ts - ts.min(axis=0)) / (ts.max(axis=0) - ts.min(axis=0))
        n_test = min(120, int(len(ts) / 5))
        return dict(train_data=ts.iloc[:-n_test], test_data=ts.iloc[-n_test:], horizon=self.horizon, calib_frac=0.2)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class M4:
    def __init__(self, subset):
        self.dataset = _M4(subset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        ts, md = self.dataset[i]
        ts = (ts - ts.min(axis=0)) / (ts.max(axis=0) - ts.min(axis=0))
        calib_frac = 0.2
        if self.dataset.subset == "Weekly":
            horizon = 26 if len(ts) > 200 else 13
            n_test = 120 if len(ts) > 400 else 60 if len(ts) > 200 else 26
        elif self.dataset.subset == "Daily":
            horizon = 28 if len(ts) > 200 else 14
            n_test = 120 if len(ts) > 400 else 60 if len(ts) > 200 else 28
            calib_frac = 0.05 if len(ts) > 4000 else 0.1 if len(ts) > 2000 else 0.2
        elif self.dataset.subset == "Hourly":
            horizon = 24
            n_test = 120
        else:
            n_test = (~md.trainval).sum()
            horizon = n_test // 2
        return dict(train_data=ts.iloc[:-n_test], test_data=ts.iloc[-n_test:], calib_frac=calib_frac, horizon=horizon)
