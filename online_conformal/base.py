#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from abc import ABC, abstractmethod
import copy
import math
from typing import Dict, Tuple

from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries
import numpy as np
import pandas as pd
import tqdm

from online_conformal.utils import quantile, Residuals


class BasePredictor(ABC):
    def __init__(
        self,
        model: ForecasterBase,
        train_data: pd.DataFrame,
        calib_frac=None,
        coverage=0.9,
        horizon=1,
        pretrained_model=False,
        verbose=False,
        calib_residuals=None,
        retrain=False,
        **kwargs,
    ):
        self.coverage = coverage
        self.horizon = horizon
        self.verbose = verbose

        # Split data into train & calibration splits after saving the full training dataset
        self.train_data = train_data
        if calib_frac is None and train_data is not None:
            delta, epsilon = 0.1, 1 - coverage
            # DKW inequality: sup_x |F_n(x) - F(x)| > epsilon w.p. at most delta if n is above the below threshold
            n_calib = -np.log(delta / 2) / 2 / epsilon**2
            # But we don't want to use more than 20% of the data for calibration unless otherwise specified
            n_calib = math.ceil(min(0.2 * len(train_data), n_calib))
        elif train_data is not None:
            calib_frac = max(0.0, min(1.0, calib_frac))
            n_calib = math.ceil(len(train_data) * calib_frac)
        else:
            n_calib = 0
        if n_calib == 0:
            calib_data = None
        else:
            calib_data = train_data.iloc[-n_calib:]
            train_data = train_data.iloc[:-n_calib]

        # Train model if needed
        if model is not None:
            assert isinstance(model, ForecasterBase)
            assert isinstance(model.target_seq_index, int) or train_data.shape[1] == 1
        else:
            pretrained_model = True
        self.model = model
        if not pretrained_model:
            self.model.reset()
            self.model.train(TimeSeries.from_pd(train_data))

        # Make predictions on the calibration data, updating the conformal wrapper in the process
        self.residuals = Residuals(self.horizon)
        if calib_residuals is not None:
            self.calib_residuals = copy.deepcopy(calib_residuals)
            for h, r in calib_residuals.horizon2residuals.items():
                if len(r) > 0:
                    self.update(pd.Series(r, dtype=float), pd.Series(np.zeros(len(r))), horizon=h)
        else:
            self.calib_residuals = None
            if calib_data is not None:
                self.forecast(calib_data, train_data)
                self.residuals.remove_outliers()
            self.calib_residuals = copy.deepcopy(self.residuals)

        if retrain and not pretrained_model:
            self.model.reset()
            self.model.train(TimeSeries.from_pd(self.train_data))

    @abstractmethod
    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon: int):
        raise NotImplementedError

    @classmethod
    def from_other(cls, other, **kwargs):
        assert isinstance(other, BasePredictor)
        all_kwargs = dict(
            model=other.model,
            pretrained_model=True,
            train_data=other.train_data,
            calib_frac=0,
            coverage=other.coverage,
            horizon=other.horizon,
            calib_residuals=other.calib_residuals,
            verbose=other.verbose,
        )
        all_kwargs.update(**kwargs)
        return cls(**all_kwargs)

    @staticmethod
    def quantile(arr, q):
        return quantile(arr, q)

    def predict(self, horizon) -> Tuple[float, float]:
        delta_ub = self.quantile(np.abs(self.residuals.get(horizon)), self.coverage)
        delta_lb = -delta_ub
        return delta_lb, delta_ub

    def forecast(
        self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None
    ) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        # Process arguments
        t0 = time_series.index[0]
        if time_series_prev is None:
            time_series_prev = self.train_data.iloc[: -self.horizon + 1]
            time_series = pd.concat((self.train_data.iloc[-self.horizon + 1 :], time_series))

        # Forecast in increments of self.horizon & update the conformal wrapper's internal state along the way.
        yhat, lb, ub = [], [], []
        for i in tqdm.trange(len(time_series), desc="Forecasting", disable=not self.verbose):
            y_t = time_series.iloc[i : i + self.horizon]
            yhat_t, _ = self.model.forecast(y_t.index, TimeSeries.from_pd(time_series_prev))
            yhat_t = yhat_t.to_pd().iloc[:, 0]
            yhat.append(yhat_t)
            lb_t, ub_t = zip(*[self.predict(h + 1) for h in range(len(yhat_t))])
            lb.append(yhat_t + np.asarray(lb_t))
            ub.append(yhat_t + np.asarray(ub_t))
            time_series_prev = pd.concat((time_series_prev, y_t.iloc[:1]))
            for h in range(len(y_t)):
                idx = self.model.target_seq_index
                if y_t.index[h] >= t0 and not np.isnan(y_t.iloc[h, idx]) and not np.isnan(yhat_t.iloc[h]):
                    self.update(y_t.iloc[h : h + 1, idx], yhat_t.iloc[h : h + 1], horizon=h + 1)

        # Aggregate & return forecasts for each horizon (along with confidence intervals)
        yhat, lb, ub = [
            {
                h + 1: pd.concat([x.iloc[h : h + 1] for x in ts if len(x) > h and x.index[h] >= t0])
                for h in range(self.horizon)
            }
            for ts in [yhat, lb, ub]
        ]
        return yhat, lb, ub
