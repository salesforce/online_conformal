#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from typing import Dict, Tuple, Union

from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries
import pandas as pd
from scipy.stats import norm
import tqdm

from online_conformal.base import BasePredictor


class ModelSigma(BasePredictor):
    """
    Use the model itself to estimate uncertainty.
    """

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
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_data=train_data,
            calib_frac=calib_frac,
            coverage=coverage,
            horizon=horizon,
            pretrained_model=pretrained_model,
            verbose=verbose,
            calib_residuals=None,
            **kwargs,
        )

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon: int):
        pass

    @classmethod
    def from_other(cls, other, **kwargs):
        all_kwargs = dict(
            model=other.model,
            pretrained_model=True,
            train_data=other.train_data,
            calib_frac=0,
            coverage=other.coverage,
            horizon=other.horizon,
            verbose=other.verbose,
        )
        all_kwargs.update(**kwargs)
        return cls(**all_kwargs)

    def forecast(
        self, time_series: Union[TimeSeries, pd.DataFrame], time_series_prev: Union[TimeSeries, pd.DataFrame] = None
    ) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        # Process arguments
        t0 = time_series.index[0]
        if time_series_prev is None:
            time_series_prev = self.train_data.iloc[: -self.horizon + 1]
            time_series = pd.concat((self.train_data.iloc[-self.horizon + 1 :], time_series))

        # Forecast in increments of self.horizon & get the error bars along the way
        alpha = 1 - self.coverage
        yhat, lb, ub = [], [], []
        for i in tqdm.trange(len(time_series), desc="Forecasting", disable=not self.verbose):
            y_t = time_series.iloc[i : i + self.horizon]
            yhat_t, err_t = self.model.forecast(y_t.index, TimeSeries.from_pd(time_series_prev))
            yhat_t = yhat_t.to_pd().iloc[:, 0]
            if err_t is None:
                raise RuntimeError(f"Model {type(self.model).__name__} does not support uncertainty estimation")
            err_t = err_t.to_pd().iloc[:, 0]
            yhat.append(yhat_t)
            lb.append(yhat_t + err_t * norm.ppf(alpha / 2))
            ub.append(yhat_t + err_t * norm.ppf(1 - alpha / 2))

        # Aggregate & return forecasts for each horizon (along with confidence intervals)
        yhat, lb, ub = [
            {
                h + 1: pd.concat([x.iloc[h : h + 1] for x in ts if len(x) > h and x.index[h] >= t0])
                for h in range(self.horizon)
            }
            for ts in [yhat, lb, ub]
        ]
        return yhat, lb, ub
