#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from abc import ABC
import copy

from merlion.models.ensemble.forecast import ForecasterEnsemble
from merlion.models.forecast.sklearn_base import SKLearnForecaster
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.transform.base import Identity
from merlion.utils import TimeSeries
import numpy as np
import pandas as pd

from online_conformal.base import BasePredictor
from online_conformal.split_conformal import SplitConformal
from online_conformal.utils import Residuals


class EnbMixIn(BasePredictor, ABC):
    """
    Mix-in class for Ensemble Prediction Intervals (EnbPI), which is the algorithm proposed by Xu & Xie, 2020,
    "Conformal prediction for time series." https://arxiv.org/abs/2010.09107.

    Inheriting from this class will transform any base predictor into one uses leave-one-out ensembles.
    """

    def __init__(
        self,
        model: SKLearnForecaster,
        train_data: pd.DataFrame,
        calib_frac=None,
        coverage=0.9,
        horizon=1,
        pretrained_model=False,
        verbose=False,
        calib_residuals=None,
        retrain=False,
        **kwargs
    ):
        if isinstance(model, SKLearnForecaster):
            model = copy.deepcopy(model)
            model.config.transform = Identity()
            model.config.prediction_stride = horizon
            if not pretrained_model:
                model.train(TimeSeries.from_pd(train_data))

            # Get the rolling window dataset
            target_idx = (
                None if train_data.shape[1] > 1 and model.prediction_stride == 1 else (model.target_seq_index or 0)
            )
            dataset = RollingWindowDataset(
                train_data,
                target_seq_index=target_idx,
                n_past=model.maxlags,
                n_future=model.prediction_stride,
                batch_size=None,
            )
            inputs, inputs_ts, labels, labels_ts = next(iter(dataset))
            non_nan = ~np.isnan(inputs).any(axis=1) & ~np.isnan(labels).any(axis=1)
            inputs, labels = inputs[non_nan], labels[non_nan]
            n = len(inputs)

            # Create copies of the models & re-train each copy on a subset of the data.
            # Obtain the forecasted values as if doing cross-validation.
            b = min(5, n)
            state = np.random.RandomState(0)
            models = [copy.deepcopy(model) for _ in range(b)]
            excluded_idx_sets = np.array_split(state.permutation(n), b)
            residuals = np.zeros((n, 1 if target_idx is None else labels.shape[1]))
            for k, (model_copy, excluded) in enumerate(zip(models, excluded_idx_sets)):
                # Train on (n - 1) / n of the data
                included = np.asarray(sorted(set(range(n)).difference(excluded)), dtype=int)
                model_copy.model.fit(inputs[included], labels[included])
                # Get residuals on the remaining 1 / n of the data
                predict = model_copy.model.predict(inputs[excluded])
                predict = predict[:, model_copy.target_seq_index] if target_idx is None else predict
                ground_truth = labels[excluded, model_copy.target_seq_index] if target_idx is None else labels[excluded]
                residuals[excluded] = (ground_truth - predict).reshape((-1, residuals.shape[1]))

            # Create an ensemble model & call the superclass initializer with it
            model = ForecasterEnsemble(models=models)
            model.train_pre_process(TimeSeries.from_pd(train_data))
            residuals = residuals[:, :horizon]
            calib_residuals = Residuals(horizon)
            for h in range(residuals.shape[1]):
                calib_residuals.extend(h + 1, residuals[:, h].tolist())

        elif not isinstance(model, ForecasterEnsemble):
            b = 5
            calib_residuals = Residuals(horizon)
            models = [copy.deepcopy(model) for _ in range(b)]
            excluded_idx_sets = np.array_split(np.arange(len(train_data), dtype=int), b + 1)[1:]
            for k, (model, excluded) in enumerate(zip(models, excluded_idx_sets)):
                # Train model on the non-excluded data points
                model.reset()
                included = np.asarray(sorted(set(range(len(train_data))).difference(excluded)))
                model.train(TimeSeries.from_pd(train_data.iloc[included]))

                # Update residuals for all horizons
                t0 = train_data.index[excluded[0]]
                ts = train_data.iloc[excluded[0] - horizon : excluded[-1] + 1]
                ts_prev = train_data.iloc[: excluded[0] - horizon]
                for i in range(len(ts)):
                    y_t = ts.iloc[i : i + horizon]
                    yhat_t = model.forecast(y_t.index, TimeSeries.from_pd(ts_prev))[0].to_pd().iloc[:, 0]
                    resid = y_t.iloc[:, model.target_seq_index] - yhat_t
                    ts_prev = pd.concat((ts_prev, y_t.iloc[:1]))
                    for h, r in enumerate(resid):
                        if y_t.index[h] >= t0:
                            calib_residuals.extend(h + 1, r)

            # Create an ensemble model to call the superclass initializer with
            model = ForecasterEnsemble(models=models)
            model.train_pre_process(TimeSeries.from_pd(train_data))

        super().__init__(
            model=model,
            train_data=train_data,
            calib_frac=0,
            coverage=coverage,
            horizon=horizon,
            pretrained_model=True,
            verbose=verbose,
            calib_residuals=calib_residuals,
            retrain=False,
            **kwargs
        )


class EnbPI(EnbMixIn, SplitConformal):
    pass
