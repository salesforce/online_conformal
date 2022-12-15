#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
import math
import matplotlib.pyplot as plt
import pandas as pd
from online_conformal.utils import coverage, interval_miscoverage


def plot(ground_truth: pd.DataFrame, pred: pd.DataFrame, lb: pd.DataFrame, ub: pd.DataFrame, title=None, ax=None):
    pred_color = "#0072B2"
    if ax is None:
        fig = plt.figure(facecolor="w", figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ax.plot(pred.index, pred.values.flatten(), c=pred_color, ls="-", zorder=0)
    ax.plot(ground_truth.index, ground_truth.values.flatten(), c="k", alpha=0.8, lw=1, zorder=1)
    ax.fill_between(pred.index, lb.values.flatten(), ub.values.flatten(), color=pred_color, alpha=0.2, zorder=2)
    ax.set_title(title)
    return fig, ax


def plot_simulated_forecast(results, horizon=None):
    ncols = 3
    nrows = math.ceil(len(results) / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5 + 1), facecolor="w")
    axs = axs.reshape(nrows, ncols)
    delta = 0
    target_cov = None
    for i, (method, result) in enumerate(results.items()):
        if method in ["ModelSigma"]:
            continue
        i = i + delta
        ground_truth = result["ground_truth"]
        forecast, lb, ub = result["forecast"]
        constructed = pd.DataFrame(0, index=ground_truth.index, columns=["yhat", "lb", "ub"])
        for k in range(len(ground_truth)):
            h = horizon if horizon else "full"
            constructed.iloc[k] = [forecast[h].iloc[k], lb[h].iloc[k], ub[h].iloc[k]]
        target_cov = result["target_cov"] if target_cov is None else target_cov
        forecast, lb, ub = constructed["yhat"], constructed["lb"], constructed["ub"]
        result = (ground_truth, forecast, lb, ub)
        title = (
            f"{method}: coverage={coverage(*result):.3f}, width={(ub - lb).mean() / 2:.2f}, "
            f"int miscov={interval_miscoverage(*result, window=20, cov=target_cov):.2f}"
        )
        plot(*result, ax=axs[i // ncols, i % ncols], title=title)
    name = f"Horizon = {horizon}" if horizon else "Simulated Forecast"
    fig.suptitle(f"{name}, Target Coverage = {target_cov}", fontsize=16)
    fig.tight_layout()
    plt.show()
