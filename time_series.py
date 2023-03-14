#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
"""
File for running all time series experiments.
"""
import argparse
from collections import OrderedDict
from functools import partial
import glob
import itertools
import logging
import math
import multiprocessing as mp
import os
import pickle
import re
import traceback

import matplotlib.pyplot as plt
from merlion.models.factory import ModelFactory
from merlion.models.utils.autosarima_utils import ndiffs
from merlion.utils import TimeSeries
import numpy as np
import pandas as pd
from scipy.stats import norm
import tqdm

from online_conformal.dataset import M4, MonashTSF
from online_conformal.saocp import SAOCP, EnbSAOCP
from online_conformal.enbpi import EnbPI, EnbMixIn
from online_conformal.faci import FACI, FACI_S, EnbFACI
from online_conformal.model_sigma import ModelSigma
from online_conformal.nex_conformal import NExConformal, EnbNEx
from online_conformal.ogd import ScaleFreeOGD, EnbOGD
from online_conformal.split_conformal import SplitConformal
from online_conformal.utils import coverage, interval_miscoverage, interval_regret, mae, width


logger = logging.getLogger(__name__)

name2dataset = dict(
    M4_Hourly=lambda: M4("Hourly"),
    M4_Daily=lambda: M4("Daily"),
    M4_Weekly=lambda: M4("Weekly"),
    NN5_Daily=lambda: MonashTSF("nn5_daily", freq="1D", horizon=30),
)

name2model = dict(
    Prophet=dict(name="Prophet", target_seq_index=0),
    LGBM=dict(name="LGBMForecaster", n_jobs=2, target_seq_index=0),
    ARIMA=dict(name="Arima", order=(10, None, 10), target_seq_index=0, transform=dict(name="Identity")),
)


def parse_args():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    parser = argparse.ArgumentParser(description="Runs conformal prediction experiments on time series datasets.")
    parser.add_argument("--dirname", type=str, default=results_dir, help="Directory where results are stored.")
    parser.add_argument("--dataset", type=str, default="M4_Hourly", choices=list(name2dataset.keys()))
    parser.add_argument("--model", type=str, default="LGBM", choices=list(name2model.keys()))
    parser.add_argument("--target_cov", type=int, nargs="*", help="The target coverages (as a percent).")
    parser.add_argument("--njobs", type=int, default=None, help="The number of parallel processes to use")
    parser.add_argument("--skip_train", action="store_true", help="Skip running models & only use saved results.")
    parser.add_argument("--start", type=int, default=0, help="The index to start at. For parallelization.")
    parser.add_argument("--end", type=int, default=None, help="The index to end at. For parallelization.")
    parser.add_argument("--ignore_checkpoint", action="store_true", help="Ignore saved results & start over.")
    parser.add_argument("--skip_model_sigma", action="store_true", help="Skip visualizing model's own uncertainty.")
    parser.add_argument("--skip_ensemble", action="store_true", help="Skip using ensemble methods.")
    args = parser.parse_args()

    # Set full dirname & convert various arguments to the forms expected downstream
    args.dirname = os.path.join(args.dirname, args.dataset, args.model)
    args.target_cov = np.asarray(args.target_cov or [80, 90, 95]) / 100
    if args.njobs is None:
        args.njobs = math.ceil(mp.cpu_count() / 2)
    if args.model == "LGBM":
        args.njobs = math.ceil(args.njobs / 2)
    args.model = name2model[args.model]
    return args


def evaluate(model, train_data, test_data, horizon, target_covs, calib_frac, ensemble=True, verbose=False, cache=None):
    cache = cache or {}
    target_idx = None
    base_predictor, base_ensemble = None, None
    if "ARIMA" in model["name"].upper():
        model["order"] = (model["order"][0], ndiffs(train_data.iloc[:, 0].dropna()), model["order"][2])
    model = ModelFactory.create(**model) if isinstance(model, dict) else model
    if not isinstance(target_covs, list):
        target_covs = [target_covs]
    predictors = OrderedDict()
    methods = [SplitConformal, NExConformal, FACI, ScaleFreeOGD, FACI_S, SAOCP, ModelSigma]
    methods += [EnbPI, EnbNEx, EnbFACI, EnbOGD, EnbSAOCP]
    method_covs = list(itertools.product(methods, target_covs))
    kwargs = dict(train_data=train_data, calib_frac=calib_frac, horizon=horizon)
    for method, cov in tqdm.tqdm(method_covs, desc="Model Training", disable=not verbose):
        # Collect all the predictors after training the base models
        method_name = method.__name__
        if cov in cache and (method_name in cache[cov] or re.sub("SAOCP", "CBCE", method_name) in cache[cov]):
            predictors[(method_name, cov)] = None
            continue
        if issubclass(method, EnbMixIn) and not ensemble:
            continue
        try:
            if issubclass(method, EnbMixIn) and base_ensemble is None:
                predictor = method(model, coverage=cov, **kwargs)
                base_ensemble = predictor
            elif not issubclass(method, EnbMixIn) and base_predictor is None:
                predictor = method(model, coverage=cov, **kwargs)
                target_idx = predictor.model.target_seq_index
                base_predictor = predictor
            else:
                other = base_ensemble if issubclass(method, EnbMixIn) else base_predictor
                predictor = method.from_other(other, coverage=cov)
            predictors[(method_name, cov)] = predictor
        except Exception as e:
            if method is ModelSigma:  # model doesn't support uncertainty estimation
                continue
            elif issubclass(method, EnbMixIn):  # model is incompatible with ensembles
                ensemble = False
                continue
            else:
                raise e

    # Do the forecasting
    t0 = test_data.index[0]
    if all(p is None for p in predictors.values()):
        target = None
    else:
        target = test_data.iloc[:, target_idx]
    if horizon > 1:
        test_data = pd.concat((train_data.iloc[-horizon + 1 :], test_data))
        train_data = train_data.iloc[: -horizon + 1]

    yhat, lb, ub = [OrderedDict((k, []) for k in predictors.keys()) for _ in range(3)]
    for i in tqdm.trange(len(test_data), desc="Forecasting", disable=not verbose):
        # Don't do anything if we've cached all the results already
        if all(p is None for p in predictors.values()):
            break
        # Get the base model's forecast for this timestamp, and then move the train data forward one step
        y_t = test_data.iloc[i : i + horizon]
        if base_predictor is not None:
            base_yhat_t, err_t = base_predictor.model.forecast(y_t.index, TimeSeries.from_pd(train_data))
            base_yhat_t = base_yhat_t.to_pd().iloc[:, 0]
            err_t = None if err_t is None else err_t.to_pd().iloc[:, 0]
        else:
            base_yhat_t = err_t = None
        if base_ensemble is not None:
            ens_yhat_t = base_ensemble.model.forecast(y_t.index, TimeSeries.from_pd(train_data))[0].to_pd().iloc[:, 0]
        else:
            ens_yhat_t = None
        train_data = pd.concat((train_data, y_t.iloc[:1]))

        # Obtain error bars from each predictor, and then update the predictor
        for (method, cov), predictor in predictors.items():
            k = (method, cov)
            if predictor is None:  # cached results
                continue
            yhat_t = ens_yhat_t if isinstance(predictor, EnbMixIn) else base_yhat_t
            if isinstance(predictor, ModelSigma):
                if err_t is None:
                    if k in yhat:
                        del yhat[k], lb[k], ub[k]
                    continue
                lb_t = yhat_t + err_t * norm.ppf((1 - cov) / 2)
                ub_t = yhat_t + err_t * norm.ppf(1 - (1 - cov) / 2)
            else:
                lb_t, ub_t = zip(*[predictor.predict(h + 1) for h in range(len(yhat_t))])
                lb_t = yhat_t + np.asarray(lb_t)
                ub_t = yhat_t + np.asarray(ub_t)
                for h in range(len(y_t)):
                    if y_t.index[h] >= t0 and not np.isnan(y_t.iloc[h, target_idx]) and not np.isnan(yhat_t.iloc[h]):
                        predictor.update(y_t.iloc[h : h + 1, target_idx], yhat_t.iloc[h : h + 1], horizon=h + 1)
            yhat[k].append(yhat_t)
            lb[k].append(lb_t)
            ub[k].append(ub_t)

    # Aggregate forecasts & error bars for each horizon
    results = OrderedDict()
    for method, cov in yhat.keys():
        if cov not in results:
            results[cov] = OrderedDict()
        if cov in cache and (method in cache[cov] or re.sub("SAOCP", "CBCE", method) in cache[cov]):
            results[cov][method] = cache[cov].get(method, cache[cov][re.sub("SAOCP", "CBCE", method)])
        else:
            yhat_k, lb_k, ub_k = [
                {
                    h + 1: pd.concat([x.iloc[h : h + 1] for x in ts if len(x) > h and x.index[h] >= t0])
                    for h in range(horizon)
                }
                for ts in [yhat[(method, cov)], lb[(method, cov)], ub[(method, cov)]]
            ]
            results[cov][method] = {"ground_truth": target, "forecast": [yhat_k, lb_k, ub_k], "target_cov": cov}

    return results


def summarize_results(all_results):
    def construct(true, pred):
        return pd.concat([pred[t % len(pred) + 1].iloc[t : t + 1] for t in range(len(true))])

    summaries = OrderedDict()
    for cov, cov_results in all_results.items():
        summary = []
        methods = [re.sub("CBCE", "SAOCP", method) for method in cov_results.keys()]
        for method, result in zip(methods, cov_results.values()):
            y = result["ground_truth"]
            yhat, lb, ub = result["forecast"]
            horizons = ["full"] + sorted(yhat.keys())
            yhat["full"], lb["full"], ub["full"] = construct(y, yhat), construct(y, lb), construct(y, ub)
            kwargs = dict(cov=result["target_cov"], window=min(20, len(y)))
            int_miscov = partial(interval_miscoverage, **kwargs)
            int_regret = partial(interval_regret, **kwargs)
            for i, fn in enumerate([coverage, width, int_miscov, int_regret, mae]):
                if len(summary) < i + 1:
                    summary.append(pd.DataFrame(0, index=pd.Index(horizons, name="Horizon"), columns=methods))
                summary[i].loc[horizons, method] = [fn(y, yhat[h], lb[h], ub[h]) for h in horizons]
        summaries[cov] = summary
    return summaries


def summarize_file(fname):
    with open(fname, "rb") as f:
        results = pickle.load(f)
    ts_target_cov = list(results.values())[0]["target_cov"]
    return ts_target_cov, summarize_results({ts_target_cov: results})[ts_target_cov]


def synthesize_results_dir(dirname: str, njobs=1):
    target_cov = None
    full_summary = []
    files = sorted(glob.glob(os.path.join(dirname, "*.pkl")), key=lambda k: int(re.search(r"(\d+)\.pkl", k).group(1)))
    if len(files) == 0:
        raise RuntimeError(f"Directory {dirname} has no .pkl files of results in it.")
    with mp.Pool(njobs) as pool:
        with tqdm.tqdm(total=len(files), desc="Analyzing Results", leave=False) as pbar:
            for ts_target_cov, summ in pool.imap_unordered(summarize_file, files):
                if target_cov is None:
                    target_cov = ts_target_cov
                assert ts_target_cov == target_cov
                if any((df > 1000).any().any() for df in summ):  # Outlier removal
                    continue
                for i, df in enumerate(summ):
                    if len(full_summary) < i + 1:
                        full_summary.append([df])
                    else:
                        full_summary[i].append(df)
                pbar.update(1)

    gbs = tuple(pd.concat(summ).groupby("Horizon", dropna=False) for summ in full_summary)
    return {target_cov: tuple(gb.mean() for gb in gbs)}, {target_cov: tuple(gb.std() for gb in gbs)}


def visualize(summaries, ensemble=False, skip_model_sigma=True, plot_regret=True):
    def skip(name):
        extra_check = name == "ModelSigma" and skip_model_sigma
        return extra_check or ("Enb" in name and not ensemble) or ("Enb" not in name and ensemble)

    figs = OrderedDict()
    for target_cov, stats in summaries.items():
        cov, subopt, miscov, regret = stats[:4]
        results = [("Coverage", target_cov, cov), ("Width", None, subopt), ("Interval Miscoverage", 0, miscov)]
        if plot_regret:
            results.append(("Interval Regret", 0, regret))
        nrows = math.ceil(len(results) / 3)
        ncols = math.ceil(len(results) / nrows)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows + 1), facecolor="w")
        axs = axs.reshape(nrows, ncols)
        for i, (title, baseline_target, df) in enumerate(results):
            i, j = i // ncols, i % ncols
            df = df.loc[[h for h in df.index if isinstance(h, int)], [c for c in df.columns if not skip(c)]]
            if baseline_target is not None:
                axs[i, j].axhline(baseline_target, ls="--", c="k", label="target")
            for k, method in enumerate(df.columns):
                c = 1 if method == "SAOCP" else k + int(k > 0)
                axs[i, j].plot(df.loc[:, method], label=method, color=f"C{c}")
            axs[i, j].set_xlabel(df.index.name, fontsize=14)
            axs[i, j].set_title(title, fontsize=16)
            if i == j == 0:
                fig.legend()
        fig.suptitle(f"Target Coverage = {target_cov:.3f}", fontsize=20)
        fig.tight_layout()
        figs[target_cov] = fig
    return figs


def main_loop(i_data_args):
    cache, fnames = {}, {}
    i, data, args = i_data_args
    covs = list(args.target_cov)
    if not args.start <= i < args.end:
        return None, None
    for cov in covs:
        fname = os.path.join(args.dirname, str(int(cov * 100)), f"{i}.pkl")
        fnames[cov] = fname
        if os.path.exists(fname) and not args.ignore_checkpoint:
            try:
                with open(fname, "rb") as f:
                    cache[cov] = pickle.load(f)
            except:
                continue
    logging.basicConfig(format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", level=logging.ERROR)
    try:
        return fnames, evaluate(args.model, **data, target_covs=covs, ensemble=not args.skip_ensemble, cache=cache)
    except Exception:
        return fnames, f"Exception on time series {i}\n{traceback.format_exc()}"


def main():
    args = parse_args()
    dataset = name2dataset[args.dataset]()
    dirnames = {cov: os.path.join(args.dirname, str(int(cov * 100))) for cov in args.target_cov}
    for dirname in dirnames.values():
        os.makedirs(dirname, exist_ok=True)
    os.makedirs(os.path.join(args.dirname, "figures"), exist_ok=True)

    if not args.skip_train:
        n = len(dataset)
        args.end = n if args.end is None else args.end
        with tqdm.trange(n, desc="Dataset") as pbar:
            with mp.Pool(args.njobs) as pool:
                for fnames, results in pool.imap_unordered(main_loop, map(lambda i: (i, dataset[i], args), range(n))):
                    if isinstance(results, str):
                        logger.error(results)
                    elif isinstance(results, dict):
                        for cov, cov_results in results.items():
                            with open(fnames[cov], "wb") as f:
                                pickle.dump(cov_results, f)
                    pbar.update(1)
        if args.start != 0 or args.end != n:
            return

    idx_cols = ["Method", "Target Coverage"]
    cols = ["Coverage", "Width", "Interval Miscoverage", "Interval Regret"]
    err_cols = ["MAE"]
    table = pd.DataFrame(columns=idx_cols + cols).set_index(idx_cols)
    enb_table = table.copy()
    mae_table = pd.DataFrame(columns=err_cols)
    for target_cov, dirname in dirnames.items():
        # Create a table & save it
        summ, sd = synthesize_results_dir(dirname, njobs=args.njobs * 2 if "LGBM" in args.model["name"] else args.njobs)
        for col_name, data, data_std in zip(cols + err_cols, *summ.values(), *sd.values()):
            if col_name in err_cols:
                enb = [m for m in data.columns if "Enb" in m]
                base = [m for m in data.columns if "Enb" not in m]
                if len(base) > 0:
                    mae_table.loc["Base", col_name] = data.loc["full", base[0]]
                    mae_table.loc["Base SD", col_name] = data_std.loc["full", base[0]]
                if len(enb) > 0:
                    mae_table.loc["Enb", col_name] = data.loc["full", enb[0]]
                    mae_table.loc["Enb SD", col_name] = data_std.loc["full", enb[0]]
                continue
            for method in data.columns:
                t = enb_table if "Enb" in method else table
                t.loc[(method, target_cov), col_name] = data.loc["full", method]
                t.loc[(method, target_cov), col_name + " SD"] = data_std.loc["full", method]
        table.to_csv(os.path.join(args.dirname, "results_base.csv"))
        enb_table.to_csv(os.path.join(args.dirname, "results_enb.csv"))
        mae_table.to_csv(os.path.join(args.dirname, "mae.csv"))

        # Make & save figures
        fig = visualize(summ, ensemble=False, skip_model_sigma=args.skip_model_sigma)[target_cov]
        fig_enb = visualize(summ, ensemble=True, skip_model_sigma=args.skip_model_sigma)[target_cov]
        fig.savefig(os.path.join(args.dirname, "figures", f"{int(target_cov * 100)}_results_base.png"))
        fig_enb.savefig(os.path.join(args.dirname, "figures", f"{int(target_cov * 100)}_results_enb.png"))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", level=logging.ERROR)
    main()
