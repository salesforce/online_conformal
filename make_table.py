#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
import argparse
import functools
import itertools
import os
import re
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", default=90, type=int)
    parser.add_argument("--window", default=20, type=int)
    parser.add_argument("--ensemble", action="store_true", default=False)
    parser.add_argument("--print_sd", action="store_true", default=False)
    parser.add_argument("--interval_table", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="M4_Hourly", help="Only used for interval analysis")
    parser.add_argument("--model", type=str, default="LGBM", help="Only used for interval analysis")
    args = parser.parse_args()
    return args


def df_to_str_df(df, interval, print_sd):
    if interval or not print_sd:
        return df.applymap(lambda x: f"{(np.round(x, 3)):.3f}")
    return df.apply(
        lambda c: c.apply(
            lambda x: (f"{(np.round(x, 3)):.3f}" if "Reg" in c.name else f"{(np.round(x, 2)):.2f}").lstrip("0")
        )
    )


def combine_sd_df(df, interval, print_sd):
    non_sd_cols = [c for c in df.columns if "SD" not in c]
    sd_cols = [c + " SD" for c in non_sd_cols]
    str_df = df_to_str_df(df.loc[:, non_sd_cols], interval=interval, print_sd=print_sd)
    if print_sd and all(c in df.columns for c in sd_cols):
        str_df = str_df + "\\textsubscript{" + df_to_str_df(df.loc[:, sd_cols], interval, True).values + "}"
    return str_df


def rename_stats(stat):
    stat = re.sub("Interval Miscoverage", r"$\\mathrm{LCE}_k$", stat)
    stat = re.sub("Interval Regret", r"$\\mathrm{SAReg}_k$", stat)
    return stat


def bold_best(v, dataset, full_df, target_cov):
    stat = v.name[1]
    if "Coverage" in full_df.columns.get_level_values(2):
        cov = full_df.loc[:, (dataset, v.name[0], "Coverage")]
        valid = np.abs(cov - target_cov) < 0.05
    else:
        valid = [True] * len(v)
    v = full_df.loc[:, (dataset, *v.name)]
    v_sort = sorted(np.round(v.loc[valid].dropna(), 3))
    if stat == "Coverage":
        return ["color: ForestGreen" if v else "color: red" for v in valid]
    else:
        best = [False] * len(v) if len(v_sort) < 1 else (np.round(v, 3) == v_sort[0]) & valid
        second_best = [False] * len(v) if len(v_sort) < 2 else (np.round(v, 3) == v_sort[1]) & valid
    return ["font-weight: bold" if b else "font-style: italic" if b2 else "" for b, b2 in zip(best, second_best)]


def tex_formatting(tex_str):
    # Rename methods to match paper
    tex_str = re.sub("SAOCP", r"\\method{}", tex_str)
    tex_str = re.sub("OGD", r"\\methodBasic{}", re.sub("ScaleFree", "", tex_str))
    tex_str = re.sub("Split", "S", re.sub("Conformal", "CP", re.sub("ACI_", "ACI-", tex_str)))
    # Update formatting. Underline second-best instead of italicize, and make index more compact
    tex_str = re.sub(r"\\itshape\s*([\d.]*)(\\textsubscript\{[\d.]*\})?", r"\\underline{\1\2}", tex_str)
    tex_str = re.sub(f"(?m)^(.*?Coverage)", r"Method\1", re.sub("Method.*?\n", "", tex_str))
    # Put methods in the right order
    lines = tex_str.split("\n")
    order = ["ModelSigma", "SCP", "NExCP", "FACI", r"\\methodBasic{}", "FACI-S", r"\\method{}"]
    order += ["EnbPI", "EnbNEx", "EnbFACI", r"Enb\\methodBasic{}", r"Enb\\method{}"]
    model_lines = sum([[i for i, line in enumerate(lines) if re.match(f"^\\s*{m}\\s*&", line)] for m in order], [])
    line_order = list(range(min(model_lines))) + model_lines + list(range(max(model_lines) + 1, len(lines)))
    tex_str = "\n".join([lines[i] for i in line_order])
    return tex_str


def primary_table(args):
    full_df, full_str_df = None, None
    models = ["LGBM", "ARIMA", "Prophet"]
    datasets = ["M4_Hourly", "M4_Daily", "M4_Weekly", "NN5_Daily"]
    mae_idx = "Enb" if args.ensemble else "Base"
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    for dataset, model in itertools.product(datasets, models):
        fname = "results_enb.csv" if args.ensemble else "results_base.csv"
        fname = os.path.join(dirname, dataset, model, f"k={args.window}", fname)
        if os.path.exists(fname):
            df = pd.read_csv(fname, index_col=0)
            df = df[df["Target Coverage"] == args.coverage / 100].drop(columns="Target Coverage")
            df = df.rename(columns=rename_stats, index=lambda s: re.sub("CBCE", "SAOCP", s))
            if df.isna().all().all() or len(df) == 0:
                continue
            for m in ["ModelSigma", "SimpleSAOCP"]:
                if m in df.index:
                    df = df.drop(labels=[m])
            str_df = combine_sd_df(df, interval=False, print_sd=args.print_sd)
            mae = pd.read_csv(os.path.join(dirname, dataset, model, "mae.csv"), index_col=0).loc[mae_idx, "MAE"]
            model = f"{model} (MAE = {mae:.2f})"
            df.columns = pd.MultiIndex.from_tuples([(re.sub("_", " ", dataset), model, c) for c in df.columns])
            str_df.columns = pd.MultiIndex.from_tuples([(re.sub("_", " ", dataset), model, c) for c in str_df.columns])
            full_df = df if full_df is None else pd.concat((full_df, df), axis=1)
            full_str_df = str_df if full_str_df is None else pd.concat((full_str_df, str_df), axis=1)

    models = full_df.columns.get_level_values(1)
    datasets = full_df.columns.get_level_values(0)
    models = models[sorted(np.unique(models, return_index=True)[1])]
    datasets = datasets[sorted(np.unique(datasets, return_index=True)[1])]
    for dataset in datasets:
        df = full_str_df.loc[:, dataset]
        highlight = functools.partial(bold_best, dataset=dataset, full_df=full_df, target_cov=args.coverage / 100)
        styler = df.style.format(na_rep="--").apply(highlight).hide(axis=1, level=2)
        _models = [m for m in models if m in df.columns.get_level_values(0)]
        tex_str = styler.to_latex(
            hrules=True,
            convert_css=True,
            multicol_align="c|",
            column_format="l" + "".join(("|" + "c" * df.loc[:, m].shape[1]) for m in _models),
        )
        print(dataset)
        # No vrule after last multicol
        tex_str = re.sub(r"(multicolumn{\d+}{c)\|(}{" + re.sub(r"\(.*?\)", ".*?", _models[-1]) + "})", r"\1\2", tex_str)
        print(tex_formatting(tex_str))


def interval_table(args):
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", args.dataset, args.model)
    fname = "results_enb.csv" if args.ensemble else "results_base.csv"
    ks = sorted([int(k[2:]) for k in os.listdir(dirname) if k.startswith("k=")])
    full_df, full_str_df = None, None
    for k in ks:
        df = pd.read_csv(os.path.join(dirname, f"k={k}", fname), index_col=0)
        df = df[df["Target Coverage"] == args.coverage / 100].drop(columns="Target Coverage")
        df = df.rename(columns=rename_stats, index=lambda m: re.sub("CBCE", "SAOCP", m))
        if df.isna().all().all() or len(df) == 0:
            continue
        for m in ["ModelSigma", "SimpleSAOCP"]:
            if m in df.index:
                df = df.drop(labels=[m])
        df = df.loc[:, [c for c in df.columns if "LCE" in c]]
        str_df = combine_sd_df(df, interval=True, print_sd=args.print_sd)
        df.columns = pd.MultiIndex.from_tuples([(args.dataset, k, c) for c in df.columns])
        str_df.columns = pd.MultiIndex.from_tuples([(k, c) for c in str_df.columns])
        full_df = df if full_df is None else pd.concat((full_df, df), axis=1)
        full_str_df = str_df if full_str_df is None else pd.concat((full_str_df, str_df), axis=1)

    if not args.print_sd:
        print(full_str_df)
    highlight = functools.partial(bold_best, dataset=args.dataset, full_df=full_df, target_cov=args.coverage / 100)
    styler = full_str_df.style.format(na_rep="--").apply(highlight).hide(axis=1, level=2)
    tex_str = styler.to_latex(hrules=True, convert_css=True)
    tex_str = "\n".join([line for line in tex_str.split("\n") if "LCE" not in line])
    tex_str = tex_formatting(tex_str)
    print(tex_str)


def main():
    args = parse_args()
    if args.interval_table:
        interval_table(args)
    else:
        primary_table(args)


if __name__ == "__main__":
    main()
