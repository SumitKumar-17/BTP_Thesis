#!/usr/bin/env python3
"""Aggregate repeated set-sweep runs with 95% confidence intervals."""

import glob
import math
import os
import re
import pandas as pd

RESULT_ROOT = "results/set_sweep"
OUT_CSV = os.path.join(RESULT_ROOT, "set_sweep_summary.csv")
OUT_MD = os.path.join(RESULT_ROOT, "set_sweep_report.md")


def ci95(values):
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    margin = 1.96 * se
    return (mean, mean - margin, mean + margin)


def parse_set_and_trial(path):
    m = re.search(r"set(\d+)_trial(\d+)\.csv$", os.path.basename(path))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_series(path):
    df = pd.read_csv(path)
    return df["latency_cpu_cycles"]


def collect_rows():
    rows = []
    set_dirs = sorted(glob.glob(os.path.join(RESULT_ROOT, "set_*")))

    for sd in set_dirs:
        attack_files = sorted(glob.glob(os.path.join(sd, "attack_set*_trial*.csv")))
        for af in attack_files:
            set_id, trial = parse_set_and_trial(af)
            if set_id is None:
                continue
            bf = os.path.join(sd, "baseline_set{}_trial{}.csv".format(set_id, trial))
            if not os.path.exists(bf):
                continue

            a = load_series(af)
            b = load_series(bf)
            attack_median = float(a.median())
            threshold = attack_median * 10.0
            evictions = int((a > threshold).sum())

            rows.append({
                "set_id": set_id,
                "trial": trial,
                "baseline_mean": float(b.mean()),
                "attack_mean": float(a.mean()),
                "mean_ratio": float(a.mean() / b.mean()),
                "baseline_p99": float(b.quantile(0.99)),
                "attack_p99": float(a.quantile(0.99)),
                "p99_ratio": float(a.quantile(0.99) / b.quantile(0.99)),
                "baseline_max": float(b.max()),
                "attack_max": float(a.max()),
                "max_ratio": float(a.max() / b.max()),
                "eviction_pct": float(100.0 * evictions / len(a)),
            })

    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for set_id in sorted(df["set_id"].unique()):
        g = df[df["set_id"] == set_id]
        mean_ratio_mean, mean_ratio_lo, mean_ratio_hi = ci95(g["mean_ratio"].tolist())
        p99_ratio_mean, p99_ratio_lo, p99_ratio_hi = ci95(g["p99_ratio"].tolist())
        max_ratio_mean, max_ratio_lo, max_ratio_hi = ci95(g["max_ratio"].tolist())
        ev_mean, ev_lo, ev_hi = ci95(g["eviction_pct"].tolist())

        out.append({
            "set_id": int(set_id),
            "trials": int(len(g)),
            "mean_ratio_mean": mean_ratio_mean,
            "mean_ratio_ci_low": mean_ratio_lo,
            "mean_ratio_ci_high": mean_ratio_hi,
            "p99_ratio_mean": p99_ratio_mean,
            "p99_ratio_ci_low": p99_ratio_lo,
            "p99_ratio_ci_high": p99_ratio_hi,
            "max_ratio_mean": max_ratio_mean,
            "max_ratio_ci_low": max_ratio_lo,
            "max_ratio_ci_high": max_ratio_hi,
            "eviction_pct_mean": ev_mean,
            "eviction_pct_ci_low": ev_lo,
            "eviction_pct_ci_high": ev_hi,
        })

    return pd.DataFrame(out).sort_values("set_id")


def write_markdown(summary_df, trial_df):
    with open(OUT_MD, "w") as f:
        f.write("# GPU-CPU LLC Set Sweep Report\\n\\n")
        f.write("Source directory: `{}`\\n\\n".format(RESULT_ROOT))
        f.write("## Per-Set Summary (95% CI)\\n\\n")
        f.write("| Set | Trials | Mean Ratio | P99 Ratio | Max Ratio | Eviction % |\\n")
        f.write("|---:|---:|---:|---:|---:|---:|\\n")

        for _, r in summary_df.iterrows():
            row = "| {} | {} | {:.2f} [{:.2f}, {:.2f}] | {:.2f} [{:.2f}, {:.2f}] | {:.2f} [{:.2f}, {:.2f}] | {:.2f}% [{:.2f}, {:.2f}] |\\n".format(
                int(r["set_id"]),
                int(r["trials"]),
                r["mean_ratio_mean"], r["mean_ratio_ci_low"], r["mean_ratio_ci_high"],
                r["p99_ratio_mean"], r["p99_ratio_ci_low"], r["p99_ratio_ci_high"],
                r["max_ratio_mean"], r["max_ratio_ci_low"], r["max_ratio_ci_high"],
                r["eviction_pct_mean"], r["eviction_pct_ci_low"], r["eviction_pct_ci_high"],
            )
            f.write(row)

        f.write("\\n## Trial-Level Metrics\\n\\n")
        f.write("| Set | Trial | Mean Ratio | P99 Ratio | Max Ratio | Eviction % |\\n")
        f.write("|---:|---:|---:|---:|---:|---:|\\n")
        for _, r in trial_df.sort_values(["set_id", "trial"]).iterrows():
            row = "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f}% |\\n".format(
                int(r["set_id"]),
                int(r["trial"]),
                r["mean_ratio"],
                r["p99_ratio"],
                r["max_ratio"],
                r["eviction_pct"],
            )
            f.write(row)


def main():
    if not os.path.isdir(RESULT_ROOT):
        raise SystemExit("Missing directory: {}".format(RESULT_ROOT))

    trial_df = collect_rows()
    if trial_df.empty:
        raise SystemExit("No sweep CSV pairs found. Run run_set_sweep_experiment.sh first.")

    summary_df = summarize(trial_df)
    trial_df.to_csv(os.path.join(RESULT_ROOT, "set_sweep_trials.csv"), index=False)
    summary_df.to_csv(OUT_CSV, index=False)
    write_markdown(summary_df, trial_df)

    print("=== SET SWEEP SUMMARY ===")
    print(summary_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("\\nSaved: {}".format(OUT_CSV))
    print("Saved: {}".format(OUT_MD))


if __name__ == "__main__":
    main()
