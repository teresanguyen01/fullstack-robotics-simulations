#!/usr/bin/env python3
"""
Compare matching sensors across multiple CSVs in one folder.

For each sensor label ("Sensor 1", "Sensor 2", ...), compare the signal in a
reference CSV to the same-named sensor in every other CSV.

Metrics saved per (sensor, other_file) row:
- Pearson r, Spearman r, Kendall tau, Cosine similarity
- RMSE % of range, MAE % of range
- Max lagged correlation (±max_lag samples) and best lag
- DTW distance on raw signals (optional)
- DTW distance on z-scored signals (shape only) (optional)
- Composite similarity % (sim_pct) built from chosen components

Outputs:
  sensor_comparison_long.csv  # one row per sensor × other_file
  sensor_comparison_wide.csv  # aggregate (mean/median/max) per sensor

Typical use:
  python sensor_compare_folder.py /path/to/folder \
    --pattern "*.csv" --time-col Time_ms --ref-index 0 --out results \
    --use-dtw --dtw-window 200 --max-lag 200 \
    --w-mag 0.4 --w-shape 0.5 --w-lag 0.1

Weights:
  w_mag   : magnitude agreement (RMSE/MAE based)
  w_shape : DTW on z-scored series (shape)
  w_lag   : max lagged correlation (cheap timing fix)

Set any weight to 0 to ignore that piece.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.stats import spearmanr, kendalltau


SENSOR_NAMES = {
    "Sensor 1":  "right elbow",
    "Sensor 2":  "right shoulder",
    "Sensor 3":  "right collarbone",
    "Sensor 4":  "bottom back",
    "Sensor 5":  "top back",
    "Sensor 6":  "left collarbone",
    "Sensor 7":  "left shoulder",
    "Sensor 8":  "left armpit (back)",
    "Sensor 9":  "left elbow",
    "Sensor 10": "left elbow (back)",
    "Sensor 11": "left armpit (front)",
    "Sensor 12": "waist left a",
    "Sensor 13": "chest l",
    "Sensor 14": "stomach l",
    "Sensor 15": "hip right",
    "Sensor 16": "waist left b",
    "Sensor 17": "right armpit (back)",
    "Sensor 18": "right elbow (front)",
    "Sensor 19": "right armpit (front)",
    "Sensor 20": "waist right a",
    "Sensor 21": "waist right b",
    "Sensor 22": "hip left",
    "Sensor 23": "stomach r",
    "Sensor 24": "chest r"
}

def zscore(v: np.ndarray) -> np.ndarray:
    m = np.mean(v)
    s = np.std(v)
    return (v - m) / s if s else np.zeros_like(v)

def dtw_distance(x, y, window=None):
    """Pure-NumPy DTW with optional Sakoe–Chiba window."""
    n, m = len(x), len(y)
    w = max(window or max(n, m), abs(n - m))
    inf = float("inf")
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - w)
        j1 = min(m, i + w)
        xi = x[i - 1]
        for j in range(j0, j1 + 1):
            cost = abs(xi - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]

def lagged_max_corr(x, y, max_lag):
    if max_lag <= 0:
        r = np.corrcoef(x, y)[0, 1]
        return r, 0
    lags = np.arange(-max_lag, max_lag + 1)
    best_r = -np.inf
    best_l = 0
    for l in lags:
        xa = x[max(0, l):len(x) + min(0, l)]
        ya = y[max(0, -l):len(y) + min(0, -l)]
        if len(xa) < 2:
            continue
        r = np.corrcoef(xa, ya)[0, 1]
        if np.isnan(r):
            continue
        if r > best_r:
            best_r = r
            best_l = l
    return best_r, best_l

def compare_sensor_vectors(x, y, use_dtw=False, dtw_window=200, max_lag=200):
    pearson = np.corrcoef(x, y)[0, 1]
    spear   = spearmanr(x, y).correlation
    kend    = kendalltau(x, y, nan_policy="omit").correlation
    cos_sim = float(np.dot(x, y) / (norm(x) * norm(y)))
    rmse    = float(np.sqrt(np.mean((x - y) ** 2)))
    mae     = float(np.mean(np.abs(x - y)))
    rng     = max(np.ptp(x), np.ptp(y))
    rmse_pct = float(rmse / rng * 100) if rng else np.nan
    mae_pct  = float(mae  / rng * 100) if rng else np.nan

    best_corr, best_lag = lagged_max_corr(x, y, max_lag)

    if use_dtw:
        dtw_raw = dtw_distance(x, y, window=dtw_window)
        zx, zy = zscore(x), zscore(y)
        dtw_z  = dtw_distance(zx, zy, window=dtw_window)
    else:
        dtw_raw = np.nan
        dtw_z   = np.nan

    return {
        "pearson_r": pearson,
        "spearman_r": spear,
        "kendall_tau": kend,
        "cosine_sim": cos_sim,
        "rmse_pct_range": rmse_pct,
        "mae_pct_range": mae_pct,
        "max_lagged_r": best_corr,
        "best_lag_samples": best_lag,
        "dtw_dist_raw": dtw_raw,
        "dtw_dist_z": dtw_z
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing CSV sensor files")
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern (default: *.csv)")
    ap.add_argument("--time-col", default="Time_ms", help="Timestamp column (default: Time_ms)")
    ap.add_argument("--ref-index", type=int, default=0,
                    help="Which file (0-based index after sorting) is the reference")
    ap.add_argument("--out", default="compare_out", help="Output folder")

    ap.add_argument("--use-dtw", action="store_true", help="Compute DTW distances (slow)")
    ap.add_argument("--dtw-window", type=int, default=200, help="DTW Sakoe-Chiba window")
    ap.add_argument("--max-lag", type=int, default=200, help="Max lag samples for lagged corr")

    ap.add_argument("--w-mag",   type=float, default=0.4, help="Weight for magnitude agreement (RMSE/MAE)")
    ap.add_argument("--w-shape", type=float, default=0.5, help="Weight for shape similarity (DTW on z-scored)")
    ap.add_argument("--w-lag",   type=float, default=0.1, help="Weight for lagged correlation")

    ap.add_argument("--w-pearson", type=float, default=0.0, help="(Deprecated) weight for Pearson in sim_pct")
    ap.add_argument("--w-rmse",    type=float, default=0.0, help="(Deprecated) weight for RMSE in old formula")
    ap.add_argument("--w-dtw",     type=float, default=0.0, help="(Deprecated) weight for DTW raw in old formula")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if len(files) < 2:
        raise RuntimeError("Need at least two CSV files in the folder.")

    if not (0 <= args.ref_index < len(files)):
        raise ValueError(f"--ref-index {args.ref_index} out of range 0..{len(files)-1}")

    ref_file = files[args.ref_index]
    other_files = [f for i, f in enumerate(files) if i != args.ref_index]

    ref_df = pd.read_csv(ref_file)
    ref_df.columns = ref_df.columns.str.strip()
    ref_df.rename(columns=SENSOR_NAMES, inplace=True)

    sensors = [c for c in ref_df.columns if c != args.time_col]

    for f in other_files:
        df = pd.read_csv(f, nrows=1)
        df.columns = df.columns.str.strip()
        df.rename(columns=SENSOR_NAMES, inplace=True)
        sensors = [s for s in sensors if s in df.columns]
    if not sensors:
        raise RuntimeError("No matching sensor columns found across files.")

    long_rows = []

    for f in other_files:
        df2 = pd.read_csv(f)
        df2.columns = df2.columns.str.strip()
        df2.rename(columns=SENSOR_NAMES, inplace=True)
        for s in sensors:
            x = ref_df[s].astype(float).values
            y = df2[s].astype(float).values
            L = min(len(x), len(y))
            x, y = x[:L], y[:L]

            stats = compare_sensor_vectors(
                x, y,
                use_dtw=args.use_dtw,
                dtw_window=args.dtw_window,
                max_lag=args.max_lag
            )
            row = {"Sensor": s, "other_file": os.path.basename(f)}
            row.update(stats)
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)

    def p95(col):
        return np.nanpercentile(long_df[col], 95) if long_df[col].notna().any() else 1.0

    rmse_p95 = p95("rmse_pct_range")
    mae_p95  = p95("mae_pct_range")
    dtw_p95  = p95("dtw_dist_z") if args.use_dtw else 1.0

    sim_mag   = 1 - np.clip(long_df["rmse_pct_range"] / rmse_p95, 0, 1)
    sim_mae   = 1 - np.clip(long_df["mae_pct_range"] / mae_p95, 0, 1)
    sim_mag   = np.nanmean(np.vstack([sim_mag, sim_mae]), axis=0)  # average RMSE & MAE pieces

    sim_shape = 1 - np.clip(long_df["dtw_dist_z"] / dtw_p95, 0, 1) if args.use_dtw else 0.0
    sim_lag   = (np.clip(long_df["max_lagged_r"], -1, 1) + 1) / 2.0

    w_total = args.w_mag + args.w_shape + args.w_lag
    if w_total == 0:
        w_total = 1.0
    sim_core = (args.w_mag   * sim_mag +
                args.w_shape * sim_shape +
                args.w_lag   * sim_lag) / w_total

    long_df["sim_pct"] = 100 * sim_core

    agg_funcs = {
        "pearson_r": ["mean", "median", "max"],
        "spearman_r": ["mean", "median"],
        "rmse_pct_range": ["mean", "median"],
        "mae_pct_range": ["mean", "median"],
        "max_lagged_r": ["mean", "max"],
        "sim_pct": ["mean", "median", "max"]
    }
    if args.use_dtw:
        agg_funcs["dtw_dist_z"] = ["mean", "median"]
        agg_funcs["dtw_dist_raw"] = ["mean", "median"]

    wide_df = long_df.groupby("Sensor").agg(agg_funcs)
    wide_df.columns = ["_".join(col).strip() for col in wide_df.columns.values]
    wide_df = wide_df.reset_index().sort_values("sim_pct_mean", ascending=False)

    long_path = os.path.join(args.out, "sensor_comparison_long.csv")
    wide_path = os.path.join(args.out, "sensor_comparison_wide.csv")
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print(f"Reference file: {os.path.basename(ref_file)}")
    print(f"Compared to {len(other_files)} file(s).")
    print("\nSummary (sorted by sim_pct_mean):")
    cols_to_show = ["Sensor", "sim_pct_mean", "rmse_pct_range_mean",
                    "mae_pct_range_mean", "max_lagged_r_mean"]
    if args.use_dtw:
        cols_to_show.append("dtw_dist_z_mean")
    print(wide_df[cols_to_show].to_string(index=False,
                                          float_format=lambda x: f"{x:0.4f}"))
    print("\nSaved:")
    print(f" - {long_path}")
    print(f" - {wide_path}")

if __name__ == "__main__":
    main()
