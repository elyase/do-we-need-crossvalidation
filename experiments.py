#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy==1.26.*",
#   "scikit-learn==1.5.*",
#   "matplotlib==3.8.*",
#   "scipy==1.13.*",
#   "seaborn==0.13.*",
#   "statsmodels==0.14.*",
# ]
# ///
"""
To Split or Not to Split — Revisited
Provocation: MDL can replace inner validation/CV for many model-selection tasks.

This script implements a clean, reproducible suite comparing MDL-style selection
against K-fold cross-validation across several tasks. We keep the simple and
didactic polynomial experiment, and add structured/noisy and temporal settings
to demonstrate when MDL shines.

Usage examples (with uv inline deps):
  uv run experiments.py --task all --runs 30 --seed 0
  uv run experiments.py --task polynomial --runs 50 --seed 1

Outputs:
  - JSON summary with per-replicate metrics and selections
  - CSV summary of paired differences and statistics
  - Forest plot of paired differences and a runtime comparison plot
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, LassoLarsIC
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings("ignore")

# Plot style
plt.style.use("default")
sns.set_palette("husl")


# -------------------------
# Config and Utilities
# -------------------------

@dataclass
class Config:
    runs: int = 30
    seed: int = 0
    folds: int = 5
    n_jobs: int | None = None
    out_json: str = "results.json"
    out_csv: str = "results_summary.csv"
    out_plot: str = "results_forest.png"
    out_runtime_plot: str = "results_runtime.png"


def get_kfold(seed: int, n_splits: int) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cohen_d_paired(x: List[float], y: List[float]) -> float:
    diffs = np.asarray(x) - np.asarray(y)
    m = diffs.mean()
    s = diffs.std(ddof=1)
    return float(m / s) if s > 0 else 0.0


def ci_mean_paired_diff(x: List[float], y: List[float], conf: float = 0.95) -> Tuple[float, float, float]:
    diffs = np.asarray(x) - np.asarray(y)
    m = diffs.mean()
    n = len(diffs)
    sem = stats.sem(diffs)
    margin = sem * stats.t.ppf((1 + conf) / 2, n - 1)
    return float(m), float(m - margin), float(m + margin)


def holm_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    for i, (name, p) in enumerate(items, start=1):
        adjusted[name] = min(1.0, p * (m - i + 1))
    # enforce monotonicity
    # map back to original order
    return adjusted


def enhanced_bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    mse = max(mse, 1e-12)
    return float(n * np.log(mse) + n_params * np.log(n))


def format_effect_size(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# -------------------------
# Experiment 1: Polynomial degree selection (keep)
# -------------------------

def run_polynomial_experiment(cfg: Config) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    results_mse = {"MDL": [], "CV": []}
    runtimes = {"MDL": [], "CV": []}
    fits = {"MDL": [], "CV": []}

    for run in range(cfg.runs):
        seed = int(cfg.seed + run)
        rs = np.random.RandomState(seed)
        X = rs.uniform(-2, 2, size=(200, 1))
        x = X[:, 0]
        y = 1 - 2 * x + 0.5 * x**2 + 0.3 * x**3 + rs.normal(0, 1.0, size=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        degrees = list(range(1, 8))

        # CV selection
        t0 = time.time()
        cv = get_kfold(seed, cfg.folds)
        cv_scores = []
        for d in degrees:
            model = make_pipeline(PolynomialFeatures(degree=d, include_bias=False), LinearRegression())
            score = -np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=cfg.n_jobs))
            cv_scores.append(score)
        best_cv_degree = int(np.argmin(cv_scores))
        runtimes["CV"].append(time.time() - t0)
        fits["CV"].append(len(degrees) * cfg.folds)

        # MDL selection via BIC on training
        t0 = time.time()
        bic_scores = []
        for d in degrees:
            model = make_pipeline(PolynomialFeatures(degree=d, include_bias=False), LinearRegression())
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_train)
            k = d + 1  # parameters including intercept in 1D
            bic_scores.append(enhanced_bic(y_train, y_pred_tr, k))
        best_mdl_degree = int(np.argmin(bic_scores))
        runtimes["MDL"].append(time.time() - t0)
        fits["MDL"].append(len(degrees))

        # Evaluate on test
        for meth, d in [("CV", best_cv_degree), ("MDL", best_mdl_degree)]:
            model = make_pipeline(PolynomialFeatures(degree=d, include_bias=False), LinearRegression())
            model.fit(X_train, y_train)
            mse = mean_squared_error(y_test, model.predict(X_test))
            results_mse[meth].append(float(mse))

    m, lo, hi = ci_mean_paired_diff(results_mse["MDL"], results_mse["CV"])
    d = cohen_d_paired(results_mse["MDL"], results_mse["CV"])  # negative favors MDL when metric is MSE
    t_stat, p_t = stats.ttest_rel(results_mse["MDL"], results_mse["CV"])
    try:
        w_stat, p_w = stats.wilcoxon(np.array(results_mse["MDL"]) - np.array(results_mse["CV"]))
    except Exception:
        p_w = float("nan")

    return {
        "name": "Polynomial",
        "metric": "Test MSE",
        "lower_is_better": True,
        "results": results_mse,
        "paired_diff": {"mean": m, "ci_lo": lo, "ci_hi": hi},
        "cohen_d": d,
        "t_pvalue": float(p_t),
        "wilcoxon_pvalue": float(p_w),
        "runtimes": {k: float(np.mean(v)) for k, v in runtimes.items()},
        "fits": {k: int(np.mean(v)) for k, v in fits.items()},
    }


# -------------------------
# Experiment 2: Sparse regression (synthetic)
# -------------------------

def simulate_sparse_linear(n: int, p: int, s: int, snr: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rs = np.random.RandomState(seed)
    X = rs.normal(0, 1, size=(n, p))
    support = rs.choice(p, size=s, replace=False)
    beta = np.zeros(p)
    beta[support] = rs.normal(0, 1, size=s)
    y_signal = X @ beta
    sigma = np.sqrt(np.var(y_signal) / max(snr, 1e-6))
    y = y_signal + rs.normal(0, sigma, size=n)
    return X, y, support


def run_sparse_regression_experiment(cfg: Config, n: int = 120, p: int = 50, s: int = 5, snr: float = 5.0) -> Dict[str, Any]:
    results_mse = {"MDL": [], "CV": []}
    support_f1 = {"MDL": [], "CV": []}
    runtimes = {"MDL": [], "CV": []}
    fits = {"MDL": [], "CV": []}

    for run in range(cfg.runs):
        seed = int(cfg.seed + run)
        X, y, support = simulate_sparse_linear(n, p, s, snr, seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # CV: LassoCV
        t0 = time.time()
        cv = get_kfold(seed, cfg.folds)
        lcv = make_pipeline(StandardScaler(with_mean=True, with_std=True), LassoCV(cv=cv, random_state=seed, n_jobs=cfg.n_jobs, max_iter=5000))
        lcv.fit(X_train, y_train)
        y_pred = lcv.predict(X_test)
        results_mse["CV"].append(float(mean_squared_error(y_test, y_pred)))
        # support
        lasso = lcv.named_steps["lassocv"]
        coef = lcv.named_steps["lassocv"].coef_
        sel = set(np.where(coef != 0)[0].tolist())
        tp = len(sel.intersection(set(support)))
        precision = tp / max(len(sel), 1)
        recall = tp / len(support)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        support_f1["CV"].append(float(f1))
        runtimes["CV"].append(time.time() - t0)
        # approximate fits: len(alphas) * folds; LassoCV chooses path adaptively
        fits["CV"].append(cfg.folds * 10)

        # MDL: LassoLarsIC with BIC
        t0 = time.time()
        lbic = make_pipeline(StandardScaler(with_mean=True, with_std=True), LassoLarsIC(criterion="bic", max_iter=5000))
        lbic.fit(X_train, y_train)
        y_pred2 = lbic.predict(X_test)
        results_mse["MDL"].append(float(mean_squared_error(y_test, y_pred2)))
        coef2 = lbic.named_steps["lassolarsic"].coef_
        sel2 = set(np.where(coef2 != 0)[0].tolist())
        tp2 = len(sel2.intersection(set(support)))
        precision2 = tp2 / max(len(sel2), 1)
        recall2 = tp2 / len(support)
        f1_2 = 0.0 if precision2 + recall2 == 0 else 2 * precision2 * recall2 / (precision2 + recall2)
        support_f1["MDL"].append(float(f1_2))
        runtimes["MDL"].append(time.time() - t0)
        fits["MDL"].append(10)

    m, lo, hi = ci_mean_paired_diff(results_mse["MDL"], results_mse["CV"])
    d = cohen_d_paired(results_mse["MDL"], results_mse["CV"])  # negative favors MDL
    t_stat, p_t = stats.ttest_rel(results_mse["MDL"], results_mse["CV"])
    try:
        w_stat, p_w = stats.wilcoxon(np.array(results_mse["MDL"]) - np.array(results_mse["CV"]))
    except Exception:
        p_w = float("nan")

    return {
        "name": "SparseRegression",
        "metric": "Test MSE",
        "lower_is_better": True,
        "results": results_mse,
        "support_f1": {"MDL": float(np.mean(support_f1["MDL"])), "CV": float(np.mean(support_f1["CV"]))},
        "paired_diff": {"mean": m, "ci_lo": lo, "ci_hi": hi},
        "cohen_d": d,
        "t_pvalue": float(p_t),
        "wilcoxon_pvalue": float(p_w),
        "runtimes": {k: float(np.mean(v)) for k, v in runtimes.items()},
        "fits": {k: int(np.mean(v)) for k, v in fits.items()},
        "params": {"n": n, "p": p, "s": s, "snr": snr},
    }


# -------------------------
# Experiment 3: Periodicity detection (noisy binary)
# -------------------------

def generate_repeating_pattern(period_length: int, total_length: int, noise_prob: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(seed)
    pattern = rs.randint(0, 2, period_length)
    full_sequence = np.tile(pattern, total_length // period_length + 1)[:total_length]
    noise_mask = rs.rand(total_length) < noise_prob
    noisy_sequence = full_sequence.copy()
    noisy_sequence[noise_mask] = 1 - noisy_sequence[noise_mask]
    return noisy_sequence.astype(int), pattern.astype(int)


def mdl_score_period(sequence: np.ndarray, period_length: int) -> float:
    n = len(sequence)
    # best-fitting pattern by majority vote
    pattern = np.zeros(period_length)
    for i in range(period_length):
        positions = np.arange(i, n, period_length)
        if len(positions) > 0:
            pattern[i] = np.round(np.mean(sequence[positions]))
    reconstructed = np.tile(pattern, n // period_length + 1)[:n]
    errors = np.sum(sequence != reconstructed)
    pattern_bits = period_length
    complexity_bits = math.log2(period_length) if period_length > 1 else 0.0
    if errors == 0:
        noise_bits = 0.0
    else:
        p_err = errors / n
        p_err = min(max(p_err, 1e-12), 1 - 1e-12)
        noise_bits = n * (-(p_err * math.log2(p_err) + (1 - p_err) * math.log2(1 - p_err)))
    return float(pattern_bits + complexity_bits + noise_bits)


def cv_score_period(sequence: np.ndarray, period_length: int) -> float:
    # Leave-one-out accuracy (fast vectorized approximation)
    n = len(sequence)
    correct = 0
    for i in range(n):
        # majority over training positions congruent to i mod p
        idx = np.arange(0, n)
        mask = (idx % period_length) == (i % period_length)
        mask[i] = False
        votes = sequence[mask]
        if votes.size == 0:
            correct += 0.5
        else:
            pred = 1 if votes.sum() > (votes.size / 2) else 0
            correct += int(pred == sequence[i])
    return correct / n


def run_periodicity_experiment(cfg: Config, true_period: int = 3, length: int = 36, noise: float = 0.1, max_p: int = 15) -> Dict[str, Any]:
    acc = {"MDL": [], "CV": []}
    runtimes = {"MDL": [], "CV": []}
    fits = {"MDL": [], "CV": []}

    for run in range(cfg.runs):
        seed = int(cfg.seed + run)
        sequence, _ = generate_repeating_pattern(true_period, length, noise, seed)
        periods = list(range(1, max_p + 1))

        t0 = time.time()
        mdl_scores = {p: mdl_score_period(sequence, p) for p in periods}
        mdl_best = min(mdl_scores.keys(), key=lambda p: mdl_scores[p])
        runtimes["MDL"].append(time.time() - t0)
        fits["MDL"].append(len(periods))

        t0 = time.time()
        cv_scores = {p: cv_score_period(sequence, p) for p in periods}
        max_acc = max(cv_scores.values())
        candidates = [p for p, a in cv_scores.items() if a == max_acc]
        cv_best = min(candidates)  # Occam tie-break
        runtimes["CV"].append(time.time() - t0)
        fits["CV"].append(len(periods) * length)  # rough proxy

        acc["MDL"].append(1.0 if mdl_best == true_period else 0.0)
        acc["CV"].append(1.0 if cv_best == true_period else 0.0)

    m, lo, hi = ci_mean_paired_diff(acc["MDL"], acc["CV"])  # here diff>0 favors MDL (accuracy higher)
    d = cohen_d_paired(acc["MDL"], acc["CV"])  # positive favors MDL
    t_stat, p_t = stats.ttest_rel(acc["MDL"], acc["CV"]) 
    try:
        w_stat, p_w = stats.wilcoxon(np.array(acc["MDL"]) - np.array(acc["CV"]))
    except Exception:
        p_w = float("nan")

    return {
        "name": "Periodicity",
        "metric": "Success Rate",
        "lower_is_better": False,
        "results": acc,
        "paired_diff": {"mean": m, "ci_lo": lo, "ci_hi": hi},
        "cohen_d": d,
        "t_pvalue": float(p_t),
        "wilcoxon_pvalue": float(p_w),
        "runtimes": {k: float(np.mean(v)) for k, v in runtimes.items()},
        "fits": {k: int(np.mean(v)) for k, v in fits.items()},
        "params": {"true_period": true_period, "length": length, "noise": noise, "max_p": max_p},
    }


# -------------------------
# Experiment 4: AR(p) order selection (time series)
# -------------------------

def simulate_ar(p: int, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from statsmodels.tsa.arima_process import ArmaProcess
    rs = np.random.RandomState(seed)
    # stable AR coefficients
    phi = 0.5 * rs.uniform(-1, 1, size=p)
    ar = np.r_[1, -phi]
    ma = np.array([1.0])
    arma = ArmaProcess(ar, ma)
    y = arma.generate_sample(nsample=n, distrvs=rs.normal, scale=1.0)
    return y, phi


def fit_ar_bic(y: np.ndarray, max_p: int) -> int:
    from statsmodels.tsa.ar_model import AutoReg
    bics = []
    for p in range(1, max_p + 1):
        try:
            model = AutoReg(y, lags=p, old_names=False).fit()
            bics.append(model.bic)
        except Exception:
            bics.append(np.inf)
    return int(np.argmin(bics) + 1)


def cv_ar_order(y: np.ndarray, max_p: int, folds: int) -> int:
    # rolling-origin CV: expand train, predict next step
    n = len(y)
    errors = {p: [] for p in range(1, max_p + 1)}
    # choose split points
    split_points = np.linspace(int(n * 0.3), int(n * 0.8), num=folds, dtype=int)
    from statsmodels.tsa.ar_model import AutoReg
    for p in range(1, max_p + 1):
        for t in split_points:
            try:
                model = AutoReg(y[:t], lags=p, old_names=False).fit()
                pred = model.predict(start=t, end=t)
                err = (y[t] - pred[0]) ** 2
                errors[p].append(err)
            except Exception:
                errors[p].append(np.inf)
    avg = {p: np.mean(v) for p, v in errors.items()}
    return int(min(avg, key=avg.get))


def run_ar_order_experiment(cfg: Config, n: int = 240, max_p: int = 10) -> Dict[str, Any]:
    acc = {"MDL": [], "CV": []}
    mse = {"MDL": [], "CV": []}
    runtimes = {"MDL": [], "CV": []}
    fits = {"MDL": [], "CV": []}

    for run in range(cfg.runs):
        seed = int(cfg.seed + run)
        y, true_phi = simulate_ar(p=3, n=n, seed=seed)
        # split outer holdout: use last 20% as test for multi-step forecast
        t_split = int(0.8 * n)
        y_tr, y_te = y[:t_split], y[t_split:]

        t0 = time.time()
        p_mdl = fit_ar_bic(y_tr, max_p)
        runtimes["MDL"].append(time.time() - t0)
        fits["MDL"].append(max_p)

        t0 = time.time()
        p_cv = cv_ar_order(y_tr, max_p, cfg.folds)
        runtimes["CV"].append(time.time() - t0)
        fits["CV"].append(max_p * cfg.folds)

        # evaluate multi-step forecast MSE on test
        from statsmodels.tsa.ar_model import AutoReg
        for meth, p in [("MDL", p_mdl), ("CV", p_cv)]:
            model = AutoReg(y_tr, lags=p, old_names=False).fit()
            preds = model.predict(start=t_split, end=len(y) - 1)
            mse_val = float(np.mean((y_te - preds) ** 2))
            mse[meth].append(mse_val)
        acc["MDL"].append(1.0 if p_mdl == 3 else 0.0)
        acc["CV"].append(1.0 if p_cv == 3 else 0.0)

    m, lo, hi = ci_mean_paired_diff(mse["MDL"], mse["CV"])  # negative favors MDL
    d = cohen_d_paired(mse["MDL"], mse["CV"])  # negative favors MDL
    t_stat, p_t = stats.ttest_rel(mse["MDL"], mse["CV"])
    try:
        w_stat, p_w = stats.wilcoxon(np.array(mse["MDL"]) - np.array(mse["CV"]))
    except Exception:
        p_w = float("nan")

    return {
        "name": "AROrder",
        "metric": "Test MSE",
        "lower_is_better": True,
        "results": mse,
        "select_acc": {"MDL": float(np.mean(acc["MDL"])), "CV": float(np.mean(acc["CV"]))},
        "paired_diff": {"mean": m, "ci_lo": lo, "ci_hi": hi},
        "cohen_d": d,
        "t_pvalue": float(p_t),
        "wilcoxon_pvalue": float(p_w),
        "runtimes": {k: float(np.mean(v)) for k, v in runtimes.items()},
        "fits": {k: int(np.mean(v)) for k, v in fits.items()},
        "params": {"n": n, "true_p": 3, "max_p": max_p},
    }


# -------------------------
# Experiment 5: Decision tree pruning (regression)
# -------------------------

def run_tree_experiment(cfg: Config, max_depth: int = 15) -> Dict[str, Any]:
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target
    results = {"MDL": [], "CV": []}
    runtimes = {"MDL": [], "CV": []}
    fits = {"MDL": [], "CV": []}

    for run in range(cfg.runs):
        seed = int(cfg.seed + run)
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)

        depths = list(range(1, max_depth + 1))

        # CV selection
        t0 = time.time()
        cv = get_kfold(seed, cfg.folds)
        cv_scores = []
        for d in depths:
            tree = DecisionTreeRegressor(max_depth=d, random_state=seed, min_samples_leaf=5)
            score = -np.mean(cross_val_score(tree, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=cfg.n_jobs))
            cv_scores.append(score)
        best_cv_depth = int(np.argmin(cv_scores) + 1)
        runtimes["CV"].append(time.time() - t0)
        fits["CV"].append(len(depths) * cfg.folds)

        # MDL selection: BIC with k = number of leaves
        t0 = time.time()
        mdl_scores = []
        for d in depths:
            tree = DecisionTreeRegressor(max_depth=d, random_state=seed, min_samples_leaf=5)
            tree.fit(X_train, y_train)
            y_pred_tr = tree.predict(X_train)
            k = tree.get_n_leaves()
            mdl_scores.append(enhanced_bic(y_train, y_pred_tr, k))
        best_mdl_depth = int(np.argmin(mdl_scores) + 1)
        runtimes["MDL"].append(time.time() - t0)
        fits["MDL"].append(len(depths))

        # Evaluate
        for meth, d in [("CV", best_cv_depth), ("MDL", best_mdl_depth)]:
            tree = DecisionTreeRegressor(max_depth=d, random_state=seed, min_samples_leaf=5)
            tree.fit(X_train, y_train)
            mse = mean_squared_error(y_test, tree.predict(X_test))
            results[meth].append(float(mse))

    m, lo, hi = ci_mean_paired_diff(results["MDL"], results["CV"])  # negative favors MDL
    d = cohen_d_paired(results["MDL"], results["CV"])  # negative favors MDL
    t_stat, p_t = stats.ttest_rel(results["MDL"], results["CV"]) 
    try:
        w_stat, p_w = stats.wilcoxon(np.array(results["MDL"]) - np.array(results["CV"]))
    except Exception:
        p_w = float("nan")

    return {
        "name": "DecisionTrees",
        "metric": "Test MSE",
        "lower_is_better": True,
        "results": results,
        "paired_diff": {"mean": m, "ci_lo": lo, "ci_hi": hi},
        "cohen_d": d,
        "t_pvalue": float(p_t),
        "wilcoxon_pvalue": float(p_w),
        "runtimes": {k: float(np.mean(v)) for k, v in runtimes.items()},
        "fits": {k: int(np.mean(v)) for k, v in fits.items()},
    }


# -------------------------
# Orchestration, analysis, and plotting
# -------------------------

def run_selected_tasks(cfg: Config, task: str) -> List[Dict[str, Any]]:
    tasks = []
    if task in ("polynomial", "all"):
        print("Running: Polynomial degree selection…")
        tasks.append(run_polynomial_experiment(cfg))
    if task in ("sparse", "all"):
        print("Running: Sparse regression (synthetic)…")
        tasks.append(run_sparse_regression_experiment(cfg))
    if task in ("periodicity", "all"):
        print("Running: Periodicity detection (noisy binary)…")
        tasks.append(run_periodicity_experiment(cfg))
    if task in ("ar", "all"):
        print("Running: AR(p) order selection (time series)…")
        tasks.append(run_ar_order_experiment(cfg))
    if task in ("trees", "all"):
        print("Running: Decision tree pruning (regression)…")
        tasks.append(run_tree_experiment(cfg))
    return tasks


def analyze_and_save(cfg: Config, task_results: List[Dict[str, Any]]):
    # Holm correction across tasks based on t-test p-values
    pvals = {tr["name"]: tr["t_pvalue"] for tr in task_results}
    adj = holm_correction(pvals)
    for tr in task_results:
        tr["holm_t_pvalue"] = adj.get(tr["name"], float("nan"))

    # Save JSON
    payload = {
        "config": asdict(cfg),
        "results": task_results,
    }
    with open(cfg.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON results to {cfg.out_json}")

    # Save CSV summary
    import csv
    with open(cfg.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "metric", "lower_is_better", "mean_diff_MDL_minus_CV", "ci_lo", "ci_hi", "cohen_d", "t_pvalue", "holm_t_pvalue", "wilcoxon_pvalue", "runtime_MDL", "runtime_CV", "fits_MDL", "fits_CV"]) 
        for tr in task_results:
            pdiff = tr["paired_diff"]
            writer.writerow([
                tr["name"], tr["metric"], tr["lower_is_better"], pdiff["mean"], pdiff["ci_lo"], pdiff["ci_hi"], tr["cohen_d"], tr["t_pvalue"], tr.get("holm_t_pvalue", float("nan")), tr["wilcoxon_pvalue"], tr["runtimes"]["MDL"], tr["runtimes"]["CV"], tr["fits"]["MDL"], tr["fits"]["CV"],
            ])
    print(f"Saved CSV summary to {cfg.out_csv}")


def plot_forest_and_runtime(cfg: Config, task_results: List[Dict[str, Any]]):
    # Forest plot of paired differences (MDL - CV)
    names = [tr["name"] for tr in task_results]
    diffs = [tr["paired_diff"]["mean"] for tr in task_results]
    los = [tr["paired_diff"]["ci_lo"] for tr in task_results]
    his = [tr["paired_diff"]["ci_hi"] for tr in task_results]
    lower_better = [tr["lower_is_better"] for tr in task_results]
    metrics = [tr["metric"] for tr in task_results]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(task_results) + 1))
    y = np.arange(len(task_results))
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    colors = ["tab:green" if (d < 0 and lb) or (d > 0 and not lb) else "tab:red" for d, lb in zip(diffs, lower_better)]
    ax.errorbar(diffs, y, xerr=[np.array(diffs) - np.array(los), np.array(his) - np.array(diffs)], fmt="o", color="black", ecolor="black", capsize=3)
    for i, c in enumerate(colors):
        ax.plot(diffs[i], y[i], "o", color=c, markersize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} ({m})" for n, m in zip(names, metrics)])
    ax.set_xlabel("Paired difference: MDL - CV")
    ax.set_title("MDL vs CV: Paired Differences with 95% CI")
    plt.tight_layout()
    plt.savefig(cfg.out_plot, dpi=150)
    print(f"Saved forest plot to {cfg.out_plot}")

    # Runtime comparison
    mdls = [tr["runtimes"]["MDL"] for tr in task_results]
    cvs = [tr["runtimes"]["CV"] for tr in task_results]
    x = np.arange(len(task_results))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(8, 0.6 * len(task_results) + 1))
    ax2.barh(x - width / 2, mdls, height=width, label="MDL")
    ax2.barh(x + width / 2, cvs, height=width, label="CV")
    ax2.set_yticks(x)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Average selection runtime (s)")
    ax2.set_title("Selection Runtime: MDL vs CV")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_runtime_plot, dpi=150)
    print(f"Saved runtime plot to {cfg.out_runtime_plot}")


def main():
    parser = argparse.ArgumentParser(description="MDL vs CV experimental suite")
    parser.add_argument("--task", type=str, default="all", choices=["all", "polynomial", "sparse", "periodicity", "ar", "trees"], help="Which task to run")
    parser.add_argument("--runs", type=int, default=30, help="Number of replicates")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for CV")
    parser.add_argument("--out-json", type=str, default="results.json", help="Path for JSON results")
    parser.add_argument("--out-csv", type=str, default="results_summary.csv", help="Path for CSV summary")
    parser.add_argument("--out-plot", type=str, default="results_forest.png", help="Path for forest plot")
    parser.add_argument("--out-runtime-plot", type=str, default="results_runtime.png", help="Path for runtime plot")
    args = parser.parse_args()

    cfg = Config(runs=args.runs, seed=args.seed, folds=args.folds, n_jobs=args.n_jobs, out_json=args.out_json, out_csv=args.out_csv, out_plot=args.out_plot, out_runtime_plot=args.out_runtime_plot)

    print("🧪 MDL vs Cross-Validation: Experimental Suite")
    print("=" * 60)
    task_results = run_selected_tasks(cfg, args.task)

    print("\nAnalyzing and saving results…")
    analyze_and_save(cfg, task_results)
    plot_forest_and_runtime(cfg, task_results)

    print("\n🎯 Done. Review the JSON/CSV and plots for findings.")


if __name__ == "__main__":
    main()
