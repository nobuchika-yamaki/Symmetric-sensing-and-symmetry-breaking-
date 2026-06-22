#!/usr/bin/env python3
"""
Revised Stage 2 analysis for the iScience bilateral symmetry manuscript.

Purpose
-------
This script replaces the single-seed Stage 2 analysis with reviewer-oriented
additional analyses:

1. Multi-seed gamma sweep with mean, SD, and 95% CI.
2. Decomposition of mutual information into:
   - I(phi; m)
   - I(sign(phi); m)
   - I(|phi|; m)
3. Sensitivity of mutual-information estimates to histogram bin number.
4. Endpoint robustness tests for assumptions requested by reviewers:
   - sensor noise
   - kernel width
   - angular range
   - asymmetric source prior
   - non-Gaussian noise
   - different signal spectra
   - different source distributions
   - different sensor spacings
5. Learned decoder control using the full cross-correlation vector C_LR(tau),
   with analysis of the learned even/odd weight components.

The script is standalone and uses only NumPy and Matplotlib.
Outputs are written by default to:
    ~/Desktop/results/env_symmetry_stage2_revised/

Typical commands
----------------
Smoke test:
    python3 -u env_symmetry_stage2_revised_additional.py --mode smoke

Reviewer-facing quick run:
    python3 -u env_symmetry_stage2_revised_additional.py --mode quick

Full run:
    python3 -u env_symmetry_stage2_revised_additional.py --mode full --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Signal
    fs: float = 2000.0
    T_sig: float = 2.0
    f_lo: float = 1.0
    f_hi: float = 8.0
    spectrum_mode: str = "baseline"  # baseline, low_band, high_band, broad_band
    source_distribution: str = "uniform"  # uniform, central_cluster, edge_cluster
    phi_prior: str = "symmetric"  # symmetric, left_heavy, right_heavy

    # Noise
    noise_sigma: float = 0.20
    noise_model: str = "gaussian"  # gaussian, laplace, student_t

    # Geometry
    d_base: float = 1.0
    c_sound: float = 1.0
    eps_shift: float = 0.0
    eps_tilt: float = 0.0
    eps_front_back: float = 0.0

    # Cross-correlation and kernels
    tau_max_frac: float = 1.2
    n_tau: int = 161
    sigma_k_frac: float = 0.4
    d0_frac: float = 0.5

    # Experiment
    n_trials: int = 600
    n_gamma: int = 21
    n_bins_mi: int = 20
    bin_sensitivity: tuple[int, ...] = (10, 15, 20, 30, 40)
    phi_max: float = math.pi / 6

    # Seeds
    seed_base: int = 20260418
    n_seeds: int = 50

    # Learned decoder
    decoder_trials: int = 1500
    decoder_seeds: int = 10
    decoder_train_frac: float = 0.70
    decoder_l2: float = 1e-2
    decoder_lr: float = 0.25
    decoder_iters: int = 500

    def tau_max(self) -> float:
        return self.tau_max_frac * (self.d_base / self.c_sound)

    def sigma_k(self) -> float:
        return self.sigma_k_frac * (self.d_base / self.c_sound)

    def d0(self) -> float:
        return self.d0_frac * (self.d_base / self.c_sound)


@dataclass
class ModePreset:
    n_trials: int
    n_seeds: int
    n_gamma: int
    n_tau: int
    fs: float
    T_sig: float
    decoder_trials: int
    decoder_seeds: int
    decoder_iters: int


MODE_PRESETS = {
    "smoke": ModePreset(
        n_trials=24,
        n_seeds=2,
        n_gamma=5,
        n_tau=41,
        fs=400.0,
        T_sig=0.35,
        decoder_trials=80,
        decoder_seeds=2,
        decoder_iters=80,
    ),
    "quick": ModePreset(
        n_trials=180,
        n_seeds=8,
        n_gamma=11,
        n_tau=101,
        fs=1000.0,
        T_sig=1.0,
        decoder_trials=600,
        decoder_seeds=4,
        decoder_iters=250,
    ),
    "full": ModePreset(
        n_trials=600,
        n_seeds=50,
        n_gamma=21,
        n_tau=161,
        fs=2000.0,
        T_sig=2.0,
        decoder_trials=1500,
        decoder_seeds=10,
        decoder_iters=500,
    ),
}


# =============================================================================
# Utilities
# =============================================================================

START_TIME = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:9.2f}s] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonable_cfg(cfg: Config) -> dict:
    d = asdict(cfg)
    d["bin_sensitivity"] = list(cfg.bin_sensitivity)
    return d


def stable_hash(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def seed_for(cfg: Config, seed_index: int, tag: str) -> int:
    h = int(hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8], 16)
    return int((cfg.seed_base + 10007 * seed_index + h) % (2**32 - 1))


def ci95(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(values, axis=axis))
    return 1.96 * np.std(values, axis=axis, ddof=1) / math.sqrt(n)


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# Geometry and signal generation
# =============================================================================


def sensor_pair(
    d: float,
    eps_shift: float = 0.0,
    eps_tilt: float = 0.0,
    eps_front_back: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.array([-d / 2.0, 0.0], dtype=float)
    x2 = np.array([+d / 2.0, 0.0], dtype=float)

    if eps_tilt != 0.0:
        c, s = np.cos(eps_tilt), np.sin(eps_tilt)
        R = np.array([[c, -s], [s, c]], dtype=float)
        x1, x2 = R @ x1, R @ x2

    if eps_shift != 0.0:
        sh = np.array([0.0, eps_shift], dtype=float)
        x1, x2 = x1 + sh, x2 + sh

    if eps_front_back != 0.0:
        x1 = x1 + np.array([0.0, -eps_front_back / 2.0], dtype=float)
        x2 = x2 + np.array([0.0, +eps_front_back / 2.0], dtype=float)

    return x1, x2


def spectrum_edges(cfg: Config) -> tuple[float, float]:
    if cfg.spectrum_mode == "baseline":
        return cfg.f_lo, cfg.f_hi
    if cfg.spectrum_mode == "low_band":
        return 0.5, 4.0
    if cfg.spectrum_mode == "high_band":
        return 4.0, 16.0
    if cfg.spectrum_mode == "broad_band":
        return 0.5, 20.0
    raise ValueError(f"unknown spectrum_mode: {cfg.spectrum_mode}")


def generate_source(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    N = max(8, int(round(cfg.T_sig * cfg.fs)))
    white = rng.standard_normal(N)
    freqs = np.fft.rfftfreq(N, d=1.0 / cfg.fs)
    f_lo, f_hi = spectrum_edges(cfg)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        # Fall back to all nonzero frequencies for very small smoke settings.
        mask = freqs > 0
    X = np.fft.rfft(white) * mask
    x = np.fft.irfft(X, n=N)
    std = np.std(x)
    return x / std if std > 1e-12 else x


def fractional_delay(x: np.ndarray, d_samples: float) -> np.ndarray:
    N = len(x)
    t = np.arange(N, dtype=float) - d_samples
    return np.interp(t, np.arange(N, dtype=float), x, left=0.0, right=0.0)


def noise_vector(cfg: Config, rng: np.random.Generator, size: int) -> np.ndarray:
    if cfg.noise_model == "gaussian":
        return rng.normal(0.0, cfg.noise_sigma, size=size)
    if cfg.noise_model == "laplace":
        # Laplace variance = 2 b^2, so b is scaled to match requested std.
        b = cfg.noise_sigma / math.sqrt(2.0)
        return rng.laplace(0.0, b, size=size)
    if cfg.noise_model == "student_t":
        # df=3 has variance 3; scale to requested std.
        return rng.standard_t(df=3, size=size) * (cfg.noise_sigma / math.sqrt(3.0))
    raise ValueError(f"unknown noise_model: {cfg.noise_model}")


def sensor_signals(
    phi: float,
    x1: np.ndarray,
    x2: np.ndarray,
    src: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.array([np.sin(phi), np.cos(phi)], dtype=float)
    tau1 = (x1 @ u) / cfg.c_sound
    tau2 = (x2 @ u) / cfg.c_sound
    s1 = fractional_delay(src, tau1 * cfg.fs)
    s2 = fractional_delay(src, tau2 * cfg.fs)
    s1 = s1 + noise_vector(cfg, rng, len(s1))
    s2 = s2 + noise_vector(cfg, rng, len(s2))
    return s1, s2


def cross_correlation(sL: np.ndarray, sR: np.ndarray, taus: np.ndarray, cfg: Config) -> np.ndarray:
    N = len(sL)
    c_full = np.correlate(sL, sR, mode="full") / N
    lags = np.arange(-(N - 1), N, dtype=float) / cfg.fs
    return np.interp(taus, lags, c_full, left=0.0, right=0.0)


# =============================================================================
# Source distributions
# =============================================================================


def sample_phi(cfg: Config, n: int, rng: np.random.Generator) -> np.ndarray:
    pm = cfg.phi_max

    if cfg.source_distribution == "uniform":
        mag = rng.uniform(0.0, pm, size=n)
    elif cfg.source_distribution == "central_cluster":
        mag = np.abs(rng.normal(loc=0.25 * pm, scale=0.12 * pm, size=n))
        mag = np.clip(mag, 0.0, pm)
    elif cfg.source_distribution == "edge_cluster":
        mag = np.abs(rng.normal(loc=0.75 * pm, scale=0.12 * pm, size=n))
        mag = np.clip(mag, 0.0, pm)
    else:
        raise ValueError(f"unknown source_distribution: {cfg.source_distribution}")

    if cfg.phi_prior == "symmetric":
        p_positive = 0.5
    elif cfg.phi_prior == "left_heavy":
        p_positive = 0.30
    elif cfg.phi_prior == "right_heavy":
        p_positive = 0.70
    else:
        raise ValueError(f"unknown phi_prior: {cfg.phi_prior}")

    signs = np.where(rng.random(n) < p_positive, 1.0, -1.0)
    # Avoid exact zero labels by imposing a small lower bound on magnitude.
    mag = np.maximum(mag, 1e-6)
    return signs * mag


# =============================================================================
# Kernels and readout
# =============================================================================


def gaussian_even(taus: np.ndarray, sigma: float) -> np.ndarray:
    g = np.exp(-(taus ** 2) / (2.0 * sigma ** 2))
    denom = np.max(np.abs(g))
    return g / denom if denom > 1e-12 else g


def gaussian_odd(taus: np.ndarray, sigma: float) -> np.ndarray:
    g = (taus / sigma) * np.exp(-(taus ** 2) / (2.0 * sigma ** 2))
    denom = np.max(np.abs(g))
    return g / denom if denom > 1e-12 else g


def kernel_integral(taus: np.ndarray, gamma: float, sigma: float) -> np.ndarray:
    return np.cos(gamma) * gaussian_even(taus, sigma) + np.sin(gamma) * gaussian_odd(taus, sigma)


def kernel_two_tap(taus: np.ndarray, gamma: float, d0: float) -> np.ndarray:
    k = np.zeros_like(taus)
    i_plus = int(np.argmin(np.abs(taus - d0)))
    i_minus = int(np.argmin(np.abs(taus + d0)))
    k[i_plus] += np.cos(gamma) + np.sin(gamma)
    k[i_minus] += np.cos(gamma) - np.sin(gamma)
    return k


def kernels_for_gammas(cfg: Config, kernel_type: str = "integral") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    taus = np.linspace(-cfg.tau_max(), cfg.tau_max(), cfg.n_tau)
    gammas = np.linspace(0.0, math.pi / 2.0, cfg.n_gamma)
    if kernel_type == "integral":
        K = np.vstack([kernel_integral(taus, g, cfg.sigma_k()) for g in gammas])
    elif kernel_type == "two_tap":
        K = np.vstack([kernel_two_tap(taus, g, cfg.d0()) for g in gammas])
    else:
        raise ValueError(f"unknown kernel_type: {kernel_type}")
    return taus, gammas, K


def apply_kernels(C: np.ndarray, K: np.ndarray, taus: np.ndarray) -> np.ndarray:
    dtau = float(taus[1] - taus[0]) if len(taus) > 1 else 1.0
    # C: trials x tau, K: gamma x tau -> M: trials x gamma
    return C @ K.T * dtau


# =============================================================================
# Diagnostics
# =============================================================================


def discretize_equal_width(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=int)
    edges = np.linspace(lo - 1e-12, hi + 1e-12, n_bins + 1)
    idx = np.digitize(x, edges) - 1
    return np.clip(idx, 0, n_bins - 1)


def mutual_information_discrete(a: np.ndarray, b: np.ndarray, n_a: int | None = None, n_b: int | None = None) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    if n_a is None:
        n_a = int(a.max()) + 1 if len(a) else 0
    if n_b is None:
        n_b = int(b.max()) + 1 if len(b) else 0
    if n_a <= 0 or n_b <= 0:
        return 0.0
    H = np.zeros((n_a, n_b), dtype=float)
    for ai, bi in zip(a, b):
        if 0 <= ai < n_a and 0 <= bi < n_b:
            H[ai, bi] += 1.0
    total = H.sum()
    if total <= 0:
        return 0.0
    Pxy = H / total
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Pxy / (Px * Py)
        log_term = np.where(Pxy > 0, np.log(ratio + 1e-30), 0.0)
    return float(np.sum(Pxy * log_term))


def mi_continuous_pair(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    xb = discretize_equal_width(x, n_bins)
    yb = discretize_equal_width(y, n_bins)
    return mutual_information_discrete(xb, yb, n_bins, n_bins)


def mi_discrete_continuous(label: np.ndarray, y: np.ndarray, n_label: int, n_bins_y: int) -> float:
    if np.std(y) < 1e-12:
        return 0.0
    yb = discretize_equal_width(y, n_bins_y)
    return mutual_information_discrete(label.astype(int), yb, n_label, n_bins_y)


def sign_labels(phi: np.ndarray) -> np.ndarray:
    return (phi > 0).astype(int)


def sign_accuracy(phi: np.ndarray, m: np.ndarray) -> float:
    if np.std(m) < 1e-12:
        return 0.5
    y = sign_labels(phi)
    pred_plus = (m > 0).astype(int)
    acc_plus = np.mean(pred_plus == y)
    acc_minus = np.mean((1 - pred_plus) == y)
    return float(max(acc_plus, acc_minus))


def balanced_sign_accuracy(phi: np.ndarray, m: np.ndarray) -> float:
    if np.std(m) < 1e-12:
        return 0.5
    y = sign_labels(phi)
    pred = (m > 0).astype(int)
    accs = []
    for polarity in (pred, 1 - pred):
        tprs = []
        for cls in (0, 1):
            mask = y == cls
            if np.any(mask):
                tprs.append(np.mean(polarity[mask] == y[mask]))
        accs.append(float(np.mean(tprs)) if tprs else 0.5)
    return float(max(accs))


def pearson_signed_and_abs(phi: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    if np.std(phi) < 1e-12 or np.std(m) < 1e-12:
        return 0.0, 0.0
    corr = float(np.corrcoef(phi, m)[0, 1])
    return corr, abs(corr)


def parity_odd_strength(phi: np.ndarray, m: np.ndarray, n_bins: int) -> float:
    """Normalized magnitude of the odd part of E[m | phi].

    This is not a formal performance measure; it is a diagnostic that checks
    whether the conditional mean has left-right antisymmetric structure.
    """
    phi = np.asarray(phi, dtype=float)
    m = np.asarray(m, dtype=float)
    pm = float(np.max(np.abs(phi)))
    if pm <= 0 or np.std(m) < 1e-12:
        return 0.0
    edges = np.linspace(-pm - 1e-12, pm + 1e-12, n_bins + 1)
    means = np.full(n_bins, np.nan)
    idx = np.clip(np.digitize(phi, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if np.any(sel):
            means[b] = np.mean(m[sel])
    # Fill empty bins by interpolation over valid bins.
    valid = np.isfinite(means)
    if valid.sum() < 2:
        return 0.0
    xs = np.arange(n_bins)
    means = np.interp(xs, xs[valid], means[valid])
    odd = np.zeros(n_bins)
    for b in range(n_bins):
        bm = n_bins - 1 - b
        odd[b] = 0.5 * (means[b] - means[bm])
    scale = np.std(means)
    return float(np.mean(np.abs(odd)) / scale) if scale > 1e-12 else 0.0


def diagnostics_for_readout(phi: np.ndarray, m: np.ndarray, n_bins: int) -> dict[str, float]:
    corr, abs_corr = pearson_signed_and_abs(phi, m)
    labels = sign_labels(phi)
    return {
        "mi_phi_m": mi_continuous_pair(phi, m, n_bins),
        "mi_sign_m": mi_discrete_continuous(labels, m, 2, n_bins),
        "mi_absphi_m": mi_continuous_pair(np.abs(phi), m, n_bins),
        "sign_acc": sign_accuracy(phi, m),
        "balanced_sign_acc": balanced_sign_accuracy(phi, m),
        "corr_phi_m": corr,
        "abs_corr_phi_m": abs_corr,
        "odd_strength": parity_odd_strength(phi, m, n_bins),
    }


# =============================================================================
# Trial data generation and caching
# =============================================================================


def condition_key(cfg: Config, seed_index: int, tag: str) -> str:
    keep = {
        "fs": cfg.fs,
        "T_sig": cfg.T_sig,
        "f_lo": cfg.f_lo,
        "f_hi": cfg.f_hi,
        "spectrum_mode": cfg.spectrum_mode,
        "source_distribution": cfg.source_distribution,
        "phi_prior": cfg.phi_prior,
        "noise_sigma": cfg.noise_sigma,
        "noise_model": cfg.noise_model,
        "d_base": cfg.d_base,
        "c_sound": cfg.c_sound,
        "eps_shift": cfg.eps_shift,
        "eps_tilt": cfg.eps_tilt,
        "eps_front_back": cfg.eps_front_back,
        "tau_max_frac": cfg.tau_max_frac,
        "n_tau": cfg.n_tau,
        "phi_max": cfg.phi_max,
        "n_trials": cfg.n_trials,
        "seed_base": cfg.seed_base,
        "seed_index": seed_index,
        "tag": tag,
    }
    return stable_hash(keep)


def generate_trial_data(
    cfg: Config,
    seed_index: int,
    outdir: Path,
    tag: str,
    resume: bool = True,
    n_trials_override: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg_local = Config(**jsonable_cfg(cfg))
    if n_trials_override is not None:
        cfg_local.n_trials = int(n_trials_override)
    key = condition_key(cfg_local, seed_index, tag)
    cache_dir = ensure_dir(outdir / "cache")
    cache_path = cache_dir / f"trialdata_{tag}_{key}.npz"
    if resume and cache_path.exists():
        z = np.load(cache_path)
        return z["phi"], z["C"], z["taus"]

    rng = np.random.default_rng(seed_for(cfg_local, seed_index, tag))
    taus = np.linspace(-cfg_local.tau_max(), cfg_local.tau_max(), cfg_local.n_tau)
    x1, x2 = sensor_pair(
        cfg_local.d_base,
        eps_shift=cfg_local.eps_shift,
        eps_tilt=cfg_local.eps_tilt,
        eps_front_back=cfg_local.eps_front_back,
    )
    phi = sample_phi(cfg_local, cfg_local.n_trials, rng)
    C = np.zeros((cfg_local.n_trials, cfg_local.n_tau), dtype=float)
    for i in range(cfg_local.n_trials):
        src = generate_source(cfg_local, rng)
        sL, sR = sensor_signals(phi[i], x1, x2, src, cfg_local, rng)
        C[i] = cross_correlation(sL, sR, taus, cfg_local)
    np.savez_compressed(cache_path, phi=phi, C=C, taus=taus)
    return phi, C, taus


# =============================================================================
# Main analyses
# =============================================================================


def run_main_multiseed(cfg: Config, outdir: Path, resume: bool) -> dict:
    log("[main] multi-seed symmetric gamma sweep")
    rows = []
    per_seed = []
    for si in range(cfg.n_seeds):
        log(f"[main] seed {si + 1}/{cfg.n_seeds}")
        phi, C, taus = generate_trial_data(cfg, si, outdir, tag="main", resume=resume)
        _, gammas, K = kernels_for_gammas(cfg, kernel_type="integral")
        M = apply_kernels(C, K, taus)
        seed_rows = []
        for gi, gamma in enumerate(gammas):
            d = diagnostics_for_readout(phi, M[:, gi], cfg.n_bins_mi)
            row = {
                "seed_index": si,
                "gamma": float(gamma),
                "A_proc": float(np.sin(gamma)),
                **d,
            }
            rows.append(row)
            seed_rows.append(row)
        per_seed.append(seed_rows)

    metrics = [
        "mi_phi_m",
        "mi_sign_m",
        "mi_absphi_m",
        "sign_acc",
        "balanced_sign_acc",
        "corr_phi_m",
        "abs_corr_phi_m",
        "odd_strength",
    ]
    gammas = np.linspace(0.0, math.pi / 2.0, cfg.n_gamma)
    A_proc = np.sin(gammas)
    summary_rows = []
    arr_by_metric = {}
    for metric in metrics:
        arr = np.array([[per_seed[si][gi][metric] for gi in range(cfg.n_gamma)] for si in range(cfg.n_seeds)])
        arr_by_metric[metric] = arr
    for gi, gamma in enumerate(gammas):
        row = {"gamma": float(gamma), "A_proc": float(A_proc[gi])}
        for metric in metrics:
            vals = arr_by_metric[metric][:, gi]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = float(ci95(vals, axis=0))
        summary_rows.append(row)

    write_csv(
        outdir / "main_gamma_sweep_per_seed.csv",
        ["seed_index", "gamma", "A_proc"] + metrics,
        ([r["seed_index"], r["gamma"], r["A_proc"]] + [r[m] for m in metrics] for r in rows),
    )
    header = ["gamma", "A_proc"]
    for metric in metrics:
        header += [f"{metric}_mean", f"{metric}_sd", f"{metric}_ci95"]
    write_csv(
        outdir / "main_gamma_sweep_summary.csv",
        header,
        ([r[h] for h in header] for r in summary_rows),
    )
    plot_main_gamma(summary_rows, outdir)

    out = {
        "metrics": metrics,
        "summary_rows": summary_rows,
    }
    save_json(outdir / "main_gamma_sweep_summary.json", out)
    return out


def run_bin_sensitivity(cfg: Config, outdir: Path, resume: bool) -> dict:
    log("[bin] MI bin sensitivity at A_proc endpoints")
    rows = []
    endpoint_gammas = {"even_Aproc0": 0.0, "odd_Aproc1": math.pi / 2.0}
    for n_bins in cfg.bin_sensitivity:
        vals_by_endpoint = {name: [] for name in endpoint_gammas}
        for si in range(cfg.n_seeds):
            phi, C, taus = generate_trial_data(cfg, si, outdir, tag="main", resume=resume)
            for name, gamma in endpoint_gammas.items():
                k = kernel_integral(taus, gamma, cfg.sigma_k())
                m = apply_kernels(C, k[None, :], taus)[:, 0]
                vals_by_endpoint[name].append(diagnostics_for_readout(phi, m, n_bins))
        for name in endpoint_gammas:
            for metric in ("mi_phi_m", "mi_sign_m", "mi_absphi_m", "balanced_sign_acc"):
                vals = np.array([v[metric] for v in vals_by_endpoint[name]], dtype=float)
                rows.append({
                    "n_bins": int(n_bins),
                    "endpoint": name,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95": float(ci95(vals, axis=0)),
                })
    write_csv(
        outdir / "mi_bin_sensitivity.csv",
        ["n_bins", "endpoint", "metric", "mean", "sd", "ci95"],
        ([r["n_bins"], r["endpoint"], r["metric"], r["mean"], r["sd"], r["ci95"]] for r in rows),
    )
    save_json(outdir / "mi_bin_sensitivity.json", {"rows": rows})
    plot_bin_sensitivity(rows, outdir)
    return {"rows": rows}


def cfg_with(cfg: Config, **kwargs) -> Config:
    d = jsonable_cfg(cfg)
    d.update(kwargs)
    if isinstance(d.get("bin_sensitivity"), list):
        d["bin_sensitivity"] = tuple(d["bin_sensitivity"])
    return Config(**d)


def robustness_conditions(cfg: Config) -> list[tuple[str, str, Config]]:
    conds: list[tuple[str, str, Config]] = []

    for val in (0.05, 0.10, 0.20, 0.40):
        conds.append(("noise_sigma", f"{val:.2f}", cfg_with(cfg, noise_sigma=val)))
    for val in (0.2, 0.4, 0.6, 0.8):
        conds.append(("sigma_k_frac", f"{val:.2f}", cfg_with(cfg, sigma_k_frac=val)))
    for val in (math.pi / 6, math.pi / 4, math.pi / 3, 0.4 * math.pi):
        conds.append(("phi_max", f"{val:.6f}", cfg_with(cfg, phi_max=val)))
    for val in ("symmetric", "left_heavy", "right_heavy"):
        conds.append(("phi_prior", val, cfg_with(cfg, phi_prior=val)))
    for val in ("gaussian", "laplace", "student_t"):
        conds.append(("noise_model", val, cfg_with(cfg, noise_model=val)))
    for val in ("baseline", "low_band", "high_band", "broad_band"):
        conds.append(("spectrum_mode", val, cfg_with(cfg, spectrum_mode=val)))
    for val in ("uniform", "central_cluster", "edge_cluster"):
        conds.append(("source_distribution", val, cfg_with(cfg, source_distribution=val)))
    for val in (0.5, 1.0, 1.5, 2.0):
        conds.append(("d_base", f"{val:.2f}", cfg_with(cfg, d_base=val)))

    # Deduplicate exact repeats while preserving order.
    seen = set()
    out = []
    for group, value, cc in conds:
        key = (group, value)
        if key not in seen:
            seen.add(key)
            out.append((group, value, cc))
    return out


def run_robustness_endpoints(cfg: Config, outdir: Path, resume: bool) -> dict:
    log("[robustness] endpoint tests for model assumptions")
    endpoint_gammas = {"even_Aproc0": 0.0, "odd_Aproc1": math.pi / 2.0}
    rows = []
    conditions = robustness_conditions(cfg)
    for ci, (group, value, cc) in enumerate(conditions):
        tag = f"robust_{group}_{value}".replace(".", "p").replace("/", "_")
        log(f"[robustness] {ci + 1}/{len(conditions)} {group}={value}")
        endpoint_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in endpoint_gammas}
        for si in range(cc.n_seeds):
            phi, C, taus = generate_trial_data(cc, si, outdir, tag=tag, resume=resume)
            for name, gamma in endpoint_gammas.items():
                k = kernel_integral(taus, gamma, cc.sigma_k())
                m = apply_kernels(C, k[None, :], taus)[:, 0]
                endpoint_metrics[name].append(diagnostics_for_readout(phi, m, cc.n_bins_mi))
        for endpoint, dicts in endpoint_metrics.items():
            for metric in ("mi_phi_m", "mi_sign_m", "mi_absphi_m", "balanced_sign_acc", "sign_acc", "abs_corr_phi_m", "odd_strength"):
                vals = np.array([d[metric] for d in dicts], dtype=float)
                rows.append({
                    "condition_group": group,
                    "condition_value": value,
                    "endpoint": endpoint,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95": float(ci95(vals, axis=0)),
                })
    write_csv(
        outdir / "robustness_endpoints_summary.csv",
        ["condition_group", "condition_value", "endpoint", "metric", "mean", "sd", "ci95"],
        ([r["condition_group"], r["condition_value"], r["endpoint"], r["metric"], r["mean"], r["sd"], r["ci95"]] for r in rows),
    )
    save_json(outdir / "robustness_endpoints_summary.json", {"rows": rows})
    plot_robustness(rows, outdir)
    return {"rows": rows}


# =============================================================================
# Learned decoder control
# =============================================================================


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def fit_logistic_l2(X: np.ndarray, y: np.ndarray, l2: float, lr: float, n_iter: int) -> tuple[np.ndarray, float]:
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    y = y.astype(float)
    for it in range(n_iter):
        pred = sigmoid(X @ w + b)
        err = pred - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(np.mean(err))
        # mild learning-rate decay for stability
        eta = lr / math.sqrt(1.0 + 0.01 * it)
        w -= eta * grad_w
        b -= eta * grad_b
    return w, b


def accuracy_binary(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y.astype(int) == pred.astype(int)))


def balanced_accuracy_binary(y: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for cls in (0, 1):
        mask = y == cls
        if np.any(mask):
            vals.append(np.mean(pred[mask] == y[mask]))
    return float(np.mean(vals)) if vals else 0.5


def even_odd_components(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rev = vec[::-1]
    even = 0.5 * (vec + rev)
    odd = 0.5 * (vec - rev)
    return even, odd


def odd_energy_fraction(vec: np.ndarray) -> float:
    even, odd = even_odd_components(vec)
    e_even = float(np.sum(even ** 2))
    e_odd = float(np.sum(odd ** 2))
    denom = e_even + e_odd
    return e_odd / denom if denom > 1e-12 else 0.0


def run_learned_decoder(cfg: Config, outdir: Path, resume: bool) -> dict:
    log("[decoder] learned full cross-correlation classifier")
    rows = []
    weight_rows = []
    all_weights = []
    all_taus = None
    cfg_dec = cfg_with(cfg, n_trials=cfg.decoder_trials)

    for si in range(cfg.decoder_seeds):
        log(f"[decoder] seed {si + 1}/{cfg.decoder_seeds}")
        phi, C, taus = generate_trial_data(cfg_dec, si, outdir, tag="decoder", resume=resume)
        all_taus = taus
        y = sign_labels(phi)
        rng = np.random.default_rng(seed_for(cfg, si, "decoder_split"))
        idx = rng.permutation(len(y))
        n_train = int(round(cfg.decoder_train_frac * len(y)))
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        X_train_raw, X_test_raw = C[train_idx], C[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test, _, _ = standardize_train_test(X_train_raw, X_test_raw)
        w, b = fit_logistic_l2(X_train, y_train, cfg.decoder_l2, cfg.decoder_lr, cfg.decoder_iters)
        prob_test = sigmoid(X_test @ w + b)
        pred_test = (prob_test >= 0.5).astype(int)
        acc = accuracy_binary(y_test, pred_test)
        bacc = balanced_accuracy_binary(y_test, pred_test)
        odd_frac = odd_energy_fraction(w)

        # Scalar endpoint controls on the same test samples.
        k_even = kernel_integral(taus, 0.0, cfg.sigma_k())
        k_odd = kernel_integral(taus, math.pi / 2.0, cfg.sigma_k())
        m_even = apply_kernels(C[test_idx], k_even[None, :], taus)[:, 0]
        m_odd = apply_kernels(C[test_idx], k_odd[None, :], taus)[:, 0]
        scalar_even_bacc = balanced_sign_accuracy(phi[test_idx], m_even)
        scalar_odd_bacc = balanced_sign_accuracy(phi[test_idx], m_odd)
        scalar_even_misign = mi_discrete_continuous(sign_labels(phi[test_idx]), m_even, 2, cfg.n_bins_mi)
        scalar_odd_misign = mi_discrete_continuous(sign_labels(phi[test_idx]), m_odd, 2, cfg.n_bins_mi)

        rows.append({
            "seed_index": si,
            "full_decoder_acc": acc,
            "full_decoder_balanced_acc": bacc,
            "full_decoder_odd_energy_fraction": odd_frac,
            "scalar_even_balanced_acc": scalar_even_bacc,
            "scalar_odd_balanced_acc": scalar_odd_bacc,
            "scalar_even_mi_sign_m": scalar_even_misign,
            "scalar_odd_mi_sign_m": scalar_odd_misign,
        })
        all_weights.append(w)
        for t, wi in zip(taus, w):
            weight_rows.append([si, float(t), float(wi)])

    metrics = [k for k in rows[0].keys() if k != "seed_index"] if rows else []
    summary = {}
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        summary[m] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci95": float(ci95(vals, axis=0)),
        }

    write_csv(outdir / "learned_decoder_per_seed.csv", ["seed_index"] + metrics, ([r["seed_index"]] + [r[m] for m in metrics] for r in rows))
    write_csv(outdir / "learned_decoder_weights.csv", ["seed_index", "tau", "weight"], weight_rows)
    write_csv(
        outdir / "learned_decoder_summary.csv",
        ["metric", "mean", "sd", "ci95"],
        ([m, summary[m]["mean"], summary[m]["sd"], summary[m]["ci95"]] for m in metrics),
    )
    if all_weights and all_taus is not None:
        W = np.vstack(all_weights)
        plot_decoder_weights(all_taus, W, outdir)
    save_json(outdir / "learned_decoder_summary.json", {"per_seed": rows, "summary": summary})
    return {"per_seed": rows, "summary": summary}


# =============================================================================
# Plotting
# =============================================================================


def plot_with_ci(ax, x: np.ndarray, mean: np.ndarray, ci: np.ndarray, label: str | None = None) -> None:
    ax.plot(x, mean, marker="o", markersize=3, label=label)
    ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)


def plot_main_gamma(summary_rows: list[dict], outdir: Path) -> Path:
    x = np.array([r["A_proc"] for r in summary_rows], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("mi_phi_m", "I(phi; m) [nats]"),
        ("mi_sign_m", "I(sign(phi); m) [nats]"),
        ("mi_absphi_m", "I(|phi|; m) [nats]"),
        ("balanced_sign_acc", "balanced sign accuracy"),
        ("abs_corr_phi_m", "|corr(phi, m)|"),
        ("odd_strength", "odd structure of E[m | phi]"),
    ]
    for ax, (metric, ylabel) in zip(axes.ravel(), panels):
        mean = np.array([r[f"{metric}_mean"] for r in summary_rows], dtype=float)
        ci = np.array([r[f"{metric}_ci95"] for r in summary_rows], dtype=float)
        plot_with_ci(ax, x, mean, ci)
        if metric == "balanced_sign_acc":
            ax.axhline(0.5, linestyle="--", linewidth=1.0)
            ax.set_ylim(0.3, 1.05)
        ax.set_xlabel("A_proc = sin(gamma)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle("Revised main sweep: sign information separated from unsigned information")
    fig.tight_layout()
    path = outdir / "figure_revised_main_gamma_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_bin_sensitivity(rows: list[dict], outdir: Path) -> Path:
    metrics = ["mi_phi_m", "mi_sign_m", "mi_absphi_m"]
    endpoints = ["even_Aproc0", "odd_Aproc1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, metrics):
        for endpoint in endpoints:
            rs = [r for r in rows if r["metric"] == metric and r["endpoint"] == endpoint]
            rs = sorted(rs, key=lambda z: z["n_bins"])
            x = np.array([r["n_bins"] for r in rs], dtype=float)
            y = np.array([r["mean"] for r in rs], dtype=float)
            ci = np.array([r["ci95"] for r in rs], dtype=float)
            plot_with_ci(ax, x, y, ci, label=endpoint)
        ax.set_xlabel("histogram bins")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Mutual-information bin sensitivity")
    fig.tight_layout()
    path = outdir / "figure_mi_bin_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_robustness(rows: list[dict], outdir: Path) -> Path:
    # Compact plot: balanced sign accuracy for even and odd endpoints.
    groups = []
    for r in rows:
        g = r["condition_group"]
        if g not in groups:
            groups.append(g)
    for group in groups:
        rs = [r for r in rows if r["condition_group"] == group and r["metric"] == "balanced_sign_acc"]
        values = []
        labels = []
        for val in [] if not rs else sorted(set(r["condition_value"] for r in rs), key=str):
            labels.append(str(val))
            e = next(r for r in rs if r["condition_value"] == val and r["endpoint"] == "even_Aproc0")
            o = next(r for r in rs if r["condition_value"] == val and r["endpoint"] == "odd_Aproc1")
            values.append((e["mean"], e["ci95"], o["mean"], o["ci95"]))
        if not values:
            continue
        x = np.arange(len(values), dtype=float)
        even_mean = np.array([v[0] for v in values])
        even_ci = np.array([v[1] for v in values])
        odd_mean = np.array([v[2] for v in values])
        odd_ci = np.array([v[3] for v in values])
        fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(values)), 4.5))
        width = 0.35
        ax.bar(x - width / 2, even_mean, width, yerr=even_ci, capsize=3, label="even A_proc=0")
        ax.bar(x + width / 2, odd_mean, width, yerr=odd_ci, capsize=3, label="odd A_proc=1")
        ax.axhline(0.5, linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("balanced sign accuracy")
        ax.set_title(f"Robustness endpoint test: {group}")
        ax.set_ylim(0.3, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / f"figure_robustness_{group}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
    return outdir / "figure_robustness_*.png"


def plot_decoder_weights(taus: np.ndarray, W: np.ndarray, outdir: Path) -> Path:
    w_mean = W.mean(axis=0)
    w_ci = ci95(W, axis=0)
    even, odd = even_odd_components(w_mean)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_with_ci(axes[0], taus, w_mean, w_ci, label="learned weight")
    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].axvline(0.0, linewidth=1.0)
    axes[0].set_xlabel("tau")
    axes[0].set_ylabel("decoder weight")
    axes[0].set_title("Learned full-correlation decoder weight")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(taus, even, marker="o", markersize=2, label="even component")
    axes[1].plot(taus, odd, marker="o", markersize=2, label="odd component")
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].axvline(0.0, linewidth=1.0)
    axes[1].set_xlabel("tau")
    axes[1].set_ylabel("component weight")
    axes[1].set_title("Even/odd decomposition of learned weight")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "figure_learned_decoder_weights.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# =============================================================================
# Output report
# =============================================================================


def make_readme(outdir: Path, cfg: Config, requested_analyses: list[str]) -> None:
    text = f"""# Revised Stage 2 outputs

Generated by: env_symmetry_stage2_revised_additional.py

## Configuration

```json
{json.dumps(jsonable_cfg(cfg), indent=2)}
```

## Analyses requested in this run

{chr(10).join('- ' + a for a in requested_analyses)}

## Key files

- `main_gamma_sweep_summary.csv`: multi-seed mean, SD, and 95% CI across A_proc.
- `main_gamma_sweep_per_seed.csv`: seed-level values for all metrics.
- `figure_revised_main_gamma_sweep.png`: revised main result figure separating sign and unsigned information.
- `mi_bin_sensitivity.csv`: mutual-information bin sensitivity.
- `robustness_endpoints_summary.csv`: endpoint robustness analyses for reviewer-requested assumptions.
- `learned_decoder_summary.csv`: full cross-correlation learned decoder control.
- `learned_decoder_weights.csv`: learned decoder weights across tau.
- `figure_learned_decoder_weights.png`: even/odd decomposition of learned decoder weights.

## Interpretation notes for the manuscript

- `mi_sign_m` is the direct information-theoretic measure for sign recovery.
- `mi_absphi_m` separates unsigned spatial information from directional sign.
- `balanced_sign_acc` should be used when priors are asymmetric because raw accuracy can be biased by class imbalance.
- `corr_phi_m` is reported with its sign, but `abs_corr_phi_m` is the polarity-invariant diagnostic.
- The learned-decoder odd-energy fraction tests whether a decoder trained on the full C_LR(tau) vector develops an odd-like readout structure.
"""
    (outdir / "README_FIRST_REVISED_STAGE2.md").write_text(text, encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revised Stage 2 reviewer analyses")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--outdir", type=str, default=str(Path.home() / "Desktop" / "results" / "env_symmetry_stage2_revised"))
    parser.add_argument("--resume", action="store_true", help="Reuse cached trial data when available")
    parser.add_argument("--seed-base", type=int, default=20260418)

    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-gamma", type=int, default=None)
    parser.add_argument("--n-tau", type=int, default=None)
    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--T-sig", type=float, default=None)
    parser.add_argument("--noise", type=float, default=0.20)
    parser.add_argument("--phi-max", type=float, default=math.pi / 6)
    parser.add_argument("--sigma-k-frac", type=float, default=0.4)

    parser.add_argument("--main", action="store_true", help="Run main multi-seed gamma sweep")
    parser.add_argument("--bin-sensitivity", action="store_true", help="Run MI bin sensitivity")
    parser.add_argument("--robustness", action="store_true", help="Run endpoint robustness tests")
    parser.add_argument("--decoder", action="store_true", help="Run learned decoder control")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    preset = MODE_PRESETS[args.mode]
    cfg = Config(
        fs=args.fs if args.fs is not None else preset.fs,
        T_sig=args.T_sig if args.T_sig is not None else preset.T_sig,
        noise_sigma=args.noise,
        phi_max=args.phi_max,
        sigma_k_frac=args.sigma_k_frac,
        n_trials=args.n_trials if args.n_trials is not None else preset.n_trials,
        n_seeds=args.n_seeds if args.n_seeds is not None else preset.n_seeds,
        n_gamma=args.n_gamma if args.n_gamma is not None else preset.n_gamma,
        n_tau=args.n_tau if args.n_tau is not None else preset.n_tau,
        seed_base=args.seed_base,
        decoder_trials=preset.decoder_trials,
        decoder_seeds=preset.decoder_seeds,
        decoder_iters=preset.decoder_iters,
    )
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    outdir = ensure_dir(Path(args.outdir).expanduser())

    if not (args.main or args.bin_sensitivity or args.robustness or args.decoder or args.all):
        # Default for each mode: all analyses. This avoids an accidental run that produces nothing.
        args.all = True

    requested = []
    log("=" * 78)
    log("Revised Stage 2 analyses for bilateral symmetry manuscript")
    log("=" * 78)
    log(f"mode      : {args.mode}")
    log(f"outdir    : {outdir}")
    log(f"resume    : {args.resume}")
    log(f"n_trials  : {cfg.n_trials}")
    log(f"n_seeds   : {cfg.n_seeds}")
    log(f"n_gamma   : {cfg.n_gamma}")
    log(f"n_tau     : {cfg.n_tau}")
    log(f"fs/T_sig  : {cfg.fs} / {cfg.T_sig}")
    save_json(outdir / "config_revised_stage2.json", jsonable_cfg(cfg))

    if args.all or args.main:
        requested.append("main multi-seed gamma sweep")
        run_main_multiseed(cfg, outdir, resume=args.resume)

    if args.all or args.bin_sensitivity:
        requested.append("mutual-information bin sensitivity")
        run_bin_sensitivity(cfg, outdir, resume=args.resume)

    if args.all or args.robustness:
        requested.append("endpoint robustness tests")
        run_robustness_endpoints(cfg, outdir, resume=args.resume)

    if args.all or args.decoder:
        requested.append("learned full cross-correlation decoder")
        run_learned_decoder(cfg, outdir, resume=args.resume)

    make_readme(outdir, cfg, requested)
    log("=" * 78)
    log(f"Done. First file to open: {outdir / 'README_FIRST_REVISED_STAGE2.md'}")
    log("=" * 78)


if __name__ == "__main__":
    main()
