"""
Environmental symmetry and bilateral body plans.

Stage 2 (v2) : Delay-based differential processing with integral kernel readout.

Main formulation (C-level, general integral kernel)
---------------------------------------------------
Sensor inputs from a symmetrically placed pair:

    s_L(t) = x(t - dtau(phi)/2) + eta_L(t)
    s_R(t) = x(t + dtau(phi)/2) + eta_R(t)

Cross-correlation of the two sensor signals:

    C_LR(tau; phi) = (1/T) * integral_t s_L(t) s_R(t - tau) dt.

Readout with a general kernel k(tau):

    m_k(phi) = integral_tau k(tau) C_LR(tau; phi) d tau.

Decompose k(tau) into even and odd parts:
    k_even(tau) = (k(tau) + k(-tau)) / 2
    k_odd(tau)  = (k(tau) - k(-tau)) / 2.

Claim (proved formally, verified numerically):
  If k is purely even (k_odd = 0), then m_k(-phi) = m_k(phi). Therefore the
  sign of phi is not recoverable from m_k.  Only the odd component of k
  conveys information about sign(phi).

Parameterize the kernel by gamma in [0, pi/2]:
    k(tau; gamma) = cos(gamma) * k_even_base(tau) + sin(gamma) * k_odd_base(tau),
with
    k_even_base(tau) = Gaussian(tau; sigma_k),
    k_odd_base(tau)  = (tau / sigma_k) * Gaussian(tau; sigma_k),
each normalized to unit max absolute value.  Processing asymmetry:
    A_proc = sin(gamma) in [0, 1].

A two-tap kernel (A-level, special case for illustration) is also provided:
    k(tau) = cos(gamma) * [delta(tau - d0) + delta(tau + d0)]
           + sin(gamma) * [delta(tau - d0) - delta(tau + d0)].
Used only in a dedicated illustrative figure.

Outputs under ~/Desktop/results/env_symmetry_stage2_v2/
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Config:
    # Signal (bandpass white noise; f in normalized units, 1 time unit =
    # maximum ITD d_base / c_sound).
    fs: float          = 2000.0
    T_sig: float       = 2.0
    f_lo: float        = 1.0        # bandpass lower edge
    f_hi: float        = 8.0        # bandpass upper edge
    noise_sigma: float = 0.20       # sensor noise std

    # Geometry
    d_base: float  = 1.0
    c_sound: float = 1.0

    # Cross-correlation window
    tau_max_frac: float = 1.2       # cover up to 1.2 * max ITD
    n_tau: int          = 161       # number of tau samples in C(tau)

    # Kernel width (as a fraction of max ITD)
    sigma_k_frac: float = 0.4

    # Two-tap kernel (illustrative A-level formulation)
    d0_frac: float = 0.5            # tap position as fraction of max ITD

    # Experiment
    n_trials: int   = 600
    n_bins_mi: int  = 20
    n_gamma: int    = 21
    n_eps: int      = 9
    phi_max: float  = np.pi / 6     # keep ITD ~ sin(phi) ~ phi (linear regime)

    # Sensitivity-analysis sweeps (main knobs)
    noise_sweep: tuple = (0.05, 0.10, 0.20, 0.40)
    sigma_k_sweep: tuple = (0.2, 0.4, 0.6, 0.8)
    phi_max_sweep: tuple = (np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2.5)

    # Reproducibility
    seed: int = 20260418

    def outdir(self) -> Path:
        p = Path.home() / "Desktop" / "results" / "env_symmetry_stage2_v2"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def tau_max(self) -> float:
        return self.tau_max_frac * (self.d_base / self.c_sound)

    def sigma_k(self) -> float:
        return self.sigma_k_frac * (self.d_base / self.c_sound)

    def d0(self) -> float:
        return self.d0_frac * (self.d_base / self.c_sound)


# ===========================================================================
# Sensor configuration
# ===========================================================================

def sensor_pair(d: float,
                eps_shift: float = 0.0,
                eps_tilt: float = 0.0,
                eps_front_back: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.array([-d / 2.0, 0.0])
    x2 = np.array([+d / 2.0, 0.0])
    if eps_tilt != 0.0:
        c, s = np.cos(eps_tilt), np.sin(eps_tilt)
        R = np.array([[c, -s], [s, c]])
        x1, x2 = R @ x1, R @ x2
    if eps_shift != 0.0:
        sh = np.array([0.0, eps_shift])
        x1, x2 = x1 + sh, x2 + sh
    if eps_front_back != 0.0:
        x1 = x1 + np.array([0.0, -eps_front_back / 2.0])
        x2 = x2 + np.array([0.0, +eps_front_back / 2.0])
    return x1, x2


# ===========================================================================
# Signal generation
# ===========================================================================

def generate_source(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    N = int(cfg.T_sig * cfg.fs)
    white = rng.standard_normal(N)
    freqs = np.fft.rfftfreq(N, d=1.0 / cfg.fs)
    mask = (freqs >= cfg.f_lo) & (freqs <= cfg.f_hi)
    X = np.fft.rfft(white) * mask
    x = np.fft.irfft(X, n=N)
    std = np.std(x)
    return x / std if std > 0 else x


def fractional_delay(x: np.ndarray, d_samples: float) -> np.ndarray:
    N = len(x)
    t = np.arange(N) - d_samples
    return np.interp(t, np.arange(N), x, left=0.0, right=0.0)


def sensor_signals(phi: float, x1: np.ndarray, x2: np.ndarray,
                   src: np.ndarray, cfg: Config,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = np.array([np.sin(phi), np.cos(phi)])
    tau1 = (x1 @ u) / cfg.c_sound
    tau2 = (x2 @ u) / cfg.c_sound
    s1 = fractional_delay(src, tau1 * cfg.fs)
    s2 = fractional_delay(src, tau2 * cfg.fs)
    s1 = s1 + rng.normal(0.0, cfg.noise_sigma, size=s1.shape)
    s2 = s2 + rng.normal(0.0, cfg.noise_sigma, size=s2.shape)
    return s1, s2


# ===========================================================================
# Cross-correlation on a dense tau grid
# ===========================================================================

def cross_correlation(sL: np.ndarray, sR: np.ndarray,
                      taus: np.ndarray, cfg: Config) -> np.ndarray:
    """Return C_LR(tau) at each tau in taus, defined as
       (1/N) sum_t s_L[t] * s_R[t - tau_samples].
    """
    N = len(sL)
    c_full = np.correlate(sL, sR, mode="full") / N
    lags = np.arange(-(N - 1), N) / cfg.fs
    return np.interp(taus, lags, c_full, left=0.0, right=0.0)


# ===========================================================================
# Kernels: general integral form (main), and two-tap form (illustrative)
# ===========================================================================

def gaussian_even(taus: np.ndarray, sigma: float) -> np.ndarray:
    g = np.exp(-(taus ** 2) / (2 * sigma ** 2))
    return g / np.max(np.abs(g))


def gaussian_odd(taus: np.ndarray, sigma: float) -> np.ndarray:
    g = (taus / sigma) * np.exp(-(taus ** 2) / (2 * sigma ** 2))
    denom = np.max(np.abs(g))
    return g / denom if denom > 0 else g


def kernel_integral(taus: np.ndarray, gamma: float,
                    sigma: float) -> np.ndarray:
    """k(tau; gamma) = cos(gamma) * k_even_base(tau)
                     + sin(gamma) * k_odd_base(tau)."""
    ke = gaussian_even(taus, sigma)
    ko = gaussian_odd(taus, sigma)
    return np.cos(gamma) * ke + np.sin(gamma) * ko


def kernel_two_tap(taus: np.ndarray, gamma: float, d0: float) -> np.ndarray:
    """Dirac two-tap kernel approximated on the tau grid by the closest tap.
    Returns a zero vector with +/- 1 entries that approximate
       cos(gamma) * [delta(tau - d0) + delta(tau + d0)]
     + sin(gamma) * [delta(tau - d0) - delta(tau + d0)].
    """
    k = np.zeros_like(taus)
    i_plus  = int(np.argmin(np.abs(taus - d0)))
    i_minus = int(np.argmin(np.abs(taus + d0)))
    k[i_plus]  += np.cos(gamma) + np.sin(gamma)
    k[i_minus] += np.cos(gamma) - np.sin(gamma)
    # No normalization: we report MI / accuracy which are invariant to scale.
    return k


def readout(C: np.ndarray, k: np.ndarray, dtau: float) -> float:
    """m = integral k(tau) C(tau) d tau, approximated by Riemann sum."""
    return float(np.dot(k, C) * dtau)


# ===========================================================================
# Diagnostics
# ===========================================================================

def estimate_mi(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    x_edges = np.linspace(x.min() - 1e-9, x.max() + 1e-9, n_bins + 1)
    y_edges = np.linspace(y.min() - 1e-9, y.max() + 1e-9, n_bins + 1)
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Pxy / (Px * Py)
        log_term = np.where(Pxy > 0, np.log(ratio + 1e-30), 0.0)
    return float(np.sum(Pxy * log_term))


def sign_accuracy(phi: np.ndarray, m: np.ndarray) -> float:
    """Binary accuracy of sign(phi) recovery from m.

    For a purely odd kernel, E[m | phi] is odd in phi, so sign(m) aligns with
    sign(phi) up to a global sign determined by the kernel's polarity.  We
    compute the accuracy after optimally aligning the sign of the readout.
    """
    mask = np.abs(phi) > 1e-6
    if mask.sum() == 0 or np.std(m) < 1e-12:
        return 0.5
    p = phi[mask]
    mm = m[mask]
    # The sign map is chosen to maximize agreement.
    acc_plus  = float(np.mean(np.sign(mm) == np.sign(p)))
    acc_minus = float(np.mean(-np.sign(mm) == np.sign(p)))
    return max(acc_plus, acc_minus)


def parity_violation(phi: np.ndarray, m: np.ndarray, n_bins: int = 20) -> float:
    """How much does <m | phi> differ from <m | -phi>?  Returns normalized
    integral of |E[m|phi] + E[m|-phi]|.  (Not |E[m|phi] - E[m|-phi]|: we
    test the symmetric part m(phi)+m(-phi), which vanishes if m is purely
    odd in phi.  But for our purposes we want the ODD part, which should
    vanish if k is purely even.  So we test E[m|phi] - E[m|-phi].)
    """
    order = np.argsort(phi)
    phi_s, m_s = phi[order], m[order]
    # Bin phi into n_bins and compute mean m per bin
    edges = np.linspace(-np.max(np.abs(phi)) - 1e-9,
                        +np.max(np.abs(phi)) + 1e-9, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    idx = np.digitize(phi_s, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            means[b] = m_s[sel].mean()
            counts[b] = sel.sum()
    # Reflection
    odd_part = np.zeros(n_bins)
    for b in range(n_bins):
        # Mirror bin
        b_mirror = n_bins - 1 - b
        odd_part[b] = 0.5 * (means[b] - means[b_mirror])
    scale = np.std(means) if np.std(means) > 1e-12 else 1.0
    return float(np.mean(np.abs(odd_part)) / scale)


# ===========================================================================
# Trial runner
# ===========================================================================

def run_condition(x1: np.ndarray, x2: np.ndarray,
                  gamma: float,
                  cfg: Config,
                  rng: np.random.Generator,
                  kernel_type: str = "integral") -> dict:
    """Run n_trials trials and return diagnostics for this (geometry, gamma,
    kernel_type) condition.
    """
    taus = np.linspace(-cfg.tau_max(), cfg.tau_max(), cfg.n_tau)
    dtau = float(taus[1] - taus[0])
    sigma_k = cfg.sigma_k()
    d0 = cfg.d0()

    if kernel_type == "integral":
        k = kernel_integral(taus, gamma, sigma_k)
    elif kernel_type == "two_tap":
        k = kernel_two_tap(taus, gamma, d0)
    else:
        raise ValueError(f"unknown kernel_type {kernel_type}")

    phi_samples = rng.uniform(-cfg.phi_max, cfg.phi_max, cfg.n_trials)
    m_samples = np.zeros(cfg.n_trials)
    for t in range(cfg.n_trials):
        src = generate_source(cfg, rng)
        sL, sR = sensor_signals(phi_samples[t], x1, x2, src, cfg, rng)
        C = cross_correlation(sL, sR, taus, cfg)
        m_samples[t] = readout(C, k, dtau)

    return {
        "phi": phi_samples,
        "m":   m_samples,
        "mi":   estimate_mi(phi_samples, m_samples, cfg.n_bins_mi),
        "acc":  sign_accuracy(phi_samples, m_samples),
        "corr": float(np.corrcoef(phi_samples, m_samples)[0, 1])
                if np.std(m_samples) > 1e-12 else 0.0,
        "parity_violation": parity_violation(phi_samples, m_samples,
                                             n_bins=cfg.n_bins_mi),
    }


# ===========================================================================
# Sweeps
# ===========================================================================

def sweep_gamma(x1: np.ndarray, x2: np.ndarray, cfg: Config,
                rng: np.random.Generator,
                kernel_type: str = "integral") -> dict:
    gammas = np.linspace(0.0, np.pi / 2, cfg.n_gamma)
    mi   = np.zeros(cfg.n_gamma)
    acc  = np.zeros(cfg.n_gamma)
    corr = np.zeros(cfg.n_gamma)
    pv   = np.zeros(cfg.n_gamma)
    for k_idx, g in enumerate(gammas):
        res = run_condition(x1, x2, g, cfg, rng, kernel_type=kernel_type)
        mi[k_idx]   = res["mi"]
        acc[k_idx]  = res["acc"]
        corr[k_idx] = res["corr"]
        pv[k_idx]   = res["parity_violation"]
    return {
        "gamma": gammas,
        "A_proc": np.sin(gammas),
        "mi": mi,
        "acc": acc,
        "corr": corr,
        "parity_violation": pv,
        "kernel_type": kernel_type,
    }


def sweep_joint(asym_type: str,
                eps_values: np.ndarray,
                cfg: Config,
                rng: np.random.Generator,
                kernel_type: str = "integral") -> dict:
    gammas = np.linspace(0.0, np.pi / 2, cfg.n_gamma)
    MI  = np.zeros((len(eps_values), cfg.n_gamma))
    ACC = np.zeros_like(MI)
    for i, eps in enumerate(eps_values):
        kw = {"eps_shift": 0.0, "eps_tilt": 0.0, "eps_front_back": 0.0}
        kw[f"eps_{asym_type}"] = float(eps)
        x1, x2 = sensor_pair(cfg.d_base, **kw)
        for j, g in enumerate(gammas):
            res = run_condition(x1, x2, g, cfg, rng, kernel_type=kernel_type)
            MI[i, j]  = res["mi"]
            ACC[i, j] = res["acc"]
    return {
        "eps": eps_values,
        "gamma": gammas,
        "A_proc": np.sin(gammas),
        "MI": MI,
        "ACC": ACC,
        "kernel_type": kernel_type,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_gamma_sweep(res: dict, title: str, outdir: Path, tag: str) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(res["A_proc"], res["mi"], "-o", markersize=4)
    axes[0].set_xlabel("A_proc")
    axes[0].set_ylabel("I(phi ; m)  [nats]")
    axes[0].set_title("Mutual information")
    axes[0].grid(alpha=0.3)

    axes[1].plot(res["A_proc"], res["acc"], "-o", markersize=4, color="C1")
    axes[1].axhline(0.5, color="k", linestyle="--", alpha=0.5,
                    label="chance")
    axes[1].set_xlabel("A_proc")
    axes[1].set_ylabel("sign accuracy")
    axes[1].set_title("Sign recovery")
    axes[1].set_ylim(0.3, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(res["A_proc"], res["corr"], "-o", markersize=4, color="C2")
    axes[2].axhline(0.0, color="k", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("A_proc")
    axes[2].set_ylabel("corr(phi, m)")
    axes[2].set_title("Linear correlation")
    axes[2].grid(alpha=0.3)

    axes[3].plot(res["A_proc"], res["parity_violation"], "-o",
                 markersize=4, color="C3")
    axes[3].set_xlabel("A_proc")
    axes[3].set_ylabel("parity violation")
    axes[3].set_title("Odd structure of E[m|phi]")
    axes[3].grid(alpha=0.3)

    plt.suptitle(title + f"  (kernel: {res['kernel_type']})")
    plt.tight_layout()
    path = outdir / f"gamma_sweep_{tag}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_joint(res: dict, asym_type: str, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, Z, title in [(axes[0], res["MI"], "MI (nats)"),
                         (axes[1], res["ACC"], "Sign accuracy")]:
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       extent=[res["A_proc"][0], res["A_proc"][-1],
                               res["eps"][0], res["eps"][-1]],
                       cmap="viridis")
        ax.set_xlabel("A_proc")
        ax.set_ylabel(f"eps_{asym_type}")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.suptitle(f"Joint sweep: {asym_type} x processing asymmetry"
                 f"   (kernel: {res['kernel_type']})")
    plt.tight_layout()
    path = outdir / f"joint_{asym_type}_{res['kernel_type']}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_kernel_shapes(cfg: Config, outdir: Path) -> Path:
    taus = np.linspace(-cfg.tau_max(), cfg.tau_max(), cfg.n_tau)
    sigma_k = cfg.sigma_k()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for g in [0.0, np.pi / 4, np.pi / 2]:
        axes[0].plot(taus, kernel_integral(taus, g, sigma_k),
                     label=f"gamma={g:.2f}")
        axes[1].plot(taus, kernel_two_tap(taus, g, cfg.d0()),
                     label=f"gamma={g:.2f}")
    axes[0].set_title("Integral kernel (Gaussian / derivative-of-Gaussian)")
    axes[0].set_xlabel("tau"); axes[0].legend()
    axes[0].axvline(0, color="k", lw=0.5); axes[0].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Two-tap kernel (illustrative)")
    axes[1].set_xlabel("tau"); axes[1].legend()
    axes[1].axvline(0, color="k", lw=0.5); axes[1].axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    path = outdir / "kernel_shapes.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ===========================================================================
# Sensitivity analysis
# ===========================================================================

def run_sensitivity(cfg_base: Config, rng: np.random.Generator) -> dict:
    """Run the gamma sweep at symmetric placement for several values of
    (noise, sigma_k, phi_max) and report MI at gamma = pi/2 and acc at
    gamma = pi/2 (pure odd readout).
    """
    results = {"noise": [], "sigma_k": [], "phi_max": []}

    # noise
    for s in cfg_base.noise_sweep:
        cfg = Config(**{**asdict(cfg_base), "noise_sigma": s})
        x1, x2 = sensor_pair(cfg.d_base)
        r = sweep_gamma(x1, x2, cfg, rng, kernel_type="integral")
        results["noise"].append({
            "noise": s,
            "mi_odd":  float(r["mi"][-1]),
            "acc_odd": float(r["acc"][-1]),
            "mi_even": float(r["mi"][0]),
            "acc_even": float(r["acc"][0]),
        })

    # sigma_k
    for sk in cfg_base.sigma_k_sweep:
        cfg = Config(**{**asdict(cfg_base), "sigma_k_frac": sk})
        x1, x2 = sensor_pair(cfg.d_base)
        r = sweep_gamma(x1, x2, cfg, rng, kernel_type="integral")
        results["sigma_k"].append({
            "sigma_k_frac": sk,
            "mi_odd":  float(r["mi"][-1]),
            "acc_odd": float(r["acc"][-1]),
            "mi_even": float(r["mi"][0]),
            "acc_even": float(r["acc"][0]),
        })

    # phi_max
    for pm in cfg_base.phi_max_sweep:
        cfg = Config(**{**asdict(cfg_base), "phi_max": pm})
        x1, x2 = sensor_pair(cfg.d_base)
        r = sweep_gamma(x1, x2, cfg, rng, kernel_type="integral")
        results["phi_max"].append({
            "phi_max": pm,
            "mi_odd":  float(r["mi"][-1]),
            "acc_odd": float(r["acc"][-1]),
            "mi_even": float(r["mi"][0]),
            "acc_even": float(r["acc"][0]),
        })

    return results


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=600)
    parser.add_argument("--n-gamma",  type=int, default=21)
    parser.add_argument("--n-eps",    type=int, default=9)
    parser.add_argument("--fs",       type=float, default=2000.0)
    parser.add_argument("--T-sig",    type=float, default=2.0)
    parser.add_argument("--noise",    type=float, default=0.20)
    parser.add_argument("--phi-max",  type=float, default=np.pi / 6,
                        help="Narrow default keeps ITD in the linear regime."
                             "  Use pi/3 for a wider range.")
    parser.add_argument("--seed",     type=int,   default=20260418)
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-two-tap", action="store_true",
                        help="Skip the illustrative two-tap kernel sweep")
    args = parser.parse_args()

    cfg = Config(fs=args.fs, T_sig=args.T_sig,
                 noise_sigma=args.noise,
                 n_trials=args.n_trials,
                 n_gamma=args.n_gamma,
                 n_eps=args.n_eps,
                 phi_max=args.phi_max,
                 seed=args.seed)
    outdir = cfg.outdir()
    rng = np.random.default_rng(cfg.seed)

    print("=" * 72)
    print("Stage 2 v2 : integral kernel readout of cross-correlation")
    print("=" * 72)
    print(f"n_trials = {cfg.n_trials}, n_gamma = {cfg.n_gamma}, "
          f"n_eps = {cfg.n_eps}")
    print(f"fs = {cfg.fs}, T_sig = {cfg.T_sig}, noise = {cfg.noise_sigma}")
    print(f"tau_max = {cfg.tau_max():.3f}, sigma_k = {cfg.sigma_k():.3f}, "
          f"d0 = {cfg.d0():.3f}")
    print(f"phi_max = {cfg.phi_max:.3f}")
    print(f"output  = {outdir}")

    # Kernel shapes reference figure
    kp = plot_kernel_shapes(cfg, outdir)
    print(f"\nKernel shapes: {kp}")

    # ----- 1. Main: symmetric placement, integral kernel gamma sweep -----
    print("\n[1] Symmetric placement, integral kernel, gamma sweep")
    x1, x2 = sensor_pair(cfg.d_base)
    res_int = sweep_gamma(x1, x2, cfg, rng, kernel_type="integral")
    print(f"   MI  (even, A_proc=0) : {res_int['mi'][0]:.4f}")
    print(f"   MI  (odd,  A_proc=1) : {res_int['mi'][-1]:.4f}")
    print(f"   acc (even, A_proc=0) : {res_int['acc'][0]:.4f}")
    print(f"   acc (odd,  A_proc=1) : {res_int['acc'][-1]:.4f}")
    p = plot_gamma_sweep(
        res_int,
        "Integral kernel readout: processing asymmetry controls sign recovery",
        outdir, "symmetric_integral")
    print(f"   saved: {p}")

    # ----- 1b. Two-tap (illustrative) -----
    if not args.skip_two_tap:
        print("\n[1b] Symmetric placement, two-tap kernel, gamma sweep")
        res_two = sweep_gamma(x1, x2, cfg, rng, kernel_type="two_tap")
        print(f"   MI (even, A_proc=0) : {res_two['mi'][0]:.4f}")
        print(f"   MI (odd,  A_proc=1) : {res_two['mi'][-1]:.4f}")
        print(f"   acc (even, A_proc=0) : {res_two['acc'][0]:.4f}")
        print(f"   acc (odd,  A_proc=1) : {res_two['acc'][-1]:.4f}")
        p = plot_gamma_sweep(
            res_two,
            "Two-tap kernel readout (special case)",
            outdir, "symmetric_two_tap")
        print(f"   saved: {p}")

    # ----- 2. Joint sweeps -----
    if not args.skip_joint:
        for atype, emax in [("shift",      0.5),
                            ("tilt",       np.pi / 6),
                            ("front_back", 0.5)]:
            print(f"\n[2] Joint sweep: {atype} x A_proc  (integral kernel)")
            eps = np.linspace(-emax, emax, cfg.n_eps)
            res = sweep_joint(atype, eps, cfg, rng, kernel_type="integral")
            print(f"   MI at (eps=0, A_proc=0) : "
                  f"{res['MI'][len(eps)//2, 0]:.4f}")
            print(f"   MI at (eps=0, A_proc=1) : "
                  f"{res['MI'][len(eps)//2, -1]:.4f}")
            print(f"   MI max over grid        : {res['MI'].max():.4f}")
            p = plot_joint(res, atype, outdir)
            print(f"   saved: {p}")

            csv_path = outdir / f"joint_{atype}_integral.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["eps", "A_proc", "MI", "ACC"])
                for i, e in enumerate(res["eps"]):
                    for j, a in enumerate(res["A_proc"]):
                        w.writerow([f"{e:.6f}", f"{a:.6f}",
                                    f"{res['MI'][i, j]:.6e}",
                                    f"{res['ACC'][i, j]:.6e}"])
            print(f"   csv:   {csv_path}")

    # ----- 3. Sensitivity analysis -----
    if not args.skip_sensitivity:
        print("\n[3] Sensitivity analysis (symmetric placement, integral kernel)")
        sens = run_sensitivity(cfg, rng)
        for key, rows in sens.items():
            print(f"   {key}:")
            for row in rows:
                print(f"     {row}")
        sens_path = outdir / "sensitivity.json"
        with open(sens_path, "w") as f:
            json.dump(sens, f, indent=2)
        print(f"   saved: {sens_path}")

    # ----- 4. Summary -----
    summary_path = outdir / "stage2_summary.json"
    summary = {
        "config": asdict(cfg),
        "symmetric_integral_sweep": {
            "gamma":  res_int["gamma"].tolist(),
            "A_proc": res_int["A_proc"].tolist(),
            "mi":     res_int["mi"].tolist(),
            "acc":    res_int["acc"].tolist(),
            "corr":   res_int["corr"].tolist(),
            "parity_violation": res_int["parity_violation"].tolist(),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()

