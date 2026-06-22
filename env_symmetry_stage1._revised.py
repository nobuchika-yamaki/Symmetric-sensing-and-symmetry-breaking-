from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    sigma: float = 0.05             # Gaussian noise std for observations
    c_sound: float = 1.0            # signal propagation speed (normalized)
    d_base: float = 1.0             # baseline sensor separation
    n_phi: int = 721                # angular or spatial resolution
    n_eps: int = 41                 # asymmetry sweep resolution
    focal: float = 1.0              # focal parameter for binocular modality
    ell: float = 0.5                # receptive field width for tactile modality
    tactile_range: float = 2.0      # body surface extent (tactile modality)

    def outdir(self) -> Path:
        p = Path.home() / "Desktop" / "results" / "env_symmetry_stage1"
        p.mkdir(parents=True, exist_ok=True)
        return p


# =============================================================================
# Sensor configuration
# =============================================================================

def sensor_pair(
    d: float,
    eps_shift: float = 0.0,
    eps_tilt: float = 0.0,
    eps_front_back: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x1, x2) positions in 2D given three asymmetry parameters.

    Baseline symmetric configuration:
        x1 = (-d/2, 0),  x2 = (+d/2, 0).

    - eps_shift       : common translation along the forward (y) axis.
                        Preserves Z_2 about the y-axis.
    - eps_tilt        : rotation angle (rad) of the pair about the origin.
                        Breaks Z_2 generically.
    - eps_front_back  : differential displacement along the y-axis.
                        x1 shifted by -eps_front_back/2 in y.
                        x2 shifted by +eps_front_back/2 in y.
                        Breaks Z_2 and models owl-like vertical ear asymmetry.
    """
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


# =============================================================================
# Fisher information profiles for three modalities
# =============================================================================

def fisher_auditory(x1, x2, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Auditory horizontal localization via ITD.

    signal    : s = (Delta . u_theta) / c + noise
    parameter : theta in [-pi, pi)
    """
    thetas = np.linspace(-np.pi, np.pi, cfg.n_phi, endpoint=False)
    delta = x2 - x1
    du = np.stack([np.cos(thetas), -np.sin(thetas)], axis=1)
    dsig = (du @ delta) / cfg.c_sound
    I = (dsig ** 2) / (cfg.sigma ** 2)
    return thetas, I


def fisher_binocular(x1, x2, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Binocular disparity for a point source in the half-plane y > 0.

    Parameter : theta in [-pi/2, pi/2) direction from midpoint, at fixed
                reference distance R_REF.  We evaluate Fisher info on theta
                which plays the symmetric role to the auditory case.

    Observation model: the difference of projection angles from each sensor
    to the source.
        phi_i = atan2(x_source_x - x_i_x, x_source_y - x_i_y)
        s = phi_1 - phi_2 + noise.
    """
    R_REF = 5.0
    thetas = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, cfg.n_phi, endpoint=True)

    src_x = R_REF * np.sin(thetas)
    src_y = R_REF * np.cos(thetas)

    phi1 = np.arctan2(src_x - x1[0], src_y - x1[1])
    phi2 = np.arctan2(src_x - x2[0], src_y - x2[1])
    disp = phi1 - phi2

    ddisp = np.gradient(disp, thetas)
    I = (ddisp ** 2) / (cfg.sigma ** 2)
    return thetas, I


def fisher_tactile(x1, x2, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Tactile two-point localization on a 1D body surface along the x-axis.

    signal   : s = G(x_p - x_1[0]) - G(x_p - x_2[0]) + noise
               G(u) = exp(-u^2 / (2 ell^2)).
    parameter: x_p in [-range, range].
    """
    xs = np.linspace(-cfg.tactile_range, cfg.tactile_range, cfg.n_phi, endpoint=True)
    u1 = xs - x1[0]
    u2 = xs - x2[0]
    G1 = np.exp(-(u1 ** 2) / (2 * cfg.ell ** 2))
    G2 = np.exp(-(u2 ** 2) / (2 * cfg.ell ** 2))

    dG1 = G1 * (-(u1) / (cfg.ell ** 2))
    dG2 = G2 * (-(u2) / (cfg.ell ** 2))
    ds = dG1 - dG2
    I = (ds ** 2) / (cfg.sigma ** 2)
    return xs, I


MODALITIES: dict[str, Callable] = {
    "auditory": fisher_auditory,
    "binocular": fisher_binocular,
    "tactile": fisher_tactile,
}


# =============================================================================
# Z_2 equivariance violation index
# =============================================================================

def z2_violation(phi: np.ndarray, I: np.ndarray) -> float:
    """Relative deviation of I from being an even function of phi.

    A(X) = mean_phi |I(phi) - I(-phi)| / mean_phi I(phi).

    A = 0 : Fisher info is Z_2-equivariant (I even in phi).
    A > 0 : Z_2 symmetry is broken.
    """
    I_mirror = np.interp(-phi, phi, I, period=2 * np.pi)
    num = np.mean(np.abs(I - I_mirror))
    den = np.mean(I)
    if den <= 0:
        return 0.0
    return float(num / den)


# =============================================================================
# Sweep
# =============================================================================

@dataclass
class SweepResult:
    modality: str
    asym_type: str
    eps: np.ndarray
    A: np.ndarray
    I_mean: np.ndarray
    phi: np.ndarray
    profiles: np.ndarray


def run_sweep(modality: str, asym_type: str, eps_values: np.ndarray, cfg: Config) -> SweepResult:
    assert modality in MODALITIES
    assert asym_type in ("shift", "tilt", "front_back")

    fisher_fn = MODALITIES[modality]
    A_vals = np.zeros_like(eps_values, dtype=float)
    I_mean = np.zeros_like(eps_values, dtype=float)
    profiles = []
    phi_ref = None

    for k, eps in enumerate(eps_values):
        kw = {"eps_shift": 0.0, "eps_tilt": 0.0, "eps_front_back": 0.0}
        kw[f"eps_{asym_type}"] = eps
        x1, x2 = sensor_pair(cfg.d_base, **kw)
        phi, I = fisher_fn(x1, x2, cfg)
        A_vals[k] = z2_violation(phi, I)
        I_mean[k] = float(I.mean())
        profiles.append(I)
        phi_ref = phi

    return SweepResult(
        modality=modality,
        asym_type=asym_type,
        eps=eps_values,
        A=A_vals,
        I_mean=I_mean,
        phi=phi_ref,
        profiles=np.array(profiles),
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_sweep(res: SweepResult, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(res.eps, res.A, "-o", markersize=3)
    axes[0].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_xlabel(f"eps_{res.asym_type}")
    axes[0].set_ylabel("A(X)  (Z_2 violation, normalized)")
    axes[0].set_title("Z_2 equivariance violation")
    axes[0].grid(alpha=0.3)

    axes[1].plot(res.eps, res.I_mean, "-o", markersize=3, color="C1")
    axes[1].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel(f"eps_{res.asym_type}")
    axes[1].set_ylabel("mean_phi I(phi)")
    axes[1].set_title("Mean Fisher information")
    axes[1].grid(alpha=0.3)

    phi = res.phi
    prof = res.profiles
    eps = res.eps
    extent = [phi[0], phi[-1], eps[0], eps[-1]]
    im = axes[2].imshow(prof, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[2].set_xlabel("phi")
    axes[2].set_ylabel(f"eps_{res.asym_type}")
    axes[2].set_title("I(phi; X) profile")
    plt.colorbar(im, ax=axes[2])

    plt.suptitle(f"{res.modality} | {res.asym_type}")
    plt.tight_layout()
    path = outdir / f"sweep_{res.modality}_{res.asym_type}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_summary_grid(results: list[SweepResult], outdir: Path) -> Path:
    modalities = ["auditory", "binocular", "tactile"]
    asym_types = ["shift", "tilt", "front_back"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex="col")

    for i, mod in enumerate(modalities):
        for j, atype in enumerate(asym_types):
            ax = axes[i, j]
            res = next((r for r in results if r.modality == mod and r.asym_type == atype), None)
            if res is None:
                ax.set_axis_off()
                continue
            ax.plot(res.eps, res.A, "-o", markersize=3)
            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.set_title(f"{mod} / {atype}")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("A(X)")
            if i == 2:
                ax.set_xlabel(f"eps_{atype}")

    plt.suptitle("Z_2 equivariance violation across modalities and asymmetry types")
    plt.tight_layout()
    path = outdir / "summary_A_grid.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# =============================================================================
# CSV / JSON saving
# =============================================================================

def save_results_csv(results: list[SweepResult], outdir: Path) -> Path:
    path = outdir / "sweeps_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modality", "asym_type", "eps", "A", "I_mean"])
        for r in results:
            for e, a, im in zip(r.eps, r.A, r.I_mean):
                w.writerow([
                    r.modality,
                    r.asym_type,
                    f"{float(e):.6f}",
                    f"{float(a):.6e}",
                    f"{float(im):.6e}",
                ])
    return path


def save_summary_json(results: list[SweepResult], cfg: Config, outdir: Path) -> Path:
    path = outdir / "sweeps_summary.json"
    summary = {"config": asdict(cfg), "sweeps": []}
    for r in results:
        mid = len(r.eps) // 2
        summary["sweeps"].append({
            "modality": r.modality,
            "asym_type": r.asym_type,
            "A_at_symmetric": float(r.A[mid]),
            "A_max": float(r.A.max()),
            "A_max_eps": float(r.eps[int(r.A.argmax())]),
            "I_mean_at_symmetric": float(r.I_mean[mid]),
            "I_mean_min": float(r.I_mean.min()),
            "I_mean_max": float(r.I_mean.max()),
        })
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-eps",
        type=int,
        default=41,
        help="Asymmetry sweep resolution (default 41).",
    )
    parser.add_argument(
        "--n-phi",
        type=int,
        default=721,
        help="Angular / spatial resolution (default 721).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Observation noise std (default 0.05).",
    )
    args = parser.parse_args()

    cfg = Config(sigma=args.sigma, n_phi=args.n_phi, n_eps=args.n_eps)
    outdir = cfg.outdir()

    print("=" * 72)
    print("Stage 1 : environmental symmetry and bilateral sensor configurations")
    print("=" * 72)
    print(f"output directory : {outdir}")
    print(f"n_eps            : {cfg.n_eps}")
    print(f"n_phi            : {cfg.n_phi}")
    print(f"sigma            : {cfg.sigma}")

    eps_ranges = {
        "shift": np.linspace(-1.0, 1.0, cfg.n_eps),
        "tilt": np.linspace(-np.pi / 2, np.pi / 2, cfg.n_eps),
        "front_back": np.linspace(-1.0, 1.0, cfg.n_eps),
    }

    results: list[SweepResult] = []
    for mod in MODALITIES:
        for atype, evals in eps_ranges.items():
            print(f"\n[{mod} / {atype}]")
            res = run_sweep(mod, atype, evals, cfg)
            mid = len(evals) // 2
            print(
                f"  A(sym)={res.A[mid]:.6f}   A_max={res.A.max():.6f}"
                f" @ eps={evals[int(res.A.argmax())]:+.4f}"
            )
            print(
                f"  I_mean(sym)={res.I_mean[mid]:.4f}"
                f"   range=[{res.I_mean.min():.4f}, {res.I_mean.max():.4f}]"
            )
            p = plot_sweep(res, outdir)
            print(f"  saved: {p}")
            results.append(res)

    gp = plot_summary_grid(results, outdir)
    print(f"\nSummary grid : {gp}")
    cp = save_results_csv(results, outdir)
    print(f"CSV          : {cp}")
    jp = save_summary_json(results, cfg, outdir)
    print(f"JSON         : {jp}")
    print("\nDone.")


if __name__ == "__main__":
    main()
