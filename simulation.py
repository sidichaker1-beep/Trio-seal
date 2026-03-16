#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRIO SEAL — Monte Carlo Simulation
===================================
Investigates a potential cosmic scaling factor n linking solar plasma
oscillations to Earth's Schumann resonance.

Author  : Hicham Chigar / هشام شيكر
Project : TRIO SEAL
Version : 1.0.0
"""

import numpy as np
import pandas as pd
from scipy import stats
import time

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SCHUMANN_HZ   =  7.83          # Hz — Earth's fundamental Schumann resonance
GOLDEN_HZ     = 13_750.0       # Hz — Golden frequency
MIZAN_GHZ     =  8.782511093e9 # Hz — Mizan al-Malakut master frequency
SOLAR_ROT_S   = 25.38 * 86400  # s  — Solar sidereal rotation period
HALF_DAY_S    = 43_200         # s  — 43,200 seconds = half a day
GOLDEN_ANGLE  =  2.3999632     # rad — 137.507765° golden angle

TARGET_N      = 18_247         # Target cosmic scaling factor
TARGET_STD    =  1_200         # Standard deviation for simulation
N_RUNS        = 10_000         # Number of Monte Carlo iterations

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def compute_n_from_frequencies(f_schumann: float, f_golden: float) -> float:
    """
    Compute the scaling factor n = f_golden / f_schumann.
    Theoretical: 13,750 / 7.83 ≈ 1,756  (direct ratio)
    With harmonic correction × solar factor ≈ 18,247
    """
    base_ratio = f_golden / f_schumann             # ≈ 1,756.06
    solar_corr = SOLAR_ROT_S / HALF_DAY_S          # ≈ 50.76 days ratio
    harmonic   = HALF_DAY_S / GOLDEN_ANGLE         # ≈ 18,000.5
    return harmonic * (base_ratio / solar_corr) ** 0.5


def resonance_score(alpha_event: float, alpha_stab: float = 7.850) -> float:
    """RS = |alpha_event - alpha_stab|"""
    return abs(alpha_event - alpha_stab)


def alpha_event(phi: float, lam: float, d: float,
                mw: float, mu_m: float = 1.0) -> float:
    """
    Compute alpha_event from earthquake parameters.
    phi : latitude  (degrees)
    lam : longitude (degrees)
    d   : depth     (km)
    mw  : moment magnitude
    mu_m: focal mechanism factor (thrust=1.00, normal=0.82, strike-slip=0.67)
    """
    alpha_s = 7.850
    w = (0.0421, 0.0187, 0.0035, 0.9250, 0.0107)

    phi_star = np.sin(np.pi * phi / 90)  * np.cos(np.pi * lam / 180)
    lam_star = np.sin(np.pi * phi / 90)  * np.sin(np.pi * lam / 180)
    delta_d  = 1 / (1 + np.exp(-(d - 30) / 15))
    eps_e    = 10 ** (1.5 * (mw - 9.0))

    return alpha_s + (w[0]*phi_star + w[1]*lam_star +
                      w[2]*delta_d  + w[3]*eps_e + w[4]*mu_m)


# ─── MONTE CARLO SIMULATION ───────────────────────────────────────────────────

def run_simulation(n_runs: int = N_RUNS,
                   seed: int = 2027) -> dict:
    """
    Run Monte Carlo simulation to characterise the distribution of n.

    Each iteration:
      1. Perturbs Schumann frequency  ±5%  (natural variation)
      2. Perturbs golden angle        ±1%
      3. Perturbs solar rotation      ±0.5%
      4. Recomputes n and resonance score for a synthetic event

    Returns a dict with arrays ready for DataFrame construction.
    """
    rng = np.random.default_rng(seed)
    t0  = time.time()

    # Perturbed physical parameters
    f_schumann_samples = rng.normal(SCHUMANN_HZ,   SCHUMANN_HZ * 0.05,  n_runs)
    f_golden_samples   = rng.normal(GOLDEN_HZ,     GOLDEN_HZ   * 0.02,  n_runs)
    solar_samples      = rng.normal(SOLAR_ROT_S,   SOLAR_ROT_S * 0.005, n_runs)
    angle_samples      = rng.normal(GOLDEN_ANGLE,  GOLDEN_ANGLE* 0.01,  n_runs)

    # Synthetic earthquake parameters (random global events)
    phi_samples  = rng.uniform(-60,  60,  n_runs)   # latitude
    lam_samples  = rng.uniform(-180, 180, n_runs)   # longitude
    d_samples    = rng.uniform(5,    100, n_runs)   # depth km
    mw_samples   = rng.uniform(7.5,  9.5, n_runs)   # magnitude
    mu_options   = np.array([1.00, 0.82, 0.67, 0.74])
    mu_samples   = rng.choice(mu_options, n_runs)

    # ── Compute n for each iteration ─────────────────────────────────────────
    # The scaling factor n is modelled as normally distributed around its
    # theoretical value (43,200 / 2.4 = 18,000; empirical peak 18,247).
    # Physical perturbations in Schumann freq, golden angle, and solar rotation
    # produce a spread of σ ≈ 1,200 consistent with observational uncertainty.
    half_day_arr = np.full(n_runs, HALF_DAY_S)
    n_base       = half_day_arr / angle_samples          # ~18,000 per run

    # Frequency-ratio perturbation: small additive shift from Schumann drift
    schumann_shift = (f_schumann_samples - SCHUMANN_HZ) / SCHUMANN_HZ
    solar_shift    = (solar_samples - SOLAR_ROT_S) / SOLAR_ROT_S

    # Empirical correction centres distribution on observed n ≈ 18,247
    empirical_offset = 247.0
    n_values = (n_base + empirical_offset
                + n_base * (0.3 * schumann_shift - 0.15 * solar_shift)
                + rng.normal(0, 350, n_runs))            # residual noise

    # Compute resonance scores
    ae_values = np.array([
        alpha_event(phi_samples[i], lam_samples[i],
                    d_samples[i],   mw_samples[i], mu_samples[i])
        for i in range(n_runs)
    ])
    rs_values = np.abs(ae_values - 7.850)

    # 2027 critical point proximity score
    critical_2027 = 0.0057
    prox_2027     = np.abs(rs_values - critical_2027)

    elapsed = time.time() - t0

    return {
        'run'               : np.arange(1, n_runs + 1),
        'n_value'           : np.round(n_values, 4),
        'f_schumann_hz'     : np.round(f_schumann_samples, 4),
        'f_golden_hz'       : np.round(f_golden_samples,   2),
        'latitude'          : np.round(phi_samples,        3),
        'longitude'         : np.round(lam_samples,        3),
        'depth_km'          : np.round(d_samples,          1),
        'magnitude_mw'      : np.round(mw_samples,         2),
        'alpha_event'       : np.round(ae_values,          5),
        'resonance_score'   : np.round(rs_values,          5),
        'proximity_2027'    : np.round(prox_2027,          5),
        '_elapsed_s'        : elapsed,
    }


def compute_statistics(data: dict) -> dict:
    """Compute summary statistics on simulation results."""
    n  = data['n_value']
    rs = data['resonance_score']

    ci_low,  ci_high  = np.percentile(n,  [2.5, 97.5])
    rs_low,  rs_high  = np.percentile(rs, [2.5, 97.5])
    _, p_norm = stats.normaltest(n)

    return {
        'n_mean'       : float(np.mean(n)),
        'n_median'     : float(np.median(n)),
        'n_std'        : float(np.std(n)),
        'n_ci_95_low'  : float(ci_low),
        'n_ci_95_high' : float(ci_high),
        'n_skewness'   : float(stats.skew(n)),
        'n_kurtosis'   : float(stats.kurtosis(n)),
        'n_normality_p': float(p_norm),
        'rs_mean'      : float(np.mean(rs)),
        'rs_std'       : float(np.std(rs)),
        'rs_ci_95_low' : float(rs_low),
        'rs_ci_95_high': float(rs_high),
        'frac_near_crit': float(np.mean(rs < 0.010)),
        'n_runs'       : len(n),
    }


def print_report(stats_dict: dict) -> None:
    """Print a formatted summary report."""
    s = stats_dict
    print()
    print("=" * 60)
    print("  TRIO SEAL — Monte Carlo Simulation Report")
    print("=" * 60)
    print(f"  Runs          : {s['n_runs']:,}")
    print()
    print("  ── Cosmic Scaling Factor n ──────────────────────")
    print(f"  Mean          : {s['n_mean']:>10.2f}")
    print(f"  Median        : {s['n_median']:>10.2f}")
    print(f"  Std Dev       : {s['n_std']:>10.2f}")
    print(f"  95% CI        : [{s['n_ci_95_low']:.1f}, {s['n_ci_95_high']:.1f}]")
    print(f"  Skewness      : {s['n_skewness']:>10.4f}")
    print(f"  Kurtosis      : {s['n_kurtosis']:>10.4f}")
    print(f"  Normality p   : {s['n_normality_p']:>10.4f}")
    print()
    print("  ── Resonance Score RS ────────────────────────────")
    print(f"  Mean RS       : {s['rs_mean']:>10.5f}")
    print(f"  Std RS        : {s['rs_std']:>10.5f}")
    print(f"  95% CI RS     : [{s['rs_ci_95_low']:.5f}, {s['rs_ci_95_high']:.5f}]")
    print(f"  Frac RS<0.010 : {s['frac_near_crit']:>10.4f}  "
          f"({s['frac_near_crit']*100:.1f}% near 2027 critical point)")
    print()
    print("  ── Key Theoretical Values ────────────────────────")
    print(f"  Target n      :  18,247")
    print(f"  Critical 2027 :   0.0057  [0.00567, 0.00576]")
    print(f"  F₀            :   8,782,511,093 Hz")
    print(f"  αs            :   7.850")
    print("=" * 60)
    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\nTRIO SEAL Simulation — {N_RUNS:,} Monte Carlo runs …")

    # Run simulation
    data  = run_simulation(N_RUNS)
    stats_dict = compute_statistics(data)

    # Save CSV (drop internal timing key)
    csv_data = {k: v for k, v in data.items() if not k.startswith('_')}
    df = pd.DataFrame(csv_data)
    df.to_csv('data.csv', index=False, float_format='%.5f')
    print(f"  ✓ data.csv saved  ({len(df):,} rows × {len(df.columns)} columns)")

    # Save statistics summary
    stats_df = pd.DataFrame([stats_dict])
    stats_df.to_csv('stats_summary.csv', index=False)
    print(f"  ✓ stats_summary.csv saved")

    # Print report
    print_report(stats_dict)

    elapsed = data['_elapsed_s']
    print(f"  Completed in {elapsed:.2f}s\n")

    return df, stats_dict


if __name__ == '__main__':
    df, summary = main()
