"""Sanity checks against the numbers reported in Baxter & King (1993)."""
import numpy as np
from fiscal_model import (calibrate_theta_L, steady_state, run_experiment,
                           public_investment_long_run)

theta_N = 0.58
delta = 0.10
r = 0.065
A = 1.0
N_target = 0.20
s_G = 0.20

# --- Baseline (theta_G=0, all spending as resource-using "public investment"
#     with zero productivity -- economically identical to the old "basic
#     purchases" category, since there's no separate G_B term anymore) ------
theta_L = calibrate_theta_L(theta_N, delta, r, A, 0.0, s_G, 0.0, False, N_target)
print(f"Calibrated theta_L = {theta_L:.4f}")

ss = steady_state(theta_N, delta, r, A, 0.0, s_G, 0.0, False, theta_L)
print("Baseline steady state:")
for k, v in ss.as_dict().items():
    print(f"  {k:10s} = {v}")
print(f"  check N == N_target: {ss.N:.5f} vs {N_target}")
print()

dG_small = 0.001

# --- Permanent, lump-sum: expect long-run multiplier ~1.13-1.16 ------------
exp_perm = run_experiment(theta_N, delta, r, A, 0.0, s_G, s_G + dG_small, 1.0, 0.0,
                           N_target, financing="lump_sum", permanent=True)
print(f"[Permanent, lump-sum] long-run multiplier dY/dG = {exp_perm.multiplier_long_run:.3f}  (paper: ~1.16)")
print(f"[Permanent, lump-sum] impact multiplier (t=0)     = {exp_perm.multiplier_impact:.3f}  (paper: ~0.86)")
print()

# --- Temporary 4-year war, lump-sum: expect impact multiplier ~0.5-0.6 -----
exp_temp4 = run_experiment(theta_N, delta, r, A, 0.0, s_G, s_G + dG_small, 1.0, 0.0,
                            N_target, financing="lump_sum",
                            permanent=False, duration_years=4)
print(f"[Temporary 4yr, lump-sum] impact multiplier (t=0) = {exp_temp4.multiplier_impact:.3f}  (paper Table 3: ~0.56)")
print()

# --- Income tax (GRH-style): expect NEGATIVE long-run multiplier ~ -1.10 --
exp_grh = run_experiment(theta_N, delta, r, A, 0.0, s_G, s_G + dG_small, 1.0, 0.0,
                          N_target, financing="income_tax", permanent=True)
print(f"[Permanent, income tax] long-run multiplier dY/dG = {exp_grh.multiplier_long_run:.3f}  (paper: -1.10)")
assert exp_grh.multiplier_long_run < 0, "income-tax financing should give a negative long-run multiplier"
print()

# --- Public investment (Table 4 style), theta_G=0 baseline must match k_adj/direct=0 at 0 ---
print("Public investment long-run table (Section VI / Table 4 style), theta_G grid:")
theta_G_grid = np.array([0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40])
tbl = public_investment_long_run(theta_N, delta, r, A, s_G, theta_L, False, theta_G_grid, s_IG=0.05)
print(f"{'theta_G':>8} {'direct':>8} {'k_adj':>8} {'both':>8}   (paper col ii, iii, iv)")
for i in range(len(theta_G_grid)):
    print(f"{tbl['theta_G'][i]:8.2f} {tbl['direct'][i]:8.2f} {tbl['k_adj'][i]:8.2f} {tbl['both'][i]:8.2f}")
assert np.all(np.diff(tbl["both"]) > 0), "multiplier should be increasing in theta_G"
print()

# --- theta_G>0 dynamic path must run without error and Kᴳ must build up ----
exp_pub = run_experiment(theta_N, delta, r, A, 0.05, s_G, s_G + 0.05, 0.3, 0.7,
                          N_target, financing="lump_sum", permanent=True)
assert exp_pub.path.KG[1] > exp_pub.path.KG[0], "public capital should build up gradually, not jump"
print(f"[theta_G=0.05, dynamic] long-run multiplier = {exp_pub.multiplier_long_run:.3f}; "
      f"KG path[0:5] = {np.round(exp_pub.path.KG[:5], 3)}")

print()
print("All checks passed.")
