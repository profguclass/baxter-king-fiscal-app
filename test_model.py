"""Sanity checks against the numbers reported in Baxter & King (1993)."""
import numpy as np
from fiscal_model import (calibrate_theta_L, steady_state, run_experiment,
                           public_investment_long_run)

theta_N = 0.58
delta = 0.10
r = 0.065
A = 1.0
tau = 0.20
s_G = 0.20
N_target = 0.20

theta_L = calibrate_theta_L(theta_N, delta, r, A, tau, s_G, N_target)
print(f"Calibrated theta_L = {theta_L:.4f}")

ss = steady_state(theta_N, delta, r, A, tau, s_G, theta_L)
print("Baseline steady state:")
for k, v in ss.as_dict().items():
    print(f"  {k:10s} = {v:.5f}")
print(f"  check N == N_target: {ss.N:.5f} vs {N_target}")
print()

# --- Permanent increase in G, lump-sum financing: expect long-run multiplier ~1.16,
#     impact multiplier ~0.86 (Baxter-King Table 2 & Table 3, duration=inf column) ---
dG_small = 0.001  # tiny s_G perturbation to approximate the marginal multiplier of eq (16)
exp_perm = run_experiment(theta_N, delta, r, A, tau, s_G, s_G + dG_small, N_target,
                           financing="lump_sum", permanent=True)
print(f"[Permanent, lump-sum] long-run multiplier dY/dG = {exp_perm.multiplier_long_run:.3f}  (paper: 1.16)")
print(f"[Permanent, lump-sum] impact multiplier (t=0)     = {exp_perm.multiplier_impact:.3f}  (paper: ~0.86)")
print(f"  Path[0]: Y={exp_perm.path.Y[0]:.3f}%  C={exp_perm.path.C[0]:.3f}%  I={exp_perm.path.I[0]:.3f}%  N={exp_perm.path.N[0]:.3f}%")
print(f"  Path[-1] (long run): Y={exp_perm.path.Y[-1]:.3f}%  C={exp_perm.path.C[-1]:.3f}%  N={exp_perm.path.N[-1]:.3f}%")
print()

# --- Temporary 4-year war, lump-sum financing: expect impact multiplier ~0.56 ---
exp_temp4 = run_experiment(theta_N, delta, r, A, tau, s_G, s_G + dG_small, N_target,
                            financing="lump_sum", permanent=False, duration_years=4)
print(f"[Temporary 4yr, lump-sum] impact multiplier (t=0) = {exp_temp4.multiplier_impact:.3f}  (paper Table 3: 0.56)")
print()

# Table 3 duration sweep
print("Duration sweep (lump-sum), benchmark column of Table 3:")
for T in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]:
    e = run_experiment(theta_N, delta, r, A, tau, s_G, s_G + dG_small, N_target,
                        financing="lump_sum", permanent=False, duration_years=T)
    print(f"  T={T:3d}  impact multiplier = {e.multiplier_impact:.3f}")

print()
# --- Distortionary (GRH) financing: expect NEGATIVE long-run multiplier ~ -1.10 ---
exp_grh = run_experiment(theta_N, delta, r, A, tau, s_G, s_G + dG_small, N_target,
                          financing="distortionary", permanent=True)
print(f"[Permanent, GRH] long-run multiplier dY/dG = {exp_grh.multiplier_long_run:.3f}  (paper: -1.10)")

exp_grh4 = run_experiment(theta_N, delta, r, A, tau, s_G, s_G + 0.05, N_target,
                           financing="distortionary", permanent=False, duration_years=4)
print("[4yr war, GRH] impact-period path (should show OUTPUT FALLING, unlike lump-sum case):")
print(f"  Y[0]={exp_grh4.path.Y[0]:.3f}%  C[0]={exp_grh4.path.C[0]:.3f}%  I[0]={exp_grh4.path.I[0]:.3f}%  N[0]={exp_grh4.path.N[0]:.3f}%")
print()

# --- Table 2 sensitivity: lower real interest rate -> higher multiplier (1.29) ---
theta_L_lowr = calibrate_theta_L(theta_N, delta, 0.03, A, tau, s_G, N_target)
exp_lowr = run_experiment(theta_N, delta, 0.03, A, tau, s_G, s_G + dG_small, N_target,
                           financing="lump_sum", permanent=True)
print(f"[Lower r=3%] long-run multiplier = {exp_lowr.multiplier_long_run:.3f}  (paper Table 2: 1.29)")

theta_L_lowdelta = calibrate_theta_L(theta_N, 0.06, r, A, tau, s_G, N_target)
exp_lowdelta = run_experiment(theta_N, 0.06, r, A, tau, s_G, s_G + dG_small, N_target,
                               financing="lump_sum", permanent=True)
print(f"[Lower delta=0.06] long-run multiplier = {exp_lowdelta.multiplier_long_run:.3f}  (paper Table 2: 1.12)")

print()
print("Public investment long-run table (Section VI / Table 4 style), theta_G grid:")
theta_G_grid = np.array([0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40])
tbl = public_investment_long_run(theta_N, delta, r, A, tau, s_G, theta_L, theta_G_grid, s_IG=0.05)
print(f"{'theta_G':>8} {'direct':>8} {'k_adj':>8} {'both':>8}   (paper col ii, iii, iv)")
for i in range(len(theta_G_grid)):
    print(f"{tbl['theta_G'][i]:8.2f} {tbl['direct'][i]:8.2f} {tbl['k_adj'][i]:8.2f} {tbl['both'][i]:8.2f}")
