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

# --- Public investment (Table 4 replication), s_IG=0.05 (paper's own calibration) --
print("Public investment long-run table (Baxter & King Table 4), theta_G grid:")
theta_G_grid = np.array([0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40])
tbl = public_investment_long_run(theta_N, delta, r, A, s_other=0.15, N_target=N_target,
                                  theta_G_grid=theta_G_grid, s_IG=0.05)
print(f"{'theta_G':>8} {'direct':>8} {'k_adj':>8} {'both':>8} {'dC':>8} {'dI':>8}   (paper: ii,iii,iv,v,vi)")
for i in range(len(theta_G_grid)):
    print(f"{tbl['theta_G'][i]:8.2f} {tbl['direct'][i]:8.2f} {tbl['k_adj'][i]:8.2f} "
          f"{tbl['both'][i]:8.2f} {tbl['dC'][i]:8.2f} {tbl['dI'][i]:8.2f}")
assert np.allclose(tbl["direct"], theta_G_grid / 0.05), "direct effect should equal theta_G/s_IG exactly"
assert np.isclose(tbl["k_adj"][3], 1.72, atol=0.01), "k_adj at theta_G=0.05 should match the paper's 1.72 exactly"
assert np.all(np.diff(tbl["both"]) > 0), "multiplier should be increasing in theta_G"
print()

# --- theta_G>0 dynamic path must run without error and Kᴳ must build up ----
exp_pub = run_experiment(theta_N, delta, r, A, 0.05, s_G, s_G + 0.05, 0.3, 0.7,
                          N_target, financing="lump_sum", permanent=True)
assert exp_pub.path.KG[1] > exp_pub.path.KG[0], "public capital should build up gradually, not jump"
print(f"[theta_G=0.05, dynamic] long-run multiplier = {exp_pub.multiplier_long_run:.3f}; "
      f"KG path[0:5] = {np.round(exp_pub.path.KG[:5], 3)}")

print()

# --- Government consumption Gᶜ valued in utility: u = ln(C + phi*Gᶜ) + theta_L ln(1-N) --
# phi=0: Gᶜ is a pure drain, identical to zero-productivity public investment.
for perm in (True, False):
    e_ig = run_experiment(theta_N, delta, r, A, 0.0, s_G, s_G + dG_small, 1.0, 0.0,
                          N_target, financing="lump_sum", permanent=perm)
    e_gc = run_experiment(theta_N, delta, r, A, 0.0, s_G, s_G + dG_small, 0.0, 0.0,
                          N_target, financing="lump_sum", permanent=perm, f_GC=1.0, phi=0.0)
    assert np.isclose(e_gc.multiplier_impact, e_ig.multiplier_impact), "phi=0 Gᶜ should match a pure drain"
    assert np.allclose(e_gc.path.Y, e_ig.path.Y), "phi=0 Gᶜ path should match a pure drain"
# phi=1, lump-sum: Gᶜ is a perfect substitute for C, so C falls one-for-one and nothing else moves.
for perm in (True, False):
    e = run_experiment(theta_N, delta, r, A, 0.0, s_G, s_G + 0.05, 0.0, 0.0,
                       N_target, financing="lump_sum", permanent=perm, f_GC=1.0, phi=1.0)
    assert abs(e.multiplier_impact) < 1e-6 and np.allclose(e.path.Y, 0.0, atol=1e-6), \
        "phi=1 lump-sum Gᶜ should leave output unchanged"
print("[G_C in utility] long-run multiplier by phi (lump-sum, permanent):")
for phi in (0.0, 0.25, 0.5, 0.75, 1.0):
    e = run_experiment(theta_N, delta, r, A, 0.0, s_G, s_G + dG_small, 0.0, 0.0,
                       N_target, financing="lump_sum", permanent=True, f_GC=1.0, phi=phi)
    print(f"  phi={phi:.2f}: long-run {e.multiplier_long_run:+.3f}, impact {e.multiplier_impact:+.3f}")

print()

# --- Two household types: share lam of hand-to-mouth households ---------------
from fiscal_model import steady_state_for_policy, _log_linear_coeffs, _tank_linearization
coef_keys = ["Phi_kk", "Phi_kc", "Phi_kg", "Phi_k_kg", "coef_c", "coef_k", "coef_g", "coef_kg", "coef_g0"]
for dist in (False, True):
    # The numerical linearization used when lam>0 must reproduce the analytic one at lam=0.
    tl = calibrate_theta_L(theta_N, delta, r, A, 0.05, s_G * 0.3, s_G * 0.4, dist, N_target,
                           s_GC=s_G * 0.3, phi=0.5)
    ss0 = steady_state_for_policy(theta_N, delta, r, A, 0.05, s_G, 0.3, 0.4, tl, dist, f_GC=0.3, phi=0.5)
    an, nu = _log_linear_coeffs(ss0), _tank_linearization(ss0)
    assert max(abs(an[k] - nu[k]) for k in coef_keys) < 1e-7, "numerical and analytic linearizations differ"

    lam = 0.3
    tl = calibrate_theta_L(theta_N, delta, r, A, 0.05, s_G * 0.3, s_G * 0.4, dist, N_target,
                           s_GC=s_G * 0.3, phi=0.5, lam=lam)
    s = steady_state_for_policy(theta_N, delta, r, A, 0.05, s_G, 0.3, 0.4, tl, dist,
                                f_GC=0.3, phi=0.5, lam=lam)
    T_ls = 0.0 if dist else s.G
    assert np.isclose(s.N, N_target), "calibration should hit the aggregate hours target"
    assert np.isclose(s.CH, s.w * s.NH + s.GT / lam - T_ls), "hand-to-mouth budget violated"
    assert np.isclose(s.C, lam * s.CH + (1 - lam) * s.CO), "aggregate consumption mismatch"
    assert np.isclose(s.Y, s.C + s.I + s.IG + s.GC), "resource constraint violated"
    if dist:
        assert np.isclose(s.tau_y * (s.Y - lam * s.w * s.NH), s.G), "income tax should fall on optimizers only"
    for perm in (True, False):
        e = run_experiment(theta_N, delta, r, A, 0.05, s_G, s_G + 0.01, 0.3, 0.4, N_target,
                           "income_tax" if dist else "lump_sum", perm, f_GC=0.3, phi=0.5, lam=lam)
        end = 100 * np.log(e.ss_new.Y / e.ss_old.Y) if perm else 0.0
        assert np.isclose(e.path.Y[-1], end, atol=1e-6), "transition should end at the right steady state"
    print(f"[lam={lam}, {'income tax' if dist else 'lump-sum'}] long-run multiplier "
          f"{e.multiplier_long_run:+.3f}; N_H={s.NH:.3f}, N_O={s.NO:.3f}, tau_y={s.tau_y:.3f}")

print()

# --- Policy performance: incomes, inequality, welfare (household_outcomes) -------
from fiscal_model import household_outcomes, welfare_gains, two_group_gini, simulate_transition
for lam in (0.0, 0.3):
    for dist in (False, True, (0.5, 0.2, 0.3), "consumption_tax"):
        tl = calibrate_theta_L(theta_N, delta, r, A, 0.1, 0.04, 0.06, dist, N_target,
                               s_GC=0.1, phi=0.1, lam=lam)
        s = steady_state_for_policy(theta_N, delta, r, A, 0.1, 0.2, 0.2, 0.3, tl, dist,
                                    f_GC=0.5, phi=0.1, lam=lam)
        o = household_outcomes(s, np.zeros((4, 1)))
        assert np.isclose(o["Y"][0], s.Y) and np.isclose(o["N"][0], s.N), "static block should reproduce the steady state"
        p_c = 1 + s.tau_c
        total = lam * o["income_H"][0] + (1 - lam) * o["income_O"][0] if lam > 0 else o["income_O"][0]
        assert np.isclose(total, p_c * s.C + s.I), "disposable incomes should add up to (1+tau_c)C + I"
        assert np.isclose(o["income_O"][0], p_c * s.CO + delta * s.K / (1 - lam)), "optimizer budget violated"
        if lam > 0:
            assert np.isclose(o["income_H"][0], p_c * s.CH), "hand-to-mouth households spend all income"
        assert all(abs(v) < 1e-8 for v in welfare_gains(o, o, 1 / (1 + r), tl, 0.1).values()), \
            "no change should mean no welfare gain"
assert np.isclose(two_group_gini(0.5, 1.0, 3.0), 0.25), "two-group Gini formula"
# Regime switch lump-sum -> income tax: the path should end at the new steady state.
tl = calibrate_theta_L(theta_N, delta, r, A, 0.1, 0.04, 0.06, False, N_target, s_GC=0.1, phi=0.1, lam=0.3)
b = steady_state_for_policy(theta_N, delta, r, A, 0.1, 0.2, 0.2, 0.3, tl, False, f_GC=0.5, phi=0.1, lam=0.3)
c = steady_state_for_policy(theta_N, delta, r, A, 0.1, 0.2, 0.2, 0.3, tl, True, f_GC=0.5, phi=0.1, lam=0.3)
p = simulate_transition(c, np.zeros(1), k0=np.log(b.K / c.K), kg0=np.log(b.KG / c.KG), T_sim=200)
o = household_outcomes(c, p.state)
assert np.isclose(o["Y"][-1], c.Y) and np.isclose(o["CH"][-1], c.CH), "path should end at the new steady state"
g = welfare_gains(household_outcomes(b, np.zeros((4, 1))), o, 1 / (1 + r), tl, 0.1)
print(f"[lump-sum -> income tax, lam=0.3] welfare gains incl. transition (%): "
      f"O {g['O']:+.2f}, H {g['H']:+.2f}, all {g['all']:+.2f}")

print()

# --- Tax mix with a consumption tax: the linear path solves the exact equations ------
# to first order, so Euler and capital-accumulation residuals shrink with the square
# of the shock (5x smaller shock -> about 25x smaller residual).
from fiscal_model import _tank_eval
for lam in (0.0, 0.3):
    mix = (0.5, 0.2, 0.3)
    tl = calibrate_theta_L(theta_N, delta, r, A, 0.1, 0.04, 0.06, mix, N_target, s_GC=0.1, phi=0.1, lam=lam)
    s = steady_state_for_policy(theta_N, delta, r, A, 0.1, 0.2, 0.2, 0.3, tl, mix, f_GC=0.5, phi=0.1, lam=lam)
    assert np.isclose(s.T + s.tau_y * (s.Y - lam * s.w * (s.NH if lam > 0 else 0.0)) + s.tau_c * s.C, s.G), \
        "the three taxes should pay for G"
    worst = []
    for eps in (0.01, 0.002):
        p = simulate_transition(s, np.r_[np.full(4, eps), 0.0], T_sim=200)
        res = []
        for t in range(30):
            e0, e1 = _tank_eval(s, *p.state[:, t]), _tank_eval(s, *p.state[:, t + 1])
            X0 = (s.CO * np.exp(p.state[1, t]) + s.phi * s.GC * np.exp(p.state[2, t])) * e0["p"]
            X1 = (s.CO * np.exp(p.state[1, t + 1]) + s.phi * s.GC * np.exp(p.state[2, t + 1])) * e1["p"]
            res += [np.log(X1 / X0) - np.log(e1["R"] / (1 + r)), np.log(e0["Knext"] / s.K) - p.state[0, t + 1]]
        worst.append(np.max(np.abs(res)))
    assert worst[0] / worst[1] > 15, "residuals should shrink quadratically with the shock"
    print(f"[tax mix 50/20/30, lam={lam}] tau_y={s.tau_y:.3f}, tau_c={s.tau_c:.3f}; "
          f"Euler/capital residuals {worst[0]:.1e} -> {worst[1]:.1e}")

print()
print("All checks passed.")
