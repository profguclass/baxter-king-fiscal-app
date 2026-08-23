"""
Core economics for the Baxter & King (1993, AER) "Fiscal Policy in General
Equilibrium" neoclassical growth model with government purchases, extended
with productive public capital (Section VI of the paper) as a first-class
part of the main model rather than a side extension.

The model:

  Preferences:      u_t = log(C_t) + theta_L * log(L_t),   L_t = 1 - N_t
  Technology:        Y_t = A * K_t^theta_K * KG_t^theta_G * N_t^theta_N
                     (theta_K = 1 - theta_N; KG = public capital stock)
  Capital accum.:    K_{t+1}  = (1-delta) K_t  + I_t
                     KG_{t+1} = (1-delta) KG_t + IG_t
  Resource constr.:  Y_t = C_t + I_t + IG_t                (transfers G_T are not resource-using;
                                                              there is no separate "basic purchases"
                                                              term, since the utility function does
                                                              not value government consumption at all)
  Government budget: tau_t * Y_t = IG_t + G_T,t,   tau_t = (IG_t+G_T,t)/Y_t
                      -- i.e. the tax rate always equals total government spending as a
                      share of output, under BOTH financing rules below.

Two financing rules (both use the SAME tax rate formula above; they differ only in
whether that tax distorts the household's labor/capital margin):

  * "lump_sum"      -- the tax is levied/rebated lump-sum: the household's after-tax
                        factor income is just w_t*N_t + r_t*K_t (no (1-tau_t) wedge).
                        Purely a resource / wealth effect.
  * "income_tax"     -- the household's after-tax factor income is (1-tau_t)*(w_t*N_t +
                        r_t*K_t): a proportional wedge on labor and capital income (this
                        is the paper's original "distortionary"/GRH channel).

Total government spending G_t is split into two shares of output that sum to the
total government-spending ratio s_G = IG/Y + G_T/Y:
  * IG   -- public investment, accumulates into a public-capital stock KG that
            raises the productivity of private capital and labor (theta_G).
  * G_T  -- lump-sum transfers back to households; resource-neutral in aggregate.

Capital (K and KG) accumulates according to the laws of motion above; the
equilibrium has predetermined capital and a forward-looking jump variable
(consumption), solved exactly via eigen-decomposition of the reduced-form
transition matrix (King-Plosser-Rebelo / Blanchard-Kahn method).

All quantities are expressed as *fractions of the time-endowment / output*
(the model's "great ratios"), so the code is scale free: only ratios (shares,
theta_L-implied hours) are pinned down, not absolute price levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Financing = Literal["lump_sum", "income_tax"]


# --------------------------------------------------------------------------
# 1. Supply side (prices and "great ratios"), with public capital KG.
# --------------------------------------------------------------------------

def _supply_side_impl(theta_N: float, delta: float, r: float, A: float, tau: float,
                       theta_G: float, s_IG: float, distortionary: bool, N: float) -> dict:
    theta_K = 1.0 - theta_N
    tax_factor = (1.0 - tau) if distortionary else 1.0
    if tax_factor <= 0:
        raise ValueError("Invalid parameters: (1-tau) must be positive under Income Tax financing.")

    kappa = ((tax_factor * theta_K * A) / (r + delta)) ** (1.0 / (1.0 - theta_K))
    alpha = A * kappa ** theta_K  # will be rescaled by KG^theta_G below
    KG = 0.0
    if theta_G > 0 and s_IG > 0:
        # Fixed point on alpha (=Y/N): KG = s_IG*Y/delta = s_IG*alpha*N/delta. Working in
        # per-N terms (kappa=K/N, KG_ppN=KG/N), dividing the technology Y=A*K^thetaK*KG^thetaG*N^thetaN
        # by N gives alpha=A*kappa^thetaK*KG^thetaG*N^(thetaK+thetaG+thetaN-1)
        # =A*kappa^thetaK*KG_ppN^thetaG*N^thetaG (thetaK+thetaN=1 cancels, but the thetaG
        # exponent on N does NOT cancel -- KG is funded as a share of AGGREGATE output,
        # which scales with N, so an explicit N^thetaG factor is required here).
        alpha_ppN = A * kappa ** theta_K  # Y/N ignoring KG, will multiply by KG_ppN^theta_G * N^theta_G
        alpha_guess = alpha_ppN
        N_pow = N ** theta_G
        for _ in range(200):
            KG_ppN = s_IG * alpha_guess / delta  # KG per unit of N
            base = (tax_factor * theta_K * A * KG_ppN ** theta_G * N_pow) / (r + delta)
            kappa = base ** (1.0 / (1.0 - theta_K))
            alpha_new = A * kappa ** theta_K * KG_ppN ** theta_G * N_pow
            if abs(alpha_new - alpha_guess) < 1e-12:
                alpha_guess = alpha_new
                break
            alpha_guess = alpha_new
        alpha = alpha_guess
        KG = s_IG * alpha / delta  # KG per unit of N
    elif s_IG > 0:
        # theta_G=0: public capital is not productive, but it still physically
        # accumulates from investment (KG=IG/delta at steady state) -- no fixed
        # point needed here since KG^0=1 means it can't feed back into alpha/kappa.
        KG = s_IG * alpha / delta
    w = theta_N * alpha
    q = tax_factor * theta_K * alpha / kappa
    s_I = delta * kappa / alpha
    return dict(theta_K=theta_K, kappa=kappa, alpha=alpha, w=w, q=q, s_I=s_I, KG=KG, tax_factor=tax_factor)


# --------------------------------------------------------------------------
# 2. Steady state (closed form / fixed-point iteration)
# --------------------------------------------------------------------------

@dataclass
class SteadyState:
    theta_N: float
    theta_K: float
    theta_G: float
    delta: float
    r: float
    A: float
    tau: float
    theta_L: float
    s_G: float       # total government spending share (IG+G_T)/Y
    s_IG: float
    s_GT: float
    distortionary: bool
    kappa: float
    alpha: float
    w: float
    q: float
    s_I: float
    s_C: float
    N: float
    L: float
    Y: float
    K: float
    KG: float
    C: float
    I: float
    G: float          # total government spending IG+G_T
    IG: float
    GT: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def calibrate_theta_L(theta_N: float, delta: float, r: float, A: float, theta_G: float,
                       s_IG: float, s_GT: float, distortionary: bool,
                       N_target: float) -> float:
    """Choose theta_L (weight on leisure) so that steady-state labor input equals
    N_target, given the other parameters (Table 1 calibration strategy)."""
    tau = s_IG + s_GT
    N = N_target
    supply = _supply_side_impl(theta_N, delta, r, A, tau, theta_G, s_IG, distortionary, N)
    s_C = 1.0 - supply["s_I"] - s_IG
    if s_C <= 0:
        raise ValueError("Government share too large: steady-state consumption share <= 0.")
    theta_L = supply["tax_factor"] * theta_N * (1.0 - N) / (s_C * N)
    if theta_L <= 0:
        raise ValueError("Implied theta_L <= 0; adjust N_target or the tax/spending settings.")
    return theta_L


def steady_state(theta_N: float, delta: float, r: float, A: float, theta_G: float,
                  s_IG: float, s_GT: float, distortionary: bool,
                  theta_L: float) -> SteadyState:
    """Exact steady state of the model for given parameters and government-spending
    composition (s_IG, s_GT).  tau = s_IG+s_GT always."""
    tau = s_IG + s_GT
    s_G = tau
    # Outer fixed point on N: when theta_G>0 and s_IG>0, the supply side (kappa,
    # alpha, KG) itself depends on N (see _supply_side_impl), while N in turn
    # depends on the supply side's implied consumption share s_C. When theta_G=0
    # or s_IG=0 this converges in a single pass, since supply doesn't depend on N.
    N = 0.3
    supply = None
    for _ in range(200):
        supply = _supply_side_impl(theta_N, delta, r, A, tau, theta_G, s_IG, distortionary, N)
        s_I = supply["s_I"]
        s_C = 1.0 - s_I - s_IG
        if s_C <= 0:
            raise ValueError("Government share too large: steady-state consumption share <= 0.")
        tax_factor = supply["tax_factor"]
        N_new = tax_factor * theta_N / (theta_L * s_C + tax_factor * theta_N)
        if not (0 < N_new < 1):
            raise ValueError(f"Implied labor supply N={N_new:.3f} is outside (0,1).")
        if abs(N_new - N) < 1e-12:
            N = N_new
            break
        N = N_new
    L = 1.0 - N

    Y = supply["alpha"] * N
    K = supply["kappa"] * N
    KG = supply["KG"] * N
    C = s_C * Y
    I = s_I * Y
    IG = s_IG * Y
    GT = s_GT * Y
    G = IG + GT

    return SteadyState(
        theta_N=theta_N, theta_K=supply["theta_K"], theta_G=theta_G, delta=delta, r=r, A=A,
        tau=tau, theta_L=theta_L, s_G=s_G, s_IG=s_IG, s_GT=s_GT,
        distortionary=distortionary, kappa=supply["kappa"], alpha=supply["alpha"],
        w=supply["w"], q=supply["q"], s_I=s_I, s_C=s_C, N=N, L=L, Y=Y, K=K, KG=KG, C=C, I=I,
        G=G, IG=IG, GT=GT,
    )


def steady_state_for_policy(theta_N: float, delta: float, r: float, A: float, theta_G: float,
                             s_G_total: float, f_IG: float, f_GT: float,
                             theta_L: float, distortionary: bool) -> SteadyState:
    """Steady state for a given total government-spending ratio s_G_total, split
    into fixed composition fractions f_IG+f_GT=1 of that total."""
    s_IG, s_GT = s_G_total * f_IG, s_G_total * f_GT
    return steady_state(theta_N, delta, r, A, theta_G, s_IG, s_GT, distortionary, theta_L)


# --------------------------------------------------------------------------
# 3. Log-linear dynamics (dynamic model type)
# --------------------------------------------------------------------------

def _log_linear_coeffs(ss: SteadyState) -> dict:
    """Coefficients of the log-linearized equilibrium system around steady state `ss`.

    Static block:   y_hat_t = y_k*k_hat_t + y_kg*kg_hat_t + y_c*c_hat_t + y_g*g_hat_t
    Capital accum.: k_hat_{t+1} = Phi_kk*k_hat_t + Phi_kc*c_hat_t + Phi_kg*g_hat_t + Phi_k_kg*kg_hat_t
    Euler equation: c_hat_t = coef_c*c_hat_{t+1} + coef_k*k_hat_{t+1} + coef_g*g_hat_{t+1}
                              + coef_kg*kg_hat_{t+1}
    """
    theta_N, theta_K, theta_G = ss.theta_N, ss.theta_K, ss.theta_G
    L = ss.L
    tau = ss.tau
    s_C, s_I, s_IG = ss.s_C, ss.s_I, ss.s_IG
    delta, r = ss.delta, ss.r

    D = 1.0 - theta_N * L
    if ss.distortionary:
        # See derivation note: y_hat*(D - theta_N*L*tau/(1-tau)) = theta_K*k_hat +
        # theta_G*kg_hat - theta_N*L*c_hat - theta_N*L*tau/(1-tau)*g_hat.
        Dp = D - theta_N * L * tau / (1.0 - tau)
        y_k = theta_K / Dp
        y_kg = theta_G / Dp
        y_c = -theta_N * L / Dp
        y_g = -theta_N * L * tau / ((1.0 - tau) * Dp)
        tax_wedge = tau / (1.0 - tau)
    else:
        y_k = theta_K / D
        y_kg = theta_G / D
        y_c = -theta_N * L / D
        y_g = 0.0
        tax_wedge = 0.0

    beta = 1.0 / (1.0 + r)
    q = r + delta
    Omega = beta * q  # = q/(1+r)

    ymk_k = y_k - 1.0
    ymk_c = y_c
    ymk_g = y_g
    ymk_kg = y_kg

    coef_c = 1.0 - Omega * ymk_c - Omega * tax_wedge * y_c
    coef_k = -Omega * ymk_k - Omega * tax_wedge * y_k
    coef_g = -Omega * ymk_g + Omega * tax_wedge * (1.0 - y_g)
    coef_kg = -Omega * ymk_kg - Omega * tax_wedge * y_kg

    Phi_kk = (1.0 - delta) + delta * y_k / s_I
    Phi_kc = delta * (y_c - s_C) / s_I
    Phi_kg = delta * (y_g - s_IG) / s_I
    Phi_k_kg = delta * y_kg / s_I

    return dict(y_k=y_k, y_kg=y_kg, y_c=y_c, y_g=y_g, s_GR=s_IG,
                coef_c=coef_c, coef_k=coef_k, coef_g=coef_g, coef_kg=coef_kg,
                Phi_kk=Phi_kk, Phi_kc=Phi_kc, Phi_kg=Phi_kg, Phi_k_kg=Phi_k_kg)


def _kg_hat_path(delta: float, g_full: np.ndarray, kg0: float) -> np.ndarray:
    """Public capital's own law of motion decouples from the K/C saddle-path system:
    kg_hat_{t+1} = (1-delta)*kg_hat_t + delta*g_hat_t, given the (exogenous) g_hat path.
    Returns T+1 values (kg_hat_0 .. kg_hat_T), the extra period letting Euler-equation
    forcing terms reference kg_hat_{T} = kg_hat one step beyond the g_hat path."""
    T = len(g_full)
    kg = np.zeros(T + 1)
    kg[0] = kg0
    g_ext = np.concatenate([g_full, [0.0]])
    for t in range(T):
        kg[t + 1] = (1.0 - delta) * kg[t] + delta * g_ext[t]
    return kg


def _solve_saddle_path(Phi_kk: float, Phi_kc: float, Phi_kg: float, Phi_k_kg: float,
                        coef_c: float, coef_k: float, coef_g: float, coef_kg: float,
                        k0: float, g_full: np.ndarray, kg_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact saddle-path solution of the linear perfect-foresight system

        k_hat_{t+1} = Phi_kk*k_hat_t + Phi_kc*c_hat_t + Phi_kg*g_hat_t + Phi_k_kg*kg_hat_t
        c_hat_t     = coef_c*c_hat_{t+1} + coef_k*k_hat_{t+1} + coef_g*g_hat_{t+1}
                      + coef_kg*kg_hat_{t+1}

    via eigen-decomposition (King-Plosser-Rebelo / Blanchard-Kahn method), for T periods,
    with kg_hat_t supplied exogenously as a length-(T+1) array (it is solved separately --
    see `_kg_hat_path` -- since its own law of motion does not depend on k_hat or c_hat;
    the extra period lets the Euler-equation forcing reference kg_hat_{t+1} at t=T-1).

    A naive forward "shooting" simulation is numerically unstable here because the
    unstable root is iterated hundreds of times, amplifying floating-point noise
    geometrically. Instead we (i) diagonalize the reduced-form transition matrix, (ii)
    pin the forward-looking (unstable) eigen-component at each date to the exact present
    value of *remaining* future forcing terms (so it never explodes), and (iii)
    propagate the predetermined (stable) eigen-component forward from k_hat_0, which is
    numerically safe since |lambda_stable| < 1.
    """
    M = np.array([[Phi_kk, Phi_kc],
                  [-coef_k * Phi_kk / coef_c, (1.0 - coef_k * Phi_kc) / coef_c]])
    vals, vecs = np.linalg.eig(M)
    if np.max(np.abs(vals.imag)) > 1e-8:
        raise RuntimeError("Complex eigenvalues encountered; parameters imply oscillatory "
                            "dynamics not supported by this simplified solver.")
    vals = vals.real
    vecs = vecs.real
    stable_idx = int(np.argmin(np.abs(vals)))
    unstable_idx = 1 - stable_idx
    lam_s, lam_u = vals[stable_idx], vals[unstable_idx]
    if not (abs(lam_s) < 1.0 < abs(lam_u)):
        raise RuntimeError("Model is not saddle-path stable for these parameters "
                            "(need exactly one root inside and one outside the unit circle).")
    V = vecs[:, [stable_idx, unstable_idx]]
    Vinv = np.linalg.inv(V)

    T = len(g_full)
    g_ext = np.concatenate([g_full, [0.0]])
    kg_now = kg_full[:T]     # kg_hat_t for t=0..T-1
    kg_next = kg_full[1:T + 1]  # kg_hat_{t+1} for t=0..T-1
    Nk = Phi_kg * g_ext[:T] + Phi_k_kg * kg_now
    Nc = (-coef_k * (Phi_kg * g_ext[:T] + Phi_k_kg * kg_now) - coef_g * g_ext[1:T + 1]
          - coef_kg * kg_next) / coef_c
    N = np.stack([Nk, Nc], axis=1)
    n = N @ Vinv.T  # forcing rotated into the eigen-basis: n[:,0]=stable comp., n[:,1]=unstable

    zeta2 = np.zeros(T)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = (n[t, 1] + running) / lam_u
        zeta2[t] = -running

    zeta1 = np.zeros(T)
    zeta1[0] = (k0 - V[0, 1] * zeta2[0]) / V[0, 0]
    for t in range(T - 1):
        zeta1[t + 1] = lam_s * zeta1[t] + n[t, 0]

    z = V @ np.vstack([zeta1, zeta2])
    return z[0, :], z[1, :]


@dataclass
class TransitionPath:
    years: np.ndarray
    Y: np.ndarray  # % deviation from the steady state being linearized around
    C: np.ndarray
    I: np.ndarray
    K: np.ndarray
    KG: np.ndarray
    N: np.ndarray
    W: np.ndarray
    G: np.ndarray
    r_bp: np.ndarray  # real interest rate, deviation in basis points (annualized)


def simulate_transition(ss_ref: SteadyState, g_path: np.ndarray,
                         k0: float = 0.0, kg0: float = 0.0, T_sim: int = 300) -> TransitionPath:
    """Simulate the perfect-foresight transition path in log-deviations from steady
    state `ss_ref`, given an exogenous path of government-spending log-deviations
    `g_path` (padded with zeros afterwards) and initial capital log-deviations k0/kg0."""
    coeffs = _log_linear_coeffs(ss_ref)
    y_k, y_kg, y_c, y_g, s_GR = coeffs["y_k"], coeffs["y_kg"], coeffs["y_c"], coeffs["y_g"], coeffs["s_GR"]

    g_full = np.zeros(T_sim)
    n_g = min(len(g_path), T_sim)
    g_full[:n_g] = g_path[:n_g]

    L = ss_ref.L
    tau = ss_ref.tau

    kg_full_ext = _kg_hat_path(ss_ref.delta, g_full, kg0)  # length T_sim+1
    k, c = _solve_saddle_path(coeffs["Phi_kk"], coeffs["Phi_kc"], coeffs["Phi_kg"], coeffs["Phi_k_kg"],
                               coeffs["coef_c"], coeffs["coef_k"], coeffs["coef_g"], coeffs["coef_kg"],
                               k0, g_full, kg_full_ext)
    kg_full = kg_full_ext[:T_sim]
    y = y_k * k + y_kg * kg_full + y_c * c + y_g * g_full

    if ss_ref.distortionary:
        n = L * (y - c) - (L * tau / (1.0 - tau)) * (g_full - y)
    else:
        n = L * (y - c)
    w = y - n  # wage = MPL, log-deviation

    i = (y - ss_ref.s_C * c - s_GR * g_full) / ss_ref.s_I

    c_pct = 100.0 * c
    y_pct = 100.0 * y
    i_pct = 100.0 * i
    k_pct = 100.0 * k
    kg_pct = 100.0 * kg_full
    n_pct = 100.0 * n
    w_pct = 100.0 * w
    g_pct = 100.0 * g_full

    r_dev_bp = np.zeros_like(c)
    r_dev_bp[:-1] = 10000.0 * (1.0 + ss_ref.r) * (c[1:] - c[:-1])
    r_dev_bp[-1] = r_dev_bp[-2]

    years = np.arange(T_sim)
    return TransitionPath(years=years, Y=y_pct, C=c_pct, I=i_pct, K=k_pct, KG=kg_pct, N=n_pct,
                           W=w_pct, G=g_pct, r_bp=r_dev_bp)


# --------------------------------------------------------------------------
# 4. Convenience: a full "policy experiment" bundling steady states + path
# --------------------------------------------------------------------------

@dataclass
class PolicyExperiment:
    ss_old: SteadyState
    ss_new: SteadyState
    path: Optional[TransitionPath]
    multiplier_long_run: float
    multiplier_impact: Optional[float]


def run_experiment(theta_N: float, delta: float, r: float, A: float, theta_G: float,
                    s_G_old: float, s_G_new: float, f_IG: float, f_GT: float,
                    N_target: float, financing: Financing,
                    permanent: bool, duration_years: int = 4, T_sim: int = 300) -> PolicyExperiment:
    """Full comparative-steady-state + transition-path experiment for a change in total
    government spending from s_G_old to s_G_new, holding the composition fractions
    (f_IG, f_GT) fixed."""
    distortionary = financing == "income_tax"
    s_IG_old, s_GT_old = s_G_old * f_IG, s_G_old * f_GT
    theta_L = calibrate_theta_L(theta_N, delta, r, A, theta_G, s_IG_old, s_GT_old,
                                 distortionary, N_target)

    ss_old = steady_state_for_policy(theta_N, delta, r, A, theta_G, s_G_old, f_IG, f_GT,
                                       theta_L, distortionary)
    ss_new = steady_state_for_policy(theta_N, delta, r, A, theta_G, s_G_new, f_IG, f_GT,
                                       theta_L, distortionary)

    dY = ss_new.Y - ss_old.Y
    dG = ss_new.G - ss_old.G
    multiplier_long_run = dY / dG if abs(dG) > 1e-12 else float("nan")

    if permanent:
        ss_ref = ss_new
        k0 = np.log(ss_old.K / ss_new.K)
        kg0 = np.log(ss_old.KG / ss_new.KG) if ss_new.KG > 0 else 0.0
        g_path = np.zeros(1)
    else:
        ss_ref = ss_old
        k0 = 0.0
        kg0 = 0.0
        dG_frac = (ss_new.G - ss_old.G) / ss_old.G if ss_old.G != 0 else 0.0
        g_level = np.log(1.0 + dG_frac)
        g_path = np.concatenate([np.full(duration_years, g_level), np.zeros(1)])

    path = simulate_transition(ss_ref, g_path, k0=k0, kg0=kg0, T_sim=T_sim)

    Y_level_0 = ss_ref.Y * float(np.exp(path.Y[0] / 100.0))
    dY0 = Y_level_0 - ss_old.Y
    dG0 = ss_new.G - ss_old.G
    multiplier_impact = dY0 / dG0 if abs(dG0) > 1e-12 else float("nan")

    def shift(field_ref, X_ref, X_old):
        return field_ref + 100.0 * np.log(X_ref / X_old)

    path.Y = shift(path.Y, ss_ref.Y, ss_old.Y)
    path.C = shift(path.C, ss_ref.C, ss_old.C)
    path.I = shift(path.I, ss_ref.I, ss_old.I)
    path.K = shift(path.K, ss_ref.K, ss_old.K)
    if ss_ref.KG > 0 and ss_old.KG > 0:
        path.KG = shift(path.KG, ss_ref.KG, ss_old.KG)
    else:
        path.KG = np.zeros_like(path.KG)
    path.N = shift(path.N, ss_ref.N, ss_old.N)
    path.W = shift(path.W, ss_ref.w, ss_old.w)
    if permanent:
        path.G = np.full_like(path.G, 100.0 * np.log(ss_new.G / ss_old.G))
    else:
        path.G = shift(path.G, ss_ref.G, ss_old.G)

    return PolicyExperiment(ss_old=ss_old, ss_new=ss_new, path=path,
                             multiplier_long_run=multiplier_long_run,
                             multiplier_impact=multiplier_impact)


# --------------------------------------------------------------------------
# 5. Table 4 / Figure 5: public-capital productivity sensitivity
# --------------------------------------------------------------------------

def public_investment_long_run(theta_N: float, delta: float, r: float, A: float,
                                 s_other: float, N_target: float,
                                 theta_G_grid: np.ndarray, s_IG: float = 0.05) -> "dict[str, np.ndarray]":
    """Replicates Table 4 of Baxter & King (1993): long-run output/consumption/
    investment effects of a marginal increase in productivity-augmenting public
    investment IG, for a grid of theta_G, at the paper's calibration:
      - s_IG = 0.05 fixed (the "share of public investment" the paper studies;
        NOT tied to the app's live composition slider, which can go to 0% and
        make the marginal step below numerically ill-posed -- Kᴳ is an *essential*
        input whenever theta_G>0, so output collapses toward 0 as Kᴳ->0).
      - s_other -- other, always-unproductive, resource-using government spending
        (what used to be "basic purchases" G_B before that category was folded
        into IG in the main model) held fixed, so the baseline total government
        share matches the sidebar's G/Y setting even though only s_IG of it is
        the productive investment under study here.
      - always lump-sum financed, exactly as the paper's Table 4 note states
        ("In each case, shifts in purchases are financed via lump-sum taxation"),
        independent of the sidebar's financing choice elsewhere in the app.
      - theta_L is calibrated once, at theta_G=0, so labor hits N_target at the
        baseline -- matching the paper's Table 1 calibration strategy.

    Because there is no productivity distinction between IG and s_other at
    theta_G=0, the theta_G=0 row reproduces the *ordinary* lump-sum government-
    purchases multiplier from Tables 2-3 (~1.16 in the paper) exactly, not zero
    -- Baxter & King's own text confirms this ("the first row of the table
    replicates the results for basic government purchases obtained in Section
    III above").

    Three cases are returned, all in ΔY/ΔIG-comparable units:
      direct  -- output effect holding private K,N fixed: theta_G/s_IG
      k_adj   -- private capital adjusts, labor fixed: direct/(1-theta_K)
      both    -- both private capital and labor adjust (full GE)
    """
    theta_K = 1.0 - theta_N

    # Calibrate theta_L once, at theta_G=0, so labor hits N_target under the
    # baseline (s_IG + s_other) total government share.
    tau0 = s_IG + s_other
    supply0 = _supply_side_impl(theta_N, delta, r, A, tau0, 0.0, s_IG, False, N_target)
    s_C0 = 1.0 - supply0["s_I"] - s_IG - s_other
    if s_C0 <= 0:
        raise ValueError("Government share too large: steady-state consumption share <= 0.")
    theta_L = theta_N * (1.0 - N_target) / (s_C0 * N_target)  # tax_factor=1 always (lump-sum)
    if theta_L <= 0:
        raise ValueError("Implied theta_L <= 0; adjust N_target or the spending settings.")

    direct = theta_G_grid / s_IG
    k_adj = direct / (1.0 - theta_K)

    both = np.zeros_like(theta_G_grid)
    dC = np.zeros_like(theta_G_grid)
    dI = np.zeros_like(theta_G_grid)
    for idx, theta_G in enumerate(theta_G_grid):
        h = min(1e-4, s_IG * 0.05)  # keep the perturbation well inside (0, s_IG)
        m1 = _public_capital_output(theta_N, theta_K, delta, r, A, s_other, theta_L,
                                     theta_G, s_IG - h)
        m2 = _public_capital_output(theta_N, theta_K, delta, r, A, s_other, theta_L,
                                     theta_G, s_IG + h)
        both[idx] = (m2["Y"] - m1["Y"]) / (m2["IG"] - m1["IG"])
        dC[idx] = (m2["C"] - m1["C"]) / (m2["IG"] - m1["IG"])
        dI[idx] = (m2["I"] - m1["I"]) / (m2["IG"] - m1["IG"])

    return dict(theta_G=theta_G_grid, direct=direct, k_adj=k_adj, both=both, dC=dC, dI=dI)


def _public_capital_output(theta_N, theta_K, delta, r, A, s_other, theta_L, theta_G, s_IG):
    """Steady state of the model with productive public capital, at a given (theta_G,
    s_IG), used for the numerical-derivative Table 4 computation. Always lump-sum
    financed (tax_factor=1), matching the paper's Table 4. s_other is other,
    always-unproductive, resource-using government spending held fixed (see
    `public_investment_long_run`'s docstring)."""
    tau = s_IG + s_other
    N = 0.3
    supply = None
    for _ in range(200):
        supply = _supply_side_impl(theta_N, delta, r, A, tau, theta_G, s_IG, False, N)
        s_I = supply["s_I"]
        s_C = 1.0 - s_I - s_IG - s_other
        if s_C <= 0:
            s_C = 1e-6
        N_new = theta_N / (theta_L * s_C + theta_N)
        if abs(N_new - N) < 1e-12:
            N = N_new
            break
        N = N_new
    Y = supply["alpha"] * N
    K = supply["kappa"] * N
    KG = supply["KG"] * N
    C = s_C * Y
    I = s_I * Y
    IG = s_IG * Y
    return dict(Y=Y, C=C, I=I, IG=IG, K=K, KG=KG, N=N)
