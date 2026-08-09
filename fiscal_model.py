"""
Core economics for the Baxter & King (1993, AER) "Fiscal Policy in General
Equilibrium" neoclassical growth model with government purchases.

The model (benchmark, "basic" government purchases -- Table 1, Part I of the
paper):

  Preferences:      u_t = log(C_t) + theta_L * log(L_t),   L_t = 1 - N_t
  Technology:       Y_t = A * K_t^theta_K * N_t^theta_N,   theta_K+theta_N=1
  Capital accum.:   K_{t+1} = (1-delta_K) K_t + I_t
  Resource constr.: Y_t = C_t + I_t + G_t
  Government:       tau_t * Y_t = G_t + TR_t   (financing rule below)

Two financing rules are implemented, matching the paper's Sections III-V:

  * "lump_sum"      -- the tax rate tau is held FIXED; transfers TR absorb the
                        government budget residually.  This is the paper's
                        benchmark experiment (Figures 2-3, Table 1-3).
  * "distortionary"  -- transfers are fixed at TR=0 and the tax rate adjusts
                        every period to balance the budget: tau_t = G_t / Y_t.
                        This is the "Gramm-Rudman-Hollings" (GRH) experiment
                        of Section V / Figure 4.

Two solution objects are provided:

  1. `steady_state(...)`   -- exact, closed-form comparative steady states
                               (Section II.A of the paper).
  2. `simulate_transition(...)` -- approximate dynamics via log-linearization
                               around a steady state, solved with a
                               perfect-foresight shooting algorithm (the
                               model has exactly one predetermined variable,
                               capital, and one jump variable, consumption,
                               so the equilibrium is saddle-path stable --
                               King, Plosser & Rebelo (1988); Baxter & King
                               (1993) footnote 4).

All quantities are expressed as *fractions of the time-endowment / output*
(the model's "great ratios"), so the code is scale free: only ratios (shares,
theta_L-implied hours) are pinned down, not absolute price levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Financing = Literal["lump_sum", "distortionary"]


# --------------------------------------------------------------------------
# 1. Supply side (prices and "great ratios" -- independent of labor input)
# --------------------------------------------------------------------------

def compute_supply_side(theta_N: float, delta: float, r: float, A: float, tau: float) -> dict:
    """Solve the capital/labor ratio kappa=K/N and associated prices.

    The household's after-tax capital-Euler condition pins down kappa
    independently of labor input (constant-returns-to-scale production):

        (1 - tau) * theta_K * A * kappa^(theta_K - 1) = r + delta
    """
    theta_K = 1.0 - theta_N
    base = (1.0 - tau) * theta_K * A / (r + delta)
    if base <= 0:
        raise ValueError("Invalid parameters: (1-tau)*theta_K*A/(r+delta) must be positive.")
    kappa = base ** (1.0 / (1.0 - theta_K))
    alpha = A * kappa ** theta_K          # Y/N, average product of labor
    w = theta_N * alpha                    # pre-tax real wage
    q = (1.0 - tau) * theta_K * alpha / kappa   # after-tax rental rate on capital (= r+delta)
    s_I = delta * kappa / alpha            # I/Y = delta*(K/Y)
    return dict(theta_K=theta_K, kappa=kappa, alpha=alpha, w=w, q=q, s_I=s_I)


# --------------------------------------------------------------------------
# 2. Steady state (closed form)
# --------------------------------------------------------------------------

@dataclass
class SteadyState:
    theta_N: float
    theta_K: float
    delta: float
    r: float
    A: float
    tau: float
    theta_L: float
    s_G: float
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
    C: float
    I: float
    G: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def calibrate_theta_L(theta_N: float, delta: float, r: float, A: float,
                       tau: float, s_G: float, N_target: float) -> float:
    """Choose theta_L (weight on leisure) so that steady-state labor input
    equals N_target, given the other parameters (Table 1 calibration
    strategy of Baxter & King)."""
    supply = compute_supply_side(theta_N, delta, r, A, tau)
    s_C = 1.0 - supply["s_I"] - s_G
    if s_C <= 0:
        raise ValueError("Government share too large: steady-state consumption share <= 0.")
    N = N_target
    theta_L = (1.0 - tau) * theta_N * (1.0 - N) / (s_C * N)
    if theta_L <= 0:
        raise ValueError("Implied theta_L <= 0; adjust N_target or tau.")
    return theta_L


def steady_state(theta_N: float, delta: float, r: float, A: float, tau: float,
                  s_G: float, theta_L: float) -> SteadyState:
    """Exact steady state of the model for given parameters and policy (s_G, tau)."""
    supply = compute_supply_side(theta_N, delta, r, A, tau)
    s_I = supply["s_I"]
    s_C = 1.0 - s_I - s_G
    if s_C <= 0:
        raise ValueError("Government share too large: steady-state consumption share <= 0.")

    N = (1.0 - tau) * theta_N / (theta_L * s_C + (1.0 - tau) * theta_N)
    if not (0 < N < 1):
        raise ValueError(f"Implied labor supply N={N:.3f} is outside (0,1).")
    L = 1.0 - N

    Y = supply["alpha"] * N
    K = supply["kappa"] * N
    C = s_C * Y
    I = s_I * Y
    G = s_G * Y

    return SteadyState(
        theta_N=theta_N, theta_K=supply["theta_K"], delta=delta, r=r, A=A, tau=tau,
        theta_L=theta_L, s_G=s_G, kappa=supply["kappa"], alpha=supply["alpha"],
        w=supply["w"], q=supply["q"], s_I=s_I, s_C=s_C, N=N, L=L, Y=Y, K=K, C=C, I=I, G=G,
    )


def steady_state_for_financing(theta_N: float, delta: float, r: float, A: float,
                                 tau_baseline: float, s_G: float, theta_L: float,
                                 financing: Financing) -> SteadyState:
    """Steady state for a given (counterfactual) s_G under a financing rule.

    lump_sum:       tau stays at tau_baseline (transfers absorb the residual).
    distortionary:  tau = s_G (balanced budget, zero transfers -- GRH rule).
    """
    tau = s_G if financing == "distortionary" else tau_baseline
    return steady_state(theta_N, delta, r, A, tau, s_G, theta_L)


# --------------------------------------------------------------------------
# 3. Log-linear dynamics
# --------------------------------------------------------------------------

def _log_linear_coeffs(ss: SteadyState, distortionary: bool) -> dict:
    """Coefficients of the log-linearized equilibrium system around steady
    state `ss`.  See module docstring / accompanying derivation notes.

    Static block:   y_hat_t = y_k*k_hat_t + y_c*c_hat_t + y_g*g_hat_t
    Capital accum.: k_hat_{t+1} = Phi_kk*k_hat_t + Phi_kc*c_hat_t + Phi_kg*g_hat_t
    Euler equation: c_hat_t = coef_c*c_hat_{t+1} + coef_k*k_hat_{t+1} + coef_g*g_hat_{t+1}
    """
    theta_N, theta_K = ss.theta_N, ss.theta_K
    L, N = ss.L, ss.N
    tau = ss.tau
    s_C, s_I, s_G = ss.s_C, ss.s_I, ss.s_G
    delta, r = ss.delta, ss.r

    D = 1.0 - theta_N * L
    if distortionary:
        # NOTE: this denominator must be D MINUS the tax-wedge cross term, not
        # plus. Derivation: log-linearizing Y=AK^theta_K N^theta_N together
        # with the labor-leisure FOC under tau_t=G_t/Y_t gives
        # y_hat*(D - theta_N*L*tau/(1-tau)) = theta_K*k_hat - theta_N*L*c_hat
        #   - theta_N*L*tau/(1-tau)*g_hat.
        # (Verified against this module's own n_hat formula in
        # simulate_transition -- L(y_hat-c_hat) - L*tau/(1-tau)*(g_hat-y_hat)
        # -- by requiring the production identity y_hat=theta_K*k_hat+theta_N*n_hat
        # hold exactly; a "+" here leaves a residual of ~0.3-1% instead of 0.)
        Dp = D - theta_N * L * tau / (1.0 - tau)
        y_k = theta_K / Dp
        y_c = -theta_N * L / Dp
        y_g = -theta_N * L * tau / ((1.0 - tau) * Dp)
        tax_wedge = tau / (1.0 - tau)
    else:
        y_k = theta_K / D
        y_c = -theta_N * L / D
        y_g = 0.0
        tax_wedge = 0.0

    beta = 1.0 / (1.0 + r)
    q = r + delta
    Omega = beta * q  # = q/(1+r)

    # (y_hat - k_hat) coefficients
    ymk_k = y_k - 1.0
    ymk_c = y_c
    ymk_g = y_g

    coef_c = 1.0 - Omega * ymk_c - Omega * tax_wedge * y_c
    coef_k = -Omega * ymk_k - Omega * tax_wedge * y_k
    coef_g = -Omega * ymk_g + Omega * tax_wedge * (1.0 - y_g)

    Phi_kk = (1.0 - delta) + delta * y_k / s_I
    Phi_kc = delta * (y_c - s_C) / s_I
    Phi_kg = delta * (y_g - s_G) / s_I

    return dict(y_k=y_k, y_c=y_c, y_g=y_g, coef_c=coef_c, coef_k=coef_k, coef_g=coef_g,
                Phi_kk=Phi_kk, Phi_kc=Phi_kc, Phi_kg=Phi_kg)


def _solve_saddle_path(Phi_kk: float, Phi_kc: float, Phi_kg: float,
                        coef_c: float, coef_k: float, coef_g: float,
                        k0: float, g_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact saddle-path solution of the linear rational-expectations system

        k_hat_{t+1} = Phi_kk*k_hat_t + Phi_kc*c_hat_t + Phi_kg*g_hat_t
        c_hat_t     = coef_c*c_hat_{t+1} + coef_k*k_hat_{t+1} + coef_g*g_hat_{t+1}

    via eigen-decomposition (King-Plosser-Rebelo / Blanchard-Kahn method),
    for T periods where `g_full[t]` gives g_hat_t (assumed 0 for all t beyond
    the array, i.e. the shock has already died out / become the new normal).

    A naive forward "shooting" simulation is numerically unstable here
    because the unstable root is iterated hundreds of times, amplifying
    floating-point noise geometrically. Instead we (i) diagonalize the
    reduced-form transition matrix, (ii) pin the forward-looking (unstable)
    eigen-component at each date to the exact present value of *remaining*
    future forcing terms (so it never explodes), and (iii) propagate the
    predetermined (stable) eigen-component forward from k_hat_0, which is
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
    Nk = Phi_kg * g_ext[:T]
    Nc = (-coef_k * Phi_kg * g_ext[:T] - coef_g * g_ext[1:T + 1]) / coef_c
    N = np.stack([Nk, Nc], axis=1)
    n = N @ Vinv.T  # forcing rotated into the eigen-basis: n[:,0]=stable comp., n[:,1]=unstable

    # Unstable (jump) component: present value of all remaining future forcing.
    zeta2 = np.zeros(T)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = (n[t, 1] + running) / lam_u
        zeta2[t] = -running

    # Stable (predetermined) component: pinned by k_hat_0, then propagated forward.
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
    N: np.ndarray
    W: np.ndarray
    G: np.ndarray
    r_bp: np.ndarray  # real interest rate, deviation in basis points (annualized)


def simulate_transition(ss_ref: SteadyState, g_path: np.ndarray, k0: float = 0.0,
                         distortionary: bool = False, T_sim: int = 300) -> TransitionPath:
    """Simulate the perfect-foresight transition path in log-deviations from
    steady state `ss_ref` (the state the system is linearized around),
    given an exogenous path of government-purchases log-deviations `g_path`
    (padded with zeros afterwards) and an initial capital log-deviation `k0`.
    """
    coeffs = _log_linear_coeffs(ss_ref, distortionary)
    y_k, y_c, y_g = coeffs["y_k"], coeffs["y_c"], coeffs["y_g"]
    Phi_kk, Phi_kc, Phi_kg = coeffs["Phi_kk"], coeffs["Phi_kc"], coeffs["Phi_kg"]
    coef_c, coef_k, coef_g = coeffs["coef_c"], coeffs["coef_k"], coeffs["coef_g"]

    g_full = np.zeros(T_sim)
    n_g = min(len(g_path), T_sim)
    g_full[:n_g] = g_path[:n_g]

    L, N = ss_ref.L, ss_ref.N
    tau = ss_ref.tau

    k, c = _solve_saddle_path(Phi_kk, Phi_kc, Phi_kg, coef_c, coef_k, coef_g, k0, g_full)

    y = y_k * k + y_c * c + y_g * g_full
    if distortionary:
        n = L * (y - c) - (L * tau / (1.0 - tau)) * (g_full - y)
    else:
        n = L * (y - c)
    w = y - n  # wage = MPL, log-deviation

    i = (y - ss_ref.s_C * c - ss_ref.s_G * g_full) / ss_ref.s_I

    c_pct = 100.0 * c
    y_pct = 100.0 * y
    i_pct = 100.0 * i
    k_pct = 100.0 * k
    n_pct = 100.0 * n
    w_pct = 100.0 * w
    g_pct = 100.0 * g_full

    r_dev_bp = np.zeros_like(c)
    r_dev_bp[:-1] = 10000.0 * (1.0 + ss_ref.r) * (c[1:] - c[:-1])
    r_dev_bp[-1] = r_dev_bp[-2]

    years = np.arange(T_sim)
    return TransitionPath(years=years, Y=y_pct, C=c_pct, I=i_pct, K=k_pct, N=n_pct,
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


def run_experiment(theta_N: float, delta: float, r: float, A: float, tau_baseline: float,
                    s_G_old: float, s_G_new: float, N_target: float,
                    financing: Financing, permanent: bool, duration_years: int = 4,
                    T_sim: int = 300) -> PolicyExperiment:
    """Full comparative-steady-state + transition-path experiment for a
    change in government purchases from s_G_old to s_G_new."""
    theta_L = calibrate_theta_L(theta_N, delta, r, A, tau_baseline, s_G_old, N_target)

    ss_old = steady_state_for_financing(theta_N, delta, r, A, tau_baseline, s_G_old,
                                          theta_L, financing)
    ss_new = steady_state_for_financing(theta_N, delta, r, A, tau_baseline, s_G_new,
                                          theta_L, financing)

    dY = ss_new.Y - ss_old.Y
    dG = ss_new.G - ss_old.G
    multiplier_long_run = dY / dG if abs(dG) > 1e-12 else float("nan")

    distortionary = financing == "distortionary"

    if permanent:
        # Linearize around the NEW steady state (where the economy converges);
        # capital is predetermined at its OLD steady-state level at t=0, and G
        # jumps immediately (and permanently) to its new level, so g_hat=0
        # throughout relative to the (new) reference steady state.
        ss_ref = ss_new
        k0 = np.log(ss_old.K / ss_new.K)
        g_path = np.zeros(1)
    else:
        # Steady state is unchanged (the shock reverts); linearize around the
        # single, common steady state and feed in a finite-duration g_hat path.
        ss_ref = ss_old
        k0 = 0.0
        dG_frac = (ss_new.G - ss_old.G) / ss_old.G if ss_old.G != 0 else 0.0
        g_level = np.log(1.0 + dG_frac)
        g_path = np.concatenate([np.full(duration_years, g_level), np.zeros(1)])

    path = simulate_transition(ss_ref, g_path, k0=k0, distortionary=distortionary, T_sim=T_sim)

    # Impact multiplier is always measured relative to the ORIGINAL (pre-shock)
    # steady state, following the paper's convention, regardless of which
    # steady state the log-linear path itself was computed around.
    Y_level_0 = ss_ref.Y * float(np.exp(path.Y[0] / 100.0))
    dY0 = Y_level_0 - ss_old.Y
    dG0 = ss_new.G - ss_old.G
    multiplier_impact = dY0 / dG0 if abs(dG0) > 1e-12 else float("nan")

    # Re-express the whole path as %-deviations from the ORIGINAL steady state
    # (matching Figures 2-4 of the paper: "percentage deviations from initial
    # steady-state values"), regardless of which steady state was used as the
    # linearization point.
    def shift(field_ref, X_ref, X_old):
        return field_ref + 100.0 * np.log(X_ref / X_old)

    path.Y = shift(path.Y, ss_ref.Y, ss_old.Y)
    path.C = shift(path.C, ss_ref.C, ss_old.C)
    path.I = shift(path.I, ss_ref.I, ss_old.I)
    path.K = shift(path.K, ss_ref.K, ss_old.K)
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
# 5. Productive public capital: long-run comparative statics (Section VI)
# --------------------------------------------------------------------------

def public_investment_long_run(theta_N: float, delta: float, r: float, A: float,
                                 tau: float, s_G: float, theta_L: float,
                                 theta_G_grid: np.ndarray, s_IG: float = 0.05) -> "dict[str, np.ndarray]":
    """Replicates Table 4 of Baxter & King: long-run output/consumption/
    investment effects of a marginal increase in productivity-augmenting
    public investment I^G, for a grid of theta_G (public-capital productivity
    parameter), holding the public-investment share s_IG = I^G/Y fixed.

    Three cases are returned:
      direct  -- output effect holding private K,N fixed (dY/dI^G = theta_G)
      k_adj   -- private capital adjusts, labor fixed
      both    -- both private capital and labor adjust (full GE)
    """
    ss0 = steady_state(theta_N, delta, r, A, tau, s_G, theta_L)
    theta_K = ss0.theta_K

    direct = theta_G_grid.copy()

    k_adj = theta_G_grid / (1.0 - theta_K)

    both = np.zeros_like(theta_G_grid)
    dC = np.zeros_like(theta_G_grid)
    dI = np.zeros_like(theta_G_grid)
    for idx, theta_G in enumerate(theta_G_grid):
        # Steady state of Y - I is maximized at s_IG = theta_G (zero net resource use benchmark);
        # here we compute the *marginal* GE multiplier via small numerical perturbation of I^G/Y.
        h = 1e-4
        m1 = _public_capital_output(theta_N, theta_K, delta, r, A, tau, s_G, theta_L,
                                     theta_G, s_IG - h)
        m2 = _public_capital_output(theta_N, theta_K, delta, r, A, tau, s_G, theta_L,
                                     theta_G, s_IG + h)
        both[idx] = (m2["Y"] - m1["Y"]) / (m2["IG"] - m1["IG"])
        dC[idx] = (m2["C"] - m1["C"]) / (m2["IG"] - m1["IG"])
        dI[idx] = (m2["I"] - m1["I"]) / (m2["IG"] - m1["IG"])

    return dict(theta_G=theta_G_grid, direct=direct, k_adj=k_adj, both=both, dC=dC, dI=dI)


def _public_capital_output(theta_N, theta_K, delta, r, A, tau, s_G, theta_L, theta_G, s_IG):
    """Steady state of the extended model with productive public capital.

    Production:  Y = A * K^theta_K * (K^G)^theta_G * N^theta_N   (theta_K+theta_N+theta_G<=1)
    K^G evolves like private capital with the same depreciation rate and is
    financed by public investment I^G = s_IG * Y (held fixed as a share of
    output, as in the paper's Table 4 / Figure 5 setup).
    """
    # After-tax private no-arbitrage condition (K^G is exogenous from the household's
    # perspective, so it does not directly enter the Euler equation beyond its effect
    # on the marginal product of private capital):
    #   (1-tau) * theta_K * A * K^theta_K-1 * KG^theta_G * N^theta_N = r+delta
    # and K^G steady state: KG = s_IG*Y/delta.
    # Solve jointly by iterating on Y (fixed point), starting from the no-public-capital case.
    kappa0 = ((1.0 - tau) * theta_K * A / (r + delta)) ** (1.0 / (1.0 - theta_K))
    Y = A * kappa0 ** theta_K  # initial guess per unit of N, refined below with N=1 normalization first pass
    N_guess = 0.2
    for _ in range(200):
        KG = s_IG * Y / delta
        base = (1.0 - tau) * theta_K * A * KG ** theta_G / (r + delta)
        kappa = base ** (1.0 / (1.0 - theta_K))
        alpha = A * kappa ** theta_K * KG ** theta_G
        w = theta_N * alpha
        s_I = delta * kappa / alpha
        s_C = 1.0 - s_I - s_G - s_IG
        if s_C <= 0:
            s_C = 1e-6
        N = (1.0 - tau) * theta_N / (theta_L * s_C + (1.0 - tau) * theta_N)
        Y_new = alpha * N
        if abs(Y_new - Y) < 1e-10:
            Y = Y_new
            N_guess = N
            break
        Y = Y_new
        N_guess = N
    K = kappa * N_guess
    C = s_C * Y
    I = s_I * Y
    IG = s_IG * Y
    return dict(Y=Y, C=C, I=I, IG=IG, K=K, N=N_guess)
