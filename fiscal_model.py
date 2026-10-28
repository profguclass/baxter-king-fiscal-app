"""
Core economics for the Baxter & King (1993, AER) "Fiscal Policy in General
Equilibrium" neoclassical growth model with government purchases, extended
with productive public capital (Section VI of the paper) as a first-class
part of the main model rather than a side extension.

The model:

  Preferences:      u_t = log(C_t + phi*GC_t) + theta_L * log(L_t),   L_t = 1 - N_t
                     (GC = government consumption; phi in [0,1] is how well it substitutes
                      for private consumption: phi=0 is a pure drain, as in Baxter-King's
                      "basic purchases"; phi=1 is a perfect substitute)
  Technology:        Y_t = A * K_t^theta_K * KG_t^theta_G * N_t^theta_N
                     (theta_K = 1 - theta_N; KG = public capital stock)
  Capital accum.:    K_{t+1}  = (1-delta) K_t  + I_t
                     KG_{t+1} = (1-delta) KG_t + IG_t
  Resource constr.:  Y_t = C_t + I_t + IG_t + GC_t         (transfers G_T are not resource-using)
  Government budget: T_t + tau_y,t*(taxed income) + tau_c,t*C_t = IG_t + GC_t + G_T,t

Three taxes, combined in a "tax mix": each raises a fixed share (f_LS, f_Y, f_C,
summing to 1) of total government spending G_t every period:

  * lump-sum        -- T_t = f_LS*G_t per person. Purely a wealth effect: it enters
                        no first-order condition.
  * income tax      -- a rate tau_y,t on (optimizers') labor and capital income: a
                        (1-tau_y) wedge on both the labor-leisure and the saving margin
                        (the paper's original "distortionary"/GRH channel).
  * consumption tax -- a rate tau_c,t on private consumption: consumers pay
                        p_t = 1+tau_c,t per unit, a wedge on the labor-leisure margin
                        only (a constant rate leaves the saving margin alone).

Total government spending G_t is split into three shares of output that sum to the
total government-spending ratio s_G = IG/Y + GC/Y + G_T/Y:
  * IG   -- public investment, accumulates into a public-capital stock KG that
            raises the productivity of private capital and labor (theta_G).
  * GC   -- government consumption, resource-using and valued by households
            through the effective consumption X_t = C_t + phi*GC_t.
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
from typing import Literal, Optional, Union

import numpy as np

Financing = Union[Literal["lump_sum", "income_tax", "consumption_tax"], tuple]


def _tax_mix(financing) -> tuple:
    """Shares (lump-sum, income tax, consumption tax) of revenue raised by each tax,
    normalized to sum to 1. Also accepts a name ("lump_sum", "income_tax",
    "consumption_tax") or a bool (True = pure income tax, False = pure lump-sum)."""
    if isinstance(financing, (bool, np.bool_)):
        return (0.0, 1.0, 0.0) if financing else (1.0, 0.0, 0.0)
    if isinstance(financing, str):
        return {"lump_sum": (1.0, 0.0, 0.0), "income_tax": (0.0, 1.0, 0.0),
                "consumption_tax": (0.0, 0.0, 1.0)}[financing]
    mix = tuple(float(v) for v in financing)
    if len(mix) != 3 or min(mix) < 0 or sum(mix) <= 0:
        raise ValueError("The tax mix needs three non-negative shares with a positive sum.")
    return tuple(v / sum(mix) for v in mix)


# --------------------------------------------------------------------------
# 1. Supply side (prices and "great ratios"), with public capital KG.
# --------------------------------------------------------------------------

def _supply_side_impl(theta_N: float, delta: float, r: float, A: float, tau_y: float,
                       theta_G: float, s_IG: float, N: float) -> dict:
    """tau_y is the income-tax rate on capital income: steady state (1-tau_y)*MPK = r+delta."""
    theta_K = 1.0 - theta_N
    tax_factor = 1.0 - tau_y
    if tax_factor <= 0:
        raise ValueError("Invalid parameters: the income-tax rate must be below 100%.")
    if theta_G > 0 and s_IG <= 0:
        raise ValueError("With productive public capital (θ_G > 0), firms need public capital "
                         "to produce: set public investment above 0% (item E), or set θ_G = 0.")

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
    tau_y: float     # income-tax rate on optimizers' income (labor + capital)
    tau_c: float     # consumption-tax rate on private consumption (both household types)
    T: float         # lump-sum tax per person (everyone pays it)
    theta_L: float
    s_G: float       # total government spending share (IG+GC+G_T)/Y
    s_IG: float
    s_GC: float
    s_GT: float
    phi: float       # utility weight of GC in effective consumption C + phi*GC
    tax_mix: tuple   # shares of revenue from (lump-sum, income tax, consumption tax)
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
    G: float          # total government spending IG+GC+G_T
    IG: float
    GC: float
    GT: float
    # Two household types (section 6): share lam of hand-to-mouth households.
    # With lam=0 there is a single (optimizing) household: CO=C, NO=N, CH=NH=nan.
    lam: float = 0.0
    CO: float = float("nan")  # per-capita consumption / hours of optimizers
    NO: float = float("nan")
    CH: float = float("nan")  # per-capita consumption / hours of hand-to-mouth households
    NH: float = float("nan")

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def calibrate_theta_L(theta_N: float, delta: float, r: float, A: float, theta_G: float,
                       s_IG: float, s_GT: float, tax_mix,
                       N_target: float, s_GC: float = 0.0, phi: float = 0.0,
                       lam: float = 0.0) -> float:
    """Choose theta_L (weight on leisure) so that steady-state labor input equals
    N_target, given the other parameters (Table 1 calibration strategy). tax_mix is
    anything `_tax_mix` accepts."""
    mix = _tax_mix(tax_mix)
    if lam > 0:
        return _calibrate_theta_L_tank(theta_N, delta, r, A, theta_G, s_IG, s_GC, s_GT,
                                        mix, N_target, phi, lam)
    s_G = s_IG + s_GC + s_GT
    tau_y = mix[1] * s_G
    N = N_target
    supply = _supply_side_impl(theta_N, delta, r, A, tau_y, theta_G, s_IG, N)
    s_C = 1.0 - supply["s_I"] - s_IG - s_GC
    if s_C <= 0:
        raise ValueError("Government share too large: steady-state consumption share <= 0.")
    s_X = s_C + phi * s_GC  # effective-consumption share (C + phi*GC)/Y
    wedge = (1.0 - tau_y) / (1.0 + mix[2] * s_G / s_C)  # after-tax wage / consumer price
    theta_L = wedge * theta_N * (1.0 - N) / (s_X * N)
    if theta_L <= 0:
        raise ValueError("Implied theta_L <= 0; adjust N_target or the tax/spending settings.")
    return theta_L


def steady_state(theta_N: float, delta: float, r: float, A: float, theta_G: float,
                  s_IG: float, s_GT: float, tax_mix,
                  theta_L: float, s_GC: float = 0.0, phi: float = 0.0,
                  lam: float = 0.0) -> SteadyState:
    """Exact steady state of the model for given parameters, government-spending
    composition (s_IG, s_GC, s_GT), and tax mix. With lam=0 the income tax raises
    f_Y*G from income Y, so tau_y = f_Y*s_G, and tau_c = f_C*s_G/s_C."""
    mix = _tax_mix(tax_mix)
    if lam > 0:
        return _steady_state_tank(theta_N, delta, r, A, theta_G, s_IG, s_GC, s_GT,
                                   mix, theta_L, phi, lam)
    s_G = s_IG + s_GC + s_GT
    tau_y = mix[1] * s_G
    # Outer fixed point on N: when theta_G>0 and s_IG>0, the supply side (kappa,
    # alpha, KG) itself depends on N (see _supply_side_impl), while N in turn
    # depends on the supply side's implied consumption share s_C. When theta_G=0
    # or s_IG=0 this converges in a single pass, since supply doesn't depend on N.
    N = 0.3
    supply = None
    for _ in range(200):
        supply = _supply_side_impl(theta_N, delta, r, A, tau_y, theta_G, s_IG, N)
        s_I = supply["s_I"]
        s_C = 1.0 - s_I - s_IG - s_GC
        if s_C <= 0:
            raise ValueError("Government share too large: steady-state consumption share <= 0.")
        tau_c = mix[2] * s_G / s_C
        wedge = (1.0 - tau_y) / (1.0 + tau_c)  # after-tax wage / consumer price
        s_X = s_C + phi * s_GC
        N_new = wedge * theta_N / (theta_L * s_X + wedge * theta_N)
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
    GC = s_GC * Y
    GT = s_GT * Y
    G = IG + GC + GT

    return SteadyState(
        theta_N=theta_N, theta_K=supply["theta_K"], theta_G=theta_G, delta=delta, r=r, A=A,
        tau_y=tau_y, tau_c=tau_c, T=mix[0] * G, theta_L=theta_L, s_G=s_G, s_IG=s_IG,
        s_GC=s_GC, s_GT=s_GT, phi=phi, tax_mix=mix, kappa=supply["kappa"], alpha=supply["alpha"],
        w=supply["w"], q=supply["q"], s_I=s_I, s_C=s_C, N=N, L=L, Y=Y, K=K, KG=KG, C=C, I=I,
        G=G, IG=IG, GC=GC, GT=GT, CO=C, NO=N,
    )


def steady_state_for_policy(theta_N: float, delta: float, r: float, A: float, theta_G: float,
                             s_G_total: float, f_IG: float, f_GT: float,
                             theta_L: float, tax_mix,
                             f_GC: float = 0.0, phi: float = 0.0, lam: float = 0.0) -> SteadyState:
    """Steady state for a given total government-spending ratio s_G_total, split
    into fixed composition fractions f_IG+f_GC+f_GT=1 of that total."""
    s_IG, s_GC, s_GT = s_G_total * f_IG, s_G_total * f_GC, s_G_total * f_GT
    return steady_state(theta_N, delta, r, A, theta_G, s_IG, s_GT, tax_mix, theta_L,
                        s_GC=s_GC, phi=phi, lam=lam)


# --------------------------------------------------------------------------
# 2b. Two household types (lam > 0)
#
#   Taxes: each of the three raises a fixed share (f_LS, f_Y, f_C) of G every period.
#     Lump-sum:     T = f_LS*G per person (everyone pays it).
#     Income tax:   tau_y = f_Y*G / (optimizers' income) = f_Y*G / (Y - lam*w*N^H).
#     Consumption:  tau_c = f_C*G / C on everyone's private consumption; p = 1+tau_c.
#   * Hand-to-mouth (share lam): no capital, consume all income each period, pay no
#     income tax, receive ALL cash welfare G_T (G_T/lam each):
#       p*C^H = w*N^H + G_T/lam - T,   theta_L*(C^H + phi*GC)/(1-N^H) = w/p
#   * Optimizers (share 1-lam): own all capital, receive no cash welfare:
#       theta_L*(C^O + phi*GC)/(1-N^O) = (1-tau_y)*w/p,  Euler in (C^O + phi*GC)*p.
#   Aggregates:  C = lam*C^H + (1-lam)*C^O,  N = lam*N^H + (1-lam)*N^O.
# With lam=0 the code above (the representative-household model) is used instead.
# --------------------------------------------------------------------------

def _htm_choice(w: float, GT: float, GC: float, T_ls: float, lam: float,
                theta_L: float, phi: float, p: float = 1.0) -> tuple[float, float]:
    """Hand-to-mouth household's static choice: maximize ln(C+phi*GC) + theta_L*ln(1-N)
    subject to p*C = w*N + GT/lam - T_ls, with p = 1 + consumption-tax rate. Returns
    (N^H, C^H); N^H is 0 (corner) if non-labor resources are so large the household
    would rather not work."""
    m = GT / lam - T_ls  # non-labor income per hand-to-mouth household
    NH = max(0.0, (w - theta_L * (m + p * phi * GC)) / (w * (1.0 + theta_L)))
    return NH, (w * NH + m) / p


def _tank_solve(theta_N, delta, r, A, theta_G, s_IG, s_GC, s_GT, tax_mix,
                theta_L, phi, lam) -> dict:
    """Unvalidated steady state of the two-household model: for a given income-tax rate
    tau_y, aggregate hours N solve lam*N^H(N) + (1-lam)*N^O(N) = N by bisection (the
    left side falls as N rises); an outer damped iteration then updates tau_y. In a
    steady state C = s_C*Y is known, so the consumption-tax rate is f_C*s_G/s_C."""
    s_G = s_IG + s_GC + s_GT
    f_LS, f_Y, f_C = tax_mix

    def given_N(N, tau_y):
        supply = _supply_side_impl(theta_N, delta, r, A, tau_y, theta_G, s_IG, N)
        Y = supply["alpha"] * N
        w = supply["w"]
        s_C = 1.0 - supply["s_I"] - s_IG - s_GC
        p = 1.0 + f_C * s_G / max(s_C, 1e-12)
        NH, CH = _htm_choice(w, s_GT * Y, s_GC * Y, f_LS * s_G * Y, lam, theta_L, phi, p)
        CO = (s_C * Y - lam * CH) / (1.0 - lam)
        NO = 1.0 - theta_L * p * (CO + phi * s_GC * Y) / (supply["tax_factor"] * w)
        return dict(supply=supply, N=N, tau_y=tau_y, p=p, Y=Y, s_C=s_C, NH=NH, CH=CH,
                    CO=CO, NO=NO, resid=lam * NH + (1.0 - lam) * NO - N)

    tau_y = f_Y * s_G
    for _ in range(500):
        lo, hi = 1e-9, 1.0 - 1e-9
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if given_N(mid, tau_y)["resid"] > 0:
                lo = mid
            else:
                hi = mid
        sol = given_N(0.5 * (lo + hi), tau_y)
        if f_Y == 0:
            break
        tau_new = f_Y * s_G * sol["Y"] / (sol["Y"] - lam * sol["supply"]["w"] * sol["NH"])
        if abs(tau_new - tau_y) < 1e-14:
            break
        tau_y = 0.5 * (tau_y + tau_new)
    sol["tau_y"] = tau_y
    return sol


def _steady_state_tank(theta_N, delta, r, A, theta_G, s_IG, s_GC, s_GT, tax_mix,
                       theta_L, phi, lam) -> SteadyState:
    if not (0 < lam < 1):
        raise ValueError("The hand-to-mouth share must be between 0 and 1.")
    sol = _tank_solve(theta_N, delta, r, A, theta_G, s_IG, s_GC, s_GT, tax_mix,
                      theta_L, phi, lam)
    supply, N, Y = sol["supply"], sol["N"], sol["Y"]
    if sol["s_C"] <= 0:
        raise ValueError("Government share too large: steady-state consumption share <= 0.")
    if not (0 < sol["NO"] < 1):
        raise ValueError(f"Implied optimizer labor supply N={sol['NO']:.3f} is outside (0,1).")
    if sol["CO"] <= 0 or sol["CH"] <= 0:
        raise ValueError("Implied consumption of one household type is <= 0; "
                         "try a different hand-to-mouth share or composition.")
    if not (0 <= sol["tau_y"] < 1):
        raise ValueError("Implied income-tax rate is outside [0,1).")
    s_G = s_IG + s_GC + s_GT
    return SteadyState(
        theta_N=theta_N, theta_K=supply["theta_K"], theta_G=theta_G, delta=delta, r=r, A=A,
        tau_y=sol["tau_y"], tau_c=sol["p"] - 1.0, T=tax_mix[0] * s_G * Y, theta_L=theta_L,
        s_G=s_G, s_IG=s_IG, s_GC=s_GC, s_GT=s_GT, phi=phi, tax_mix=tax_mix,
        kappa=supply["kappa"], alpha=supply["alpha"], w=supply["w"], q=supply["q"],
        s_I=supply["s_I"], s_C=sol["s_C"], N=N, L=1.0 - N, Y=Y, K=supply["kappa"] * N,
        KG=supply["KG"] * N, C=sol["s_C"] * Y, I=supply["s_I"] * Y, G=s_G * Y,
        IG=s_IG * Y, GC=s_GC * Y, GT=s_GT * Y,
        lam=lam, CO=sol["CO"], NO=sol["NO"], CH=sol["CH"], NH=sol["NH"],
    )


def _calibrate_theta_L_tank(theta_N, delta, r, A, theta_G, s_IG, s_GC, s_GT, tax_mix,
                             N_target, phi, lam) -> float:
    """Bisection on log(theta_L): aggregate steady-state hours fall as theta_L rises."""
    lo, hi = np.log(1e-4), np.log(1e4)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        try:
            N = _tank_solve(theta_N, delta, r, A, theta_G, s_IG, s_GC, s_GT, tax_mix,
                            np.exp(mid), phi, lam)["N"]
        except ValueError:
            N = 0.0  # infeasible region: treat as "too little work" (theta_L too high)
        if N > N_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return float(np.exp(0.5 * (lo + hi)))


# --------------------------------------------------------------------------
# 3. Log-linear dynamics (dynamic model type)
# --------------------------------------------------------------------------

def _log_linear_coeffs(ss: SteadyState) -> dict:
    """Coefficients of the log-linearized equilibrium system around steady state `ss`.

    Static block:   y_hat_t = y_k*k_hat_t + y_kg*kg_hat_t + y_c*c_hat_t + y_g*g_hat_t
    Capital accum.: k_hat_{t+1} = Phi_kk*k_hat_t + Phi_kc*c_hat_t + Phi_kg*g_hat_t + Phi_k_kg*kg_hat_t
    Euler equation: c_hat_t = coef_c*c_hat_{t+1} + coef_k*k_hat_{t+1} + coef_g*g_hat_{t+1}
                              + coef_kg*kg_hat_{t+1} + coef_g0*g_hat_t

    The labor-leisure condition and the Euler equation are in effective consumption
    X = C + phi*GC, whose log-deviation is x_hat = a*c_hat + b*g_hat (GC moves with
    total G under a fixed composition); a=1, b=0 recovers the model without GC.

    Covers the one-household model (lam=0) with pure lump-sum or pure income-tax
    financing only; it is kept as an analytic check on `_tank_linearization`, which
    `simulate_transition` uses for every case.
    """
    if ss.tax_mix not in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        raise ValueError("Analytic coefficients cover pure lump-sum or pure income-tax financing only.")
    distortionary = ss.tax_mix[1] == 1.0
    theta_N, theta_K, theta_G = ss.theta_N, ss.theta_K, ss.theta_G
    L = ss.L
    tau = ss.tau_y
    s_C, s_I, s_IG, s_GC = ss.s_C, ss.s_I, ss.s_IG, ss.s_GC
    delta, r = ss.delta, ss.r

    s_X = s_C + ss.phi * s_GC
    a = s_C / s_X
    b = ss.phi * s_GC / s_X

    D = 1.0 - theta_N * L
    if distortionary:
        # See derivation note: y_hat*(D - theta_N*L*tau/(1-tau)) = theta_K*k_hat +
        # theta_G*kg_hat - theta_N*L*x_hat - theta_N*L*tau/(1-tau)*g_hat.
        Dp = D - theta_N * L * tau / (1.0 - tau)
        y_k = theta_K / Dp
        y_kg = theta_G / Dp
        y_x = -theta_N * L / Dp
        y_g = -theta_N * L * tau / ((1.0 - tau) * Dp) + y_x * b
        tax_wedge = tau / (1.0 - tau)
    else:
        y_k = theta_K / D
        y_kg = theta_G / D
        y_x = -theta_N * L / D
        y_g = y_x * b
        tax_wedge = 0.0
    y_c = y_x * a

    beta = 1.0 / (1.0 + r)
    q = r + delta
    Omega = beta * q  # = q/(1+r)

    ymk_k = y_k - 1.0
    ymk_c = y_c
    ymk_g = y_g
    ymk_kg = y_kg

    # Euler in x_hat: a*c_t + b*g_t = a*c_{t+1} + b*g_{t+1} - Omega*[...]_{t+1}; divide by a.
    coef_c = (a - Omega * ymk_c - Omega * tax_wedge * y_c) / a
    coef_k = (-Omega * ymk_k - Omega * tax_wedge * y_k) / a
    coef_g = (b - Omega * ymk_g + Omega * tax_wedge * (1.0 - y_g)) / a
    coef_kg = (-Omega * ymk_kg - Omega * tax_wedge * y_kg) / a
    coef_g0 = -b / a

    s_GR = s_IG + s_GC  # resource-using government spending share
    Phi_kk = (1.0 - delta) + delta * y_k / s_I
    Phi_kc = delta * (y_c - s_C) / s_I
    Phi_kg = delta * (y_g - s_GR) / s_I
    Phi_k_kg = delta * y_kg / s_I

    return dict(y_k=y_k, y_kg=y_kg, y_c=y_c, y_g=y_g, s_GR=s_GR, a=a, b=b,
                coef_c=coef_c, coef_k=coef_k, coef_g=coef_g, coef_kg=coef_kg, coef_g0=coef_g0,
                Phi_kk=Phi_kk, Phi_kc=Phi_kc, Phi_kg=Phi_kg, Phi_k_kg=Phi_k_kg)


def _tank_eval(ss: SteadyState, k: float, c: float, g: float, kg: float) -> dict:
    """Static block plus capital accumulation and the Euler return, at log-deviations
    (k, c=optimizer consumption, g, kg) from `ss`, in levels; works for lam=0 too.
    Solves aggregate hours N by Newton's method on lam*N^H + (1-lam)*N^O = N, and the
    consumer price p = 1+tau_c by bisection on tau_c*C = f_C*G."""
    lam, theta_L, phi = ss.lam, ss.theta_L, ss.phi
    f_LS, f_Y, f_C = ss.tax_mix
    K, CO, G, KG = ss.K * np.exp(k), ss.CO * np.exp(c), ss.G * np.exp(g), ss.KG * np.exp(kg)
    f_IG, f_GC, f_GT = (ss.s_IG / ss.s_G, ss.s_GC / ss.s_G, ss.s_GT / ss.s_G) if ss.s_G > 0 else (0.0, 0.0, 0.0)
    GC, GT = f_GC * G, f_GT * G
    T_ls = f_LS * G
    kg_factor = KG ** ss.theta_G if (ss.theta_G > 0 and KG > 0) else 1.0

    def block(N):
        Y = ss.A * K ** ss.theta_K * kg_factor * N ** ss.theta_N
        w = ss.theta_N * Y / N
        if lam > 0:
            def htm(p):
                return _htm_choice(w, GT, GC, T_ls, lam, theta_L, phi, p)
            p = 1.0
            if f_C > 0:
                # (p-1)*C = f_C*G, with C = lam*C^H(p) + (1-lam)*C^O; p <= hi since C^H >= 0.
                lo, hi = 1.0, 1.0 + f_C * G / ((1.0 - lam) * CO)
                for _ in range(100):
                    p = 0.5 * (lo + hi)
                    if (p - 1.0) * (lam * htm(p)[1] + (1.0 - lam) * CO) < f_C * G:
                        lo = p
                    else:
                        hi = p
            NH, CH = htm(p)
        else:
            p, NH, CH = 1.0 + f_C * G / CO, 0.0, 0.0
        tau_y = f_Y * G / (Y - lam * w * NH)
        NO = 1.0 - theta_L * p * (CO + phi * GC) / ((1.0 - tau_y) * w)
        return Y, w, NH, CH, NO, tau_y, p, lam * NH + (1.0 - lam) * NO - N

    N = ss.N
    for _ in range(100):
        resid = block(N)[-1]
        if abs(resid) < 1e-15:
            break
        h = 1e-7 * N
        N -= resid / ((block(N + h)[-1] - resid) / h)
    Y, w, NH, CH, NO, tau_y, p, _ = block(N)
    C = lam * CH + (1.0 - lam) * CO
    I = Y - C - (f_IG + f_GC) * G
    return dict(Y=Y, N=N, NH=NH, CH=CH, NO=NO, C=C, I=I, w=w, tau_y=tau_y, p=p,
                Knext=(1.0 - ss.delta) * K + I,
                R=1.0 + (1.0 - tau_y) * ss.theta_K * Y / K - ss.delta)


def _tank_linearization(ss: SteadyState) -> dict:
    """Numerical first-order coefficients of the model around `ss` (any lam and tax mix),
    in the form `_solve_saddle_path` uses, plus a gradient row (w.r.t. k, c, g, kg) for
    each reported variable's log-deviation.

    Euler equation in logs: x_t + pi_t = x_{t+1} + pi_{t+1} - rho_{t+1}, with x the
    optimizers' effective consumption, pi the consumer price 1+tau_c, rho the gross
    after-tax return; pi and rho are linear in the state through their gradients."""
    base = dict(k=0.0, c=0.0, g=0.0, kg=0.0)
    h = 1e-6
    grads = {}
    for name in ("k", "c", "g", "kg"):
        up, dn = dict(base), dict(base)
        up[name], dn[name] = h, -h
        e_up, e_dn = _tank_eval(ss, **up), _tank_eval(ss, **dn)
        for var in ("Y", "N", "NO", "C", "I", "Knext", "R", "p", "CH", "NH"):
            if var in ("CH", "NH") and ss.lam == 0:
                continue  # no hand-to-mouth households
            if var == "NH" and ss.NH <= 0:
                continue  # corner N^H=0: hours stay at 0 for small deviations
            ref = {"Knext": ss.K, "R": 1.0 + ss.r, "p": 1.0 + ss.tau_c}.get(var)
            ref = getattr(ss, var) if ref is None else ref
            grads.setdefault(var, np.zeros(4))[("k", "c", "g", "kg").index(name)] = \
                (np.log(e_up[var] / ref) - np.log(e_dn[var] / ref)) / (2 * h)

    X_O = ss.CO + ss.phi * ss.GC
    a, b = ss.CO / X_O, ss.phi * ss.GC / X_O
    Fk, Fc, Fg, Fkg = grads["R"]
    Pk, Pc, Pg, Pkg = grads["p"]
    Phi_kk, Phi_kc, Phi_kg, Phi_k_kg = grads["Knext"]
    A0 = a + Pc  # coefficient on c_t
    return dict(a=a, b=b, grads=grads,
                coef_c=(a + Pc - Fc) / A0, coef_k=(Pk - Fk) / A0, coef_g=(b + Pg - Fg) / A0,
                coef_kg=(Pkg - Fkg) / A0, coef_g0=-(b + Pg) / A0,
                coef_k0=-Pk / A0, coef_kg0=-Pkg / A0,
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
                        coef_g0: float,
                        k0: float, g_full: np.ndarray, kg_full: np.ndarray,
                        coef_k0: float = 0.0, coef_kg0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact saddle-path solution of the linear perfect-foresight system

        k_hat_{t+1} = Phi_kk*k_hat_t + Phi_kc*c_hat_t + Phi_kg*g_hat_t + Phi_k_kg*kg_hat_t
        c_hat_t     = coef_c*c_hat_{t+1} + coef_k*k_hat_{t+1} + coef_g*g_hat_{t+1}
                      + coef_kg*kg_hat_{t+1} + coef_g0*g_hat_t + coef_k0*k_hat_t
                      + coef_kg0*kg_hat_t

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
                  [-(coef_k * Phi_kk + coef_k0) / coef_c, (1.0 - coef_k * Phi_kc) / coef_c]])
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
          - coef_kg * kg_next - coef_g0 * g_ext[:T] - coef_kg0 * kg_now) / coef_c
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
    # Two household types only (lam > 0); None otherwise. Per-capita, % deviations.
    CO: Optional[np.ndarray] = None
    CH: Optional[np.ndarray] = None
    NO: Optional[np.ndarray] = None
    NH: Optional[np.ndarray] = None
    # Log-deviations of (k, optimizers' c, g, kg) from the steady state the path is
    # linearized around, shape (4, T): the input to `household_outcomes`.
    state: Optional[np.ndarray] = None


def simulate_transition(ss_ref: SteadyState, g_path: np.ndarray,
                         k0: float = 0.0, kg0: float = 0.0, T_sim: int = 300) -> TransitionPath:
    """Simulate the perfect-foresight transition path in log-deviations from steady
    state `ss_ref`, given an exogenous path of government-spending log-deviations
    `g_path` (padded with zeros afterwards) and initial capital log-deviations k0/kg0."""
    tank = ss_ref.lam > 0
    coeffs = _tank_linearization(ss_ref)

    g_full = np.zeros(T_sim)
    n_g = min(len(g_path), T_sim)
    g_full[:n_g] = g_path[:n_g]

    kg_full_ext = _kg_hat_path(ss_ref.delta, g_full, kg0)  # length T_sim+1
    k, c = _solve_saddle_path(coeffs["Phi_kk"], coeffs["Phi_kc"], coeffs["Phi_kg"], coeffs["Phi_k_kg"],
                               coeffs["coef_c"], coeffs["coef_k"], coeffs["coef_g"], coeffs["coef_kg"],
                               coeffs["coef_g0"], k0, g_full, kg_full_ext,
                               coef_k0=coeffs["coef_k0"], coef_kg0=coeffs["coef_kg0"])
    kg_full = kg_full_ext[:T_sim]
    x = coeffs["a"] * c + coeffs["b"] * g_full  # effective consumption C + phi*GC (optimizers)

    state = np.vstack([k, c, g_full, kg_full])
    gr = coeffs["grads"]
    y, n, i, c_agg = gr["Y"] @ state, gr["N"] @ state, gr["I"] @ state, gr["C"] @ state
    x = x + gr["p"] @ state  # marginal utility of a unit of spending: (C^O + phi*GC)*p
    extra = {}
    if tank:
        extra = dict(CO=100.0 * c, CH=100.0 * (gr["CH"] @ state), NO=100.0 * (gr["NO"] @ state),
                     NH=100.0 * (gr["NH"] @ state) if "NH" in gr else np.zeros(T_sim))
    w = y - n  # wage = MPL, log-deviation

    c_pct = 100.0 * c_agg
    y_pct = 100.0 * y
    i_pct = 100.0 * i
    k_pct = 100.0 * k
    kg_pct = 100.0 * kg_full
    n_pct = 100.0 * n
    w_pct = 100.0 * w
    g_pct = 100.0 * g_full

    r_dev_bp = np.zeros_like(c)
    r_dev_bp[:-1] = 10000.0 * (1.0 + ss_ref.r) * (x[1:] - x[:-1])
    r_dev_bp[-1] = r_dev_bp[-2]

    years = np.arange(T_sim)
    return TransitionPath(years=years, Y=y_pct, C=c_pct, I=i_pct, K=k_pct, KG=kg_pct, N=n_pct,
                           W=w_pct, G=g_pct, r_bp=r_dev_bp, state=state, **extra)


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
                    permanent: bool, duration_years: int = 4, T_sim: int = 300,
                    f_GC: float = 0.0, phi: float = 0.0, lam: float = 0.0,
                    theta_L: Optional[float] = None) -> PolicyExperiment:
    """Full comparative-steady-state + transition-path experiment for a change in total
    government spending from s_G_old to s_G_new, holding the composition fractions
    (f_IG, f_GC, f_GT) fixed. theta_L is calibrated so that hours hit N_target at
    s_G_old, unless a theta_L is given."""
    mix = _tax_mix(financing)
    if theta_L is None:
        s_IG_old, s_GC_old, s_GT_old = s_G_old * f_IG, s_G_old * f_GC, s_G_old * f_GT
        theta_L = calibrate_theta_L(theta_N, delta, r, A, theta_G, s_IG_old, s_GT_old,
                                     mix, N_target, s_GC=s_GC_old, phi=phi, lam=lam)

    ss_old = steady_state_for_policy(theta_N, delta, r, A, theta_G, s_G_old, f_IG, f_GT,
                                       theta_L, mix, f_GC=f_GC, phi=phi, lam=lam)
    ss_new = steady_state_for_policy(theta_N, delta, r, A, theta_G, s_G_new, f_IG, f_GT,
                                       theta_L, mix, f_GC=f_GC, phi=phi, lam=lam)

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
    if lam > 0:
        path.CO = shift(path.CO, ss_ref.CO, ss_old.CO)
        path.CH = shift(path.CH, ss_ref.CH, ss_old.CH)
        path.NO = shift(path.NO, ss_ref.NO, ss_old.NO)
        # N^H can sit at the corner 0, where a % deviation is undefined.
        path.NH = shift(path.NH, ss_ref.NH, ss_old.NH) if (ss_ref.NH > 0 and ss_old.NH > 0) \
            else np.zeros_like(path.NH)

    return PolicyExperiment(ss_old=ss_old, ss_new=ss_new, path=path,
                             multiplier_long_run=multiplier_long_run,
                             multiplier_impact=multiplier_impact)


# --------------------------------------------------------------------------
# 4b. Policy performance: incomes, inequality, and welfare along a path
# --------------------------------------------------------------------------

def household_outcomes(ss: SteadyState, state: np.ndarray) -> dict:
    """Levels of output, prices, and each household type's incomes, consumption, and
    hours along a path. `state` holds log-deviations (k, optimizers' c, g, kg) from
    `ss`, one column per year (TransitionPath.state); each year's static block is solved
    exactly. Incomes are per person: market labor and capital income (before tax), and
    disposable income = the right-hand side of each household's budget constraint
    (spent on consumption including the consumption tax, and on investment).
    With lam=0 everyone is an optimizer, who then also receives the cash welfare."""
    lam = ss.lam
    rows = []
    for k, c, g, kg in np.asarray(state).T:
        e = _tank_eval(ss, k, c, g, kg)
        K, GC, GT = ss.K * np.exp(k), ss.GC * np.exp(g), ss.GT * np.exp(g)
        tau = e["tau_y"]
        T_ls = ss.tax_mix[0] * ss.G * np.exp(g)
        R = ss.theta_K * e["Y"] / K  # rental rate of capital
        KO = K / (1.0 - lam)  # capital per optimizer
        rows.append(dict(
            Y=e["Y"], N=e["N"], w=e["w"], R=R, tau_y=tau, tau_c=e["p"] - 1.0, T=T_ls, GC=GC,
            CO=ss.CO * np.exp(c), NO=e["NO"],
            CH=e["CH"] if lam > 0 else np.nan, NH=e["NH"] if lam > 0 else np.nan,
            labor=e["w"] * e["N"], capital=R * K,
            labor_O=e["w"] * e["NO"], capital_O=R * KO,
            labor_H=e["w"] * e["NH"] if lam > 0 else np.nan,
            income_O=(1.0 - tau) * (e["w"] * e["NO"] + R * KO) - T_ls + (GT if lam == 0 else 0.0),
            income_H=e["w"] * e["NH"] + GT / lam - T_ls if lam > 0 else np.nan,
        ))
    out = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    out["lam"] = lam
    return out


def two_group_gini(lam: float, y_H: np.ndarray, y_O: np.ndarray) -> np.ndarray:
    """Gini coefficient of a population with two groups (shares lam and 1-lam) whose
    members have incomes y_H and y_O: lam*(1-lam)*|y_O - y_H| / mean income."""
    mean = lam * y_H + (1.0 - lam) * y_O
    return lam * (1.0 - lam) * np.abs(y_O - y_H) / mean


def period_utility(C, N, GC, theta_L: float, phi: float):
    """u = ln(C + phi*GC) + theta_L*ln(1-N), the same for both household types."""
    return np.log(C + phi * GC) + theta_L * np.log(1.0 - N)


def welfare_gains(bench: dict, out: dict, beta: float, theta_L: float, phi: float,
                  long_run: bool = False) -> dict:
    """Consumption-equivalent welfare gains, in %, of the path `out` relative to staying
    at the steady state `bench` forever (both from `household_outcomes`, `bench` with a
    single column): the permanent % change in benchmark private consumption that would
    give a household the same lifetime utility as the path. Lifetime utility discounts
    with beta and holds the path's last year forever; long_run=True uses that last year
    only (steady state vs steady state). Returns gains for optimizers "O", hand-to-mouth
    households "H" (when both have lam>0), and "all": one common % change for everyone
    that gives the same population-weighted (with out's lam) lifetime utility."""
    types = ["O", "H"] if (out["lam"] > 0 and bench["lam"] > 0) else ["O"]

    def avg_u(i):  # (1-beta) * lifetime utility, i.e. its per-year equivalent
        u = period_utility(out["C" + i], out["N" + i], out["GC"], theta_L, phi)
        if long_run:
            return u[-1]
        disc = beta ** np.arange(len(u))
        return (1.0 - beta) * np.sum(disc * u) + beta ** len(u) * u[-1]

    def bench_u(i, mu):
        return period_utility(bench["C" + i][0] * (1.0 + mu), bench["N" + i][0],
                              bench["GC"][0], theta_L, phi)

    target = {i: avg_u(i) for i in types}
    gains = {}
    for i in types:
        x = np.exp(target[i] - theta_L * np.log(1.0 - bench["N" + i][0])) - phi * bench["GC"][0]
        gains[i] = 100.0 * (x / bench["C" + i][0] - 1.0)
    if len(types) == 1:
        gains["all"] = gains["O"]
        return gains

    lam = out["lam"]
    goal = lam * target["H"] + (1.0 - lam) * target["O"]
    if not np.isfinite(goal):  # some year of the path has no valid allocation
        gains["all"] = np.nan
        return gains

    def excess(mu):
        return lam * bench_u("H", mu) + (1.0 - lam) * bench_u("O", mu) - goal

    lo, hi = -1.0 + 1e-12, 1.0
    while excess(hi) < 0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) < 0:
            lo = mid
        else:
            hi = mid
    gains["all"] = 100.0 * 0.5 * (lo + hi)
    return gains


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
    supply0 = _supply_side_impl(theta_N, delta, r, A, 0.0, 0.0, s_IG, N_target)
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
        supply = _supply_side_impl(theta_N, delta, r, A, 0.0, theta_G, s_IG, N)
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
