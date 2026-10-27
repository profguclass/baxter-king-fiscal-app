"""
Streamlit teaching app: Baxter & King (1993, AER) "Fiscal Policy in General
Equilibrium" -- an interactive neoclassical growth model with government
purchases, extended with productive public capital, for exploring how fiscal
policy parameters and the *size*, *composition*, and *financing* of
government spending shape the whole economy.

Run locally with:   streamlit run app.py
Deploy for free on Streamlit Community Cloud by pointing it at this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fiscal_model import (
    calibrate_theta_L,
    household_outcomes,
    period_utility,
    run_experiment,
    simulate_transition,
    steady_state_for_policy,
    two_group_gini,
    welfare_gains,
)

st.set_page_config(page_title="Fiscal Policy in General Equilibrium",
                    page_icon="\U0001F3DB️", layout="wide")

# --------------------------------------------------------------------------
# Shared chart helpers (used by both the "Cumulative Effects" and "Marginal
# Effects" time-series panels below).
# --------------------------------------------------------------------------


def focus_range(arrays):
    """A y-axis range hugging the plotted values (with a small margin), so the
    lines fill the chart instead of being squeezed against a zero line."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = vals.min(), vals.max()
    pad = 0.08 * (hi - lo) if hi > lo else max(0.05 * abs(lo), 0.05)
    return [lo - pad, hi + pad]


def line_fig(title, series_specs, yaxis_title, yrs):
    fig = go.Figure()
    for name, arr, color in series_specs:
        fig.add_trace(go.Scatter(x=yrs, y=arr[:len(yrs)], mode="lines+markers",
                                  name=name, line=dict(color=color, width=2.5), marker=dict(size=5)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"),
                       xaxis_title="Years after the change", yaxis_title=yaxis_title,
                       height=540, legend=dict(orientation="h", y=1.18, yanchor="bottom"),
                       margin=dict(t=110, b=20))
    fig.update_yaxes(range=focus_range([arr[:len(yrs)] for _, arr, _ in series_specs]))
    return fig


def render_transition_tabs(path, slider_key, baseline, caption_commodity, caption_labor, caption_financial):
    """Renders the Commodity/Labor/Financial-market tabbed transition-path charts
    shared by both time-series panels, given any TransitionPath-like object."""
    years_to_show = st.slider("Years to display", 5, 100, 25, 5, key=slider_key)
    yrs = path.years[:years_to_show]

    tab1, tab2, tab3 = st.tabs(["Commodity market", "Labor market", "Financial market"])

    # Per-household series exist only for the two-household model (λ>0).
    split_c = [("Optimizers' consumption (Cᴼ)", path.CO, "#0d9488"),
               ("Hand-to-mouth consumption (Cᴴ)", path.CH, "#db2777")] if path.CO is not None else []
    split_n = [("Optimizers' hours (Nᴼ)", path.NO, "#0d9488"),
               ("Hand-to-mouth hours (Nᴴ)", path.NH, "#db2777")] if path.NO is not None else []

    with tab1:
        st.plotly_chart(line_fig(
            "Output, consumption, investment, government spending",
            [("Output (Y)", path.Y, "#2563eb"), ("Consumption (C)", path.C, "#16a34a"),
             ("Investment (I)", path.I, "#f59e0b"), ("Government spending (G)", path.G, "#6b7280")]
            + split_c,
            f"% deviation from {baseline}", yrs), use_container_width=True,
            key=f"{slider_key}_commodity")
        st.caption(caption_commodity)

    with tab2:
        st.plotly_chart(line_fig(
            "Labor input and the real wage",
            [("Labor input (N)", path.N, "#2563eb"), ("Real wage (w)", path.W, "#dc2626")] + split_n,
            f"% deviation from {baseline}", yrs), use_container_width=True,
            key=f"{slider_key}_labor")
        st.caption(caption_labor)

    with tab3:
        st.plotly_chart(line_fig(
            "Private and public capital stocks",
            [("Private capital (K)", path.K, "#2563eb"), ("Public capital (Kᴳ)", path.KG, "#16a34a")],
            f"% deviation from {baseline}", yrs), use_container_width=True,
            key=f"{slider_key}_capital")
        fig_r = go.Figure(go.Scatter(x=yrs, y=path.r_bp[:years_to_show], mode="lines+markers",
                                      line=dict(color="#7c3aed", width=2.5), marker=dict(size=5)))
        fig_r.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_r.update_layout(title=f"Real interest rate, deviation from {baseline}",
                             xaxis_title="Years after the change", yaxis_title="Basis points",
                             height=450, margin=dict(t=50, b=20))
        fig_r.update_yaxes(range=focus_range([path.r_bp[:years_to_show]]))
        st.plotly_chart(fig_r, use_container_width=True, key=f"{slider_key}_rate")
        st.caption(caption_financial)


def regime_transition_path(ss_from, ss_to, ss_reverted=None, permanent=True,
                            duration_years=4, T_sim=200):
    """Perfect-foresight transition path for a switch from the structural regime/
    policy of ss_from to that of ss_to -- used for the "Cumulative Effects" panel's
    time series, where financing/composition/θ_G/r can all differ at once (not just
    G/Y, unlike the ΔG/Y-specific experiment). Financing, composition, θ_G, and r are
    always treated as permanent, immediate switches, since this model has no
    mechanism for a structural parameter to "revert." Only total government spending
    G respects the sidebar's Duration control (item 4.3).

    If permanent, this linearizes around ss_to (the destination regime, already at
    its final G/Y) with g_path=0 throughout -- G jumps to ss_to's share immediately
    and stays. If temporary, G instead reverts to ss_from's share after
    duration_years -- but the economy does NOT return to ss_from itself (financing/
    composition/θ_G don't revert), so the true long-run anchor is a THIRD steady
    state, ss_reverted (required when permanent=False): ss_to's regime at ss_from's
    G/Y level. Mirrors exactly how run_experiment's own temporary-shock case
    linearizes around ss_old (its analogous "reference/target" state) rather than
    ss_new, with g_path returning to exactly 0 (not a nonzero constant) after
    duration_years -- a nonzero forcing that never decays back to 0 breaks the
    saddle-path solver's implicit terminal condition and produces an explosive,
    non-converging path instead of a transition to a well-defined steady state."""
    ss_ref = ss_to if permanent else ss_reverted
    if not permanent and ss_reverted is None:
        raise ValueError("ss_reverted is required when permanent=False")

    k0 = np.log(ss_from.K / ss_ref.K)
    kg0 = np.log(ss_from.KG / ss_ref.KG) if (ss_from.KG > 0 and ss_ref.KG > 0) else 0.0
    if permanent:
        g_path = np.zeros(1)
    else:
        g_level = np.log(ss_to.G / ss_ref.G)
        g_path = np.concatenate([np.full(duration_years, g_level), np.zeros(1)])
    path = simulate_transition(ss_ref, g_path, k0=k0, kg0=kg0, T_sim=T_sim)

    def shift(field_ref, X_ref, X_old):
        return field_ref + 100.0 * np.log(X_ref / X_old)

    path.Y = shift(path.Y, ss_ref.Y, ss_from.Y)
    path.C = shift(path.C, ss_ref.C, ss_from.C)
    path.I = shift(path.I, ss_ref.I, ss_from.I)
    path.K = shift(path.K, ss_ref.K, ss_from.K)
    if ss_ref.KG > 0 and ss_from.KG > 0:
        path.KG = shift(path.KG, ss_ref.KG, ss_from.KG)
    else:
        path.KG = np.zeros_like(path.KG)
    path.N = shift(path.N, ss_ref.N, ss_from.N)
    path.W = shift(path.W, ss_ref.w, ss_from.w)
    if permanent:
        path.G = np.full_like(path.G, 100.0 * np.log(ss_to.G / ss_from.G))
    else:
        path.G = shift(path.G, ss_ref.G, ss_from.G)
    # Per-type series are shown only when both regimes have hand-to-mouth
    # households; otherwise one side has no per-type level to measure against.
    if ss_ref.lam > 0 and ss_from.lam > 0:
        path.CO = shift(path.CO, ss_ref.CO, ss_from.CO)
        path.CH = shift(path.CH, ss_ref.CH, ss_from.CH)
        path.NO = shift(path.NO, ss_ref.NO, ss_from.NO)
        path.NH = shift(path.NH, ss_ref.NH, ss_from.NH) if (ss_ref.NH > 0 and ss_from.NH > 0) \
            else np.zeros_like(path.NH)
    else:
        path.CO = path.CH = path.NO = path.NH = None
    return path

# --------------------------------------------------------------------------
# Sidebar -- model controls
# --------------------------------------------------------------------------

st.sidebar.title("\U0001F3DB️ Model controls")

if "reset_version" not in st.session_state:
    st.session_state["reset_version"] = 0

def wkey(name: str) -> str:
    """Version every widget key with the reset counter, so clicking Reset forces
    Streamlit to fully remount each widget at its coded default -- merely deleting
    session_state and rerunning does not reliably redraw slider handles/labels
    client-side, even though the underlying computed value does reset correctly."""
    return f"{name}_{st.session_state['reset_version']}"

if st.sidebar.button("↺ Reset to benchmark", use_container_width=True):
    st.session_state["reset_version"] += 1
    st.rerun()

# Fixed at the paper's calibration (Table 1): labor share of income,
# depreciation rate, and target steady-state hours worked are no longer sidebar
# controls.
theta_N = 0.58
delta_pct = 10.0
N_target_pct = 20.0

# Benchmark policy scenario: the sidebar's defaults (and Reset target), and the
# fixed reference point Panel 1 compares the current settings against.
BENCH_f_LS_pct, BENCH_f_Y_pct, BENCH_f_C_pct = 50.0, 20.0, 30.0  # shares of revenue
BENCH_theta_G = 0.10
BENCH_r_pct = 6.5
BENCH_s_G_pct = 20.0
BENCH_f_IG_pct, BENCH_f_GC_pct, BENCH_f_GT_pct = 20.0, 50.0, 30.0
BENCH_phi = 0.10
BENCH_lam = 0.30

# 1. Financing rule -----------------------------------------------------
st.sidebar.subheader("1. How is the tax collected?")
st.sidebar.caption("Enter each tax's share of tax revenue (%). Together the three taxes "
                    "always pay for all government spending G. **Lump-sum**: the same "
                    "amount per person, whatever one does. **Income tax**: a rate on "
                    "optimizers' labor and capital income. **Consumption tax**: a rate on "
                    "everyone's private consumption.")
tax_c1, tax_c2, tax_c3 = st.sidebar.columns(3)
in_LS = tax_c1.number_input("Lump-sum (%)", 0.0, 100.0, BENCH_f_LS_pct, 5.0, key=wkey("in_LS"))
in_Y = tax_c2.number_input("Income (%)", 0.0, 100.0, BENCH_f_Y_pct, 5.0, key=wkey("in_Y"))
in_C = tax_c3.number_input("Consumption (%)", 0.0, 100.0, BENCH_f_C_pct, 5.0, key=wkey("in_C"))
tax_total = in_LS + in_Y + in_C
if tax_total <= 0:
    st.sidebar.error("At least one share must be positive; using 100% lump-sum tax.")
    in_LS, in_Y, in_C, tax_total = 100.0, 0.0, 0.0, 100.0
elif abs(tax_total - 100.0) > 1e-9:
    st.sidebar.warning(f"The shares add up to {tax_total:.0f}%, not 100%, so they are "
                        f"rescaled proportionally.")
tax_mix_v = (in_LS / tax_total, in_Y / tax_total, in_C / tax_total)
st.sidebar.caption(f"Lump-sum **{100 * tax_mix_v[0]:.0f}%** · income tax "
                    f"**{100 * tax_mix_v[1]:.0f}%** · consumption tax "
                    f"**{100 * tax_mix_v[2]:.0f}%** of revenue.")

# 2. theta_G -----------------------------------------------------------------
st.sidebar.subheader("2. Productive public capital")
theta_G = st.sidebar.slider(
    "Public-capital productivity, θ_G", 0.00, 0.20, BENCH_theta_G, 0.01,
    help="Y = K^θK·(Kᴳ/K̄ᴳ)^θG·N^θN, with public capital measured relative to a fixed "
         "reference K̄ᴳ of about half the benchmark's public capital, so a higher θG makes "
         "public capital more productive and raises output. θG=0 means public investment "
         "is a pure resource cost with no productivity effect. With θG>0, public "
         "investment (item 5) must be above 0%.",
    key=wkey("theta_G"),
)

# 3. Steady-state real interest rate -----------------------------------------
st.sidebar.subheader("3. Real interest rate")
r_pct = st.sidebar.slider("Steady-state real interest rate, r (%/yr)", 1.0, 12.0, BENCH_r_pct, 0.25, key=wkey("r_pct"))

# 4. Total government spending -----------------------------------------
st.sidebar.subheader("4. Total government spending")
s_G_old_pct = BENCH_s_G_pct  # the benchmark's G/Y, Section 1's reference level
s_G_new_pct = st.sidebar.slider(
    "4.1 Total government spending, G/Y (%)", 5.0, 40.0, BENCH_s_G_pct, 1.0,
    help="The current level of government spending. The benchmark is G/Y = 20%: "
         "Section 1 compares the current settings against it, and Section 2 raises "
         "G/Y from this current level (item 4.2).",
    key=wkey("s_G_new_pct"))
delta_s_G_pct = s_G_new_pct - s_G_old_pct
st.sidebar.caption(f"The taxes in item 1 rise with G/Y to pay for it; the tables show "
                    f"the resulting tax rates. "
                    f"ΔG/Y vs. the {BENCH_s_G_pct:.0f}% benchmark = **{delta_s_G_pct:+.1f}** "
                    f"percentage points.")
dG_pct = st.sidebar.number_input(
    "4.2 Increase in G/Y for Section 2 (percentage points)", 0.5, 10.0, 1.0, 0.5,
    help="Sections 2 and 3 raise G/Y by this much from its current level (item 4.1), "
         "holding every other setting as it is now.",
    key=wkey("dG_pct"))

# 4.3 Duration of the change, alongside G/Y within the same section, since it
# only governs how long a ΔG/Y change lasts. --------------------------------
duration_label = st.sidebar.radio("4.3 Duration of the change", ["Permanent", "Temporary"], key=wkey("duration_label"))
permanent = duration_label == "Permanent"
duration_years = 4
if not permanent:
    duration_years = st.sidebar.slider("Duration (years)", 1, 20, 4, 1, key=wkey("duration_years"))

# 5. Composition of G ------------------------------------------------------
st.sidebar.subheader("5. Composition of G")
st.sidebar.caption("Enter each piece's share of government spending (%): public "
                    "investment Iᴳ (builds public capital Kᴳ), in-kind welfare Gᶜ (government "
                    "consumption such as health care, education, or food programs, which "
                    "households value), and cash welfare Gᵀ (transfers paid to households).")
comp_c1, comp_c2, comp_c3 = st.sidebar.columns(3)
in_IG = comp_c1.number_input("Investment Iᴳ (%)", 0.0, 100.0, BENCH_f_IG_pct, 5.0, key=wkey("in_IG"))
in_GC = comp_c2.number_input("In-kind Gᶜ (%)", 0.0, 100.0, BENCH_f_GC_pct, 5.0, key=wkey("in_GC"))
in_GT = comp_c3.number_input("Cash Gᵀ (%)", 0.0, 100.0, BENCH_f_GT_pct, 5.0, key=wkey("in_GT"))
comp_total = in_IG + in_GC + in_GT
if comp_total <= 0:
    st.sidebar.error("At least one share must be positive; using 100% cash welfare.")
    in_IG, in_GC, in_GT, comp_total = 0.0, 0.0, 100.0, 100.0
elif abs(comp_total - 100.0) > 1e-9:
    st.sidebar.warning(f"The shares add up to {comp_total:.0f}%, not 100%, so they are "
                        f"rescaled proportionally.")
f_IG_pct = 100.0 * in_IG / comp_total
f_GC_pct = 100.0 * in_GC / comp_total
f_GT_pct = 100.0 * in_GT / comp_total

fig_comp = go.Figure()
for name, val, color in [("Investment Iᴳ", f_IG_pct, "#16a34a"),
                          ("In-kind Gᶜ", f_GC_pct, "#f59e0b"),
                          ("Cash Gᵀ", f_GT_pct, "#7c3aed")]:
    fig_comp.add_trace(go.Bar(y=["G/Y"], x=[val], name=name, orientation="h",
                               marker_color=color,
                               text=f"{name.split()[-1]} {val:.0f}%" if val >= 10 else "",
                               textposition="inside", insidetextanchor="middle",
                               hoverinfo="skip"))
fig_comp.update_layout(barmode="stack", height=60, showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(range=[0, 100], showticklabels=False, fixedrange=True),
                        yaxis=dict(showticklabels=False, fixedrange=True))
st.sidebar.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
st.sidebar.caption(f"Public investment Iᴳ **{f_IG_pct:.0f}%** · in-kind welfare Gᶜ "
                    f"**{f_GC_pct:.0f}%** · cash welfare Gᵀ **{f_GT_pct:.0f}%** of G.")

phi = st.sidebar.slider(
    "Value of Gᶜ to households, φ", 0.0, 1.0, BENCH_phi, 0.05,
    help="Utility is ln(C + φ·Gᶜ) + θ_L·ln(1−N). φ=0: Gᶜ is a pure drain (Baxter & King's "
         "\"basic purchases\"). φ=1: Gᶜ is a perfect substitute for private consumption, "
         "so under lump-sum financing it crowds out C one-for-one and leaves output unchanged.",
    key=wkey("phi"),
)

# 6. Household types ----------------------------------------------------------
st.sidebar.subheader("6. Household types")
lam = st.sidebar.slider(
    "Hand-to-mouth share of households, λ", 0.0, 0.9, BENCH_lam, 0.05,
    help="Hand-to-mouth (non-Ricardian) working families own no capital, spend all their "
         "income each period, pay no income tax, and receive all cash welfare Gᵀ. They "
         "pay the lump-sum tax and the consumption tax like everyone else. Optimizers own "
         "the capital, pay all three taxes, and receive no cash welfare. Both types "
         "benefit from Gᶜ. λ=0 is the paper's single representative household.",
    key=wkey("lam"),
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Reading the results:** all quantities in the charts are *percent "
    "deviations from the original (pre-policy) equilibrium*. Because this app "
    "solves the model's exact nonlinear steady state (rather than the paper's "
    "closed-form elasticity approximation, eq. 16), multiplier values will be "
    "close to, but not identical to, the paper's point estimates."
)

# --------------------------------------------------------------------------
# Solve the model
# --------------------------------------------------------------------------

theta_N_v = theta_N
delta_v = delta_pct / 100.0
r_v = r_pct / 100.0
theta_G_v = theta_G
s_G_old_v = s_G_old_pct / 100.0
s_G_new_v = s_G_new_pct / 100.0
N_target_v = N_target_pct / 100.0
f_IG_v, f_GC_v, f_GT_v = f_IG_pct / 100.0, f_GC_pct / 100.0, f_GT_pct / 100.0
phi_v = phi
lam_v = lam
# Section 2's experiment: from the current G/Y (item 4.1) up by dG (item 4.2).
s_G_up_v = s_G_new_v + dG_pct / 100.0

BENCH_f_IG, BENCH_f_GC, BENCH_f_GT = BENCH_f_IG_pct / 100.0, BENCH_f_GC_pct / 100.0, BENCH_f_GT_pct / 100.0
BENCH_s_G = BENCH_s_G_pct / 100.0
BENCH_tax_mix = (BENCH_f_LS_pct / 100.0, BENCH_f_Y_pct / 100.0, BENCH_f_C_pct / 100.0)
# theta_L is calibrated once, so hours are N_target in the benchmark scenario, and
# then held fixed for every other setting -- if each sidebar configuration
# recalibrated its own theta_L, hours would always land back on exactly N_target by
# construction, silently masking any real labor-supply response to composition/
# financing/θ_G/r/φ/λ. Holding theta_L fixed lets N move honestly, and makes
# Section 2's starting point the same as Section 1's "Current settings".
#
# Public capital enters production relative to a fixed reference level K̄ᴳ,
# Y = K^θK·(Kᴳ/K̄ᴳ)^θG·N^θN, which is the model's Y = A·K^θK·(Kᴳ)^θG·N^θN with
# A = K̄ᴳ^(-θG). K̄ᴳ is half the public capital of the benchmark economy without the
# public-capital effect (θG=0), so the benchmark has Kᴳ/K̄ᴳ > 1 and a higher θG raises
# output. theta_L is then calibrated in the benchmark itself.
_theta_L_0 = calibrate_theta_L(theta_N, delta_pct / 100.0, BENCH_r_pct / 100.0, 1.0, 0.0,
                                BENCH_s_G * BENCH_f_IG, BENCH_s_G * BENCH_f_GT,
                                BENCH_tax_mix, N_target_pct / 100.0,
                                s_GC=BENCH_s_G * BENCH_f_GC, phi=BENCH_phi, lam=BENCH_lam)
KG_ref = 0.5 * steady_state_for_policy(theta_N, delta_pct / 100.0, BENCH_r_pct / 100.0, 1.0, 0.0,
                                       BENCH_s_G, BENCH_f_IG, BENCH_f_GT, _theta_L_0, BENCH_tax_mix,
                                       f_GC=BENCH_f_GC, phi=BENCH_phi, lam=BENCH_lam).KG
A_bench = KG_ref ** -BENCH_theta_G
A_v = KG_ref ** -theta_G_v
theta_L_bench = calibrate_theta_L(theta_N, delta_pct / 100.0, BENCH_r_pct / 100.0, A_bench, BENCH_theta_G,
                                   BENCH_s_G * BENCH_f_IG, BENCH_s_G * BENCH_f_GT,
                                   BENCH_tax_mix, N_target_pct / 100.0,
                                   s_GC=BENCH_s_G * BENCH_f_GC, phi=BENCH_phi, lam=BENCH_lam)

error = None
path_error = None
experiment = None
ss_old = ss_new = path = None


def section2_experiment(**kw):
    """Section 2/3's ΔG/Y experiment at the current sidebar settings."""
    return run_experiment(
        theta_N=theta_N_v, delta=delta_v, r=r_v, A=A_v, theta_G=theta_G_v,
        s_G_old=s_G_new_v, s_G_new=s_G_up_v, f_IG=f_IG_v, f_GT=f_GT_v,
        N_target=N_target_v, financing=tax_mix_v, T_sim=200,
        f_GC=f_GC_v, phi=phi_v, lam=lam_v, theta_L=theta_L_bench, **kw)


try:
    experiment = section2_experiment(permanent=permanent, duration_years=duration_years)
    ss_old, ss_new, path = experiment.ss_old, experiment.ss_new, experiment.path
except ValueError as exc:  # the steady state itself is infeasible
    error = str(exc)
except RuntimeError as exc:  # raised by _solve_saddle_path only
    # The transition-path solver can lose saddle-path stability at extreme tax
    # rates (very high income-tax G/Y, especially with a low investment share)
    # even though the two steady states themselves are perfectly well-defined --
    # fall back to the steady states alone, so Panel 1 and Panel 2's comparison
    # table still work; only the transition-dependent sections (impact
    # multiplier, transition charts, duration sensitivity, Panel 3's impact
    # column) show a scoped note below instead of taking down the whole page.
    try:
        ss_old = steady_state_for_policy(theta_N_v, delta_v, r_v, A_v, theta_G_v,
                                          s_G_new_v, f_IG_v, f_GT_v, theta_L_bench,
                                          tax_mix_v, f_GC=f_GC_v, phi=phi_v,
                                          lam=lam_v)
        ss_new = steady_state_for_policy(theta_N_v, delta_v, r_v, A_v, theta_G_v,
                                          s_G_up_v, f_IG_v, f_GT_v, theta_L_bench,
                                          tax_mix_v, f_GC=f_GC_v, phi=phi_v,
                                          lam=lam_v)
        path_error = str(exc)
    except Exception as exc2:  # noqa: BLE001 -- steady state itself is infeasible
        error = str(exc2)

st.title("Fiscal Policy in General Equilibrium")
st.caption("Public Economics · Prof. Dr. Gyun Cheol Gu")

if error:
    st.error(f"These parameters don't yield a valid equilibrium: {error}\n\n"
             "Try a smaller change in G/Y, a lower baseline G/Y, or a different composition.")
    st.stop()

if path_error:
    st.warning(
        f"⚠️ At these settings, the transition-path solver lost **saddle-path "
        f"stability**: *{path_error}* This happens at sufficiently extreme "
        f"income-tax rates (especially combined with a low public-investment "
        f"share), where the linearized model no longer has a well-defined "
        f"perfect-foresight path -- it is a genuine limit of this local "
        f"linearization at extreme calibrations, not a bug. The **steady-state "
        f"comparisons and long-run multipliers below are still exact and "
        f"valid**; only the transition-path charts and impact multipliers need a "
        f"stable path and are skipped. "
        f"Try a lower total government spending (item 4.1), a larger lump-sum "
        f"share (item 1), or a higher public-investment share (item 5) -- all three "
        f"push the instability threshold higher."
    )

# --------------------------------------------------------------------------
# Fixed benchmark steady state (the BENCH_* scenario defined at the top of the
# sidebar): a genuinely fixed reference point, independent of every sidebar
# control, so panel 1 below always shows *some* difference when composition,
# financing, θ_G, r, φ, or λ are moved -- not just when G/Y is.
# --------------------------------------------------------------------------

benchmark_error = None
ss_benchmark = None
ss_current = None
ss_reverted = None
cumulative_path = None
try:
    ss_benchmark = steady_state_for_policy(theta_N, delta_pct / 100.0, BENCH_r_pct / 100.0, A_bench, BENCH_theta_G,
                                            BENCH_s_G, BENCH_f_IG, BENCH_f_GT,
                                            theta_L_bench, BENCH_tax_mix,
                                            f_GC=BENCH_f_GC, phi=BENCH_phi, lam=BENCH_lam)
    # The same (benchmark) theta_L is used for "current settings" -- see theta_L_bench.
    ss_current = steady_state_for_policy(theta_N_v, delta_v, r_v, A_v, theta_G_v,
                                          s_G_new_v, f_IG_v, f_GT_v,
                                          theta_L_bench, tax_mix_v,
                                          f_GC=f_GC_v, phi=phi_v, lam=lam_v)
    # "Reverted" state for the Cumulative Effects time series under a TEMPORARY
    # duration: current settings' own regime (financing/composition/θ_G/r), but at
    # the benchmark's G/Y level -- the true long-run anchor once G/Y reverts, since
    # financing/composition/θ_G don't revert along with it (see
    # regime_transition_path's docstring for why this matters).
    ss_reverted = steady_state_for_policy(theta_N_v, delta_v, r_v, A_v, theta_G_v,
                                           s_G_old_v, f_IG_v, f_GT_v,
                                           theta_L_bench, tax_mix_v,
                                           f_GC=f_GC_v, phi=phi_v, lam=lam_v)
except Exception as exc:  # noqa: BLE001
    benchmark_error = str(exc)

st.header("1. Cumulative Effects")
st.write("Exact, closed-form solution of the model's long-run (“great ratios”) "
         "equilibrium: the **fixed benchmark scenario** (tax revenue 50% lump-sum, 20% "
         "income tax, 30% consumption tax; θ_G=0.10, "
         "r=6.5%, G/Y=20% split into 20% public investment, 50% in-kind welfare and "
         "30% cash welfare, φ=0.10, hand-to-mouth share λ=0.3; θ_N=0.58 and δ=10% as "
         "in Baxter & King) vs. "
         "**whatever the sidebar is currently set to**. Moving *any* sidebar control -- "
         "duration aside -- changes the right-hand column, including the tax mix alone "
         "(item 1): shifting revenue toward the income or consumption tax distorts "
         "households' choices even if G/Y and composition don't move. \"Cumulative\" because this bundles the net "
         "effect of *every* control that currently differs from the benchmark, not just "
         "one at a time -- see \"2. The Marginal Effects of Government Spendings\" below to isolate a single lever.")

if benchmark_error:
    st.error(f"Could not compute the benchmark steady state: {benchmark_error}")
else:
    bench_left_label, bench_right_label = "Benchmark scenario", "Current settings"

    bench_table = pd.DataFrame({
        "Variable": ["Output Y", "Consumption C", "Investment I", "Private capital K",
                     "Public capital Kᴳ", "Government spending G (total)",
                     "  Public investment Iᴳ", "  In-kind welfare Gᶜ", "  Cash welfare Gᵀ",
                     "Labor input N (% of time)", "Real wage w", "Income-tax rate τʸ (%)",
                     "Consumption-tax rate τᶜ (%)", "Lump-sum tax T (per person)"],
        bench_left_label: [ss_benchmark.Y, ss_benchmark.C, ss_benchmark.I, ss_benchmark.K, ss_benchmark.KG,
                            ss_benchmark.G, ss_benchmark.IG, ss_benchmark.GC, ss_benchmark.GT, ss_benchmark.N * 100,
                            ss_benchmark.w, ss_benchmark.tau_y * 100, ss_benchmark.tau_c * 100,
                            ss_benchmark.T],
        bench_right_label: [ss_current.Y, ss_current.C, ss_current.I, ss_current.K, ss_current.KG, ss_current.G,
                             ss_current.IG, ss_current.GC, ss_current.GT, ss_current.N * 100,
                             ss_current.w, ss_current.tau_y * 100, ss_current.tau_c * 100,
                             ss_current.T],
    })
    if ss_benchmark.lam > 0 and ss_current.lam > 0:
        bench_table = pd.concat([bench_table, pd.DataFrame({
            "Variable": ["Optimizers' consumption Cᴼ (per person)", "Hand-to-mouth consumption Cᴴ (per person)",
                         "Optimizers' hours Nᴼ (% of time)", "Hand-to-mouth hours Nᴴ (% of time)"],
            bench_left_label: [ss_benchmark.CO, ss_benchmark.CH, ss_benchmark.NO * 100, ss_benchmark.NH * 100],
            bench_right_label: [ss_current.CO, ss_current.CH, ss_current.NO * 100, ss_current.NH * 100],
        })], ignore_index=True)
    bench_both_zero = (bench_table[bench_left_label] == 0) & (bench_table[bench_right_label] == 0)
    bench_table["% change"] = 100 * (bench_table[bench_right_label] / bench_table[bench_left_label].replace(0, np.nan) - 1)
    bench_table.loc[bench_both_zero, "% change"] = 0.0

    st.dataframe(
        bench_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            bench_left_label: st.column_config.NumberColumn(format="%.6f"),
            bench_right_label: st.column_config.NumberColumn(format="%.6f"),
            "% change": st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )

    bench_bar_vars = ["Output Y", "Consumption C", "Investment I", "Private capital K", "Public capital Kᴳ"]
    bench_pct_change = bench_table.set_index("Variable").loc[bench_bar_vars, "% change"]
    bench_bar_heights = bench_pct_change.fillna(0.0).values
    bench_bar_text = [("n/a" if pd.isna(v) else f"{v:+.2f}%") for v in bench_pct_change.values]
    fig_bench_bar = go.Figure(go.Bar(x=bench_bar_vars, y=bench_bar_heights,
                                      marker_color=["#2563eb" if v >= 0 else "#dc2626" for v in bench_bar_heights],
                                      text=bench_bar_text, textposition="outside"))
    fig_bench_bar.update_layout(title="% change from the benchmark scenario",
                                 yaxis_title="% change", height=380, margin=dict(t=50, b=20))
    st.plotly_chart(fig_bench_bar, use_container_width=True)

    st.subheader("Transition path: benchmark → current settings")
    st.write("Perfect-foresight transition path (log-linearized around the **current "
             "settings**, solved exactly via eigen-decomposition), imagining the "
             "economy starts with the benchmark's capital stocks and this new "
             "regime -- tax mix, composition, φ, θ_G, r, and household types -- is put permanently in "
             "place at year 0. **G/Y specifically respects the Duration control "
             "(item 4.3)**: Permanent means it jumps to the current G/Y and stays; "
             "Temporary means it reverts to the benchmark's 20% share after the "
             "chosen number of years (everything else stays at current settings). "
             "All series are % deviations from the **benchmark** level.")
    try:
        cumulative_path = regime_transition_path(ss_benchmark, ss_current, ss_reverted=ss_reverted,
                                                  permanent=permanent, duration_years=duration_years,
                                                  T_sim=200)
        render_transition_tabs(
            cumulative_path, wkey("years_to_show_cumulative"), "the benchmark",
            caption_commodity="Compare to Baxter & King Figures 2-4, but here the "
                "\"shock\" is the *entire* gap between the benchmark and the sidebar's "
                "current settings, not a single isolated policy lever.",
            caption_labor="Labor and the wage jump immediately to reflect the new "
                "regime's prices; only capital is predetermined and adjusts gradually.",
            caption_financial="Watch how far and how fast private (and, if θ_G>0, "
                "public) capital has to move to reach the new regime's steady state.",
        )
    except Exception as exc:  # noqa: BLE001
        cumulative_path = None
        st.error(f"Could not compute the cumulative transition path: {exc}")

# --------------------------------------------------------------------------
# 2. Marginal Effects: an increase in G/Y (item 4.2) starting from the current
# situation -- every sidebar setting, including the current G/Y, as it is now.
# Reuses ss_old/ss_new/path from the ΔG/Y experiment above; ss_old is the same
# steady state as Section 1's "Current settings".
# --------------------------------------------------------------------------

st.header("2. The Marginal Effects of Government Spendings")
st.write(f"What happens if government spending **rises by {dG_pct:.1f} percentage "
         f"points of output (item 4.2), starting from the current situation**: G/Y goes "
         f"from {s_G_new_pct:.1f}% to {s_G_new_pct + dG_pct:.1f}%, while the tax mix, "
         "composition, φ, θ_G, r, and household types stay exactly as the sidebar has "
         "them now. \"Marginal\" because this isolates the effect of *one* lever -- more "
         "government spending -- in whatever economy the sidebar describes. Change any "
         "other setting (for example λ in item 6) and the starting point changes, so "
         "these effects change too. The left column is the same steady state as "
         "Section 1's \"Current settings\".")

marg_left_label = f"Current situation (G/Y = {s_G_new_pct:.1f}%)"
marg_right_label = f"After the increase (G/Y = {s_G_new_pct + dG_pct:.1f}%)"

marg_table = pd.DataFrame({
    "Variable": ["Output Y", "Consumption C", "Investment I", "Private capital K",
                 "Public capital Kᴳ", "Government spending G (total)",
                 "  Public investment Iᴳ", "  In-kind welfare Gᶜ", "  Cash welfare Gᵀ",
                 "Labor input N (% of time)", "Real wage w", "Income-tax rate τʸ (%)",
                 "Consumption-tax rate τᶜ (%)", "Lump-sum tax T (per person)"],
    marg_left_label: [ss_old.Y, ss_old.C, ss_old.I, ss_old.K, ss_old.KG,
                       ss_old.G, ss_old.IG, ss_old.GC, ss_old.GT, ss_old.N * 100,
                       ss_old.w, ss_old.tau_y * 100, ss_old.tau_c * 100, ss_old.T],
    marg_right_label: [ss_new.Y, ss_new.C, ss_new.I, ss_new.K, ss_new.KG, ss_new.G,
                        ss_new.IG, ss_new.GC, ss_new.GT, ss_new.N * 100,
                        ss_new.w, ss_new.tau_y * 100, ss_new.tau_c * 100, ss_new.T],
})
if lam_v > 0:
    marg_table = pd.concat([marg_table, pd.DataFrame({
        "Variable": ["Optimizers' consumption Cᴼ (per person)", "Hand-to-mouth consumption Cᴴ (per person)",
                     "Optimizers' hours Nᴼ (% of time)", "Hand-to-mouth hours Nᴴ (% of time)"],
        marg_left_label: [ss_old.CO, ss_old.CH, ss_old.NO * 100, ss_old.NH * 100],
        marg_right_label: [ss_new.CO, ss_new.CH, ss_new.NO * 100, ss_new.NH * 100],
    })], ignore_index=True)
marg_both_zero = (marg_table[marg_left_label] == 0) & (marg_table[marg_right_label] == 0)
marg_table["% change"] = 100 * (marg_table[marg_right_label] / marg_table[marg_left_label].replace(0, np.nan) - 1)
marg_table.loc[marg_both_zero, "% change"] = 0.0

st.dataframe(
    marg_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        marg_left_label: st.column_config.NumberColumn(format="%.6f"),
        marg_right_label: st.column_config.NumberColumn(format="%.6f"),
        "% change": st.column_config.NumberColumn(format="%+.2f%%"),
    },
)

marg_bar_vars = ["Output Y", "Consumption C", "Investment I", "Private capital K", "Public capital Kᴳ"]
marg_pct_change = marg_table.set_index("Variable").loc[marg_bar_vars, "% change"]
marg_bar_heights = marg_pct_change.fillna(0.0).values
marg_bar_text = [("n/a" if pd.isna(v) else f"{v:+.2f}%") for v in marg_pct_change.values]
fig_marg_bar = go.Figure(go.Bar(x=marg_bar_vars, y=marg_bar_heights,
                                 marker_color=["#2563eb" if v >= 0 else "#dc2626" for v in marg_bar_heights],
                                 text=marg_bar_text, textposition="outside"))
fig_marg_bar.update_layout(title=f"% change from the current situation after G/Y rises by {dG_pct:.1f} points",
                            yaxis_title="% change", height=380, margin=dict(t=50, b=20))
st.plotly_chart(fig_marg_bar, use_container_width=True)

dG_headline = ss_new.G - ss_old.G
multiplier_long_run = (ss_new.Y - ss_old.Y) / dG_headline if abs(dG_headline) > 1e-12 else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Long-run output multiplier ΔY/ΔG", f"{multiplier_long_run:+.2f}")
c2.metric("Impact (year-0) output multiplier",
          f"{experiment.multiplier_impact:+.2f}" if experiment else "n/a (unstable path)")
c3.metric("Steady-state hours worked N",
          f"{ss_old.N*100:.1f}% → {ss_new.N*100:.1f}%")
c4.metric("Income / consumption tax rate",
          f"{ss_old.tau_y*100:.1f}% / {ss_old.tau_c*100:.1f}% → "
          f"{ss_new.tau_y*100:.1f}% / {ss_new.tau_c*100:.1f}%")

if multiplier_long_run > 1:
    st.success(
        f"**A one-dollar permanent increase in government spending raises long-run "
        f"output by ≈ ${multiplier_long_run:.2f}.** This exceed-1 "
        f"multiplier is the paper's central, surprising result: it comes from labor "
        f"supply rising (negative wealth effect), and — if θ_G>0 — public investment "
        f"directly raising the productivity of private capital and labor."
    )
elif multiplier_long_run < 0:
    st.warning(
        f"**Output *falls* by ≈ ${-multiplier_long_run:.2f} for every dollar "
        f"of new spending.** The higher income-tax rate (on labor and capital income) "
        f"and consumption-tax rate (which lowers what an hour of work buys) depress work "
        f"effort and capital formation by more than the direct resource cost of the "
        f"spending itself."
    )

if path is None:
    st.info("Transition path and impact multipliers are unavailable at these settings "
             "(see the saddle-path note above) -- only long-run, steady-state "
             "comparisons are shown.")
else:
    st.subheader("Transition path: isolating the ΔG/Y shock")
    st.write("Perfect-foresight transition path (log-linearized around the relevant steady "
             "state, solved exactly via eigen-decomposition -- no simulation noise). "
             "All series are % deviations from the *current situation* (the steady state "
             "set by the sidebar, before the extra rise in G/Y), matching "
             "Figures 2-4 of the paper.")
    render_transition_tabs(
        path, wkey("years_to_show_marginal"), "the current situation",
        caption_commodity="Compare to Baxter & King Figure 2 (permanent) / Figure 3 "
            "(temporary war) / Figure 4 (GRH). Watch the investment 'accelerator boom' "
            "on impact when the shock is permanent and lump-sum financed.",
        caption_labor="A permanent, lump-sum-financed increase in G is a negative wealth "
            "effect: households work more and consume less, so labor rises and (with "
            "capital predetermined) the wage falls on impact.",
        caption_financial="An unanticipated permanent increase in G should raise short "
            "real rates on impact -- the model's sharpest, most testable empirical "
            "prediction (Section III.E of the paper).",
    )

# --------------------------------------------------------------------------
# 3. Multiplier effects of government spending, by variable
# --------------------------------------------------------------------------

st.header("3. Multiplier Effects")
st.write(
    "How much does the increase in G/Y (item 4.2) move each variable, per dollar of "
    r"new spending: $\Delta X/\Delta G$ for $X\in\{Y,C,I,K,K^G\}$, both "
    "**long-run** (comparing the two steady states) and **on impact** (year 0 of "
    "the transition path above) -- same experiment as \"2. The Marginal Effects of Government Spendings\", "
    "holding the tax mix, composition, φ, θ_G, r, and household types fixed at the "
    "sidebar's current settings. With hand-to-mouth households (item 6), the two "
    "household types' consumption rows are group totals, so they add up to the "
    "consumption row."
)

if path is None:
    st.info("Impact multipliers are unavailable at these settings (see the "
             "saddle-path note above) -- only long-run multipliers are shown.")

dG_total = ss_new.G - ss_old.G

def _long_run_mult(x_old, x_new):
    return (x_new - x_old) / dG_total if abs(dG_total) > 1e-12 else float("nan")

def _impact_mult(field_path, x_old):
    if field_path is None:
        return float("nan")
    # field_path is already expressed as a % deviation from ss_old (see
    # run_experiment's `shift` step), so this recovers the year-0 level directly.
    x0 = x_old * float(np.exp(field_path[0] / 100.0))
    return (x0 - x_old) / dG_total if abs(dG_total) > 1e-12 else float("nan")

mult_vars = [
    ("Output Y", ss_old.Y, ss_new.Y, path.Y if path else None),
    ("Consumption C", ss_old.C, ss_new.C, path.C if path else None),
    ("Investment I", ss_old.I, ss_new.I, path.I if path else None),
    ("Private capital K", ss_old.K, ss_new.K, path.K if path else None),
    ("Public capital Kᴳ", ss_old.KG, ss_new.KG, path.KG if path else None),
]
if lam_v > 0:
    # Group totals, so the two rows add up to the aggregate consumption row.
    mult_vars += [
        ("Optimizers' consumption (1−λ)·Cᴼ", (1 - lam_v) * ss_old.CO, (1 - lam_v) * ss_new.CO,
         path.CO if path else None),
        ("Hand-to-mouth consumption λ·Cᴴ", lam_v * ss_old.CH, lam_v * ss_new.CH,
         path.CH if path else None),
    ]
mult_table = pd.DataFrame({
    "Variable": [name for name, *_ in mult_vars],
    "Long-run ΔX/ΔG": [_long_run_mult(x_old, x_new) for _, x_old, x_new, _ in mult_vars],
    "Impact (year-0) ΔX/ΔG": [_impact_mult(fp, x_old) for _, x_old, _, fp in mult_vars],
})

st.dataframe(
    mult_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Long-run ΔX/ΔG": st.column_config.NumberColumn(format="%+.2f"),
        "Impact (year-0) ΔX/ΔG": st.column_config.NumberColumn(format="%+.2f"),
    },
)

fig_mult = go.Figure()
fig_mult.add_trace(go.Bar(name="Long-run", x=mult_table["Variable"],
                           y=mult_table["Long-run ΔX/ΔG"], marker_color="#2563eb"))
fig_mult.add_trace(go.Bar(name="Impact (year-0)", x=mult_table["Variable"],
                           y=mult_table["Impact (year-0) ΔX/ΔG"], marker_color="#f59e0b"))
fig_mult.add_hline(y=0, line_dash="dot", line_color="gray")
fig_mult.update_layout(barmode="group", title="Multipliers ΔX/ΔG: change in each variable per dollar of new government spending",
                        yaxis_title="ΔX / ΔG", height=420, margin=dict(t=50, b=20))
st.plotly_chart(fig_mult, use_container_width=True)

st.caption(
    "A multiplier above +1.00 for Output means a marginal dollar of government "
    "spending raises long-run output by more than a dollar -- the paper's central, "
    "surprising result under lump-sum financing (labor supply rises from the "
    "negative wealth effect, and, if θ_G>0, public investment directly raises "
    "productivity). Investment's impact multiplier can be large even when its "
    "long-run multiplier is small: capital 'overshoots' toward the new steady "
    "state on impact (the accelerator boom, Baxter & King Figure 2), then its net "
    "long-run change is only what is needed to sustain the new higher K."
)

# --------------------------------------------------------------------------
# 4. Policy performance along Section 1's transition (benchmark -> current
# settings): growth, incomes by household type, inequality, and welfare.
# --------------------------------------------------------------------------

st.header("4. Policy Performance: Growth, Inequality, and Welfare")
st.write(
    "How does the policy in Section 1 -- moving from the **benchmark scenario** to the "
    "**current settings** -- perform year by year? This section follows the same "
    "transition path as Section 1 and asks four questions: Does the economy grow? "
    "Whose income grows? Does inequality between the two household types rise or "
    "fall? And are households better or worse off overall? Year 0 is the year the "
    "policy changes; the economy was at the benchmark steady state before that. "
    "At the default settings the current settings *are* the benchmark, so every line "
    "is flat: change a sidebar setting to create a policy to evaluate."
)
st.caption(
    "Levels here come from the model's exact equations evaluated along Section 1's "
    "path of capital and optimizers' consumption, so for large policy changes the "
    "first years can differ a little from Section 1's charts, which use a linear "
    "approximation for every variable. Both end at the same new steady state."
)

if benchmark_error or cumulative_path is None:
    st.info("Policy performance needs Section 1's transition path, which is "
            "unavailable at these settings.")
else:
    perf = household_outcomes(ss_current if permanent else ss_reverted, cumulative_path.state)
    perf_bench = household_outcomes(ss_benchmark, np.zeros((4, 1)))
    for outcomes in (perf, perf_bench):
        outcomes["prod"] = outcomes["Y"] / outcomes["N"]  # labor productivity
    two_types = ss_benchmark.lam > 0 and ss_current.lam > 0
    perf_bad = ~np.isfinite(perf["Y"]) | ~np.isfinite(perf["income_O"])
    if two_types:
        perf_bad |= ~np.isfinite(perf["CH"])
    if perf_bad.any():
        st.warning(
            f"In {int(perf_bad.sum())} year(s) of the transition (from year "
            f"{int(np.argmax(perf_bad))}) the model's exact equations have no solution: "
            "the economy starts so far from its new steady state (for example, with far "
            "more capital than it needs) that some household would have to work negative "
            "hours. Those years are left blank and welfare cannot be computed. Try a "
            "smaller policy change."
        )
    perf_years = st.slider("Years to display", 5, 100, 25, 5, key=wkey("years_to_show_performance"))
    yrs = np.arange(perf_years)
    yrs_lvl = np.arange(-1, perf_years)  # year -1 = the benchmark steady state

    def yoy(key):
        """Year-over-year growth (%), year 0 measured against the benchmark."""
        x = perf[key]
        prev = np.concatenate([perf_bench[key], x[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            g = 100.0 * (x / prev - 1.0)
        return np.where(np.isfinite(g), g, np.nan)

    def level(arr_path, arr_bench):
        """A level series with the benchmark value prepended as year -1."""
        return np.concatenate([np.atleast_1d(arr_bench), arr_path])

    cumulative_rules = {
        "growth": ("Cumulative growth", "compounded growth over the years shown, "
                   "i.e. the total % change from the benchmark to the last year"),
        "change": ("Cumulative change", "the change from the benchmark (year −1) to the last year"),
        "sum": ("Cumulative sum", "the sum over the years shown"),
    }

    def cumulative(arr, how):
        a = np.asarray(arr, dtype=float)
        if how == "growth":
            return 100.0 * (np.prod(1.0 + a / 100.0) - 1.0)
        if how == "change":
            return a[-1] - a[0]
        return a.sum()

    def perf_chart(title, specs, yaxis_title, x, key, how):
        """A Section 4 line chart plus, below it, a table of the same numbers
        with a final row for the cumulative effect over the years shown."""
        st.plotly_chart(line_fig(title, specs, yaxis_title, x), use_container_width=True, key=key)
        label, rule = cumulative_rules[how]
        table = pd.DataFrame({"Year": [str(v) for v in x] + [f"{label}, {x[0]} to {x[-1]}"],
                              **{name: np.append(arr[:len(x)], cumulative(arr[:len(x)], how))
                                 for name, arr, _ in specs}})
        with st.expander(f"Data table: {title}"):
            st.caption(f"Units: {yaxis_title}." + (" Year −1 is the benchmark." if x[0] < 0 else "")
                       + f" The last row is {rule}.")
            st.dataframe(table, hide_index=True, use_container_width=True,
                         column_config={name: st.column_config.NumberColumn(format="%.3f")
                                        for name, _, _ in specs})

    st.caption("All incomes, hours, and welfare below are **per person** (per capita): "
               "the population is normalized to 1, so the economy-wide series are averages "
               "over everyone, and each household type's series is for one member of that "
               "type.")
    if not two_types:
        st.info("The current settings have no hand-to-mouth households (λ = 0, item 6), "
                "so only economy-wide results are shown. Set λ above 0 to compare the "
                "two household types.")

    tab_g, tab_i, tab_h, tab_q, tab_w = st.tabs(
        ["Growth", "Incomes", "Hours", "Inequality", "Welfare"])

    with tab_g:
        perf_chart("Output and labor productivity: year-over-year growth",
            [("Output per person Y", yoy("Y"), "#2563eb"),
             ("Labor productivity Y/N", yoy("prod"), "#16a34a"),
             ("Hours per person N", yoy("N"), "#f59e0b")],
            "% growth from the previous year", yrs, "perf_growth", "growth")
        st.caption(
            "Output growth splits into two parts: growth in labor productivity (output per "
            "hour, Y/N) and growth in hours worked. Productivity grows when capital per "
            "hour rises -- private capital K through saving, public capital Kᴳ through "
            "public investment. Year 0 shows the jump when the policy starts: hours can "
            "change at once, but capital cannot. After that, growth comes only from "
            "capital adjusting, so it fades to zero as the economy reaches its new "
            "steady state (this model has no long-run technological progress)."
        )

    with tab_i:
        labor_specs = [("Labor income per person, average wN", yoy("labor"), "#2563eb"),
                       ("Capital income per optimizer RKᴼ", yoy("capital_O"), "#7c3aed")]
        if two_types:
            labor_specs[1:1] = [("Labor income per optimizer wNᴼ", yoy("labor_O"), "#0d9488"),
                                ("Labor income per hand-to-mouth household wNᴴ", yoy("labor_H"), "#db2777")]
        perf_chart("Labor and capital income per person (before tax): year-over-year growth",
            labor_specs, "% growth from the previous year", yrs, "perf_factor_income", "growth")
        disp_specs = [("Per optimizer", yoy("income_O"), "#0d9488")]
        if two_types:
            disp_specs.append(("Per hand-to-mouth household", yoy("income_H"), "#db2777"))
        perf_chart("Disposable income per person: year-over-year growth",
            disp_specs, "% growth from the previous year", yrs, "perf_disp_income", "growth")
        st.caption(
            "Labor income is the wage times hours; capital income is the rental rate "
            "times the capital stock. Only optimizers own capital, so all capital income "
            "is theirs: each optimizer owns Kᴼ = K/(1−λ). **Disposable income** is what each household can spend, the "
            "right-hand side of its budget constraint (Appendix A.3-A.4): for optimizers, "
            "after-income-tax labor and capital income minus the lump-sum tax; for "
            "hand-to-mouth households, their wage income plus cash welfare minus the "
            "lump-sum tax. Both then pay the consumption tax on what they consume out "
            "of it. A policy that raises the wage helps both groups' labor "
            "income; a policy that raises the capital stock helps optimizers' capital "
            "income; cash welfare and taxes move disposable income directly."
        )

    with tab_h:
        if two_types:
            perf_chart("Hours per person, by household type (% of time)",
                [("Per optimizer Nᴼ", 100 * level(perf["NO"], perf_bench["NO"]), "#0d9488"),
                 ("Per hand-to-mouth household Nᴴ", 100 * level(perf["NH"], perf_bench["NH"]), "#db2777"),
                 ("Per person, average N", 100 * level(perf["N"], perf_bench["N"]), "#2563eb")],
                "% of time endowment", yrs_lvl, "perf_hours", "change")
            perf_chart("Hours gap: optimizers minus hand-to-mouth (Nᴼ − Nᴴ)",
                [("Hours gap", 100 * level(perf["NO"] - perf["NH"],
                                            perf_bench["NO"] - perf_bench["NH"]), "#6b7280")],
                "Percentage points of time", yrs_lvl, "perf_hours_gap", "change")
            st.caption(
                "Year −1 is the benchmark. Both types face the same wage, so differences "
                "in hours come from differences in wealth and taxes. Cash welfare and a "
                "lower lump-sum tax make hand-to-mouth households richer, so they work "
                "less. An income tax lowers optimizers' after-tax wage, so they work "
                "less; hand-to-mouth households pay no income tax, so the tax alone does "
                "not discourage their work. A consumption tax lowers what an hour of work "
                "buys for both types alike. A positive gap means optimizers work more."
            )
        else:
            st.info("Hours by household type need λ > 0 (item 6).")

    with tab_q:
        if two_types:
            perf_chart("Optimizer / hand-to-mouth ratios (per person)",
                [("Disposable income ratio", level(perf["income_O"] / perf["income_H"],
                                                   perf_bench["income_O"] / perf_bench["income_H"]), "#2563eb"),
                 ("Consumption ratio Cᴼ/Cᴴ", level(perf["CO"] / perf["CH"],
                                                    perf_bench["CO"] / perf_bench["CH"]), "#16a34a")],
                "Ratio (1 = equal)", yrs_lvl, "perf_ratios", "change")
            perf_chart("Gini coefficient of disposable income (two groups)",
                [("Gini", level(two_group_gini(ss_current.lam, perf["income_H"], perf["income_O"]),
                                two_group_gini(ss_benchmark.lam, perf_bench["income_H"], perf_bench["income_O"])),
                  "#dc2626")],
                "Gini (0 = equal)", yrs_lvl, "perf_gini", "change")
            st.caption(
                "Two ways to measure inequality between the two household types. The "
                "**ratio** compares an average optimizer with an average hand-to-mouth "
                "household (2 means twice as much). The **Gini coefficient** runs from 0 "
                "(everyone has the same income) to 1; with only two groups it equals "
                "λ(1−λ)·|yᴼ − yᴴ| / (average income), so it also depends on how large "
                "each group is. It ignores inequality *within* each group, so it is a "
                "lower bound on overall inequality. Optimizers' disposable income "
                "includes the part they save and invest, so the consumption ratio is "
                "often a better guide to living standards. Year −1 is the benchmark."
            )
        else:
            st.info("Inequality between household types needs λ > 0 (item 6).")

    with tab_w:
        if two_types:
            beta_w = 1.0 / (1.0 + r_v)
            gains_all = welfare_gains(perf_bench, perf, beta_w, theta_L_bench, phi_v)
            gains_lr = welfare_gains(perf_bench, perf, beta_w, theta_L_bench, phi_v, long_run=True)
            welfare_table = pd.DataFrame({
                "Household": ["Per optimizer", "Per hand-to-mouth household",
                              "Per person, average over everyone"],
                "Including the transition": [gains_all["O"], gains_all["H"], gains_all["all"]],
                "Long run only (new steady state)": [gains_lr["O"], gains_lr["H"], gains_lr["all"]],
            })
            w1, w2, w3 = st.columns(3)
            for col, label, key in ((w1, "Per optimizer", "O"), (w2, "Per hand-to-mouth household", "H"),
                                    (w3, "Per person (average)", "all")):
                col.metric(label, f"{gains_all[key]:+.2f}%" if np.isfinite(gains_all[key]) else "n/a")
            st.dataframe(welfare_table, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%+.2f%%")
                                        for c in welfare_table.columns[1:]})
            st.markdown(
                "**Consumption-equivalent welfare gain.** Utility has no natural units, so "
                "welfare is measured in consumption: *by what percent would a household's "
                "consumption have to rise, every year forever, in the benchmark economy, to "
                "make it exactly as happy as living through this policy?* A positive "
                "number means the policy makes the household better off. **Including the "
                "transition** adds up utility year by year from year 0 on, discounted "
                "with β = 1/(1+r); this is what a household alive when the policy starts "
                "gets. **Long run only** compares the two steady states and ignores the "
                "years of adjustment, which can matter a lot -- for example, when capital "
                "is run down, households consume more in the early years. **Per person "
                "(average)** is one common percent for both groups that gives the same "
                "average utility per person, weighting the types by their population "
                "shares (1−λ and λ)."
            )
            u_O = period_utility(perf["CO"], perf["NO"], perf["GC"], theta_L_bench, phi_v)
            u_H = period_utility(perf["CH"], perf["NH"], perf["GC"], theta_L_bench, phi_v)
            u_O_b = period_utility(perf_bench["CO"], perf_bench["NO"], perf_bench["GC"], theta_L_bench, phi_v)
            u_H_b = period_utility(perf_bench["CH"], perf_bench["NH"], perf_bench["GC"], theta_L_bench, phi_v)
            # Average utility per person, each scenario with its own population shares.
            u_avg = ss_current.lam * u_H + (1 - ss_current.lam) * u_O
            u_avg_b = ss_benchmark.lam * u_H_b + (1 - ss_benchmark.lam) * u_O_b
            perf_chart("Period utility per person, relative to the benchmark",
                [("Per optimizer", u_O - u_O_b, "#0d9488"),
                 ("Per hand-to-mouth household", u_H - u_H_b, "#db2777"),
                 ("Per person, average", u_avg - u_avg_b, "#2563eb")],
                "Change in utility (utils)", yrs, "perf_utility", "sum")
            st.caption(
                "Each year's utility, ln(C + φGᶜ) + θ_L·ln(1 − N), minus its benchmark "
                "value. It rises with own consumption, in-kind welfare, and leisure. Above "
                "zero means that year is better than the benchmark. A change of 0.01 is "
                "worth about 1% more consumption-plus-in-kind-welfare (C + φGᶜ) in that "
                "year. Both the benchmark and the policy are valued with the current φ "
                "(item 5), so the comparison uses one set of preferences."
            )
        else:
            st.info("Welfare by household type needs λ > 0 (item 6), since the benchmark "
                    "has two household types and the current settings have one.")

# --------------------------------------------------------------------------
# Footer / methodology note
# --------------------------------------------------------------------------

st.divider()
with st.expander("ℹ️ Methodology notes"):
    st.markdown(
        """
- **Steady state / equilibrium** is solved exactly (closed form, plus a small
  fixed-point iteration when θ_G>0 since public capital Kᴳ=Iᴳ/δ depends on Y) -- no
  numerical root-finding failure modes.
- **Transition dynamics** are the standard log-linearization around a steady state,
  solved via eigen-decomposition of the reduced-form transition matrix
  (King-Plosser-Rebelo / Blanchard-Kahn method); public capital's own law of motion
  decouples from the private-capital/consumption saddle path (it follows the exogenous
  spending path directly) so no 3-variable generalization of the eigen-decomposition is
  needed.
- **Tax mix (item 1)**: three taxes together raise revenue equal to total government
  spending G every year, each a fixed share of it (benchmark 50% lump-sum, 20% income
  tax, 30% consumption tax); their rates adjust year by year so that they do. The
  lump-sum tax is a true non-distorting poll tax. The income tax puts a (1-τʸ) wedge
  on both the labor-leisure and the saving (Euler) margin. The consumption tax puts a
  1/(1+τᶜ) wedge on the labor-leisure margin only: it lowers what an hour of work
  buys, while a rate that stays constant over time leaves saving undistorted. Because
  the wedges sit on the *margin*, not on net revenue, changing the tax mix alone
  reshapes the whole steady state.
- **Composition (item 5)**: total government spending G/Y is split into public
  investment Iᴳ (accumulates into Kᴳ, productivity-enhancing if θ_G>0; Kᴳ enters
  production as (Kᴳ/K̄ᴳ)^θG relative to a fixed reference K̄ᴳ of about half the
  benchmark's public capital, so a higher θ_G raises output, and θ_G>0 needs a
  positive Iᴳ share), in-kind
  welfare Gᶜ (government consumption, valued by households through C+φ·Gᶜ), and cash
  welfare Gᵀ (transfers paid to households). The three shares are typed in as
  percentages; if they don't add up to 100% they are rescaled proportionally. Iᴳ and
  Gᶜ use up output (Y=C+I+Iᴳ+Gᶜ); Gᵀ only moves income to households. With φ=0, Gᶜ
  is Baxter & King's pure-drain "basic purchases"; with φ=1 and lump-sum financing it
  crowds out private consumption one-for-one and leaves output unchanged.
- **Household types (item 6)**: a share λ of households are hand-to-mouth working
  families (no capital, no saving, no income tax, all of the cash welfare Gᵀ); the
  rest are optimizers (own the capital, pay the income tax, no cash welfare).
  Everyone pays the same per-person lump-sum tax and the consumption tax. Both types
  benefit from Gᶜ. Transition paths are linearized *numerically* around the steady
  state; with λ=0 and a single lump-sum or income tax this reproduces the analytic
  linearization. See Appendix A.3-A.4.
- **Panel 1 (Cumulative Effects)** compares a *fixed* benchmark scenario (tax revenue
  50% lump-sum, 20% income tax, 30% consumption tax; θ_G=0.10, r=6.5%, G/Y=20% split
  20/50/30 into public investment,
  in-kind welfare and cash welfare, φ=0.10, λ=0.3; θ_N=0.58, δ=10% and 20% target
  hours as in Baxter & King) against whatever the sidebar is *currently*
  set to, so every control (tax mix, composition, φ, θ_G, r, G/Y, household types)
  shows up as a difference there -- including the tax mix alone, since the income-
  and consumption-tax wedges distort households' margins independent of net revenue. Its transition-path
  chart imagines the economy starting at the benchmark's capital stocks with the
  current regime put permanently in place.
- **Panel 2 (The Marginal Effects of Government Spendings)** instead isolates the
  effect of more government spending in the *current* situation: starting from the
  current settings (including the current G/Y, item 4.1), G/Y rises by the amount in
  item 4.2 while every other setting (tax mix, composition, φ, θ_G, r, household
  types) stays as it is. Its starting point is the same steady state as Panel 1's
  "Current settings", so changing any setting changes these marginal effects.
- **Panel 3 (Multiplier Effects)** uses the same G/Y increase and reports, for each
  variable X (Y, C, I, K, Kᴳ, and with λ>0 each household type's consumption), the
  multiplier ΔX/ΔG -- the change in X per dollar of new government spending -- both
  the long-run multiplier (from the two steady states) and the impact multiplier
  (from year 0 of the transition path) for each.
- **Panel 4 (Policy Performance)** follows Panel 1's transition path (benchmark →
  current settings) and reports, year by year, growth of output, labor productivity,
  hours, and labor/capital/disposable income by household type; the hours gap;
  inequality between the types (income and consumption ratios and a two-group Gini);
  and welfare (period utility and consumption-equivalent gains, with and without the
  transition). Levels come from solving each year's static equations exactly at the
  path's capital and optimizers' consumption. Welfare uses the current φ for both
  scenarios and discounts with β = 1/(1+r).
- Reported multipliers are close to, but will not exactly reproduce, the point
  estimates in the paper's Tables 2-4, which are built from a closed-form elasticity
  approximation (equation 16/16'); this app instead solves the *exact* nonlinear
  steady state and the *exact* linear saddle path implied by the same first-order
  conditions.

**Citation:** Baxter, Marianne, and Robert G. King. 1993. "Fiscal Policy in
General Equilibrium." *American Economic Review* 83 (3): 315-334.
        """
    )

# --------------------------------------------------------------------------
# Appendix: the model's equations, written out and explained
# --------------------------------------------------------------------------

st.divider()
st.header("Appendix: The model, agent by agent")
st.markdown(
    "The economy has four kinds of agents: **firms**, **optimizing households**, "
    "**hand-to-mouth households**, and the **government**. Each one solves its own "
    "problem: it chooses something to do the best it can, subject to a constraint. "
    "An **equilibrium** is a set of prices and quantities at which every agent's "
    "problem is solved at the same time and every market clears. Each section below "
    "states one agent's **problem** and then its **solution** (the conditions its "
    "best choice must satisfy)."
)

st.subheader("A.1 Notation")
st.markdown(
    r"""
| Symbol | Meaning |
|---|---|
| $Y_t,\ C_t,\ I_t$ | Output, private consumption, private investment |
| $K_t,\ N_t$ | Private capital, hours worked (a fraction of the time endowment; $1-N_t$ is leisure) |
| $w_t,\ R_t$ | Real wage, rental rate of capital |
| $G_t = I^G_t+G^C_t+G^T_t$ | Government spending: public investment + in-kind welfare + cash welfare |
| $K^G_t$ | Public capital (built by public investment $I^G_t$) |
| $\phi$ | How much households value in-kind welfare, relative to their own consumption ($0\le\phi\le1$) |
| $\lambda$ | Share of households that are hand-to-mouth |
| $T_t,\ \tau^Y_t,\ \tau^C_t$ | Lump-sum tax per person, income-tax rate, consumption-tax rate |
| $\beta=1/(1+r),\ \delta$ | Discount factor ($r$ = steady-state real interest rate), depreciation rate |
| $\theta_N,\ \theta_K=1-\theta_N,\ \theta_G$ | Output elasticities of labor, private capital, public capital |
| $\theta_L$ | Weight on leisure in utility |
| superscript $O$ / $H$ | An optimizing / a hand-to-mouth household |
"""
)

st.subheader("A.2 Firms")
st.markdown(
    "**Problem.** A firm rents capital and hires labor to maximize profit. Public "
    "capital $K^G_t$ is provided by the government, so the firm uses it for free:"
)
st.latex(r"\max_{K_t,\,N_t}\ \ K_t^{\theta_K}\left(\frac{K^G_t}{\bar K^G}\right)^{\theta_G}N_t^{\theta_N} - w_t N_t - R_t K_t")
st.markdown(
    "Public capital is measured relative to a fixed reference amount $\\bar K^G$, about "
    "half of the benchmark's public capital. The benchmark therefore has "
    "$K^G/\\bar K^G>1$, so the public-capital term is above 1: public capital makes "
    "firms more productive, and a higher $\\theta_G$ (more productive public capital) "
    "raises output. $\\theta_G$ is also the percent rise in output from a 1% rise in "
    "public capital. Firms cannot produce without public capital when $\\theta_G>0$, "
    "so public investment must then be positive."
)
st.markdown("**Solution.** Each input is paid its marginal product:")
st.latex(r"w_t = \theta_N\,\frac{Y_t}{N_t}, \qquad R_t = \theta_K\,\frac{Y_t}{K_t}")
st.markdown(
    "Because $\\theta_K+\\theta_N=1$, wages plus rents use up all of output: "
    "$w_tN_t + R_tK_t = Y_t$. More public capital ($\\theta_G>0$) raises output, and "
    "therefore both the wage and the rental rate."
)

st.subheader("A.3 Optimizing households (share 1 − λ)")
st.markdown(
    "**Problem.** Optimizers own all the capital, so they receive all capital income. "
    "They choose consumption, hours, and investment to maximize lifetime utility. "
    "They pay all three taxes and receive no cash welfare:"
)
st.latex(r"\max\ \sum_{t=0}^{\infty} \beta^t \Big[\ln\big(C^O_t + \phi\,G^C_t\big) + \theta_L \ln\big(1-N^O_t\big)\Big]")
st.latex(r"\text{subject to}\quad (1+\tau^C_t)\,C^O_t + I^O_t = (1-\tau^Y_t)\big(w_t N^O_t + R_t K^O_t\big) - T_t, \qquad K^O_{t+1} = (1-\delta)\,K^O_t + I^O_t")
st.markdown(
    "They pay the lump-sum tax $T_t$, the income tax at rate $\\tau^Y_t$ on their labor "
    "and capital income, and the consumption tax at rate $\\tau^C_t$ on what they "
    "consume (each unit of consumption costs $1+\\tau^C_t$). "
    "In-kind welfare enters utility next to their own consumption: one unit of "
    "$G^C_t$ is worth $\\phi$ units of $C^O_t$. (With $\\lambda=0$ there are no "
    "hand-to-mouth households, so optimizers are everyone and also receive the cash "
    "welfare $G^T_t$ in their budget: the Baxter & King representative household.)"
)
st.markdown("**Solution.** Two conditions:")
st.latex(r"\text{Labor-leisure:}\qquad \theta_L\,\frac{C^O_t+\phi\,G^C_t}{1-N^O_t} = \frac{(1-\tau^Y_t)\,w_t}{1+\tau^C_t}")
st.latex(r"\text{Euler equation:}\qquad \frac{1}{(1+\tau^C_t)(C^O_t+\phi\,G^C_t)} = \beta\,\frac{1+(1-\tau^Y_{t+1})\,R_{t+1}-\delta}{(1+\tau^C_{t+1})(C^O_{t+1}+\phi\,G^C_{t+1})}")
st.markdown(
    "*Labor-leisure*: the value of an extra hour of leisure (left) equals what an "
    "extra hour of work earns after tax (right). *Euler*: giving up one unit of "
    "consumption today to invest must be worth exactly as much as the extra "
    "consumption it buys tomorrow. The income tax appears in both conditions, so it "
    "discourages both work and saving. The consumption tax lowers what an hour of work "
    "buys, so it discourages work much like a tax on wages; when its rate is the same "
    "today and tomorrow it cancels out of the Euler equation, so it does not discourage "
    "saving. A lump-sum tax appears in neither condition, so it only makes the "
    "household poorer: it consumes less and works more, but its choices at the margin "
    "are not distorted."
)

st.subheader("A.4 Hand-to-mouth households (share λ)")
st.markdown(
    "**Problem.** Hand-to-mouth households own no capital and cannot save or borrow, "
    "so they spend this period's income this period. Their income is their wage plus "
    "all of the cash welfare ($G^T_t/\\lambda$ each), minus the lump-sum tax. They pay "
    "no income tax, but they pay the consumption tax on what they buy:"
)
st.latex(r"\max_{C^H_t,\,N^H_t}\ \ln\big(C^H_t + \phi\,G^C_t\big) + \theta_L \ln\big(1-N^H_t\big) \qquad \text{subject to}\quad (1+\tau^C_t)\,C^H_t = w_t N^H_t + \frac{G^T_t}{\lambda} - T_t")
st.markdown(
    "Nothing they do today affects tomorrow, so they solve this problem one period "
    "at a time."
)
st.markdown("**Solution.** The same labor-leisure condition as optimizers, but without the income tax:")
st.latex(r"\theta_L\,\frac{C^H_t+\phi\,G^C_t}{1-N^H_t} = \frac{w_t}{1+\tau^C_t}")
st.markdown("Substituting the budget constraint gives their hours directly:")
st.latex(r"N^H_t = \frac{w_t - \theta_L\left(\dfrac{G^T_t}{\lambda} - T_t + (1+\tau^C_t)\,\phi\,G^C_t\right)}{(1+\theta_L)\,w_t}")
st.markdown(
    "(or $N^H_t=0$ if this is negative). More cash welfare or in-kind welfare makes "
    "them richer, so they work less; a lump-sum tax makes them poorer, so they work "
    "more. Their consumption then follows from the budget constraint."
)

st.subheader("A.5 Government")
st.markdown(
    "**Policy.** The government sets total spending $G_t$ as a share of output (sidebar "
    "item 4) and splits it into three parts (item 5). Public investment builds "
    "public capital:"
)
st.latex(r"G_t = I^G_t + G^C_t + G^T_t, \qquad K^G_{t+1} = (1-\delta)\,K^G_t + I^G_t")
st.markdown(
    "**Budget.** The government cannot borrow, so the three taxes together pay for all "
    "spending every period. Each tax raises a fixed share of $G_t$ (sidebar item 1: "
    "$f^{LS}+f^Y+f^C=1$), and the tax rates adjust each year so that it does. Every "
    "person pays the same lump-sum tax (the population is 1), only optimizers' income "
    "is taxed, and both household types pay the consumption tax:"
)
st.latex(r"T_t = f^{LS}\,G_t, \qquad \tau^Y_t\,(1-\lambda)\big(w_t N^O_t + R_t K^O_t\big) = f^Y G_t, \qquad \tau^C_t\,C_t = f^C G_t")

st.subheader("A.6 Market clearing and equilibrium")
st.markdown("Aggregate quantities add up the two household types:")
st.latex(r"C_t = \lambda\,C^H_t + (1-\lambda)\,C^O_t, \quad N_t = \lambda\,N^H_t + (1-\lambda)\,N^O_t, \quad K_t = (1-\lambda)\,K^O_t, \quad I_t = (1-\lambda)\,I^O_t")
st.markdown(
    "Adding up both households' budgets and the government's budget gives the "
    "**goods-market** condition: output is used for private consumption, private "
    "investment, public investment, and in-kind welfare:"
)
st.latex(r"Y_t = C_t + I_t + I^G_t + G^C_t")
st.markdown(
    "Cash welfare $G^T_t$ is not in this equation: it only moves income from "
    "taxpayers to recipients, who then spend it on $C_t$. An **equilibrium** is a "
    "path of prices ($w_t$, $R_t$) and quantities such that firms (A.2), both "
    "household types (A.3, A.4), and the government (A.5) all satisfy their "
    "conditions, and the labor, capital, and goods markets clear."
)

st.subheader("A.7 Steady state and transition")
st.markdown(
    "In a **steady state** nothing changes over time. The optimizers' Euler equation "
    "then becomes"
)
st.latex(r"(1-\tau^Y)\,R = r + \delta")
st.markdown(
    "The after-tax return on capital must cover the interest rate plus depreciation. "
    "This pins down the rental rate, and so the capital-labor ratio $K/N$. A higher "
    "income-tax rate means a higher required $R$, and so less capital per worker; the "
    "consumption tax does not appear here. "
    "Public capital settles at $K^G = I^G/\\delta$. The two labor-leisure conditions "
    "and the budget constraints then determine hours and consumption. $\\theta_L$ is "
    "chosen so that total hours are 20% of the time endowment before the policy "
    "change (Baxter & King's calibration target)."
)
st.markdown(
    "After a policy change, capital can only change slowly through investment, while "
    "consumption, hours, and prices can jump. The app computes the **transition "
    "path**: the one path along which every condition above holds in every year and "
    "the economy converges to the new steady state. That path is what the charts in "
    "panels 1 and 2 show."
)
st.markdown(
    "**Special case.** With $\\lambda=0$, a single tax (all lump-sum or all income "
    "tax), and all government spending used for goods "
    "that households do not value -- in-kind welfare with $\\phi=0$, or public "
    "investment with $\\theta_G=0$ -- this is Baxter & King's (1993) model. "
    "One result to explore: without an income tax (only lump-sum and consumption "
    "taxes), shifting cash welfare toward "
    "hand-to-mouth households changes *who* works but not *total* long-run output, as "
    "long as they still work. With log utility, the two types' labor-leisure "
    "conditions add up to a single one for the whole economy. With an income tax, "
    "which only optimizers pay, this is no longer true."
)

st.divider()
_emblem_path = Path(__file__).parent / "kgu_emblem.png"
if _emblem_path.exists():
    _, _emblem_col, _ = st.columns([1, 1, 1])
    with _emblem_col:
        st.image(str(_emblem_path), use_container_width=True)
