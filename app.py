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
    run_experiment,
    simulate_transition,
    steady_state_for_policy,
)

st.set_page_config(page_title="Fiscal Policy in General Equilibrium",
                    page_icon="\U0001F3DB️", layout="wide")

# --------------------------------------------------------------------------
# Shared chart helpers (used by both the "Cumulative Effects" and "Marginal
# Effects" time-series panels below).
# --------------------------------------------------------------------------


def line_fig(title, series_specs, yaxis_title, yrs):
    fig = go.Figure()
    for name, arr, color in series_specs:
        fig.add_trace(go.Scatter(x=yrs, y=arr[:len(yrs)], mode="lines+markers",
                                  name=name, line=dict(color=color, width=2), marker=dict(size=4)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"),
                       xaxis_title="Years after the change", yaxis_title=yaxis_title,
                       height=420, legend=dict(orientation="h", y=1.18, yanchor="bottom"),
                       margin=dict(t=110, b=20))
    return fig


def render_transition_tabs(path, slider_key, caption_commodity, caption_labor, caption_financial):
    """Renders the Commodity/Labor/Financial-market tabbed transition-path charts
    shared by both time-series panels, given any TransitionPath-like object."""
    years_to_show = st.slider("Years to display", 5, 100, 25, 5, key=slider_key)
    yrs = path.years[:years_to_show]

    tab1, tab2, tab3 = st.tabs(["Commodity market", "Labor market", "Financial market"])

    with tab1:
        st.plotly_chart(line_fig(
            "Output, consumption, investment, government spending",
            [("Output (Y)", path.Y, "#2563eb"), ("Consumption (C)", path.C, "#16a34a"),
             ("Investment (I)", path.I, "#f59e0b"), ("Government spending (G)", path.G, "#6b7280")],
            "% deviation from original steady state", yrs), use_container_width=True,
            key=f"{slider_key}_commodity")
        st.caption(caption_commodity)

    with tab2:
        st.plotly_chart(line_fig(
            "Labor input and the real wage",
            [("Labor input (N)", path.N, "#2563eb"), ("Real wage (w)", path.W, "#dc2626")],
            "% deviation from original steady state", yrs), use_container_width=True,
            key=f"{slider_key}_labor")
        st.caption(caption_labor)

    with tab3:
        st.plotly_chart(line_fig(
            "Private and public capital stocks",
            [("Private capital (K)", path.K, "#2563eb"), ("Public capital (Kᴳ)", path.KG, "#16a34a")],
            "% deviation from original steady state", yrs), use_container_width=True,
            key=f"{slider_key}_capital")
        fig_r = go.Figure(go.Scatter(x=yrs, y=path.r_bp[:years_to_show], mode="lines+markers",
                                      line=dict(color="#7c3aed", width=2)))
        fig_r.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_r.update_layout(title="Real interest rate, deviation from steady state",
                             xaxis_title="Years after the change", yaxis_title="Basis points",
                             height=350, margin=dict(t=50, b=20))
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
    G respects the sidebar's Duration control (item 4.2).

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
    return path

# --------------------------------------------------------------------------
# Sidebar -- model controls
# --------------------------------------------------------------------------

st.sidebar.title("\U0001F3DB️ Model controls")
st.sidebar.caption("Baxter & King (1993, *AER*), “Fiscal Policy in General "
                    "Equilibrium.” Adjust anything below and every panel updates.")

if "reset_version" not in st.session_state:
    st.session_state["reset_version"] = 0

def wkey(name: str) -> str:
    """Version every widget key with the reset counter, so clicking Reset forces
    Streamlit to fully remount each widget at its coded default -- merely deleting
    session_state and rerunning does not reliably redraw slider handles/labels
    client-side, even though the underlying computed value does reset correctly."""
    return f"{name}_{st.session_state['reset_version']}"

if st.sidebar.button("↺ Reset to paper's benchmark calibration", use_container_width=True):
    st.session_state["reset_version"] += 1
    st.rerun()

# Fixed at the paper's benchmark calibration (Table 1): labor share of income,
# depreciation rate, and target steady-state hours worked are no longer sidebar
# controls.
theta_N = 0.58
delta_pct = 10.0
N_target_pct = 20.0

# 1. Financing rule -----------------------------------------------------
st.sidebar.subheader("1. How is the tax collected?")
financing_label = st.sidebar.radio(
    "Financing rule",
    ["Lump-sum (tax revenue collected without distorting labor/capital margins)",
     "Income tax (tax is a (1-τ) wedge on labor and capital income)"],
    help="Both options collect the SAME tax revenue: τ = G/Y always. They differ only "
         "in whether that tax distorts the household's labor-leisure and capital-Euler "
         "first-order conditions.",
    key=wkey("financing_label"),
)
financing = "lump_sum" if financing_label.startswith("Lump-sum") else "income_tax"

# 2. theta_G -----------------------------------------------------------------
st.sidebar.subheader("2. Productive public capital")
theta_G = st.sidebar.slider(
    "Public-capital productivity, θ_G", 0.00, 0.40, 0.00, 0.01,
    help="Y = A·K^θK·(Kᴳ)^θG·N^θN. θG=0 means public investment is a pure resource "
         "cost with no productivity effect (Baxter & King's Table 4 grid runs 0-0.40).",
    key=wkey("theta_G"),
)

# 3. Steady-state real interest rate -----------------------------------------
st.sidebar.subheader("3. Real interest rate")
r_pct = st.sidebar.slider("Steady-state real interest rate, r (%/yr)", 1.0, 12.0, 6.5, 0.25, key=wkey("r_pct"))

# 4. Total government spending -----------------------------------------
st.sidebar.subheader("4. Total government spending")
BENCH_s_G_pct = 20.0  # paper's benchmark calibration (Table 1) -- the fixed baseline
                       # every experiment/comparison below is measured against.
s_G_old_pct = BENCH_s_G_pct
s_G_new_pct = st.sidebar.slider(
    "4.1 Total government spending, G/Y (%)", 5.0, 40.0, BENCH_s_G_pct, 1.0,
    help="The paper's benchmark calibration is G/Y = 20% (dashed reference). Move this "
         "slider to change total government spending; every result in this app compares "
         "against that fixed 20% baseline. Composition shares (item 5) stay fixed.",
    key=wkey("s_G_new_pct"))
delta_s_G_pct = s_G_new_pct - s_G_old_pct
st.sidebar.caption(f"Tax rate τ = G/Y always, under both financing rules above (item 1). "
                    f"ΔG/Y vs. the {BENCH_s_G_pct:.0f}% benchmark = **{delta_s_G_pct:+.1f}** "
                    f"percentage points.")

# 4.2 Duration of the change, alongside G/Y within the same section, since it
# only governs how long a ΔG/Y experiment (item 4.1) lasts. -----------------
duration_label = st.sidebar.radio("4.2 Duration of the change", ["Permanent", "Temporary"], key=wkey("duration_label"))
permanent = duration_label == "Permanent"
duration_years = 4
if not permanent:
    duration_years = st.sidebar.slider("Duration (years)", 1, 20, 4, 1, key=wkey("duration_years"))

# 5. Composition of G ------------------------------------------------------
st.sidebar.subheader("5. Composition of G")
st.sidebar.caption("Government spending has no separate \"basic purchases\" term -- the "
                    "utility function never values government consumption directly -- so "
                    "it splits into just two pieces: public investment Iᴳ and transfers G_T.")
f_IG_pct = st.sidebar.slider("Public investment share of G, Iᴳ/G (%)", 0.0, 100.0, 0.0, 5.0, key=wkey("f_IG_pct"))
f_GT_pct = 100.0 - f_IG_pct
st.sidebar.caption(f"⇒ Transfers G_T/G = **{f_GT_pct:.0f}%** is the residual (100% − Iᴳ/G).")

fig_comp = go.Figure()
for name, val, color in [("Iᴳ", f_IG_pct, "#16a34a"),
                          ("G_T", f_GT_pct, "#7c3aed")]:
    fig_comp.add_trace(go.Bar(y=["G/Y"], x=[val], name=name, orientation="h",
                               marker_color=color,
                               text=f"{name} {val:.0f}%" if val > 3 else "",
                               textposition="inside", insidetextanchor="middle"))
fig_comp.update_layout(barmode="stack", height=110, showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(range=[0, 100], showticklabels=False),
                        yaxis=dict(showticklabels=False))
st.sidebar.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

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
f_IG_v, f_GT_v = f_IG_pct / 100.0, f_GT_pct / 100.0

error = None
path_error = None
experiment = None
ss_old = ss_new = path = None
try:
    experiment = run_experiment(
        theta_N=theta_N_v, delta=delta_v, r=r_v, A=1.0, theta_G=theta_G_v,
        s_G_old=s_G_old_v, s_G_new=s_G_new_v, f_IG=f_IG_v, f_GT=f_GT_v,
        N_target=N_target_v, financing=financing,
        permanent=permanent, duration_years=duration_years, T_sim=200,
    )
    ss_old, ss_new, path = experiment.ss_old, experiment.ss_new, experiment.path
except Exception as exc:  # noqa: BLE001
    # The transition-path solver can lose saddle-path stability at extreme tax
    # rates (very high income-tax G/Y, especially with a low investment share)
    # even though the two steady states themselves are perfectly well-defined --
    # fall back to the steady states alone, so Panel 1 and Panel 2's comparison
    # table still work; only the transition-dependent sections (impact
    # multiplier, transition charts, duration sensitivity, Panel 3's impact
    # column) show a scoped note below instead of taking down the whole page.
    try:
        theta_L_fallback = calibrate_theta_L(theta_N_v, delta_v, r_v, 1.0, theta_G_v,
                                              s_G_old_v * f_IG_v, s_G_old_v * f_GT_v,
                                              financing == "income_tax", N_target_v)
        ss_old = steady_state_for_policy(theta_N_v, delta_v, r_v, 1.0, theta_G_v,
                                          s_G_old_v, f_IG_v, f_GT_v, theta_L_fallback,
                                          financing == "income_tax")
        ss_new = steady_state_for_policy(theta_N_v, delta_v, r_v, 1.0, theta_G_v,
                                          s_G_new_v, f_IG_v, f_GT_v, theta_L_fallback,
                                          financing == "income_tax")
        path_error = str(exc)
    except Exception as exc2:  # noqa: BLE001 -- steady state itself is infeasible
        error = str(exc2)

st.title("Fiscal Policy in General Equilibrium")
st.caption("An interactive replication of Baxter & King (1993, *American Economic Review*) "
           "83(3): 315-334 — built for classroom exploration.")

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
        f"valid**; only the transition-path charts, impact multipliers, and "
        f"duration-sensitivity replication need a stable path and are skipped. "
        f"Try a lower total government spending (item 4.1), Lump-sum financing "
        f"(item 1), or a higher public-investment share (item 5) -- all three "
        f"push the instability threshold higher."
    )

# --------------------------------------------------------------------------
# Fixed benchmark steady state (paper's Table 1 calibration): a genuinely
# fixed reference point, independent of every sidebar control, so panel 1
# below always shows *some* difference when composition, financing, θ_G, or r
# are moved -- not just when G/Y is.
# --------------------------------------------------------------------------

BENCH_theta_N, BENCH_delta, BENCH_r = 0.58, 0.10, 0.065
BENCH_theta_G, BENCH_N_target = 0.0, 0.20
BENCH_f_IG, BENCH_f_GT = 0.0, 1.0  # benchmark: no public investment, all spending is transfers
BENCH_distortionary = False

benchmark_error = None
ss_benchmark = None
ss_current = None
ss_reverted = None
try:
    theta_L_bench = calibrate_theta_L(BENCH_theta_N, BENCH_delta, BENCH_r, 1.0, BENCH_theta_G,
                                       (BENCH_s_G_pct / 100.0) * BENCH_f_IG,
                                       (BENCH_s_G_pct / 100.0) * BENCH_f_GT,
                                       BENCH_distortionary, BENCH_N_target)
    ss_benchmark = steady_state_for_policy(BENCH_theta_N, BENCH_delta, BENCH_r, 1.0, BENCH_theta_G,
                                            BENCH_s_G_pct / 100.0, BENCH_f_IG, BENCH_f_GT,
                                            theta_L_bench, BENCH_distortionary)
    # Use the SAME (benchmark) theta_L for "current settings" -- if we instead let each
    # sidebar configuration recalibrate its own theta_L, hours worked would always land
    # back on exactly N_target by construction, silently masking any real labor-supply
    # response to composition/financing/θ_G/r and dumping the whole adjustment onto
    # consumption. Holding theta_L fixed at the benchmark's value lets N move honestly.
    ss_current = steady_state_for_policy(theta_N_v, delta_v, r_v, 1.0, theta_G_v,
                                          s_G_new_v, f_IG_v, f_GT_v,
                                          theta_L_bench, financing == "income_tax")
    # "Reverted" state for the Cumulative Effects time series under a TEMPORARY
    # duration: current settings' own regime (financing/composition/θ_G/r), but at
    # the benchmark's G/Y level -- the true long-run anchor once G/Y reverts, since
    # financing/composition/θ_G don't revert along with it (see
    # regime_transition_path's docstring for why this matters).
    ss_reverted = steady_state_for_policy(theta_N_v, delta_v, r_v, 1.0, theta_G_v,
                                           s_G_old_v, f_IG_v, f_GT_v,
                                           theta_L_bench, financing == "income_tax")
except Exception as exc:  # noqa: BLE001
    benchmark_error = str(exc)

st.header("1. Cumulative Effects")
st.write("Exact, closed-form solution of the model's long-run (“great ratios”) "
         "equilibrium: the paper's **fixed benchmark calibration** (θ_N=0.58, δ=10%, "
         "r=6.5%, G/Y=20%, all of it transfers, lump-sum financing, θ_G=0) vs. "
         "**whatever the sidebar is currently set to**. Moving *any* sidebar control -- "
         "duration aside -- changes the right-hand column, including financing alone "
         "(item 1): switching to Income tax distorts the household's margins even if "
         "G/Y and composition don't move. \"Cumulative\" because this bundles the net "
         "effect of *every* control that currently differs from the benchmark, not just "
         "one at a time -- see \"2. The Marginal Effects of Government Spendings\" below to isolate a single lever.")

if benchmark_error:
    st.error(f"Could not compute the benchmark steady state: {benchmark_error}")
else:
    bench_left_label, bench_right_label = "Paper's benchmark calibration", "Current settings"

    bench_table = pd.DataFrame({
        "Variable": ["Output Y", "Consumption C", "Investment I", "Private capital K",
                     "Public capital Kᴳ", "Government spending G (total)",
                     "  Public investment Iᴳ", "  Transfers G_T",
                     "Labor input N (% of time)", "Real wage w", "Tax rate τ (%)"],
        bench_left_label: [ss_benchmark.Y, ss_benchmark.C, ss_benchmark.I, ss_benchmark.K, ss_benchmark.KG,
                            ss_benchmark.G, ss_benchmark.IG, ss_benchmark.GT, ss_benchmark.N * 100,
                            ss_benchmark.w, ss_benchmark.tau * 100],
        bench_right_label: [ss_current.Y, ss_current.C, ss_current.I, ss_current.K, ss_current.KG, ss_current.G,
                             ss_current.IG, ss_current.GT, ss_current.N * 100,
                             ss_current.w, ss_current.tau * 100],
    })
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
    fig_bench_bar.update_layout(title="% change from the paper's benchmark calibration",
                                 yaxis_title="% change", height=380, margin=dict(t=50, b=20))
    st.plotly_chart(fig_bench_bar, use_container_width=True)

    st.subheader("Transition path: benchmark → current settings")
    st.write("Perfect-foresight transition path (log-linearized around the **current "
             "settings**, solved exactly via eigen-decomposition), imagining the "
             "economy starts with the benchmark's capital stocks and this new "
             "regime -- financing, composition, θ_G, and r -- is put permanently in "
             "place at year 0. **G/Y specifically respects the Duration control "
             "(item 4.2)**: Permanent means it jumps to the current G/Y and stays; "
             "Temporary means it reverts to the benchmark's 20% share after the "
             "chosen number of years (everything else stays at current settings). "
             "All series are % deviations from the **benchmark** level.")
    try:
        cumulative_path = regime_transition_path(ss_benchmark, ss_current, ss_reverted=ss_reverted,
                                                  permanent=permanent, duration_years=duration_years,
                                                  T_sim=200)
        render_transition_tabs(
            cumulative_path, wkey("years_to_show_cumulative"),
            caption_commodity="Compare to Baxter & King Figures 2-4, but here the "
                "\"shock\" is the *entire* gap between the benchmark and the sidebar's "
                "current settings, not a single isolated policy lever.",
            caption_labor="Labor and the wage jump immediately to reflect the new "
                "regime's prices; only capital is predetermined and adjusts gradually.",
            caption_financial="Watch how far and how fast private (and, if θ_G>0, "
                "public) capital has to move to reach the new regime's steady state.",
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute the cumulative transition path: {exc}")

# --------------------------------------------------------------------------
# 2. Marginal Effects: isolating the ΔG/Y lever specifically, holding every
# OTHER sidebar setting (financing, composition, θ_G, r) fixed at whatever
# it is currently set to. Reuses ss_old/ss_new/path from the ΔG/Y experiment
# above, so resetting G/Y to 20% always drives the steady-state comparison
# back to exactly zero.
# --------------------------------------------------------------------------

st.header("2. The Marginal Effects of Government Spendings")
st.write("Same closed-form solution, but comparing **G/Y = 20%** against **the total "
         "government spending set in the sidebar (item 4.1)**, holding financing, "
         "composition, θ_G, and r fixed at whatever the sidebar currently has them at "
         "on *both* sides. \"Marginal\" because this isolates the effect of *one* lever "
         "at a time -- resetting item 4.1 back to 20% always brings the numbers below "
         "back to zero, regardless of what items 1/2/3/5 are set to.")

marg_left_label, marg_right_label = f"G/Y = {BENCH_s_G_pct:.0f}% (at current other settings)", "Current G/Y"

marg_table = pd.DataFrame({
    "Variable": ["Output Y", "Consumption C", "Investment I", "Private capital K",
                 "Public capital Kᴳ", "Government spending G (total)",
                 "  Public investment Iᴳ", "  Transfers G_T",
                 "Labor input N (% of time)", "Real wage w", "Tax rate τ (%)"],
    marg_left_label: [ss_old.Y, ss_old.C, ss_old.I, ss_old.K, ss_old.KG,
                       ss_old.G, ss_old.IG, ss_old.GT, ss_old.N * 100,
                       ss_old.w, ss_old.tau * 100],
    marg_right_label: [ss_new.Y, ss_new.C, ss_new.I, ss_new.K, ss_new.KG, ss_new.G,
                        ss_new.IG, ss_new.GT, ss_new.N * 100,
                        ss_new.w, ss_new.tau * 100],
})
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
fig_marg_bar.update_layout(title=f"% change from the G/Y = {BENCH_s_G_pct:.0f}% baseline (ΔG/Y effect only)",
                            yaxis_title="% change", height=380, margin=dict(t=50, b=20))
st.plotly_chart(fig_marg_bar, use_container_width=True)

if abs(delta_s_G_pct) < 1e-9:
    st.info("ΔG/Y is set to 0 — the long-run/impact **multipliers**, the **transition "
            "dynamics**, and **\"3. Multiplier Effects\"** below all need a nonzero "
            "spending change to be defined (they isolate the effect of *that one* "
            "policy lever, holding everything else fixed). Move the **ΔG/Y** slider "
            "away from 0 in the sidebar (item 4.1) to see them. Panel 1 above doesn't "
            "need ΔG/Y and is already showing your current settings.")
else:
    dG_headline = ss_new.G - ss_old.G
    multiplier_long_run = (ss_new.Y - ss_old.Y) / dG_headline if abs(dG_headline) > 1e-12 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Long-run output multiplier ΔY/ΔG", f"{multiplier_long_run:+.2f}")
    c2.metric("Impact (year-0) output multiplier",
              f"{experiment.multiplier_impact:+.2f}" if experiment else "n/a (unstable path)")
    c3.metric("Steady-state hours worked N",
              f"{ss_old.N*100:.1f}% → {ss_new.N*100:.1f}%")
    c4.metric("Tax rate τ = G/Y", f"{ss_old.tau*100:.1f}% → {ss_new.tau*100:.1f}%")

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
            f"of new spending.** Under Income Tax financing the tax wedge on labor and "
            f"capital income depresses work effort and capital formation by more than the "
            f"direct resource cost of the spending itself."
        )

    if path is None:
        st.info("Transition path, impact multipliers, and the duration-sensitivity "
                 "replication are unavailable at these settings (see the saddle-path "
                 "note above) -- only long-run, steady-state comparisons are shown.")
    else:
        st.subheader("Transition path: isolating the ΔG/Y shock")
        st.write("Perfect-foresight transition path (log-linearized around the relevant steady "
                 "state, solved exactly via eigen-decomposition -- no simulation noise). "
                 "All series are % deviations from the *original* steady state, matching "
                 "Figures 2-4 of the paper.")
        render_transition_tabs(
            path, wkey("years_to_show_marginal"),
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
        # Duration sensitivity (Table 3 style)
        # --------------------------------------------------------------------------

        with st.expander("\U0001F4CA How much does the *duration* of a temporary shock matter? (Table 3 replication)"):
            st.write("Holding the financing rule fixed, how does the impact-period output "
                     "multiplier change as a temporary spending increase is made to last longer?")
            durations = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
            mults = []
            for T in durations:
                try:
                    e = run_experiment(theta_N_v, delta_v, r_v, 1.0, theta_G_v, s_G_old_v, s_G_new_v,
                                        f_IG_v, f_GT_v, N_target_v, financing,
                                        permanent=False, duration_years=T, T_sim=200)
                    mults.append(e.multiplier_impact)
                except Exception:
                    mults.append(np.nan)
            try:
                perm_e = run_experiment(theta_N_v, delta_v, r_v, 1.0, theta_G_v, s_G_old_v, s_G_new_v,
                                         f_IG_v, f_GT_v, N_target_v, financing,
                                         permanent=True, T_sim=200)
                perm_mult = perm_e.multiplier_impact
            except Exception:
                perm_mult = None
            fig_dur = go.Figure()
            fig_dur.add_trace(go.Scatter(x=durations, y=mults, mode="lines+markers", name="Temporary shock"))
            if perm_mult is not None:
                fig_dur.add_hline(y=perm_mult, line_dash="dash", line_color="#dc2626",
                                   annotation_text="Permanent-shock impact multiplier")
            fig_dur.update_layout(title="Impact multiplier vs. duration of the spending increase",
                                   xaxis_title="Duration (years)", yaxis_title="ΔY/ΔG on impact",
                                   height=380)
            st.plotly_chart(fig_dur, use_container_width=True)
            st.caption("Baxter & King's key point (Section IV): more *persistent* shocks produce "
                       "larger short-run multipliers because consumers cannot smooth as easily "
                       "when higher spending is known to last longer -- the opposite of the "
                       "Barro-Hall intuition that temporary shocks should have larger effects.")

    # --------------------------------------------------------------------------
    # 3. Multiplier effects of government spending, by variable
    # --------------------------------------------------------------------------

    st.header("3. Multiplier Effects")
    st.write(
        "How much does the ΔG/Y change (item 4.1) move each variable, per dollar of "
        r"new spending: $\Delta X/\Delta G$ for $X\in\{Y,C,I,K,K^G\}$, both "
        "**long-run** (comparing the two steady states) and **on impact** (year 0 of "
        "the transition path above) -- same experiment as \"2. The Marginal Effects of Government Spendings\", "
        "holding financing, composition, θ_G, and r fixed at the sidebar's current "
        "settings."
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
    fig_mult.update_layout(barmode="group", title="Multiplier effects of ΔG/Y, by variable",
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
- **Financing (item 1)**: both "Lump-sum" and "Income tax" set τ = G/Y identically.
  They differ only in whether that tax enters the household's labor-leisure and
  capital-Euler first-order conditions as a (1-τ) wedge (Income tax) or not (Lump-sum,
  a true non-distorting poll tax for revenue purposes). Because the wedge sits on the
  *margin*, not on net revenue, switching financing alone reshapes the whole steady
  state even at 100% transfers (item 5) -- there is no "neutral" financing switch the
  way there is for a lump-sum-financed, all-transfers *composition* change.
- **Composition (item 5)**: total government spending G/Y is split into public
  investment Iᴳ (accumulates into Kᴳ, productivity-enhancing if θ_G>0) and transfers
  G_T (resource-neutral, returned to households). There is no separate "basic
  purchases" category, since the utility function never assigns households any value
  from government consumption directly -- only Iᴳ enters the economy-wide resource
  constraint Y=C+I+Iᴳ; G_T nets out in aggregate.
- **Panel 1 (Cumulative Effects)** compares the paper's *fixed* benchmark calibration
  (θ_N=0.58, δ=10%, r=6.5%, G/Y=20%, all of it transfers, lump-sum financing, θ_G=0)
  against whatever the sidebar is *currently* set to, so every control (financing,
  composition, θ_G, r, G/Y) shows up as a difference there -- including financing
  alone, since the (1-τ) wedge distorts the household's margins independent of net
  revenue. Its transition-path chart imagines the economy starting at the
  benchmark's capital stocks with the current regime put permanently in place.
- **Panel 2 (The Marginal Effects of Government Spendings)** instead isolates the ΔG/Y policy lever specifically,
  holding every other setting (financing, composition, θ_G, r) fixed on both sides
  of the comparison -- resetting G/Y to 20% always brings it back to exactly zero,
  and its multipliers/transition-path chart need ΔG/Y ≠ 0 to be defined for the
  same reason.
- **Panel 3 (Multiplier Effects)** breaks the same ΔG/Y experiment down by variable
  (Y, C, I, K, Kᴳ), reporting both the long-run multiplier (from the two steady
  states) and the impact multiplier (from year 0 of the transition path) for each.
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
st.header("Appendix: The model's equations")
st.write(
    "Every number in this app comes from the equations below. They are written out "
    "in full here -- with the economic intuition behind each one -- so you can see "
    "exactly what is being solved, not just the results."
)

st.subheader("A.1 Notation")
st.markdown(
    r"""
| Symbol | Meaning |
|---|---|
| $C_t$ | Consumption |
| $N_t$ | Labor supplied (fraction of the time endowment), $L_t=1-N_t$ is leisure |
| $Y_t$ | Output |
| $K_t$ | Private capital stock |
| $K^G_t$ | Public (government) capital stock |
| $I_t,\ I^G_t$ | Private and public investment |
| $G_{T,t}$ | Lump-sum government transfers |
| $w_t,\ r_t$ | Real wage and real rental rate on capital |
| $\tau_t$ | Tax rate |
| $\theta_N,\ \theta_K=1-\theta_N$ | Labor's and capital's shares of income |
| $\theta_G$ | Productivity of public capital (Section VI extension) |
| $\theta_L$ | Weight on leisure in utility (calibrated, not chosen by the user directly) |
| $\delta$ | Depreciation rate (same for $K$ and $K^G$) |
| $A$ | Total factor productivity, normalized to 1 |

A hat over a variable, e.g. $\hat y_t$, denotes a **percent deviation from steady
state** — the units used in every transition-path chart in this app.
"""
)

st.subheader("A.2 Preferences")
st.latex(r"u(C_t, N_t) = \ln C_t + \theta_L \ln(1-N_t)")
st.markdown(
    "Households value consumption and leisure, with **log utility** in each — the "
    "standard King-Plosser-Rebelo preference specification that keeps labor supply "
    "responses well-behaved along a balanced growth path. Notice government spending "
    "does **not** appear here at all -- households derive no direct utility from "
    "$G_{T,t}$ or $I^G_t$, which is exactly why there is no separate \"basic "
    "purchases\" term anywhere in this model: a category of spending the household "
    "valued directly would need to enter this utility function, and none does. "
    "$\\theta_L$ is not a free slider in this app; instead it is *calibrated* (see "
    "A.6) so that steady-state labor supply matches the target hours-worked share at "
    "the baseline policy — exactly as Baxter & King do in their Table 1."
)

st.subheader("A.3 Technology")
st.latex(r"Y_t = A\,K_t^{\theta_K}\,(K^G_t)^{\theta_G}\,N_t^{\theta_N}, \qquad \theta_K = 1-\theta_N")
st.markdown(
    "Output is Cobb-Douglas in private capital and labor, exactly as in the paper's "
    "baseline model. The extension in Section VI (and the sidebar's θ_G slider) adds "
    "public capital $K^G_t$ as a third input: it raises the productivity of *every* "
    "private input without the government having to pay for it again next period — "
    "a non-rival \"public good\" flavor. Setting $\\theta_G=0$ collapses this exactly "
    "to the paper's baseline model, where public investment is a pure resource "
    "cost with no productivity effect."
)

st.subheader("A.4 Capital accumulation")
st.latex(r"K_{t+1} = (1-\delta)K_t + I_t \qquad\qquad K^G_{t+1} = (1-\delta)K^G_t + I^G_t")
st.markdown(
    "Both private and public capital accumulate by the standard **perpetual "
    "inventory** rule, and (for simplicity) depreciate at the same rate $\\delta$. "
    "Public capital is financed by public investment $I^G_t$, one of the two "
    "components of total government spending (sidebar item 5)."
)

st.subheader("A.5 Resource constraint and the government budget")
st.latex(r"Y_t = C_t + I_t + I^G_t")
st.markdown(
    "Output is used for private consumption, private investment, and public "
    "investment. **Transfers $G_{T,t}$ do not appear here** — a transfer just moves "
    "resources from the government's books to a household's pocket without using up "
    "any output, so it nets out of the economy-wide resource constraint even though "
    "it is part of the government's budget below. (There is likewise no \"basic "
    "purchases\" term: since it would never enter the utility function of A.2 either, "
    "keeping it in the model would just be another resource-using category "
    "economically indistinguishable from public investment, so this app folds all "
    "resource-using spending into $I^G_t$.)"
)
st.latex(r"\tau_t Y_t = I^G_t + G_{T,t}, \qquad \tau_t = \dfrac{I^G_t+G_{T,t}}{Y_t}")
st.markdown(
    "The government's budget always balances **every period** (no debt in this "
    "model), and the tax rate is *defined* as total government spending divided "
    "by output. This holds **identically under both financing rules** in the "
    "sidebar (item 1) — what differs between them is not how much revenue is "
    "raised, but whether that revenue collection distorts the household's "
    "decisions, in the household budget constraint below."
)
st.latex(r"\text{Income tax:}\quad C_t+I_t = (1-\tau_t)\big(w_tN_t+r_tK_t\big) + G_{T,t}")
st.latex(r"\text{Lump-sum:}\quad C_t+I_t = w_tN_t+r_tK_t - \tau_tY_t + G_{T,t}")
st.markdown(
    "Under **Income tax** financing, $(1-\\tau_t)$ is a proportional wedge that "
    "shrinks the *marginal* return to working and saving — this is what makes the "
    "tax distortionary. Under **Lump-sum** financing, the same total revenue "
    "$\\tau_t Y_t$ is instead collected as a poll tax that does not depend on how "
    "much the household chooses to work or save, so the household simply faces "
    "$w_t$ and $r_t$ directly (no $(1-\\tau_t)$ term at all), creating a pure income "
    "(wealth) effect with no substitution effect at the margin."
)

st.subheader("A.6 Household optimization")
st.markdown("Maximizing lifetime utility subject to the budget constraint above gives two first-order conditions, one per financing rule.")
st.latex(r"\text{Labor-leisure, income tax:}\qquad \theta_L\,\frac{C_t}{1-N_t} = (1-\tau_t)\,w_t")
st.latex(r"\text{Labor-leisure, lump-sum:}\qquad \theta_L\,\frac{C_t}{1-N_t} = w_t")
st.markdown(
    "The household works until the marginal disutility of an extra hour (left side) "
    "equals its marginal after-tax benefit (right side). Under income tax financing "
    "that benefit is shrunk by $(1-\\tau_t)$; under lump-sum financing there is no "
    "such wedge at all — this single difference is *the* mechanism generating the "
    "app's negative multiplier result under Income tax."
)
st.latex(r"\text{Euler, income tax:}\qquad \frac{1}{C_t} = \beta\,\frac{1+(1-\tau_{t+1})\,\mathrm{MPK}_{t+1}-\delta}{C_{t+1}}")
st.latex(r"\text{Euler, lump-sum:}\qquad \frac{1}{C_t} = \beta\,\frac{1+\mathrm{MPK}_{t+1}-\delta}{C_{t+1}},\qquad \beta=\frac{1}{1+r}")
st.markdown(
    "There is no expectation operator here: this model has no stochastic element "
    "(no random shocks to technology, policy, or anything else), so every "
    "\"future\" variable is simply its known, perfect-foresight value. The "
    "household is not *guessing* what $C_{t+1}$ or $\\mathrm{MPK}_{t+1}$ will "
    "be — under a given policy path it *knows*, and the Euler equation is an "
    "ordinary deterministic difference equation linking today's consumption to "
    "tomorrow's."
)
st.markdown(
    "The household is indifferent between consuming one more unit today and "
    "saving it to consume $(1+\\text{after-tax net return})$ units tomorrow — with "
    "the same $(1-\\tau_{t+1})$ wedge on the capital return under income tax, and "
    "no wedge under lump-sum. This is the equation that makes consumption "
    "**forward-looking** — the *entire future* path of the economy is baked into "
    "today's consumption choice, which is exactly why the transition path (A.8) "
    "has to be solved with a saddle-path method rather than simulated forward "
    "period by period."
)
st.markdown(
    r"Prices are just marginal products: the wage $w_t=\theta_N Y_t/N_t$ (marginal "
    r"product of labor) and the rental rate $\mathrm{MPK}_t = \theta_K Y_t/K_t$ "
    r"(marginal product of capital); in steady state, under income tax financing, "
    r"$(1-\tau)\mathrm{MPK}=r+\delta$, and under lump-sum financing, $\mathrm{MPK}=r+\delta$."
)

st.subheader("A.7 Steady state (closed form)")
st.markdown(
    "Because the production function has constant returns to scale in $(K,N)$ "
    "*alone* ($\\theta_K+\\theta_N=1$), dividing the technology equation through by "
    "$N$ leaves the capital/labor ratio $\\kappa=K/N$ pinned down by a capital-Euler "
    "condition written in *per-worker* public capital $K^G/N$ — **but with an "
    "extra $N^{\\theta_G}$ factor that does not cancel**, since $K^G$ is itself funded "
    "as a share of *aggregate* output $Y=(Y/N)\\cdot N$, not of output per worker. "
    "Under income tax financing:"
)
st.latex(r"(1-\tau)\,\theta_K\,A\,(K^G/N)^{\theta_G}\,N^{\theta_G}\,\kappa^{\theta_K-1} = r+\delta")
st.markdown("and under lump-sum financing (no tax wedge at all):")
st.latex(r"\theta_K\,A\,(K^G/N)^{\theta_G}\,N^{\theta_G}\,\kappa^{\theta_K-1} = r+\delta")
st.markdown(
    "(When $\\theta_G=0$ the $N^{\\theta_G}$ factor is just 1 and this collapses to "
    "the familiar CRS-in-$(K,N)$ formula. When $\\theta_G>0$, this is solved by a "
    "fixed-point iteration *nested inside* another fixed-point iteration on $N$ "
    "itself — since $K^G=I^G/\\delta$ depends on output, and now the equation above "
    "depends on $N$ directly too — see the code's `_supply_side_impl` and its "
    "caller `steady_state`.) Everything else in the steady state then follows in "
    "closed form, with $s_C=1-s_I-s_{I^G}$ the steady-state consumption "
    "share (note transfers $G_T$ never reduce $s_C$, since they aren't resource-using) "
    "and $s_I=\\delta\\kappa/(Y/N)$ the investment share. Under income tax financing:"
)
st.latex(r"N = \frac{(1-\tau)\,\theta_N}{\theta_L\,s_C + (1-\tau)\,\theta_N}")
st.markdown("and under lump-sum financing:")
st.latex(r"N = \frac{\theta_N}{\theta_L\,s_C + \theta_N}")
st.latex(r"Y=\frac{Y}{N}\cdot N,\qquad K=\kappa N,\qquad C=s_C\,Y,\qquad I=s_I\,Y")
st.markdown(
    "This is the exact equation solved for the **\"1. Cumulative Effects\"** and "
    "**\"2. The Marginal Effects of Government Spendings\"** tables above: once $N$ is known, every other "
    "steady-state quantity is a simple multiple of it. This closed-form "
    "solve is why the app never fails to converge the way a numerical "
    "root-finder might."
)

st.subheader("A.8 Transition dynamics (log-linearized)")
st.markdown(
    "Around a steady state, every equation above can be approximated to first "
    "order in percent deviations (hats). The production function and labor-supply "
    "condition combine into one static relationship:"
)
st.latex(r"\hat y_t = y_k\,\hat k_t + y_{kg}\,\hat{k}^G_t + y_c\,\hat c_t + y_g\,\hat g_t")
st.markdown(
    "Capital accumulation becomes a linear law of motion for the "
    "**predetermined** state variable $\\hat k_t$ (its value today was fixed by "
    "yesterday's investment, so it cannot jump):"
)
st.latex(r"\hat k_{t+1} = \Phi_{kk}\,\hat k_t + \Phi_{kc}\,\hat c_t + \Phi_{kg}\,\hat g_t + \Phi_{k,kg}\,\hat{k}^G_t")
st.markdown(
    "and the Euler equation becomes a linear relationship pinning down the "
    "**forward-looking (\"jump\")** variable $\\hat c_t$ in terms of *tomorrow's* "
    "capital and government spending:"
)
st.latex(r"\hat c_t = \text{coef}_c\,\hat c_{t+1} + \text{coef}_k\,\hat k_{t+1} + \text{coef}_g\,\hat g_{t+1} + \text{coef}_{kg}\,\hat{k}^G_{t+1}")
st.markdown(
    "Public capital's own law of motion is exogenous — it just tracks the "
    "government-spending shock directly and does not depend on $\\hat k_t$ or "
    "$\\hat c_t$ — so it can be solved on its own and fed into the two equations "
    "above as a known forcing term:"
)
st.latex(r"\hat{k}^G_{t+1} = (1-\delta)\,\hat{k}^G_t + \delta\,\hat g_t")
st.markdown(
    "With one predetermined variable ($\\hat k_t$) and one jump variable "
    "($\\hat c_t$), the system is **saddle-path stable**: there is exactly one "
    "stable and one unstable eigenvalue of the reduced-form transition matrix. "
    "The app diagonalizes that matrix and pins the unstable (forward-looking) "
    "component to the present value of all *future* government-spending shocks — "
    "the King-Plosser-Rebelo / Blanchard-Kahn method — rather than simulating "
    "forward naively, which would blow up numerically as the unstable root gets "
    "amplified hundreds of times over a long horizon. This is exactly the "
    "computation behind every chart in **\"2. Transition dynamics\"** above."
)

st.divider()
_emblem_path = Path(__file__).parent / "kgu_emblem.png"
if _emblem_path.exists():
    _, _emblem_col, _ = st.columns([1, 1, 1])
    with _emblem_col:
        st.image(str(_emblem_path), use_container_width=True)
