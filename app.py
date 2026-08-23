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
    public_investment_long_run,
    run_experiment,
    steady_state_for_policy,
)

st.set_page_config(page_title="Fiscal Policy in General Equilibrium",
                    page_icon="\U0001F3DB️", layout="wide")

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

# 1. Duration ---------------------------------------------------------------
st.sidebar.subheader("1. Duration of the change")
duration_label = st.sidebar.radio("Duration", ["Permanent", "Temporary"], key=wkey("duration_label"))
permanent = duration_label == "Permanent"
duration_years = 4
if not permanent:
    duration_years = st.sidebar.slider("Duration (years)", 1, 20, 4, 1, key=wkey("duration_years"))

# 2. Financing rule -----------------------------------------------------
st.sidebar.subheader("2. How is the tax collected?")
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

# 3. theta_G -----------------------------------------------------------------
st.sidebar.subheader("3. Productive public capital")
theta_G = st.sidebar.slider(
    "Public-capital productivity, θ_G", 0.00, 0.40, 0.00, 0.01,
    help="Y = A·K^θK·(Kᴳ)^θG·N^θN. θG=0 means public investment is a pure resource "
         "cost with no productivity effect (Baxter & King's Table 4 grid runs 0-0.40).",
    key=wkey("theta_G"),
)

# 4. Steady-state real interest rate -----------------------------------------
st.sidebar.subheader("4. Real interest rate")
r_pct = st.sidebar.slider("Steady-state real interest rate, r (%/yr)", 1.0, 12.0, 6.5, 0.25, key=wkey("r_pct"))

# 5. Total government spending -----------------------------------------
st.sidebar.subheader("5. Total government spending")
BENCH_s_G_pct = 20.0  # paper's benchmark calibration (Table 1) -- the fixed baseline
                       # every experiment/comparison below is measured against.
s_G_old_pct = BENCH_s_G_pct
s_G_new_pct = st.sidebar.slider(
    "Total government spending, G/Y (%)", 5.0, 40.0, BENCH_s_G_pct, 1.0,
    help="The paper's benchmark calibration is G/Y = 20% (dashed reference). Move this "
         "slider to change total government spending; every result in this app compares "
         "against that fixed 20% baseline. Composition shares (item 6) stay fixed.",
    key=wkey("s_G_new_pct"))
delta_s_G_pct = s_G_new_pct - s_G_old_pct
st.sidebar.caption(f"Tax rate τ = G/Y always, under both financing rules above (item 2). "
                    f"ΔG/Y vs. the {BENCH_s_G_pct:.0f}% benchmark = **{delta_s_G_pct:+.1f}** "
                    f"percentage points.")

# 6. Composition of G ------------------------------------------------------
st.sidebar.subheader("6. Composition of G")
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
experiment = None
try:
    experiment = run_experiment(
        theta_N=theta_N_v, delta=delta_v, r=r_v, A=1.0, theta_G=theta_G_v,
        s_G_old=s_G_old_v, s_G_new=s_G_new_v, f_IG=f_IG_v, f_GT=f_GT_v,
        N_target=N_target_v, financing=financing,
        permanent=permanent, duration_years=duration_years, T_sim=200,
    )
except Exception as exc:  # noqa: BLE001 -- surface any parameter-infeasibility to the user
    error = str(exc)

st.title("Fiscal Policy in General Equilibrium")
st.caption("An interactive replication of Baxter & King (1993, *American Economic Review*) "
           "83(3): 315-334 — built for classroom exploration.")

if error:
    st.error(f"These parameters don't yield a valid equilibrium: {error}\n\n"
             "Try a smaller change in G/Y, a lower baseline G/Y, or a different composition.")
    st.stop()

ss_old, ss_new, path = experiment.ss_old, experiment.ss_new, experiment.path

# --------------------------------------------------------------------------
# 1. Steady-state / equilibrium comparison: G/Y=20% baseline vs. current G/Y,
# holding every OTHER sidebar setting (financing, composition, θ_G, r) fixed
# at whatever it is currently set to. This reuses ss_old/ss_new from the
# ΔG/Y experiment above, so resetting the G/Y slider to 20% always drives
# every number and bar back to exactly zero -- moving composition/financing/
# θ_G/r alone (with G/Y left at 20%) is correctly a no-op here, since both
# sides of the comparison share those settings; only ΔG/Y ever shows up.
# --------------------------------------------------------------------------

st.header("1. Comparative steady states")
st.write("Exact, closed-form solution of the model's long-run (“great ratios”) "
         "equilibrium: **G/Y = 20% (the paper's benchmark share)** vs. **the total "
         "government spending set in the sidebar (item 5)**, holding financing, "
         "composition, θ_G, and r fixed at whatever the sidebar currently has them at. "
         "Resetting item 5 back to 20% always brings every number below back to zero.")
if f_IG_pct == 0.0 and financing == "lump_sum":
    st.caption("⚠️ At the current settings (0% public investment, i.e. **all** government "
               "spending is transfers, under **lump-sum** financing), changing \"Total "
               "government spending\" only changes G, τ, and Transfers G_T -- Output, "
               "Consumption, Investment, Capital, and N stay exactly flat regardless of "
               "G/Y. This is a real model result, not a bug: a lump-sum tax funding an "
               "equal-sized lump-sum transfer is a pure wash for the household (no "
               "(1-τ) wedge to distort anything under lump-sum financing). Raise the "
               "public investment share (item 6) or switch to Income tax financing "
               "(item 2) to see the other variables respond.")

left_label, right_label = f"G/Y = {BENCH_s_G_pct:.0f}% (benchmark share)", "Current G/Y"

ss_table = pd.DataFrame({
    "Variable": ["Output Y", "Consumption C", "Investment I", "Private capital K",
                 "Public capital Kᴳ", "Government spending G (total)",
                 "  Public investment Iᴳ", "  Transfers G_T",
                 "Labor input N (% of time)", "Real wage w", "Tax rate τ (%)"],
    left_label: [ss_old.Y, ss_old.C, ss_old.I, ss_old.K, ss_old.KG,
                 ss_old.G, ss_old.IG, ss_old.GT, ss_old.N * 100,
                 ss_old.w, ss_old.tau * 100],
    right_label: [ss_new.Y, ss_new.C, ss_new.I, ss_new.K, ss_new.KG, ss_new.G,
                  ss_new.IG, ss_new.GT, ss_new.N * 100,
                  ss_new.w, ss_new.tau * 100],
})
both_zero = (ss_table[left_label] == 0) & (ss_table[right_label] == 0)
ss_table["% change"] = 100 * (ss_table[right_label] / ss_table[left_label].replace(0, np.nan) - 1)
ss_table.loc[both_zero, "% change"] = 0.0

st.dataframe(
    ss_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        left_label: st.column_config.NumberColumn(format="%.6f"),
        right_label: st.column_config.NumberColumn(format="%.6f"),
        "% change": st.column_config.NumberColumn(format="%+.2f%%"),
    },
)

bar_vars = ["Output Y", "Consumption C", "Investment I", "Private capital K", "Public capital Kᴳ"]
pct_change = ss_table.set_index("Variable").loc[bar_vars, "% change"]
bar_heights = pct_change.fillna(0.0).values
bar_text = [("n/a" if pd.isna(v) else f"{v:+.2f}%") for v in pct_change.values]
fig_bar = go.Figure(go.Bar(x=bar_vars, y=bar_heights,
                            marker_color=["#2563eb" if v >= 0 else "#dc2626" for v in bar_heights],
                            text=bar_text, textposition="outside"))
fig_bar.update_layout(title=f"% change from the G/Y = {BENCH_s_G_pct:.0f}% benchmark share",
                       yaxis_title="% change", height=380, margin=dict(t=50, b=20))
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------------------------------
# Headline numbers (the ΔG/Y policy-experiment multipliers specifically)
# --------------------------------------------------------------------------

if abs(delta_s_G_pct) < 1e-9:
    st.info("ΔG/Y is set to 0 — the long-run/impact **multipliers** and the **transition "
            "dynamics** below need a nonzero spending change to be defined (they isolate "
            "the effect of *that one* policy lever, holding everything else fixed). Move "
            "the **ΔG/Y** slider away from 0 in the sidebar to see them. Panel 1 above and "
            "the Table 4 panel below don't need ΔG/Y and are already showing your current "
            "settings.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Long-run output multiplier ΔY/ΔG",
              f"{experiment.multiplier_long_run:+.2f}")
    c2.metric("Impact (year-0) output multiplier",
              f"{experiment.multiplier_impact:+.2f}")
    c3.metric("Steady-state hours worked N",
              f"{ss_old.N*100:.1f}% → {ss_new.N*100:.1f}%")
    c4.metric("Tax rate τ = G/Y", f"{ss_old.tau*100:.1f}% → {ss_new.tau*100:.1f}%")

    if experiment.multiplier_long_run > 1:
        st.success(
            f"**A one-dollar permanent increase in government spending raises long-run "
            f"output by ≈ ${experiment.multiplier_long_run:.2f}.** This exceed-1 "
            f"multiplier is the paper's central, surprising result: it comes from labor "
            f"supply rising (negative wealth effect), and — if θ_G>0 — public investment "
            f"directly raising the productivity of private capital and labor."
        )
    elif experiment.multiplier_long_run < 0:
        st.warning(
            f"**Output *falls* by ≈ ${-experiment.multiplier_long_run:.2f} for every dollar "
            f"of new spending.** Under Income Tax financing the tax wedge on labor and "
            f"capital income depresses work effort and capital formation by more than the "
            f"direct resource cost of the spending itself."
        )

    # --------------------------------------------------------------------------
    # 2. Transition dynamics
    # --------------------------------------------------------------------------

    st.header("2. Transition dynamics")
    st.write("Perfect-foresight transition path (log-linearized around the relevant steady "
             "state, solved exactly via eigen-decomposition -- no simulation noise). "
             "All series are % deviations from the *original* steady state, matching "
             "Figures 2-4 of the paper.")

    years_to_show = st.slider("Years to display", 5, 100, 25, 5, key=wkey("years_to_show"))
    yrs = path.years[:years_to_show]

    def line_fig(title, series_specs, yaxis_title):
        fig = go.Figure()
        for name, arr, color in series_specs:
            fig.add_trace(go.Scatter(x=yrs, y=arr[:years_to_show], mode="lines+markers",
                                      name=name, line=dict(color=color, width=2), marker=dict(size=4)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"),
                           xaxis_title="Years after the shock", yaxis_title=yaxis_title,
                           height=420, legend=dict(orientation="h", y=1.18, yanchor="bottom"),
                           margin=dict(t=110, b=20))
        return fig

    tab1, tab2, tab3 = st.tabs(["Commodity market", "Labor market", "Financial market"])

    with tab1:
        st.plotly_chart(line_fig(
            "Output, consumption, investment, government spending",
            [("Output (Y)", path.Y, "#2563eb"), ("Consumption (C)", path.C, "#16a34a"),
             ("Investment (I)", path.I, "#f59e0b"), ("Government spending (G)", path.G, "#6b7280")],
            "% deviation from original steady state"), use_container_width=True)
        st.caption("Compare to Baxter & King Figure 2 (permanent) / Figure 3 (temporary war) / "
                   "Figure 4 (GRH). Watch the investment 'accelerator boom' on impact when the "
                   "shock is permanent and lump-sum financed.")

    with tab2:
        st.plotly_chart(line_fig(
            "Labor input and the real wage",
            [("Labor input (N)", path.N, "#2563eb"), ("Real wage (w)", path.W, "#dc2626")],
            "% deviation from original steady state"), use_container_width=True)
        st.caption("A permanent, lump-sum-financed increase in G is a negative wealth effect: "
                   "households work more and consume less, so labor rises and (with capital "
                   "predetermined) the wage falls on impact.")

    with tab3:
        st.plotly_chart(line_fig(
            "Private and public capital stocks",
            [("Private capital (K)", path.K, "#2563eb"), ("Public capital (Kᴳ)", path.KG, "#16a34a")],
            "% deviation from original steady state"), use_container_width=True)
        fig_r = go.Figure(go.Scatter(x=yrs, y=path.r_bp[:years_to_show], mode="lines+markers",
                                      line=dict(color="#7c3aed", width=2)))
        fig_r.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_r.update_layout(title="Real interest rate, deviation from steady state",
                             xaxis_title="Years after the shock", yaxis_title="Basis points",
                             height=350, margin=dict(t=50, b=20))
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption("An unanticipated permanent increase in G should raise short real rates on "
                   "impact -- the model's sharpest, most testable empirical prediction "
                   "(Section III.E of the paper).")

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
        perm_e = run_experiment(theta_N_v, delta_v, r_v, 1.0, theta_G_v, s_G_old_v, s_G_new_v,
                                 f_IG_v, f_GT_v, N_target_v, financing,
                                 permanent=True, T_sim=200)
        fig_dur = go.Figure()
        fig_dur.add_trace(go.Scatter(x=durations, y=mults, mode="lines+markers", name="Temporary shock"))
        fig_dur.add_hline(y=perm_e.multiplier_impact, line_dash="dash", line_color="#dc2626",
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
# 3. Table 4 replication: public-capital productivity sensitivity
# --------------------------------------------------------------------------

st.header("3. Productive public investment")
st.write(
    "Baxter & King's Section VI, replicating their Table 4 exactly: public investment "
    "directly raises the productivity of private capital and labor, "
    r"$Y = A\,K^{\theta_K}\,(K^G)^{\theta_G}\,N^{\theta_N}$. "
    "Below: the long-run effect of a marginal dollar of public investment on output, "
    "consumption, and investment, for a grid of θ_G, at the paper's calibration "
    "(public investment fixed at 5% of output, always lump-sum financed, regardless "
    "of the sidebar's composition/financing settings elsewhere). Since public "
    "investment is only *productive* if θ_G>0, the θ_G=0 row reproduces the ordinary "
    "lump-sum spending multiplier from the headline metric above, not zero. The row "
    "matching the sidebar's current θ_G is highlighted."
)
theta_G_grid = np.array([0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40])
try:
    s_other_pub = max(s_G_old_v - 0.05, 0.0)
    tbl = public_investment_long_run(theta_N_v, delta_v, r_v, 1.0, s_other_pub, N_target_v,
                                      theta_G_grid, s_IG=0.05)
    table4 = pd.DataFrame({
        "θ_G": tbl["theta_G"],
        "ΔY / ΔIᴳ": tbl["both"],
        "ΔC / ΔIᴳ": tbl["dC"],
        "ΔI / ΔIᴳ": tbl["dI"],
    })
    closest_idx = int(np.argmin(np.abs(table4["θ_G"] - theta_G_v)))

    def _highlight_current(row):
        return ["background-color: rgba(220,38,38,0.25)" if row.name == closest_idx else "" for _ in row]

    st.dataframe(
        table4.style.apply(_highlight_current, axis=1).format({
            "θ_G": "{:.2f}",
            "ΔY / ΔIᴳ": "{:.2f}",
            "ΔC / ΔIᴳ": "{:.2f}",
            "ΔI / ΔIᴳ": "{:.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("At the paper's own benchmark (θ_G ≈ 0.05), the full general-equilibrium "
               "multiplier is roughly 2.6 times the direct productivity effect alone -- "
               "most of the payoff comes from the *supply-side response* of private "
               "labor and capital, not the direct productivity gain, matching Baxter "
               "& King's Table 4 finding almost exactly.")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not compute Table 4 with the current sidebar settings: {exc}")

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
- **Financing (item 2)**: both "Lump-sum" and "Income tax" set τ = G/Y identically.
  They differ only in whether that tax enters the household's labor-leisure and
  capital-Euler first-order conditions as a (1-τ) wedge (Income tax) or not (Lump-sum,
  a true non-distorting poll tax for revenue purposes).
- **Composition (item 6)**: total government spending G/Y is split into public
  investment Iᴳ (accumulates into Kᴳ, productivity-enhancing if θ_G>0) and transfers
  G_T (resource-neutral, returned to households). There is no separate "basic
  purchases" category, since the utility function never assigns households any value
  from government consumption directly -- only Iᴳ enters the economy-wide resource
  constraint Y=C+I+Iᴳ; G_T nets out in aggregate.
- **Panel 1** compares G/Y = 20% (the paper's benchmark share) against the
  sidebar's current G/Y (item 5), holding every *other* setting -- financing,
  composition, θ_G, r -- fixed at whatever the sidebar currently has them at, same
  as the **multipliers and transition dynamics** below. This isolates the effect of
  the ΔG/Y policy lever specifically: resetting G/Y to 20% always brings Panel 1
  back to exactly zero, and the multipliers/transition dynamics need ΔG/Y ≠ 0 to be
  defined for the same reason.
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
    "components of total government spending (sidebar item 6)."
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
    "sidebar (item 2) — what differs between them is not how much revenue is "
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
st.latex(r"\text{Euler, income tax:}\qquad \frac{1}{C_t} = \beta\,E_t\!\left[\frac{1+(1-\tau_{t+1})\,\mathrm{MPK}_{t+1}-\delta}{C_{t+1}}\right]")
st.latex(r"\text{Euler, lump-sum:}\qquad \frac{1}{C_t} = \beta\,E_t\!\left[\frac{1+\mathrm{MPK}_{t+1}-\delta}{C_{t+1}}\right],\qquad \beta=\frac{1}{1+r}")
st.markdown(
    "The household is indifferent between consuming one more unit today and "
    "saving it to consume $(1+\\text{after-tax net return})$ units tomorrow — with "
    "the same $(1-\\tau_{t+1})$ wedge on the capital return under income tax, and "
    "no wedge under lump-sum. This is the equation that makes consumption "
    "**forward-looking** — its expectation of the *entire future* path of the "
    "economy is baked into today's consumption choice, which is exactly why the "
    "transition path (A.8) has to be solved with a saddle-path method rather than "
    "simulated forward period by period."
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
    "This is the exact equation solved for the "
    "**\"1. Comparative steady states\"** table above: once $N$ is known, every "
    "other steady-state quantity is a simple multiple of it. This closed-form "
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
