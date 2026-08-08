"""
Streamlit teaching app: Baxter & King (1993, AER) "Fiscal Policy in General
Equilibrium" -- an interactive neoclassical growth model with government
purchases, for exploring how fiscal policy parameters and the *size* and
*financing* of government spending shape the whole economy.

Run locally with:   streamlit run app.py
Deploy for free on Streamlit Community Cloud by pointing it at this file.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fiscal_model import (
    calibrate_theta_L,
    public_investment_long_run,
    run_experiment,
    steady_state,
)

st.set_page_config(page_title="Fiscal Policy in General Equilibrium",
                    page_icon="\U0001F3DB️", layout="wide")

# --------------------------------------------------------------------------
# Sidebar -- model parameters and policy experiment
# --------------------------------------------------------------------------

st.sidebar.title("\U0001F3DB️ Model controls")
st.sidebar.caption("Baxter & King (1993, *AER*), “Fiscal Policy in General "
                    "Equilibrium.” Adjust anything below and every chart updates.")

if st.sidebar.button("↺ Reset to paper's benchmark calibration", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.subheader("Preferences & technology")
theta_N = st.sidebar.slider("Labor share of income, θₙ", 0.30, 0.80, 0.58, 0.01,
                             help="Cobb-Douglas exponent on labor. Capital's share is 1-θₙ.",
                             key="theta_N")
delta_pct = st.sidebar.slider("Depreciation rate, δₖ (%/yr)", 2.0, 20.0, 10.0, 0.5, key="delta_pct")
r_pct = st.sidebar.slider("Steady-state real interest rate, r (%/yr)", 1.0, 12.0, 6.5, 0.25, key="r_pct")
N_target_pct = st.sidebar.slider("Target steady-state hours worked, N (% of time endowment)",
                                  5.0, 45.0, 20.0, 1.0,
                                  help="Pins down the leisure-preference weight θ_L "
                                       "so that steady-state labor supply hits this target "
                                       "at the baseline policy.", key="N_target_pct")

st.sidebar.subheader("Baseline fiscal policy")
s_G_old_pct = st.sidebar.slider("Baseline government purchases, G/Y (%)", 5.0, 40.0, 20.0, 1.0, key="s_G_old_pct")
tau_pct = st.sidebar.slider("Baseline tax rate, τ (%)", 0.0, 45.0, 20.0, 1.0,
                             help="Used directly under lump-sum financing; under "
                                  "balanced-budget financing τ is instead set equal to G/Y "
                                  "every period.", key="tau_pct")

st.sidebar.subheader("Policy experiment")
financing_label = st.sidebar.radio(
    "How is the change in G financed?",
    ["Lump-sum (tax rate fixed, transfers absorb the residual)",
     "Balanced-budget distortionary tax (τₜ = Gₜ / Yₜ, the “GRH” rule)"],
    key="financing_label",
)
financing = "lump_sum" if financing_label.startswith("Lump-sum") else "distortionary"

s_G_new_pct = st.sidebar.slider("New government purchases, G/Y (%)", 5.0, 40.0, 25.0, 1.0, key="s_G_new_pct")

duration_label = st.sidebar.radio("Duration of the change", ["Permanent", "Temporary"], key="duration_label")
permanent = duration_label == "Permanent"
duration_years = 4
if not permanent:
    duration_years = st.sidebar.slider("Duration (years)", 1, 20, 4, 1, key="duration_years")

st.sidebar.divider()
st.sidebar.markdown(
    "**Reading the results:** all quantities in the charts are *percent "
    "deviations from the original (pre-policy) steady state* -- exactly as "
    "in Figures 2-4 of the paper. Because this app solves the model's exact "
    "nonlinear steady state (rather than the paper's closed-form elasticity "
    "approximation, eq. 16), multiplier values will be close to, but not "
    "identical to, the point estimates in the paper's tables."
)

# --------------------------------------------------------------------------
# Solve the model
# --------------------------------------------------------------------------

theta_N_v = theta_N
delta_v = delta_pct / 100.0
r_v = r_pct / 100.0
tau_v = tau_pct / 100.0
s_G_old_v = s_G_old_pct / 100.0
s_G_new_v = s_G_new_pct / 100.0
N_target_v = N_target_pct / 100.0

error = None
experiment = None
try:
    experiment = run_experiment(
        theta_N=theta_N_v, delta=delta_v, r=r_v, A=1.0, tau_baseline=tau_v,
        s_G_old=s_G_old_v, s_G_new=s_G_new_v, N_target=N_target_v,
        financing=financing, permanent=permanent, duration_years=duration_years,
        T_sim=200,
    )
except Exception as exc:  # noqa: BLE001 -- surface any parameter-infeasibility to the user
    error = str(exc)

st.title("Fiscal Policy in General Equilibrium")
st.caption("An interactive replication of Baxter & King (1993, *American Economic Review*) "
           "83(3): 315-334 — built for classroom exploration.")

if error:
    st.error(f"These parameters don't yield a valid steady state: {error}\n\n"
             "Try a smaller change in G/Y, a lower baseline G/Y, or a lower tax rate.")
    st.stop()

ss_old, ss_new, path = experiment.ss_old, experiment.ss_new, experiment.path

# --------------------------------------------------------------------------
# Headline numbers
# --------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Long-run output multiplier ΔY/ΔG",
          f"{experiment.multiplier_long_run:+.2f}")
c2.metric("Impact (year-0) output multiplier",
          f"{experiment.multiplier_impact:+.2f}")
c3.metric("Steady-state hours worked N",
          f"{ss_old.N*100:.1f}% → {ss_new.N*100:.1f}%")
c4.metric("Calibrated leisure weight θ_L", f"{ss_old.theta_L:.2f}")

if experiment.multiplier_long_run > 1:
    st.success(
        f"**A one-dollar permanent increase in government purchases raises long-run "
        f"output by ≈ ${experiment.multiplier_long_run:.2f}.** This exceed-1 "
        f"multiplier is the paper's central, surprising result: it comes from labor "
        f"supply rising (negative wealth effect) which raises the marginal product of "
        f"capital and pulls in *more* private investment (the paper's “amplification "
        f"effect”) — not from any Keynesian demand channel."
    )
elif experiment.multiplier_long_run < 0:
    st.warning(
        f"**Output *falls* by ≈ ${-experiment.multiplier_long_run:.2f} for every dollar "
        f"of new spending.** Financing the increase with a balanced-budget distortionary "
        f"tax rate turns the multiplier negative: the tax wedge on labor and capital "
        f"income depresses work effort and capital formation by more than the direct "
        f"resource cost of the spending itself. This is the paper's finding that *how* "
        f"spending is financed matters more than its resource cost."
    )

# --------------------------------------------------------------------------
# Steady-state comparison
# --------------------------------------------------------------------------

st.header("1. Comparative steady states")
st.write("Exact, closed-form solution of the model's long-run (“great ratios”) "
         "equilibrium, before vs. after the policy change.")

ss_table = pd.DataFrame({
    "Variable": ["Output Y", "Consumption C", "Investment I", "Capital K",
                 "Government purchases G", "Labor input N (% of time)",
                 "Real wage w", "Tax rate τ (%)"],
    "Original steady state": [ss_old.Y, ss_old.C, ss_old.I, ss_old.K, ss_old.G,
                               ss_old.N * 100, ss_old.w, ss_old.tau * 100],
    "New steady state": [ss_new.Y, ss_new.C, ss_new.I, ss_new.K, ss_new.G,
                          ss_new.N * 100, ss_new.w, ss_new.tau * 100],
})
ss_table["% change"] = 100 * (ss_table["New steady state"] / ss_table["Original steady state"] - 1)

st.dataframe(
    ss_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Original steady state": st.column_config.NumberColumn(format="%.4f"),
        "New steady state": st.column_config.NumberColumn(format="%.4f"),
        "% change": st.column_config.NumberColumn(format="%+.2f%%"),
    },
)

col_right = st.container()
with col_right:
    bar_vars = ["Output Y", "Consumption C", "Investment I", "Capital K"]
    pct_change = ss_table.set_index("Variable").loc[bar_vars, "% change"]
    fig_bar = go.Figure(go.Bar(x=bar_vars, y=pct_change.values,
                                marker_color=["#2563eb" if v >= 0 else "#dc2626" for v in pct_change.values],
                                text=[f"{v:+.2f}%" for v in pct_change.values], textposition="outside"))
    fig_bar.update_layout(title="Long-run % change from original steady state",
                           yaxis_title="% change", height=380, margin=dict(t=50, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------------------------------
# Transition path
# --------------------------------------------------------------------------

st.header("2. Transition dynamics")
st.write("Perfect-foresight transition path (log-linearized around the relevant steady "
         "state, solved exactly via eigen-decomposition -- no simulation noise). "
         "All series are % deviations from the *original* steady state, matching "
         "Figures 2-4 of the paper.")

years_to_show = st.slider("Years to display", 5, 100, 25, 5, key="years_to_show")
yrs = path.years[:years_to_show]

def line_fig(title, series_specs, yaxis_title):
    fig = go.Figure()
    for name, arr, color in series_specs:
        fig.add_trace(go.Scatter(x=yrs, y=arr[:years_to_show], mode="lines+markers",
                                  name=name, line=dict(color=color, width=2), marker=dict(size=4)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title=title, xaxis_title="Years after the shock", yaxis_title=yaxis_title,
                       height=380, legend=dict(orientation="h", y=1.15), margin=dict(t=70, b=20))
    return fig

tab1, tab2, tab3 = st.tabs(["Commodity market", "Labor market", "Financial market"])

with tab1:
    st.plotly_chart(line_fig(
        "Output, consumption, investment, government purchases",
        [("Output (Y)", path.Y, "#2563eb"), ("Consumption (C)", path.C, "#16a34a"),
         ("Investment (I)", path.I, "#f59e0b"), ("Government purchases (G)", path.G, "#6b7280")],
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
        "Capital stock and the real interest rate",
        [("Capital stock (K)", path.K, "#2563eb")],
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
            e = run_experiment(theta_N_v, delta_v, r_v, 1.0, tau_v, s_G_old_v, s_G_new_v,
                                N_target_v, financing, permanent=False, duration_years=T, T_sim=200)
            mults.append(e.multiplier_impact)
        except Exception:
            mults.append(np.nan)
    perm_e = run_experiment(theta_N_v, delta_v, r_v, 1.0, tau_v, s_G_old_v, s_G_new_v,
                             N_target_v, financing, permanent=True, T_sim=200)
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
# Productive public capital (Section VI)
# --------------------------------------------------------------------------

with st.expander("\U0001F3D7️ Bonus: productive public investment (Section VI / Table 4)"):
    st.write(
        "So far, government purchases are purely a resource cost (“basic” "
        "purchases). Baxter & King also study **public investment** that directly "
        "raises the productivity of private capital and labor: "
        r"$Y = A\,K^{\theta_K}\,(K^G)^{\theta_G}\,N^{\theta_N}$. "
        "Below: the long-run output effect of a marginal dollar of public investment, "
        "for different values of the public-capital productivity parameter θ_G."
    )
    s_IG_pct = st.slider("Public investment share of output, Iᴳ/Y (%)", 1.0, 15.0, 5.0, 0.5, key="s_IG_pct")
    theta_G_grid = np.array([0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40])
    try:
        theta_L_pub = calibrate_theta_L(theta_N_v, delta_v, r_v, 1.0, tau_v,
                                         s_G_old_v + s_IG_pct / 100.0, N_target_v)
        tbl = public_investment_long_run(theta_N_v, delta_v, r_v, 1.0, tau_v, s_G_old_v,
                                          theta_L_pub, theta_G_grid, s_IG=s_IG_pct / 100.0)
        fig_pub = go.Figure()
        fig_pub.add_trace(go.Scatter(x=tbl["theta_G"], y=tbl["direct"], mode="lines+markers",
                                      name="Direct effect (K, N fixed)"))
        fig_pub.add_trace(go.Scatter(x=tbl["theta_G"], y=tbl["k_adj"], mode="lines+markers",
                                      name="Private capital adjusts (N fixed)"))
        fig_pub.add_trace(go.Scatter(x=tbl["theta_G"], y=tbl["both"], mode="lines+markers",
                                      name="Full general equilibrium (K and N adjust)"))
        fig_pub.update_layout(title="ΔY / ΔIᴳ as public-capital productivity θ_G rises",
                               xaxis_title="θ_G (public-capital productivity parameter)",
                               yaxis_title="Long-run output multiplier", height=420,
                               legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_pub, use_container_width=True)
        st.caption("Even mildly productive public capital (θ_G ≈ 0.03-0.05, "
                   "Baxter & King's benchmark) generates a long-run multiplier several "
                   "times larger than basic government purchases, and most of the effect "
                   "comes from the *supply-side response* of private labor and capital, "
                   "not the direct productivity gain.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute this panel with the current sidebar settings: {exc}")

# --------------------------------------------------------------------------
# Footer / methodology note
# --------------------------------------------------------------------------

st.divider()
with st.expander("ℹ️ Methodology notes"):
    st.markdown(
        """
- **Steady state** is solved exactly (closed form) from the model's static
  first-order conditions -- no numerical root-finding, so it never fails to
  converge.
- **Transition dynamics** are the standard log-linearization around a steady
  state, solved via eigen-decomposition of the reduced-form transition
  matrix (King-Plosser-Rebelo / Blanchard-Kahn method). Because the leisure
  weight enters only through steady-state shares, this is exact to first
  order for any shock size, and does not suffer the numerical blow-up that a
  naive forward "shooting" simulation would over long horizons.
- **Two financing rules** are implemented: lump-sum (tax rate fixed;
  transfers absorb the government budget residually) and balanced-budget
  distortionary taxation (tax rate reset each period to `G_t / Y_t`,
  Baxter & King's stylized Gramm-Rudman-Hollings experiment).
- Reported multipliers are close to, but will not exactly reproduce, the
  point estimates in the paper's Tables 2-4, which are built from a
  closed-form elasticity approximation (equation 16/16'); this app instead
  solves the *exact* nonlinear steady state and the *exact* linear saddle
  path implied by the same first-order conditions.

**Citation:** Baxter, Marianne, and Robert G. King. 1993. "Fiscal Policy in
General Equilibrium." *American Economic Review* 83 (3): 315-334.
        """
    )
