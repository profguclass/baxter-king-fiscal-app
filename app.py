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
)

st.set_page_config(page_title="Fiscal Policy in General Equilibrium",
                    page_icon="\U0001F3DB️", layout="wide")

# --------------------------------------------------------------------------
# Sidebar -- model controls
# --------------------------------------------------------------------------

st.sidebar.title("\U0001F3DB️ Model controls")
st.sidebar.caption("Baxter & King (1993, *AER*), “Fiscal Policy in General "
                    "Equilibrium.” Adjust anything below and every panel updates.")

if st.sidebar.button("↺ Reset to paper's benchmark calibration", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Fixed at the paper's benchmark calibration (Table 1): labor share of income,
# depreciation rate, and target steady-state hours worked are no longer sidebar
# controls.
theta_N = 0.58
delta_pct = 10.0
N_target_pct = 40.0

# 1. Duration ---------------------------------------------------------------
st.sidebar.subheader("1. Duration of the change")
duration_label = st.sidebar.radio("Duration", ["Permanent", "Temporary"], key="duration_label")
permanent = duration_label == "Permanent"
duration_years = 4
if not permanent:
    duration_years = st.sidebar.slider("Duration (years)", 1, 20, 4, 1, key="duration_years")

# 2. Financing rule -----------------------------------------------------
st.sidebar.subheader("2. How is the tax collected?")
financing_label = st.sidebar.radio(
    "Financing rule",
    ["Lump-sum (tax revenue collected without distorting labor/capital margins)",
     "Income tax (tax is a (1-τ) wedge on labor and capital income)"],
    help="Both options collect the SAME tax revenue: τ = G/Y always. They differ only "
         "in whether that tax distorts the household's labor-leisure and capital-Euler "
         "first-order conditions.",
    key="financing_label",
)
financing = "lump_sum" if financing_label.startswith("Lump-sum") else "income_tax"

# 3. theta_G -----------------------------------------------------------------
st.sidebar.subheader("3. Productive public capital")
theta_G = st.sidebar.slider(
    "Public-capital productivity, θ_G", 0.00, 0.40, 0.05, 0.01,
    help="Y = A·K^θK·(Kᴳ)^θG·N^θN. θG=0 means public investment is a pure resource "
         "cost with no productivity effect (Baxter & King's Table 4 grid runs 0-0.40).",
    key="theta_G",
)

# 4. Steady-state real interest rate -----------------------------------------
st.sidebar.subheader("4. Real interest rate")
r_pct = st.sidebar.slider("Steady-state real interest rate, r (%/yr)", 1.0, 12.0, 6.5, 0.25, key="r_pct")

# 5. Baseline fiscal policy: size and change ------------------------
st.sidebar.subheader("5. Baseline government purchases")
s_G_old_pct = st.sidebar.slider("Baseline total government purchases, G/Y (%)", 5.0, 40.0, 20.0, 1.0, key="s_G_old_pct")
st.sidebar.caption("Tax rate τ = G/Y always, under both financing rules above (item 2).")

delta_s_G_pct = st.sidebar.slider(
    "Change in government purchases, ΔG/Y (percentage points)", -10.0, 15.0, 5.0, 0.5,
    help="Expressed relative to the baseline G/Y set above, so the experiment is "
         "never accidentally a zero-size change. Composition shares (item 6) stay fixed.",
    key="delta_s_G_pct")
s_G_new_pct_raw = s_G_old_pct + delta_s_G_pct
s_G_new_pct = min(max(s_G_new_pct_raw, 1.0), 48.0)
if s_G_new_pct != s_G_new_pct_raw:
    st.sidebar.warning(f"New G/Y clamped to {s_G_new_pct:.1f}% (must stay between 1% and 48%).")
st.sidebar.caption(f"New G/Y = {s_G_old_pct:.1f}% + {delta_s_G_pct:+.1f} = **{s_G_new_pct:.1f}%**")

# 6. Composition of G ------------------------------------------------------
st.sidebar.subheader("6. Composition of G")
f_IG_pct = st.sidebar.slider("Public investment share of G, Iᴳ/G (%)", 0.0, 100.0, 25.0, 5.0, key="f_IG_pct")
f_TR_pct = st.sidebar.slider("Transfers share of G, TR/G (%)", 0.0, 100.0 - f_IG_pct, 0.0, 5.0, key="f_TR_pct")
f_GB_pct = 100.0 - f_IG_pct - f_TR_pct
st.sidebar.caption(f"Basic purchases G_B/G = **{f_GB_pct:.0f}%** is the residual determined "
                    f"by the two sliders above (100% − Iᴳ/G − TR/G).")

fig_comp = go.Figure()
for name, val, color in [("G_B", f_GB_pct, "#6b7280"),
                          ("Iᴳ", f_IG_pct, "#16a34a"),
                          ("TR", f_TR_pct, "#7c3aed")]:
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
f_GB_v, f_IG_v, f_TR_v = f_GB_pct / 100.0, f_IG_pct / 100.0, f_TR_pct / 100.0

error = None
experiment = None
try:
    experiment = run_experiment(
        theta_N=theta_N_v, delta=delta_v, r=r_v, A=1.0, theta_G=theta_G_v,
        s_G_old=s_G_old_v, s_G_new=s_G_new_v, f_GB=f_GB_v, f_IG=f_IG_v, f_TR=f_TR_v,
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
# Headline numbers
# --------------------------------------------------------------------------

if abs(delta_s_G_pct) < 1e-9:
    st.info("ΔG/Y is set to 0 — the baseline and new policy are identical, so there's no "
            "experiment to show yet. Move the **ΔG/Y** slider away from 0 in the sidebar.")
    st.stop()

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
        f"**A one-dollar permanent increase in government purchases raises long-run "
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
# 1. Steady-state / equilibrium comparison
# --------------------------------------------------------------------------

st.header("1. Comparative steady states")
st.write("Exact, closed-form solution of the model's long-run (“great ratios”) "
         "equilibrium, before vs. after the policy change.")
left_label, right_label = "Original steady state", "New steady state"

ss_table = pd.DataFrame({
    "Variable": ["Output Y", "Consumption C", "Investment I", "Private capital K",
                 "Public capital Kᴳ", "Government purchases G (total)",
                 "Real wage w", "Tax rate τ (%)"],
    left_label: [ss_old.Y, ss_old.C, ss_old.I, ss_old.K, ss_old.KG, ss_old.G,
                 ss_old.w, ss_old.tau * 100],
    right_label: [ss_new.Y, ss_new.C, ss_new.I, ss_new.K, ss_new.KG, ss_new.G,
                  ss_new.w, ss_new.tau * 100],
})
ss_table["% change"] = 100 * (ss_table[right_label] / ss_table[left_label].replace(0, np.nan) - 1)

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
fig_bar = go.Figure(go.Bar(x=bar_vars, y=pct_change.values,
                            marker_color=["#2563eb" if v >= 0 else "#dc2626" for v in pct_change.values],
                            text=[f"{v:+.2f}%" for v in pct_change.values], textposition="outside"))
fig_bar.update_layout(title=f"% change from the {left_label.lower()}",
                       yaxis_title="% change", height=380, margin=dict(t=50, b=20))
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------------------------------
# 2. Transition dynamics
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
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"),
                       xaxis_title="Years after the shock", yaxis_title=yaxis_title,
                       height=420, legend=dict(orientation="h", y=1.18, yanchor="bottom"),
                       margin=dict(t=110, b=20))
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
                                f_GB_v, f_IG_v, f_TR_v, N_target_v, financing,
                                permanent=False, duration_years=T, T_sim=200)
            mults.append(e.multiplier_impact)
        except Exception:
            mults.append(np.nan)
    perm_e = run_experiment(theta_N_v, delta_v, r_v, 1.0, theta_G_v, s_G_old_v, s_G_new_v,
                             f_GB_v, f_IG_v, f_TR_v, N_target_v, financing,
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
    "Baxter & King's Section VI: public investment directly raises the productivity of "
    "private capital and labor, "
    r"$Y = A\,K^{\theta_K}\,(K^G)^{\theta_G}\,N^{\theta_N}$. "
    "Below: the long-run effect of a marginal dollar of public investment on output, "
    "consumption, and investment, for a grid of θ_G. The row matching the sidebar's "
    "current θ_G is highlighted."
)
theta_G_grid = np.array([0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40])
try:
    theta_L_pub = calibrate_theta_L(theta_N_v, delta_v, r_v, 1.0, theta_G_v,
                                     s_G_old_v * f_GB_v, s_G_old_v * f_IG_v, s_G_old_v * f_TR_v,
                                     financing == "income_tax", N_target_v)
    tbl = public_investment_long_run(theta_N_v, delta_v, r_v, 1.0, s_G_old_v * f_GB_v, theta_L_pub,
                                      financing == "income_tax", theta_G_grid,
                                      s_IG=max(s_G_old_v * f_IG_v, 1e-4))
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
    st.caption("Even mildly productive public capital (θ_G ≈ 0.03-0.05, "
               "Baxter & King's benchmark) generates a long-run output multiplier several "
               "times larger than basic government purchases, driven by the full "
               "general-equilibrium response of private capital and labor.")
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
- **Composition (item 6)**: total government purchases G/Y is split into basic
  purchases G_B (pure resource cost), public investment Iᴳ (accumulates into Kᴳ,
  productivity-enhancing if θ_G>0), and transfers TR (resource-neutral, returned to
  households). Only G_B and Iᴳ enter the economy-wide resource constraint
  Y=C+I+G_B+Iᴳ; TR nets out in aggregate.
- Reported multipliers are close to, but will not exactly reproduce, the point
  estimates in the paper's Tables 2-4, which are built from a closed-form elasticity
  approximation (equation 16/16'); this app instead solves the *exact* nonlinear
  steady state and the *exact* linear saddle path implied by the same first-order
  conditions.

**Citation:** Baxter, Marianne, and Robert G. King. 1993. "Fiscal Policy in
General Equilibrium." *American Economic Review* 83 (3): 315-334.
        """
    )

st.divider()
_emblem_path = Path(__file__).parent / "kgu_emblem.png"
if _emblem_path.exists():
    _, _emblem_col, _ = st.columns([1, 1, 1])
    with _emblem_col:
        st.image(str(_emblem_path), use_container_width=True)
