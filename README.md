# Fiscal Policy in General Equilibrium — a classroom simulator

An interactive replication of **Baxter, Marianne, and Robert G. King. 1993.
"Fiscal Policy in General Equilibrium." *American Economic Review* 83(3):
315–334.**

The paper's core model is a one-sector neoclassical growth model with
government purchases: representative-agent preferences over consumption and
leisure, Cobb-Douglas production with variable labor *and* endogenous
capital accumulation, and a government budget that can be financed either by
lump-sum taxation or by a distortionary balanced-budget tax rule. Students
can move any structural parameter or fiscal-policy setting and immediately
see the effect on the economy's steady state and its transition path.

## What's inside

| File | Purpose |
|---|---|
| `fiscal_model.py` | The economics: closed-form steady-state solver, exact log-linear transition-path solver (eigen-decomposition / saddle-path method), and the Section VI productive-public-capital extension. No UI code. |
| `app.py` | The Streamlit front end: sliders for every structural and policy parameter, steady-state comparison table/chart, transition-path charts (commodity / labor / financial markets), a duration-sensitivity panel (Table 3 style), and a productive-public-investment panel (Table 4 style). |
| `test_model.py` | Sanity checks against the numbers reported in the paper's Tables 1–4 and Figures 2–5. |
| `requirements.txt` | Python dependencies. |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Publishing to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **"New app."**
3. Point it at your repo, branch, and `app.py` as the entry point.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically and
   gives you a shareable `https://<something>.streamlit.app` URL that your
   students can use directly, no installation required.

Any time you push a new commit, the deployed app redeploys automatically.

## The economics, briefly

- **Steady state** is solved in closed form (no numerical root-finding): the
  capital/labor ratio, wage, and rental rate are pinned down by the supply
  side alone (independent of labor input), and labor supply is then solved
  from the consumption/leisure first-order condition given the government's
  claim on resources.
- **Transition dynamics** use the standard log-linearization around a
  steady state. The equilibrium has exactly one predetermined variable
  (capital) and one forward-looking "jump" variable (consumption), so it is
  saddle-path stable. The app solves this **exactly** via eigen-decomposition
  of the reduced-form transition matrix (the King–Plosser–Rebelo /
  Blanchard–Kahn approach), rather than a naive forward "shooting"
  simulation — the latter is numerically unstable here because the unstable
  root gets iterated many times, amplifying floating-point noise
  geometrically.
- **Two financing rules** replicate the paper's central finding that *how*
  a spending increase is financed matters more than its direct resource
  cost:
  - *Lump-sum*: the tax rate is held fixed; transfers absorb the government
    budget residually. Permanent increases in G typically raise long-run
    output by *more* than one-for-one (multiplier > 1) because higher labor
    supply raises the marginal product of capital and pulls in additional
    private investment.
  - *Distortionary / balanced-budget ("GRH")*: the tax rate is reset every
    period to `G_t / Y_t`. The added tax wedge on labor and capital income
    typically makes the multiplier **negative** — output falls by more than
    the dollar value of the new spending.
- **Productive public capital** (Section VI extension, exposed as a bonus
  panel) lets government investment directly raise the marginal product of
  private capital and labor, generating long-run multipliers far above 1
  even for modest values of the public-capital productivity parameter.

Because the app solves the *exact* nonlinear steady state and the *exact*
linear saddle path (rather than the paper's closed-form elasticity
approximation, its equation 16/16'), multiplier values will be close to,
but will not exactly reproduce, the specific point estimates in the paper's
Tables 2–4. The qualitative patterns — multiplier > 1 with lump-sum
financing, multiplier < 0 under GRH, more persistent shocks producing
larger short-run effects, productive public capital dramatically amplifying
long-run output — all replicate closely.
