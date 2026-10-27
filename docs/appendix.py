"""The Appendix: the model, agent by agent.

Each part of the model is told first as a story, then given in symbols in a
bordered box. app.py shows it on its own page (?view=appendix), which the button
at the bottom of the main page opens in a separate browser window.
"""

import streamlit as st


def render():
    # Any equation still wider than a narrow window scrolls inside its box. The
    # padding absorbs KaTeX's 1-2px overhang, so equations that fit show no scrollbar.
    st.markdown(
        "<style>"
        '[data-testid="stMarkdownContainer"]:has(> .katex-display)'
        " { min-width: 0; max-width: 100%; overflow-x: auto; overflow-y: hidden; }"
        " .katex-display { padding: 0 4px; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("Appendix: The model, agent by agent")
    st.markdown(
        """
    Each part of the model is told first as a story, in words only. The box that follows,
    **In symbols**, gives the same part as a formal equation.

    Picture an island whose population adds up to exactly one. On it live four kinds of
    agents. There are **firms**, which make goods. There are **optimizing households**,
    who look ahead, own all the island's private machines, and save for the future. There
    are **hand-to-mouth households**, who own nothing, cannot save or borrow, and spend
    each year's income in that same year. And there is a **government**, which spends,
    builds, and taxes.

    Each of these agents faces its own problem. It wants to do the best it can, but it is
    held back by some constraint. The story of this island is the story of a set of prices
    and quantities at which every agent has solved its own problem at the same moment, and
    at which every market, for labor, for the use of capital, and for goods, balances
    exactly. Economists call that state of affairs an **equilibrium**. In each section
    below, we first tell what an agent wants and what holds it back, which is its
    **problem**, and then what its best choice must look like, which is its **solution**.
    """
    )

    st.subheader("A.1 The cast and the words we use")
    st.markdown(
        """
    Every year on the island, three private quantities matter most. There is output, the
    total amount of goods made. There is private consumption, the goods households use up.
    And there is private investment, the goods set aside to build new machines.

    Two inputs make the goods. The first is the stock of private capital, the machines the
    households own. The second is hours worked. Every person has one unit of time each
    year; hours worked are the fraction of that time spent working, and whatever is left
    over is leisure.

    Two prices tie the inputs to their owners. The wage is what a firm pays for one hour of
    work. The rental rate is what a firm pays to use one machine for one year.

    The government's total spending is made of three parts added together. The first is
    public investment, which builds public capital such as roads and bridges. The second
    is in-kind welfare, goods and services the government hands to people directly. The
    third is cash welfare, money paid to people. Public capital is a stock, built up over
    time by public investment.

    Households may value in-kind welfare less than goods they buy for themselves. A number
    between zero and one, which we will call the in-kind value, says how many units of
    their own consumption one unit of in-kind welfare is worth to them. When it is one, a
    meal from the government is as good as a meal they buy; when it is zero, the
    government's gift is worth nothing to them.

    A second number between zero and one, the hand-to-mouth share, tells what fraction of
    the population is hand-to-mouth. The rest, one minus that share, are optimizers.

    The government has three taxes. The lump-sum tax is a fixed amount that every person
    pays no matter what they do. The income-tax rate is the fraction of income the
    government takes. The consumption-tax rate is the fraction added on top of the price of
    everything a household buys.

    Optimizers care less about the future than the present. The discount factor tells how
    much they value a unit of happiness one year from now compared with the same unit
    today; it equals one divided by one plus the steady-state real interest rate. The
    depreciation rate is the fraction of any stock of capital, private or public, that
    wears out each year.

    Three output elasticities describe the technology. The labor elasticity says by what
    percent output rises when hours rise by one percent. The private-capital elasticity
    does the same for private capital, and it equals one minus the labor elasticity, so
    the two always add up to one. The public-capital elasticity does the same for public
    capital. Finally, the leisure weight says how much households care about leisure
    compared with consumption.

    When we need to tell the two kinds of household apart, we call one "the optimizer" and
    the other "the hand-to-mouth household."
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
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
        """
    **What the firms want.** A firm on the island rents machines from their owners and
    hires hours of work, and it chooses how many of each to use so as to make its profit as
    large as possible. Profit is what is left of output after the firm pays the wage for
    every hour it hires and the rental rate for every machine it rents.

    Output is made by a recipe that combines three ingredients. The first is private
    capital: when the firm uses one percent more of it, output rises by a percentage equal
    to the private-capital elasticity. The second is hours: one percent more hours raises
    output by a percentage equal to the labor elasticity. The third is public capital,
    which the government provides and the firm uses for free. Public capital enters the
    recipe not by its raw size but by how large it is compared with a fixed reference
    amount, set at about half of the public capital in the benchmark economy. One percent
    more public capital raises output by a percentage equal to the public-capital
    elasticity. The three effects multiply together. Since the two private elasticities add
    up to one, a firm that doubles both its machines and its hours, while public capital
    stays where it is, exactly doubles its output.

    Because the benchmark economy has about twice the reference amount of public capital,
    the public-capital part of the recipe is larger than one there, and so public capital
    makes firms more productive than they would be at the reference amount. The more
    productive public capital is, that is, the higher the public-capital elasticity, the
    more output the same machines and hours produce. If the public-capital elasticity is
    above zero and public capital fell all the way to nothing, nothing at all could be
    produced; so whenever public capital is productive, the government must keep public
    investment above zero.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"\max_{K_t,\,N_t}\ \ K_t^{\theta_K}\left(\frac{K^G_t}{\bar K^G}\right)^{\theta_G}N_t^{\theta_N} - w_t N_t - R_t K_t")
        st.markdown(
            r"The first term is output $Y_t$, with $\theta_K+\theta_N=1$. $\bar K^G$ is a fixed reference amount of public "
            r"capital, about half of the benchmark's, so $K^G/\bar K^G>1$ in the benchmark. "
            r"$\theta_G=\partial\ln Y_t/\partial\ln K^G_t$, and $Y_t=0$ when $K^G_t=0$ and "
            r"$\theta_G>0$, so then $I^G_t>0$ is required."
        )
    st.markdown(
        """
    **What the firms do.** The firm keeps hiring until the last hour adds exactly as much
    to output as it costs. That last hour's contribution turns out to be the labor
    elasticity times output per hour worked, so the wage equals the labor elasticity times
    output divided by hours. In the same way, the firm keeps renting until the last machine
    adds exactly its rent, so the rental rate equals the private-capital elasticity times
    output divided by the stock of private capital.

    These two rules have a tidy consequence. The wage bill is the labor elasticity's share
    of output, and the rental bill is the private-capital elasticity's share of output.
    Since the two shares add up to one, the wage bill and the rental bill together use up
    all of output, and the firm earns no profit at all. More public capital raises output
    for the same hours and machines, and so it raises both the wage and the rental rate.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"w_t = \theta_N\,\frac{Y_t}{N_t}, \qquad R_t = \theta_K\,\frac{Y_t}{K_t}")
        st.latex(r"\Longrightarrow\quad w_tN_t + R_tK_t = (\theta_N+\theta_K)\,Y_t = Y_t")

    st.subheader("A.3 Optimizing households (share 1 − λ)")
    st.markdown(
        """
    **What the optimizers want.** The optimizers make up the share of the population that
    is not hand-to-mouth. They own every machine on the island, so every unit of rent goes
    to them. Each year they choose how much to consume, how many hours to work, and how
    much to invest, and they do so to make the happiness of their whole, never-ending
    future as large as possible.

    Their happiness in any one year has two parts. The first part comes from what we may
    call effective consumption: the goods they buy for themselves, plus in-kind welfare
    counted at its in-kind value, so that one unit of in-kind welfare counts as that many
    units of their own goods. Happiness grows with the logarithm of effective consumption,
    which means that each one-percent rise in effective consumption adds the same amount of
    happiness, whether they are rich or poor. The second part comes from leisure: the
    logarithm of leisure, multiplied by the leisure weight. To value their whole future,
    the optimizers add up the happiness of every year from now on, but they shrink next
    year's happiness by the discount factor, the year after that by the discount factor
    twice over, and so on forever.

    Each year the optimizers face a budget. What they spend goes to two uses: consumption,
    where every unit costs one plus the consumption-tax rate, and investment in new
    machines. What they have to spend is their labor income, the wage times their hours,
    plus their capital income, the rental rate times their machines; the government takes
    the income-tax rate's fraction of this total, and then takes the lump-sum tax on top.
    Spending must equal what is left. They receive no cash welfare, unless the
    hand-to-mouth share is zero: in that case the optimizers are the whole population, cash
    welfare is paid to them and added to what they have to spend, and the island becomes
    the single representative household of Baxter and King.

    Their machines also change from year to year. Next year's stock of machines equals
    this year's stock, minus the depreciation rate's fraction that wears out, plus this
    year's investment.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"\max\ \sum_{t=0}^{\infty} \beta^t \Big[\ln\big(C^O_t + \phi\,G^C_t\big) + \theta_L \ln\big(1-N^O_t\big)\Big]")
        st.markdown(r"subject to")
        st.latex(r"(1+\tau^C_t)\,C^O_t + I^O_t = (1-\tau^Y_t)\big(w_t N^O_t + R_t K^O_t\big) - T_t")
        st.latex(r"K^O_{t+1} = (1-\delta)\,K^O_t + I^O_t")
        st.markdown(r"When $\lambda=0$, the term $+\,G^T_t$ is added to the right-hand side of the budget.")
    st.markdown(
        """
    **What the optimizers do.** Two rules describe their best choices.

    The first rule balances work against leisure. Take the leisure weight, multiply it by
    effective consumption, and divide by leisure. That is what one more hour of leisure is
    worth, measured in goods. It must equal what one more hour of work brings in goods: the
    wage, reduced by the income-tax fraction, and then divided by one plus the
    consumption-tax rate, since every good bought carries that tax. If an hour of leisure
    were worth more than this, they would work less; if it were worth less, they would work
    more.

    The second rule, the Euler equation, balances today against tomorrow. Giving up one
    unit of goods today means giving up one divided by one plus today's consumption-tax
    rate units of consumption, and each of those units is worth one divided by today's
    effective consumption in happiness. That unit of goods, invested instead, becomes a
    machine that next year earns the rental rate, of which the government keeps next
    year's income-tax fraction; the machine itself is still there, except for the
    depreciation-rate fraction that wore out. So the unit returns one, plus the after-tax
    rental rate, minus the depreciation rate. Those goods buy consumption at next year's
    price, one plus next year's consumption-tax rate, and each unit of consumption is worth
    one divided by next year's effective consumption. Finally the whole of next year's gain
    is shrunk by the discount factor. At the best plan, the happiness lost today equals the
    discounted happiness gained tomorrow exactly.

    These two rules show how each tax bends behavior. The income tax appears in both rules:
    it shrinks the reward for an hour of work and the reward for saving, and so it
    discourages both. The consumption tax lowers how many goods an hour of work can buy, so
    it discourages work much as a wage tax would. But when its rate is the same today and
    tomorrow, it raises the price of consumption equally in both years and cancels out of
    the Euler equation, so it does not discourage saving. The lump-sum tax appears in
    neither rule. It only makes the optimizers poorer, so they consume less and work more,
    but it does not distort the trade-offs they make at the margin.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.markdown(r"Labor-leisure:")
        st.latex(r"\theta_L\,\frac{C^O_t+\phi\,G^C_t}{1-N^O_t} = \frac{(1-\tau^Y_t)\,w_t}{1+\tau^C_t}")
        st.markdown(r"Euler equation:")
        st.latex(r"\frac{1}{(1+\tau^C_t)(C^O_t+\phi\,G^C_t)} = \beta\,\frac{1+(1-\tau^Y_{t+1})\,R_{t+1}-\delta}{(1+\tau^C_{t+1})(C^O_{t+1}+\phi\,G^C_{t+1})}")
        st.markdown(
            r"$\tau^Y$ enters both conditions; $\tau^C$ enters the first and cancels from the "
            r"second when $\tau^C_t=\tau^C_{t+1}$; $T_t$ enters neither."
        )

    st.subheader("A.4 Hand-to-mouth households (share λ)")
    st.markdown(
        """
    **What the hand-to-mouth households want.** The hand-to-mouth households own no
    machines and can neither save nor borrow, so whatever they receive this year they spend
    this year. Their happiness has the same two parts as the optimizers': the logarithm of
    effective consumption, their own goods plus in-kind welfare counted at its in-kind
    value, and the leisure weight times the logarithm of leisure. Each year they choose how
    much to consume and how many hours to work to make that year's happiness as large as
    possible.

    Their budget is simple. What they spend on consumption, at one plus the consumption-tax
    rate per unit, equals their wage times their hours, plus their share of the cash
    welfare, minus the lump-sum tax. All of the island's cash welfare goes to them, so each
    hand-to-mouth household receives total cash welfare divided by the hand-to-mouth share.
    They pay no income tax, but every good they buy carries the consumption tax. Nothing
    they do this year changes what they can do next year, so they solve their problem one
    year at a time.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"\max_{C^H_t,\,N^H_t}\ \ln\big(C^H_t + \phi\,G^C_t\big) + \theta_L \ln\big(1-N^H_t\big)")
        st.markdown(r"subject to")
        st.latex(r"(1+\tau^C_t)\,C^H_t = w_t N^H_t + \frac{G^T_t}{\lambda} - T_t")
    st.markdown(
        """
    **What the hand-to-mouth households do.** They follow the same work-versus-leisure rule
    as the optimizers, with one change: the income tax is missing. The leisure weight times
    their effective consumption, divided by their leisure, must equal the wage divided by
    one plus the consumption-tax rate.

    Putting this rule together with their budget tells us their hours directly. First add
    up their resources other than work: the cash welfare each one receives, minus the
    lump-sum tax, plus the in-kind welfare counted at its in-kind value and priced as if it
    were bought at the after-tax price, that is, multiplied by one plus the consumption-tax
    rate. Multiply these other resources by the leisure weight and subtract the result from
    the wage. Then divide by the wage times one plus the leisure weight. What comes out is
    the fraction of their time they work. If it comes out below zero, they do not work at
    all, and their hours are zero.

    The rule says that the richer they are without working, the less they work. More cash
    welfare or more in-kind welfare means fewer hours; a larger lump-sum tax makes them
    poorer and means more hours. Once their hours are known, their consumption follows from
    their budget.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"\theta_L\,\frac{C^H_t+\phi\,G^C_t}{1-N^H_t} = \frac{w_t}{1+\tau^C_t}")
        st.markdown("Substituting the budget constraint:")
        st.latex(r"N^H_t = \max\left\{0,\ \frac{w_t - \theta_L\left(\dfrac{G^T_t}{\lambda} - T_t + (1+\tau^C_t)\,\phi\,G^C_t\right)}{(1+\theta_L)\,w_t}\right\}")
        st.latex(r"C^H_t = \frac{w_t N^H_t + G^T_t/\lambda - T_t}{1+\tau^C_t}")

    st.subheader("A.5 Government")
    st.markdown(
        """
    **What the government does.** The government sets its total spending as a share of
    output (sidebar item 4) and divides it into its three parts (item 5). Its total
    spending each year is exactly public investment plus in-kind welfare plus cash welfare.

    Public investment builds public capital the same way private investment builds
    machines. Next year's public capital equals this year's public capital, minus the
    depreciation-rate fraction that wears out, plus this year's public investment.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"G_t = I^G_t + G^C_t + G^T_t, \qquad K^G_{t+1} = (1-\delta)\,K^G_t + I^G_t")
    st.markdown(
        """
    **How the government pays.** The government cannot borrow, so every year the three
    taxes together must pay for all of its spending. Sidebar item 1 fixes the share of
    spending that each tax pays for, and the three shares add up to one. The tax rates then
    adjust each year so that each tax raises exactly its share.

    The lump-sum tax is paid by everyone in the same amount. Since the population is one,
    the lump-sum tax each person pays equals the lump-sum share times total spending.

    The income tax falls only on the optimizers. The income-tax rate, times the optimizers'
    population share, times each optimizer's income from wages and rents, must equal the
    income-tax share times total spending.

    Both kinds of household pay the consumption tax. The consumption-tax rate times the
    island's total private consumption must equal the consumption-tax share times total
    spending.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.markdown(r"With $f^{LS}+f^Y+f^C=1$:")
        st.latex(r"T_t = f^{LS}\,G_t, \qquad \tau^C_t\,C_t = f^C G_t")
        st.latex(r"\tau^Y_t\,(1-\lambda)\big(w_t N^O_t + R_t K^O_t\big) = f^Y G_t")

    st.subheader("A.6 Market clearing and equilibrium")
    st.markdown(
        """
    **Adding up the households.** The island's totals come from weighting each kind of
    household by its share of the population. Total private consumption is the
    hand-to-mouth share times what each hand-to-mouth household consumes, plus the
    optimizers' share times what each optimizer consumes. Total hours are built the same
    way from the two kinds' hours. Since only optimizers own machines, the island's private
    capital is the optimizers' share times each optimizer's machines, and the island's
    private investment is the optimizers' share times each optimizer's investment.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"C_t = \lambda\,C^H_t + (1-\lambda)\,C^O_t, \qquad N_t = \lambda\,N^H_t + (1-\lambda)\,N^O_t")
        st.latex(r"K_t = (1-\lambda)\,K^O_t, \qquad I_t = (1-\lambda)\,I^O_t")
    st.markdown(
        """
    **The goods market.** When we add together the two households' budgets and the
    government's budget, the taxes paid by one side cancel the taxes received by the other,
    and the wages and rents cancel the firm's costs. What remains is a statement about
    goods: everything the island makes each year is used in exactly four ways. Output
    equals private consumption, plus private investment, plus public investment, plus
    in-kind welfare.

    Cash welfare is absent from this list. It does not use up any goods; it only moves
    income from taxpayers to the people who receive it, and they then spend it, so it
    already shows up inside private consumption.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.latex(r"Y_t = C_t + I_t + I^G_t + G^C_t \qquad (G^T_t \text{ does not appear})")
    st.markdown(
        """
    **Equilibrium.** An equilibrium is a path through time of the wage, the rental rate,
    and every quantity, along which the firms follow their two hiring rules, the optimizers
    follow their two rules and their budget, the hand-to-mouth households follow their rule
    and their budget, the government follows its spending and tax rules, and the markets
    for labor, for capital, and for goods all clear in every year.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.markdown(
            r"An equilibrium is a sequence $\{w_t, R_t, C^O_t, C^H_t, N^O_t, N^H_t, I^O_t, "
            r"K^O_{t+1}, K^G_{t+1}, \tau^Y_t, \tau^C_t, T_t\}_{t\ge0}$ satisfying the "
            r"conditions of A.2–A.5, the aggregation identities, and the goods-market "
            r"condition, for every $t$."
        )

    st.subheader("A.7 Steady state and transition")
    st.markdown(
        """
    **The steady state.** Suppose the island has settled down so that nothing changes from
    one year to the next. Then today's and tomorrow's effective consumption are the same,
    the tax rates are the same, and the Euler equation becomes very simple: the rental
    rate, after the income tax has taken its fraction, must equal the real interest rate
    plus the depreciation rate. In words, the after-tax return on a machine must be just
    enough to pay the interest on the funds tied up in it and to replace the part that
    wears out.

    This rule fixes the rental rate, and through the firms' rental rule it fixes how many
    machines are used per hour of work. A higher income-tax rate means the before-tax
    rental rate must be higher to leave the same after-tax return, and a higher rental rate
    means fewer machines per worker. The consumption tax does not appear in this rule at
    all.

    Public capital also settles down. It stops changing when public investment just
    replaces what wears out, so in the long run public capital equals public investment
    divided by the depreciation rate.

    With the rental rate and the machines per hour known, the two work-versus-leisure rules
    and the households' budgets determine how many hours each kind of household works and
    how much each consumes. The leisure weight is chosen so that, before the policy
    changes, total hours worked are one fifth, twenty percent, of everyone's time, which is
    the target Baxter and King used.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.markdown(r"With all variables constant and $\beta=1/(1+r)$, the Euler equation becomes")
        st.latex(r"(1-\tau^Y)\,R = r + \delta")
        st.markdown(r"With $R=\theta_K Y/K$ and the production function, this fixes the capital-labor ratio:")
        st.latex(r"\frac{K}{N} = \left[\frac{\theta_K\,(K^G/\bar K^G)^{\theta_G}}{R}\right]^{1/\theta_N}")
        st.latex(r"K^G = \frac{I^G}{\delta}")
        st.markdown(r"and $\theta_L$ is set so that $N = 0.20$ before the policy change.")
    st.markdown(
        """
    **The transition.** When the government changes its policy, the island does not jump
    straight to its new steady state. Machines and public capital can change only slowly,
    year by year, through investment. Consumption, hours, the wage, and the rental rate, on
    the other hand, can jump at once. The app finds the one path along which every rule in
    this story holds in every single year and the island converges to its new steady
    state. That path is what the charts in panels 1 and 2 show.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.markdown(
            r"$K_0$ and $K^G_0$ are given by the old steady state; $C_t$, $N_t$, $w_t$, $R_t$ "
            r"may jump at $t=0$. The transition path satisfies every condition of A.2–A.6 for "
            r"all $t\ge0$ and converges to the new steady state as $t\to\infty$."
        )
    st.markdown(
        """
    **Special case.** Strip the island down. Make the hand-to-mouth share zero, so that
    everyone is an optimizer. Use only one tax, either all lump-sum or all income tax. And
    spend all government money on things households do not value: in-kind welfare with an
    in-kind value of zero, or public investment with a public-capital elasticity of zero.
    What is left is exactly the model of Baxter & King (1993).
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.markdown(
            r"Baxter & King (1993): $\lambda=0$; $f^{LS}=1$ or $f^Y=1$; spending on $G^C$ "
            r"with $\phi=0$, or on $I^G$ with $\theta_G=0$."
        )
    st.markdown(
        """
    **One result to explore.** Suppose there is no income tax, so the government uses only
    the lump-sum and consumption taxes. Then shifting cash welfare toward the hand-to-mouth
    households changes *who* does the island's work, but not *total* long-run output, as
    long as the hand-to-mouth households still work some hours. The reason is that, with
    logarithmic happiness, the two kinds of household's work-versus-leisure rules add up to
    a single rule for the whole island, and a transfer from one kind to the other does not
    change that combined rule. Once there is an income tax, which only the optimizers pay,
    the two rules no longer face the same after-tax wage, they no longer add up to one
    rule, and the result no longer holds.
    """
    )
    with st.container(border=True):
        st.markdown("**In symbols**")
        st.markdown(r"If $\tau^Y=0$ and $N^H>0$, both types $i=O,H$ satisfy")
        st.latex(r"\theta_L\,(C^i+\phi G^C) = \frac{w}{1+\tau^C}\,(1-N^i)")
        st.markdown(r"Weighting by $1-\lambda$ and $\lambda$ and adding gives one aggregate condition,")
        st.latex(r"\theta_L\,(C+\phi G^C) = \frac{w}{1+\tau^C}\,(1-N)")
        st.markdown(
            r"which does not depend on how income is split between the types. With "
            r"$\tau^Y>0$ the optimizers face $(1-\tau^Y)w$ instead of $w$, and the "
            r"conditions no longer aggregate."
        )
