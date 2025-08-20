---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# The Fisher Two-Period Optimal Consumption Problem

[![badge](https://img.shields.io/badge/Launch%20using%20-Econ--ARK-blue)](https://econ-ark.org/materials/fishertwoperiod#launch)

```python
# Some initial setup
import ipywidgets as widgets
from ipywidgets import interact, fixed
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")
palette = plt.get_cmap("Dark2")

def mystr(number):
    return "{:.3f}".format(number)
```

Irving Fisher first analyzed the optimization problem of a consumer who faces no uncertainty and lives for two periods. This notebook creates interactive widgets that let you explore this foundational problem and see how changes in interest rates affect consumption through income, substitution, and human wealth effects.

## The Consumer's Problem

In its most general form, the household's lifetime value function is:
$$
V(c_1, c_2)
$$
where $c_1$ is consumption in 'youth' and $c_2$ is consumption in 'old age'. We assume the consumer always prefers more consumption ($V_1, V_2 > 0$) but at a diminishing rate ($V_{11}, V_{22} < 0$).

The consumer begins period 1 with bank balances $b_1$ and income $y_1$. These resources are divided between consumption $c_1$ and end-of-period assets $a_1$:
$$
b_1 + y_1 = c_1 + a_1
$$

In period 2, assets earn interest at rate $R$ and the consumer receives income $y_2$:
$$
c_2 = a_1 R + y_2 = (b_1 + y_1 - c_1)R + y_2
$$

## The Optimal Choice

### Basic Plot: the optimal $(c_1,c_2)$ bundle.

The maximization problem is stated below:

$$
\max_{\{c_1, c_2\}} V(c_1, c_2)
$$
subject to:
$$
c_2 = (b_1 + y_1 - c_1)R + y_2
$$

To solve this optimization problem, we use the method of Lagrange multipliers. The first-order conditions yield the fundamental **Euler equation**:
$$
V_1(c_1) = R V_2(c_2)
$$

This equation says that the marginal utility of consumption today must equal the interest-adjusted marginal utility of consumption tomorrow. If this condition is violated, the consumer could increase total utility by shifting consumption between periods.

For **time-separable utility** with identical period utility functions, $V(c_1,c_2) = u(c_1) + \beta u(c_2)$, where $\beta$ is the time preference factor, the Euler equation becomes:
$$
u'(c_1) = R \beta u'(c_2)
$$

The interactive plots below allow you to explore how changes in interest rates, income, and other parameters affect the optimal consumption choice.

```python
# The first step in creating a widget is defining
# a function that will receive the user's input as
# parameters and generate the entity that we want
# to analyze: in this case, a figure of the optimmal
# consumption bundle given income, assets, and interest
# rates

def FisherPlot(Y_1, Y_2, B_1, R, C_1_Max, C_2_Max):
    # Basic setup of perfect foresight consumer

    # We first create an instance of the class
    # PerfForesightConsumerType, with its standard parameters.
    PFexample = PerfForesightConsumerType()

    PFexample.cycles = 1  # let the agent live the cycle of periods just once
    PFexample.T_cycle = 1  # Number of non-terminal periods in the cycle
    # No automatic growth in income across periods
    PFexample.PermGroFac = [1.0]
    PFexample.LivPrb = [1.0]  # No chance of dying before the second period
    PFexample.aNrmInitStd = 0.0
    PFexample.AgentCount = 1
    CRRA = PFexample.CRRA
    beta = PFexample.DiscFac

    # Set interest rate and bank balances from input.
    PFexample.Rfree = [R]
    PFexample.aNrmInitMean = B_1

    # Solve the model: this generates the optimal consumption function.

    # Try-except blocks "try" to execute the code in the try block. If an
    # error occurs, the except block is executed and the application does
    # not halt
    try:
        PFexample.solve()
    except:
        print("Those parameter values violate a condition required for solution!")

    # Create the figure
    C_1_Min = 0.0
    C_2_Min = 0.0
    plt.figure(figsize=(8, 8))
    plt.xlabel("Period 1 Consumption $C_1$")
    plt.ylabel("Period 2 Consumption $C_2$")
    plt.ylim([C_2_Min, C_2_Max])
    plt.xlim([C_1_Min, C_1_Max])

    # Plot the budget constraint
    C_1_bc = np.linspace(C_1_Min, B_1 + Y_1 + Y_2 / R, 10, endpoint=True)
    C_2_bc = (Y_1 + B_1 - C_1_bc) * R + Y_2
    plt.plot(C_1_bc, C_2_bc, "k-", label="Budget Constraint")

    # Plot the optimal consumption bundle
    C_1 = PFexample.solution[0].cFunc(B_1 + Y_1 + Y_2 / R)
    C_2 = PFexample.solution[1].cFunc((Y_1 + B_1 - C_1) * R + Y_2)
    plt.plot(C_1, C_2, "ro", label="Optimal Consumption")

    # Plot the indifference curve
    V = C_1 ** (1 - CRRA) / (1 - CRRA) + beta * C_2 ** (1 - CRRA) / (
        1 - CRRA
    )  # Get max utility
    C_1_V = np.linspace(((1 - CRRA) * V) ** (1 / (1 - CRRA)) + 0.5, C_1_Max, 1000)
    C_2_V = (((1 - CRRA) * V - C_1_V ** (1 - CRRA)) / beta) ** (1 / (1 - CRRA))
    plt.plot(C_1_V, C_2_V, "b-", label="Indiferrence Curve")

    # Add a legend and display the plot
    plt.legend()
    plt.show()

    return None
```

```python
# We now define the controls that will receive and transmit the
# user's input.
# These are sliders for which the range, step, display, and
# behavior are defined.

# Define a slider for the interest rate
Rfree_widget = widgets.FloatSlider(
    min=1.0,
    max=2.0,
    step=0.001,
    value=1.05,
    continuous_update=True,
    readout_format=".4f",
    description="$R$",
)

# Define a slider for Y_1
Y_1_widget = widgets.FloatSlider(
    min=0.0,
    max=100.0,
    step=0.1,
    value=50.0,
    continuous_update=True,
    readout_format="4f",
    description="$Y_1$",
)

# Define a slider for Y_2
Y_2_widget = widgets.FloatSlider(
    min=0.0,
    max=100.0,
    step=0.1,
    value=50.0,
    continuous_update=True,
    readout_format="4f",
    description="$Y_2$",
)

# Define a slider for B_1
B_1_widget = widgets.FloatSlider(
    min=0.0,
    max=4.0,
    step=0.01,
    value=2.0,
    continuous_update=True,
    readout_format="4f",
    description="$B_1$",
)

# Define a textbox for C_1 max
C_1_Max_widget = widgets.FloatText(
    value=120, step=1.0, description="$C_1$ max", disabled=False
)

# Define a textbox for C_2 max
C_2_Max_widget = widgets.FloatText(
    value=120, step=1.0, description="$C_2$ max", disabled=False
)
```

```python
# Make the widget
interact(
    FisherPlot,
    Y_1=Y_1_widget,
    Y_2=Y_2_widget,
    B_1=B_1_widget,
    R=Rfree_widget,
    C_1_Max=C_1_Max_widget,
    C_2_Max=C_2_Max_widget,
)
```

## CRRA Utility and the Analytical Solution

For most economic analysis, we assume **Constant Relative Risk Aversion (CRRA)** utility:
$$
u(c) = \frac{c^{1-\rho}}{1-\rho}
$$
with marginal utility $u'(c) = c^{-\rho}$, where $\rho$ is the coefficient of relative risk aversion.

With CRRA utility, the Euler equation becomes:
$$
c_1^{-\rho} = R\beta c_2^{-\rho}
$$
which gives us the optimal consumption growth rate:
$$
\frac{c_2}{c_1} = (R\beta)^{1/\rho}
$$

The parameter $1/\rho$ is the **intertemporal elasticity of substitution** - it measures how willing consumers are to substitute consumption across time periods in response to interest rate changes.

Using the budget constraint $c_1 + c_2/R = b_1 + y_1 + y_2/R$, we can solve for the **consumption function**:
$$
c_1 = \frac{b_1 + y_1 + y_2/R}{1 + R^{-1}(R\beta)^{1/\rho}}
$$

This shows that consumption depends on **total lifetime wealth** (the sum of current assets, current income, and the present value of future income).

### Interest Rate Effects: Income, Substitution, and Human Wealth

When interest rates change, three distinct effects operate simultaneously:

```python
# This follows the same process as the previous plot, but now the problem
# is solved at two different interest rates in order to illustrate their effect.

# Define a function that plots something given some bits
def FisherPlot1(Y_1, Y_2, B_1, RHi, RLo, C_1_Max, C_2_Max):
    # Basic setup of perfect foresight consumer
    PFexample = (
        PerfForesightConsumerType()
    )  # set up a consumer type and use default parameteres
    PFexample.cycles = 1  # let the agent live the cycle of periods just once
    PFexample.T_cycle = 1  # Number of non-terminal periods in the cycle
    # No automatic growth in income across periods
    PFexample.PermGroFac = [1.0]
    PFexample.LivPrb = [1.0]  # No chance of dying before the second period
    PFexample.aNrmInitStd = 0.0
    PFexample.AgentCount = 1
    CRRA = 2.0
    beta = PFexample.DiscFac

    # Set the parameters we enter
    PFexample.aNrmInitMean = B_1

    # Create two models, one for RfreeHigh and one for RfreeLow
    PFexampleRHi = deepcopy(PFexample)
    PFexampleRHi.Rfree = [RHi]
    PFexampleRLo = deepcopy(PFexample)
    PFexampleRLo.Rfree = [RLo]

    C_1_Min = 0.0
    C_2_Min = 0.0

    # Solve the model for RfreeHigh and RfreeLow
    try:
        PFexampleRHi.solve()
        PFexampleRLo.solve()
    except:
        print("Those parameter values violate a condition required for solution!")

    # Plot the chart
    plt.figure(figsize=(8, 8))
    plt.xlabel("Period 1 Consumption $C_1$")
    plt.ylabel("Period 2 Consumption $C_2$")
    plt.ylim([C_2_Min, C_2_Max])
    plt.xlim([C_1_Min, C_1_Max])

    # Plot the budget constraints
    C_1_bc_RLo = np.linspace(C_1_Min, B_1 + Y_1 + Y_2 / RLo, 10, endpoint=True)
    C_2_bc_RLo = (Y_1 + B_1 - C_1_bc_RLo) * RLo + Y_2
    plt.plot(C_1_bc_RLo, C_2_bc_RLo, "k-", label="Budget Constraint R Low")

    C_1_bc_RHi = np.linspace(C_1_Min, B_1 + Y_1 + Y_2 / RHi, 10, endpoint=True)
    C_2_bc_RHi = (Y_1 + B_1 - C_1_bc_RHi) * RHi + Y_2
    plt.plot(C_1_bc_RHi, C_2_bc_RHi, "k--", label="Budget Constraint R High")

    # The optimal consumption bundles
    C_1_opt_RLo = PFexampleRLo.solution[0].cFunc(B_1 + Y_1 + Y_2 / RLo)
    C_2_opt_RLo = PFexampleRLo.solution[1].cFunc((Y_1 + B_1 - C_1_opt_RLo) * RLo + Y_2)

    C_1_opt_RHi = PFexampleRHi.solution[0].cFunc(B_1 + Y_1 + Y_2 / RHi)
    C_2_opt_RHi = PFexampleRHi.solution[1].cFunc((Y_1 + B_1 - C_1_opt_RHi) * RHi + Y_2)

    # Plot the indifference curves
    V_RLo = C_1_opt_RLo ** (1 - CRRA) / (1 - CRRA) + beta * C_2_opt_RLo ** (1 - CRRA) / (
        1 - CRRA
    )  # Get max utility
    C_1_V_RLo = np.linspace(((1 - CRRA) * V_RLo) ** (1 / (1 - CRRA)) + 0.5, C_1_Max, 1000)
    C_2_V_RLo = (((1 - CRRA) * V_RLo - C_1_V_RLo ** (1 - CRRA)) / beta) ** (
        1 / (1 - CRRA)
    )
    plt.plot(C_1_V_RLo, C_2_V_RLo, "b-", label="Indiferrence Curve R Low")

    V_RHi = C_1_opt_RHi ** (1 - CRRA) / (1 - CRRA) + beta * C_2_opt_RHi ** (1 - CRRA) / (
        1 - CRRA
    )  # Get max utility
    C_1_V_RHi = np.linspace(((1 - CRRA) * V_RHi) ** (1 / (1 - CRRA)) + 0.5, C_1_Max, 1000)
    C_2_V_RHi = (((1 - CRRA) * V_RHi - C_1_V_RHi ** (1 - CRRA)) / beta) ** (
        1 / (1 - CRRA)
    )
    plt.plot(C_1_V_RHi, C_2_V_RHi, "b--", label="Indiferrence Curve R High")

    # The substitution effect
    C_1_Subs = (
        V_RHi * (1 - CRRA) / (1 + beta * (RLo * beta) ** ((1 - CRRA) / CRRA))
    ) ** (1 / (1 - CRRA))
    C_2_Subs = C_1_Subs * (RLo * beta) ** (1 / CRRA)

    C_1_bc_Subs = np.linspace(C_1_Min, B_1 + Y_1 + Y_2 / RHi, 10, endpoint=True)
    C_2_bc_Subs = (C_1_Subs + C_2_Subs / RLo - C_1_bc_Subs) * RLo + Y_2
    plt.plot(C_1_bc_Subs, C_2_bc_Subs, "k-")

    # Plot the points of interest
    plt.plot(C_1_opt_RLo, C_2_opt_RLo, "ro", label="A: Optimal Consumption R Low")
    plt.plot(
        C_1_Subs,
        C_2_Subs,
        "go",
        label="B: Income effect AB \n     Substitution effect BC ",
    )
    plt.plot(C_1_opt_RHi, C_2_opt_RHi, "mo", label="C: Optimal Consumption R High")

    plt.legend()
    plt.show()

    return None
```

```python
# Define some widgets to control the plot

# Define a slider for the high interest rate
RHi_widget1 = widgets.FloatSlider(
    min=1.0,
    max=4.0,
    step=0.01,
    value=2,
    continuous_update=True,
    readout_format=".2f",
    description="$R High$",
)

# Define a slider for the low interest rate
RLo_widget1 = widgets.FloatSlider(
    min=0.0,
    max=4.0,
    step=0.01,
    value=1.0,
    continuous_update=True,
    readout_format=".2f",
    description="$R Low$",
)

# Define a slider for Y_1
Y_1_widget1 = widgets.FloatSlider(
    min=0.0,
    max=100.0,
    step=0.1,
    value=100.0,
    continuous_update=True,
    readout_format=".1f",
    description="$Y_1$",
)

# Define a slider for Y_2
Y_2_widget1 = widgets.FloatSlider(
    min=0.0,
    max=100.0,
    step=0.1,
    value=0.0,
    continuous_update=True,
    readout_format=".1f",
    description="$Y_2$",
)

# Define a slider for B_1
B_1_widget1 = widgets.FloatSlider(
    min=0.0,
    max=4.0,
    step=0.01,
    value=0.0,
    continuous_update=True,
    readout_format=".1f",
    description="$B_1$",
)

# Define a textbox for C_1 max
c_1_Max_widget1 = widgets.FloatText(
    value=120, step=1.0, description="$C_1$ max", disabled=False
)

# Define a textbox for C_2 max
c_2_Max_widget1 = widgets.FloatText(
    value=120, step=1.0, description="$C_2$ max", disabled=False
)
```

```python
# Make the widget

interact(
    FisherPlot1,
    Y_1=Y_1_widget1,
    Y_2=fixed(0.0),
    B_1=B_1_widget1,
    RHi=RHi_widget1,
    RLo=RLo_widget1,
    C_1_Max=c_1_Max_widget1,
    C_2_Max=c_2_Max_widget1,
)
```

## Understanding Interest Rate Effects

The previous interactive plot showed how consumption choices change when interest rates rise for a consumer whose income is earned in the first period. The movement from the low-rate optimum (point A) to the high-rate optimum (point C) can be decomposed into two key effects:

1. **Income Effect (A$\to$B)**: Higher interest rates increase the return on savings, making the consumer effectively wealthier. This tends to increase consumption in both periods.

2. **Substitution Effect (B$\to$C)**: Higher interest rates make future consumption relatively cheaper compared to current consumption, encouraging the consumer to save more today and consume more tomorrow. This tends to decrease current consumption.

The **net effect** on current consumption depends on which effect dominates. For most parameter values, the substitution effect dominates, leading to lower current consumption when interest rates rise.

### The Human Wealth Effect

When income is earned in the future (as in the next interactive example), a third effect emerges:

3. **Human Wealth Effect**: Higher interest rates reduce the present discounted value of future income, making the consumer effectively poorer. This tends to decrease consumption in both periods.

The concept of **human wealth** - the present value of future labor income - is crucial:
$$
h_1 = y_1 + \frac{y_2}{R}
$$

For consumers who earn most of their income in the future, changes in interest rates primarily operate through this human wealth channel rather than through returns on financial assets.

### Third plot: interest rate shifts with lifetime income earned in second period

```python
# This follows the same process, but we now fix Y_1 at 0

# Define a function that plots something given some bits
def FisherPlot2(Y_1, Y_2, B_1, RHi, RLo, C_1_Max, C_2_Max):
    # Basic setup of perfect foresight consumer
    PFexample = (
        PerfForesightConsumerType()
    )  # set up a consumer type and use default parameteres
    PFexample.cycles = 1  # let the agent live the cycle of periods just once
    PFexample.T_cycle = 1  # Number of non-terminal periods in the cycle
    # No automatic growth in income across periods
    PFexample.PermGroFac = [1.0]
    PFexample.LivPrb = [1.0]  # No chance of dying before the second period
    PFexample.aNrmInitStd = 0.0
    PFexample.AgentCount = 1
    CRRA = 2.0
    beta = PFexample.DiscFac

    # Set the parameters we enter
    PFexample.aNrmInitMean = B_1

    # Create two models, one for RfreeHigh and one for RfreeLow
    PFexampleRHi = deepcopy(PFexample)
    PFexampleRHi.Rfree = [RHi]
    PFexampleRLo = deepcopy(PFexample)
    PFexampleRLo.Rfree = [RLo]

    C_1_Min = 0.0
    C_2_Min = 0.0

    # Solve the model for RfreeHigh
    try:
        PFexampleRHi.solve()
        PFexampleRLo.solve()
    except:
        print("Those parameter values violate a condition required for solution!")

    # Plot the chart
    plt.figure(figsize=(8, 8))
    plt.xlabel("Period 1 Consumption $C_1$")
    plt.ylabel("Period 2 Consumption $C_2$")
    plt.ylim([C_2_Min, C_2_Max])
    plt.xlim([C_1_Min, C_1_Max])

    # Plot the budget constraints
    C_1_bc_RLo = np.linspace(C_1_Min, B_1 + Y_1 + Y_2 / RLo, 10, endpoint=True)
    C_2_bc_RLo = (Y_1 + B_1 - C_1_bc_RLo) * RLo + Y_2
    plt.plot(C_1_bc_RLo, C_2_bc_RLo, "k-", label="Budget Constraint R Low")

    C_1_bc_RHi = np.linspace(C_1_Min, B_1 + Y_1 + Y_2 / RHi, 10, endpoint=True)
    C_2_bc_RHi = (Y_1 + B_1 - C_1_bc_RHi) * RHi + Y_2
    plt.plot(C_1_bc_RHi, C_2_bc_RHi, "k--", label="Budget Constraint R High")

    # The optimal consumption bundles
    C_1_opt_RLo = PFexampleRLo.solution[0].cFunc(B_1 + Y_1 + Y_2 / RLo)
    C_2_opt_RLo = PFexampleRLo.solution[1].cFunc((Y_1 + B_1 - C_1_opt_RLo) * RLo + Y_2)

    C_1_opt_RHi = PFexampleRHi.solution[0].cFunc(B_1 + Y_1 + Y_2 / RHi)
    C_2_opt_RHi = PFexampleRHi.solution[1].cFunc((Y_1 + B_1 - C_1_opt_RHi) * RHi + Y_2)

    # Plot the indifference curves
    V_RLo = C_1_opt_RLo ** (1 - CRRA) / (1 - CRRA) + beta * C_2_opt_RLo ** (1 - CRRA) / (
        1 - CRRA
    )  # Get max utility
    C_1_V_RLo = np.linspace(((1 - CRRA) * V_RLo) ** (1 / (1 - CRRA)) + 0.5, C_1_Max, 1000)
    C_2_V_RLo = (((1 - CRRA) * V_RLo - C_1_V_RLo ** (1 - CRRA)) / beta) ** (
        1 / (1 - CRRA)
    )
    plt.plot(C_1_V_RLo, C_2_V_RLo, "b-", label="Indiferrence Curve R Low")

    V_RHi = C_1_opt_RHi ** (1 - CRRA) / (1 - CRRA) + beta * C_2_opt_RHi ** (1 - CRRA) / (
        1 - CRRA
    )  # Get max utility
    C_1_V_RHi = np.linspace(((1 - CRRA) * V_RHi) ** (1 / (1 - CRRA)) + 0.5, C_1_Max, 1000)
    C_2_V_RHi = (((1 - CRRA) * V_RHi - C_1_V_RHi ** (1 - CRRA)) / beta) ** (
        1 / (1 - CRRA)
    )
    plt.plot(C_1_V_RHi, C_2_V_RHi, "b--", label="Indiferrence Curve R High")

    # The substitution effect
    C_1_Subs = (
        V_RHi * (1 - CRRA) / (1 + beta * (RLo * beta) ** ((1 - CRRA) / CRRA))
    ) ** (1 / (1 - CRRA))
    C_2_Subs = C_1_Subs * (RLo * beta) ** (1 / CRRA)

    # The Human wealth effect
    Y_2HW = Y_2 * RLo / RHi
    C_1HW = Y_2HW / (RLo + (RLo) ** (1 / CRRA))
    C_2HW = C_1HW * (RLo * beta) ** (1 / CRRA)

    C_1_bc_HW = np.linspace(C_1_Min, B_1 + Y_1 + Y_2HW / RLo, 10, endpoint=True)
    C_2_bc_HW = (Y_1 + B_1 - C_1_bc_HW) * RLo + Y_2HW
    plt.plot(C_1_bc_HW, C_2_bc_HW, "k:")

    VHW = (C_1HW ** (1 - CRRA)) / (1 - CRRA) + (beta * C_2HW ** (1 - CRRA)) / (1 - CRRA)
    C_1_V_HW = np.linspace(((1 - CRRA) * VHW) ** (1 / (1 - CRRA)) + 0.5, C_1_Max, 1000)
    C_2_V_HW = (((1 - CRRA) * VHW - C_1_V_HW ** (1 - CRRA)) / beta) ** (1 / (1 - CRRA))
    plt.plot(C_1_V_HW, C_2_V_HW, "b:")

    # Plot the points of interest
    plt.plot(C_1_opt_RLo, C_2_opt_RLo, "ro", label="A: Optimal Consumption R Low")
    plt.plot(
        C_1_Subs, C_2_Subs, "go", label="B: Income effect DB \n    Substitution effect BC"
    )
    plt.plot(C_1_opt_RHi, C_2_opt_RHi, "mo", label="C: Optimal Consumption R High")
    plt.plot(C_1HW, C_2HW, "co", label="D: HW effect AD")

    plt.legend()
    plt.show()

    return None
```

```python
# Define some widgets to control the plot

# Define a slider for the high interest rate
RHi_widget2 = widgets.FloatSlider(
    min=1.0,
    max=4.0,
    step=0.01,
    value=2,
    continuous_update=True,
    readout_format=".2f",
    description="$R High$",
)

# Define a slider for the low interest rate
RLo_widget2 = widgets.FloatSlider(
    min=0.0,
    max=4.0,
    step=0.01,
    value=1.0,
    continuous_update=True,
    readout_format=".2f",
    description="$R Low$",
)

# Define a slider for Y_1
Y_1_widget2 = widgets.FloatSlider(
    min=0.0,
    max=100.0,
    step=0.1,
    value=100.0,
    continuous_update=True,
    readout_format=".1f",
    description="$Y_1$",
)

# Define a slider for Y_2
Y_2_widget2 = widgets.FloatSlider(
    min=0.0,
    max=100.0,
    step=0.1,
    value=100.0,
    continuous_update=True,
    readout_format=".1f",
    description="$Y_2$",
)

# Define a slider for B_1
B_1_widget2 = widgets.FloatSlider(
    min=0.0,
    max=4.0,
    step=0.01,
    value=0.0,
    continuous_update=True,
    readout_format=".1f",
    description="$B_1$",
)

# Define a textbox for C_1 max
c_1_Max_widget2 = widgets.FloatText(
    value=120, step=1.0, description="$C_1$ max", disabled=False
)

# Define a textbox for C_2 max
c_2_Max_widget2 = widgets.FloatText(
    value=120, step=1.0, description="$C_2$ max", disabled=False
)
```

```python
# Make the widget

interact(
    FisherPlot2,
    Y_1=fixed(0.0),
    Y_2=Y_2_widget2,
    B_1=fixed(0.0),
    RHi=RHi_widget2,
    RLo=RLo_widget2,
    C_1_Max=c_1_Max_widget2,
    C_2_Max=c_2_Max_widget2,
)
```

# Fourth plot: the effects of changes in the coefficient of risk aversion

For this exercise, we assume that no income is received in the second period. The relevant parameter is therefore $M_1$, the total market resources before consumption in period 1.

```python
# Create plotting function
def FisherPlot3(M_1, R, beta, CRRA, C_1_Max, C_2_Max):
    # Basic setup of perfect foresight consumer

    # We first create an instance of the class
    # PerfForesightConsumerType, with its standard parameters.
    PFexample = PerfForesightConsumerType()

    PFexample.cycles = 1  # let the agent live the cycle of periods just once
    PFexample.T_cycle = 1  # One single non-terminal period
    PFexample.PermGroFac = [0]  # No income in the second period
    PFexample.LivPrb = [1.0]  # No chance of dying before the second period

    # Set interest rate and bank balances from input.
    PFexample.Rfree = R
    PFexample.CRRA = CRRA
    PFexample.DiscFac = beta

    # Solve the model: this generates the optimal consumption function.

    # Try-except blocks "try" to execute the code in the try block. If an
    # error occurs, the except block is executed and the application does
    # not halt
    try:
        PFexample.solve()
    except:
        print("Those parameter values violate a condition required for solution!")
        return

    # Create the figure
    C_1_Min = 0.0
    C_2_Min = 0.0
    plt.figure(figsize=(8, 8))
    plt.xlabel("Period 1 Consumption $C_1$")
    plt.ylabel("Period 2 Consumption $C_2$")
    plt.ylim([C_2_Min, C_2_Max])
    plt.xlim([C_1_Min, C_1_Max])

    # Plot the budget constraint
    C_1_bc = np.linspace(C_1_Min, M_1, 10, endpoint=True)
    C_2_bc = (M_1 - C_1_bc) * R
    plt.plot(C_1_bc, C_2_bc, "k-", label="Budget Constraint")

    # Plot the optimal consumption bundle
    C_1 = PFexample.solution[0].cFunc(M_1)
    C_2 = PFexample.solution[1].cFunc((M_1 - C_1) * R)
    plt.plot(C_1, C_2, "ro", label="Optimal Consumption")

    # Plot the indifference curve
    V = C_1 ** (1 - CRRA) / (1 - CRRA) + beta * C_2 ** (1 - CRRA) / (
        1 - CRRA
    )  # Get max utility
    C_1_V = np.linspace(((1 - CRRA) * V) ** (1 / (1 - CRRA)) + 0.1, C_1_Max, 1000)
    C_2_V = (((1 - CRRA) * V - C_1_V ** (1 - CRRA)) / beta) ** (1 / (1 - CRRA))
    plt.plot(C_1_V, C_2_V, "b-", label="Indiferrence Curve")

    # Add a legend and display the plot
    plt.legend()
    plt.show()

    return None
```

```python
# We now define the controls that will receive and transmit the
# user's input.
# These are sliders for which the range, step, display, and
# behavior are defined.

# Define a slider for the interest rate
Rfree_widget = widgets.FloatSlider(
    min=1.0,
    max=2.0,
    step=0.001,
    value=1.02,
    continuous_update=True,
    readout_format=".4f",
    description="$R$",
)

# Define a slider for the discount factor
Beta_widget = widgets.FloatSlider(
    min=0.5,
    max=0.9,
    step=0.001,
    value=0.9,
    continuous_update=True,
    readout_format=".4f",
    description="$\beta$",
)

# Define a slider for the interest rate
Beta_widget = widgets.FloatSlider(
    min=0.5,
    max=0.9,
    step=0.001,
    value=0.9,
    continuous_update=True,
    readout_format=".4f",
    description="$\\beta$",
)

# Define a slider for the CRRA
CRRA_widget = widgets.FloatSlider(
    min=0.9,
    max=2.0,
    step=0.1,
    value=2,
    continuous_update=True,
    readout_format=".4f",
    description="$\\rho$",
)

# Define a slider for M_1
M_1_widget = widgets.FloatSlider(
    min=0.1,
    max=10.0,
    step=0.1,
    value=50.0,
    continuous_update=True,
    readout_format="4f",
    description="$M_1$",
)

# Define a textbox for C_1 max
C_1_Max_widget = widgets.FloatText(
    value=20, step=1.0, description="$C_1$ max", disabled=False
)

# Define a textbox for C_2 max
C_2_Max_widget = widgets.FloatText(
    value=20, step=1.0, description="$C_2$ max", disabled=False
)

# Make the widget
interact(
    FisherPlot3,
    M_1=M_1_widget,
    R=Rfree_widget,
    beta=Beta_widget,
    CRRA=CRRA_widget,
    C_1_Max=C_1_Max_widget,
    C_2_Max=C_2_Max_widget,
)
```

## Key Insights: Fisherian Separation

One of the most surprising results from the Fisher two-period model is **Fisherian Separation**: The optimal consumption growth rate $c_2/c_1 = (R\beta)^{1/\rho}$ depends only on preferences ($\rho$, $\beta$) and market conditions ($R$), **not on the timing of income**.

This means that two consumers with the same total lifetime wealth will have identical consumption profiles, regardless of whether their wealth comes from:
- High income when young and low income when old
- Low income when young and high income when old  
- Any other income timing pattern

### Practical Implications

**Fisherian Separation** has profound implications for economic policy and theory:

1. **Tax Timing Neutrality**: In a frictionless world, it doesn't matter when you tax income - consumers will smooth consumption optimally

2. **Life Cycle Planning**: Young people with high future earning potential should borrow against that future income to smooth consumption

3. **Capital Markets**: Perfect capital markets allow optimal consumption smoothing regardless of income timing

### Limitations and Extensions

The Fisher model's assumptions of:
- Perfect foresight (no uncertainty)
- Perfect capital markets (no borrowing constraints)
- Time-separable utility

...are strong and often violated in reality. Modern consumption models extend this framework by adding:
- Income uncertainty $\to$ Buffer stock models
- Borrowing constraints $\to$ Liquidity constraints  
- Non-separable preferences $\to$ Habit formation models

These extensions, explored in subsequent notebooks, help explain observed deviations from the simple Fisher predictions while maintaining the fundamental insights about intertemporal optimization.

