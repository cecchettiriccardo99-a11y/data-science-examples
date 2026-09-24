# data-science-examples
Life Cycle Assesment evaluetad exercise 
"""
LCA Allocation Methods for Co-Production Systems
=================================================

This script demonstrates how allocation issues in Life Cycle Assessment (LCA)
can be solved using linear-algebra-based input-output modelling (the Leontief
inverse), applied to a combined heat and power (CHP) plant.

A CHP plant is a classic multi-output process: it burns natural gas to produce
both electricity and heat, and the resulting CO2 emissions must be allocated
between the two co-products. This script implements and compares two common
allocation approaches used in LCA and environmental accounting:

1. Substitution (system expansion): credit the CHP system for the heat/electricity
   it displaces from a reference (stand-alone) production process.
2. Partitioning: split emissions proportionally, either by energy content or by
   economic value (revenue) of the two outputs.

Core technique: for any linear production system described by a technology
matrix A (technical coefficients) and an environmental intervention matrix S
(emissions per unit output), the total emissions embodied in a final demand
vector y are:

    d = C @ S @ L @ y

where L = (I - A)^-1 is the Leontief inverse, giving the total (direct +
indirect) output required per unit of final demand, and C is a characterisation
matrix converting emissions into an impact category (e.g. Global Warming
Potential, GWP).

Note: all numerical values below are illustrative example data, not the
original assignment dataset.
"""

import numpy as np
import pandas as pd


def leontief_inverse(A: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Leontief inverse L = (I - A)^-1 of a technology matrix A.

    L gives the total (direct + indirect) output of every process required
    to deliver one unit of final demand of each product - the backbone of
    input-output-based LCA and economic input-output analysis.
    """
    I = np.eye(len(A))
    L = pd.DataFrame(np.linalg.inv(I - A.values), index=A.index, columns=A.columns)
    return L


def compute_impact(C: pd.DataFrame, S: pd.DataFrame, L: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total environmental impact for a given final demand vector y:
        impact = C @ S @ L @ y
    """
    impact = C @ S @ L @ y
    return impact.rename(columns={y.columns[0]: "Impact"})


def build_example_system(allocation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a simplified example technology matrix (A) and emissions matrix (S)
    for a CHP plant, under a given allocation method.

    allocation : one of "substitution_heat", "substitution_electricity",
                 "partition_energy", "partition_cost"
    """
    processes = ["CHP (kWh)", "Heat (kWh)", "Electricity (kWh)", "N.Gas (MJ)"]
    A = pd.DataFrame(0.0, index=processes, columns=processes)
    S = pd.DataFrame(0.0, index=["CO2 (kg)"], columns=processes)

    # Efficiencies: 1 kWh of gas input -> 0.31 kWh electricity + 0.56 kWh heat
    eff_elec, eff_heat = 0.31, 0.56
    direct_emission_factor = 0.230  # kg CO2 per kWh of useful energy output

    A.loc["N.Gas (MJ)", "CHP (kWh)"] = 3.6 / 0.87  # gas input per kWh CHP output
    S.loc["CO2 (kg)", "CHP (kWh)"] = direct_emission_factor

    if allocation == "substitution_heat":
        # Credit the CHP system for the heat it substitutes; residual burden -> electricity
        A.loc["CHP (kWh)", "Electricity (kWh)"] = 0.87 / eff_elec
        A.loc["Heat (kWh)", "Electricity (kWh)"] = -eff_heat / eff_elec
        S.loc["CO2 (kg)", "Heat (kWh)"] = 0.299  # emissions of the substituted heat source

    elif allocation == "substitution_electricity":
        A.loc["CHP (kWh)", "Heat (kWh)"] = 0.87 / eff_heat
        A.loc["Electricity (kWh)", "Heat (kWh)"] = -eff_elec / eff_heat
        S.loc["CO2 (kg)", "Electricity (kWh)"] = 0.0105  # substituted electricity source

    elif allocation == "partition_energy":
        A.loc["CHP (kWh)", "Electricity (kWh)"] = 1.0
        A.loc["CHP (kWh)", "Heat (kWh)"] = 1.0

    elif allocation == "partition_cost":
        revenue_elec, revenue_heat = eff_elec * 1.0, eff_heat * 0.7
        total_revenue = revenue_elec + revenue_heat
        A.loc["CHP (kWh)", "Electricity (kWh)"] = (revenue_elec / total_revenue) * (0.87 / eff_elec)
        A.loc["CHP (kWh)", "Heat (kWh)"] = (revenue_heat / total_revenue) * (0.87 / eff_heat)

    else:
        raise ValueError(f"Unknown allocation method: {allocation}")

    A = A.fillna(0.0)
    return A, S


if __name__ == "__main__":
    # Characterisation vector: 1 kg CO2 = 1 kg CO2-eq (GWP)
    C = pd.DataFrame([[1]], index=["GWP"], columns=["CO2 (kg)"])

    # Example final demand: 1000 kWh of electricity
    processes = ["CHP (kWh)", "Heat (kWh)", "Electricity (kWh)", "N.Gas (MJ)"]
    y = pd.DataFrame(0.0, index=processes, columns=["Final demand"])
    y.loc["Electricity (kWh)", "Final demand"] = 1000

    for method in ["substitution_heat", "substitution_electricity", "partition_energy", "partition_cost"]:
        A, S = build_example_system(method)
        L = leontief_inverse(A)
        impact = compute_impact(C, S, L, y)
        print(f"\nAllocation method: {method}")
        print(impact)
