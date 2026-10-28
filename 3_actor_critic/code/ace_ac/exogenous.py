"""Exogenous paths of ACE-DICE (labor, TFP, carbon intensity, abatement costs,
backstop price and land-use emissions).

All paths are indexed by the period t = 0, 1, 2, ... and follow the GAMS code of
DICE 2016 and DICE 2023, as in the original implementation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ExogenousPaths:
    N: np.ndarray  # Labor (millions)
    A: np.ndarray  # Total factor productivity
    sigma: np.ndarray  # Carbon intensity of gross output (GtCO2 per trillion USD)
    theta_1: np.ndarray  # Abatement cost coefficient
    p_back: np.ndarray  # Backstop price (USD per tCO2)
    E_land: np.ndarray  # Land-use emissions (GtCO2 per year)


def _labor(p: dict, n: int) -> np.ndarray:
    labor = np.zeros(n)
    labor[0] = p["pop0"]
    for t in range(n - 1):
        labor[t + 1] = labor[t] * (p["popasym"] / labor[t]) ** p["popadj"]
    return labor


def _tfp(p: dict, n: int) -> np.ndarray:
    ga = p["ga0"] * np.exp(-p["dela"] * p["timestep"] * np.arange(n))
    tfp = np.zeros(n)
    tfp[0] = p["a0"]
    for t in range(n - 1):
        # Matches the original implementation, which uses ga(t+1) in step t
        tfp[t + 1] = tfp[t] / (1 - ga[t + 1])
    return tfp


def _land_emissions(p: dict, n: int) -> np.ndarray:
    return p["e_land0"] * (1 - p["de_land"]) ** np.arange(n)


def _dice_2016(p: dict, n: int):
    growth = np.zeros(n)
    sigma = np.zeros(n)
    growth[0] = p["g_0_sigma"]
    sigma[0] = p["E_0"] / (p["Y_0"] * (1 - p["mu_0"]))
    for t in range(1, n):
        growth[t] = growth[t - 1] * (1 + p["delta_sigma"]) ** p["timestep"]
        sigma[t] = sigma[t - 1] * np.exp(growth[t - 1] * p["timestep"])
    p_back = p["p_back"] * (1 - p["g_back"]) ** np.arange(n)
    theta_1 = p_back * sigma / p["theta_2"] / 1000
    return sigma, theta_1, p_back


def _dice_2023(p: dict, n: int):
    t = np.arange(1, n + 1, dtype=float)
    p_back = np.where(
        t > 7,
        p["pback2050"] * np.exp(-p["timestep"] * 0.001 * (t - 7)),
        p["pback2050"] * np.exp(-p["timestep"] * 0.01 * (t - 7)),
    )
    gsig = np.minimum(p["gsigma1"] * p["delgsig"] ** (t - 1), p["asymgsig"])
    sigma0 = p["E_1"] / (p["Y_1"] * (1 - p["mu_1"]))
    sigma = sigma0 * np.cumprod(np.concatenate([[1.0], np.exp(5 * gsig[:-1])]))
    emissrat = np.where(
        t <= 16,
        p["emissrat2020"] + (p["emissrat2100"] - p["emissrat2020"]) / 16 * (t - 1),
        p["emissrat2100"],
    )
    sigma_tot = sigma * emissrat
    theta_1 = p_back * sigma_tot / (p["theta_2"] * 1000)
    return sigma_tot, theta_1, p_back


def build_exogenous_paths(version: str, economy: dict, n: int) -> ExogenousPaths:
    if version == "2016":
        sigma, theta_1, p_back = _dice_2016(economy, n)
    elif version == "2023":
        sigma, theta_1, p_back = _dice_2023(economy, n)
    else:
        raise ValueError(f"Unknown DICE version: {version}")

    return ExogenousPaths(
        N=_labor(economy, n),
        A=_tfp(economy, n),
        sigma=sigma,
        theta_1=theta_1,
        p_back=p_back,
        E_land=_land_emissions(economy, n),
    )
