"""
Mathematical ecological models: exponential growth, logistic growth, and Lotka–Volterra.

Provides small, dependency-light simulation helpers returning pandas DataFrames for easy plotting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Callable


@dataclass
class ExponentialParams:
    N0: float = 100.0   # initial population
    r: float = 0.2      # intrinsic growth rate per time unit
    t_max: float = 50.0
    dt: float = 0.1


@dataclass
class LogisticParams:
    N0: float = 100.0
    r: float = 0.2
    K: float = 1000.0   # carrying capacity
    t_max: float = 50.0
    dt: float = 0.1


@dataclass
class LotkaVolterraParams:
    prey0: float = 40.0
    pred0: float = 9.0
    alpha: float = 1.1   # prey growth rate
    beta: float = 0.4    # predation rate
    delta: float = 0.1   # predator reproduction rate per prey eaten
    gamma: float = 0.4   # predator mortality
    t_max: float = 50.0
    dt: float = 0.02


def _rk4_step(f: Callable[[np.ndarray, float], np.ndarray], y: np.ndarray, t: float, dt: float) -> np.ndarray:
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_exponential(params: ExponentialParams) -> pd.DataFrame:
    """dN/dt = r N"""
    N = params.N0
    t_values = np.arange(0.0, params.t_max + params.dt, params.dt)
    series = []
    for t in t_values:
        series.append((t, N))
        N = max(0.0, N + params.r * N * params.dt)
    return pd.DataFrame(series, columns=["time", "population"])  # type: ignore


def simulate_logistic(params: LogisticParams) -> pd.DataFrame:
    """dN/dt = r N (1 - N/K)"""
    N = params.N0
    t_values = np.arange(0.0, params.t_max + params.dt, params.dt)
    series = []
    for t in t_values:
        series.append((t, N))
        growth = params.r * N * (1.0 - N / max(1e-9, params.K))
        N = max(0.0, N + growth * params.dt)
    return pd.DataFrame(series, columns=["time", "population"])  # type: ignore


def simulate_lotka_volterra(params: LotkaVolterraParams) -> pd.DataFrame:
    """
    dX/dt = alpha X - beta X Y
    dY/dt = delta X Y - gamma Y
    where X=prey, Y=predator
    """
    def f(y: np.ndarray, _t: float) -> np.ndarray:
        X, Y = y
        dX = params.alpha * X - params.beta * X * Y
        dY = params.delta * X * Y - params.gamma * Y
        return np.array([dX, dY], dtype=float)

    y = np.array([params.prey0, params.pred0], dtype=float)
    t = 0.0
    series: list[Tuple[float, float, float]] = [(t, y[0], y[1])]
    while t < params.t_max:
        y = np.maximum(0.0, _rk4_step(f, y, t, params.dt))
        t += params.dt
        series.append((t, y[0], y[1]))
    return pd.DataFrame(series, columns=["time", "prey", "predator"])  # type: ignore
