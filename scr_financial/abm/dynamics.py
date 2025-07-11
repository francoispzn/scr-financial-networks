"""
Pluggable stochastic dynamics for bank regulatory ratios.

Provides OU, jump-diffusion, and regime-switching processes
that can be assigned to BankAgent instances.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RatioDynamics(ABC):
    """Base class for stochastic ratio evolution."""

    @abstractmethod
    def step(self, x: float, mu: float, rng: np.random.Generator, dt: float = 1.0,
             stress_level: float = 0.0) -> float:
        """Evolve ratio x by one time step.

        Args:
            x: Current ratio value.
            mu: Long-run mean (target level).
            rng: Random number generator.
            dt: Time step size.
            stress_level: System-wide stress indicator (0=calm, 1=crisis).

        Returns:
            Updated ratio value.
        """

    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        """Return current parameter values for serialization."""


class OUDynamics(RatioDynamics):
    """Standard Ornstein-Uhlenbeck process.

    dx = theta * (mu - x) * dt + sigma * sqrt(dt) * N(0,1)
    """

    def __init__(self, theta: float = 0.05, sigma: float = 0.25,
                 clamp_lo: float = 0.0, clamp_hi: float = 30.0):
        self.theta = theta
        self.sigma = sigma
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi

    def step(self, x, mu, rng, dt=1.0, stress_level=0.0):
        dx = self.theta * (mu - x) * dt + self.sigma * np.sqrt(dt) * rng.standard_normal()
        return float(np.clip(x + dx, self.clamp_lo, self.clamp_hi))

    def get_params(self):
        return {"theta": self.theta, "sigma": self.sigma,
                "clamp_lo": self.clamp_lo, "clamp_hi": self.clamp_hi}


class JumpDiffusionDynamics(RatioDynamics):
    """Merton-style jump-diffusion OU process.

    dx = theta * (mu - x) * dt + sigma * sqrt(dt) * N(0,1) + J * Poisson(lambda_j * dt)

    where J ~ N(jump_mean, jump_std) represents sudden shocks
    (e.g., credit events, regulatory interventions).
    """

    def __init__(self, theta: float = 0.05, sigma: float = 0.25,
                 lambda_j: float = 0.02, jump_mean: float = -2.0,
                 jump_std: float = 1.5,
                 clamp_lo: float = 0.0, clamp_hi: float = 30.0):
        self.theta = theta
        self.sigma = sigma
        self.lambda_j = lambda_j  # Jump intensity (expected jumps per unit time)
        self.jump_mean = jump_mean  # Negative = adverse jump (CET1 drops)
        self.jump_std = jump_std
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi

    def step(self, x, mu, rng, dt=1.0, stress_level=0.0):
        # OU diffusion component
        dx = self.theta * (mu - x) * dt + self.sigma * np.sqrt(dt) * rng.standard_normal()

        # Poisson jump component (stress-dependent intensity)
        effective_lambda = self.lambda_j * (1.0 + 2.0 * stress_level)
        n_jumps = rng.poisson(effective_lambda * dt)
        if n_jumps > 0:
            jump_size = sum(rng.normal(self.jump_mean, self.jump_std) for _ in range(n_jumps))
            dx += jump_size

        return float(np.clip(x + dx, self.clamp_lo, self.clamp_hi))

    def get_params(self):
        return {"theta": self.theta, "sigma": self.sigma,
                "lambda_j": self.lambda_j, "jump_mean": self.jump_mean,
                "jump_std": self.jump_std,
                "clamp_lo": self.clamp_lo, "clamp_hi": self.clamp_hi}


class RegimeSwitchingDynamics(RatioDynamics):
    """Two-regime Markov-switching OU process.

    Regime 0 (calm): low volatility, fast mean reversion
    Regime 1 (crisis): high volatility, slow mean reversion

    Transition probabilities:
        P(0->1) = p_calm_to_crisis (per time step)
        P(1->0) = p_crisis_to_calm (per time step)
    """

    def __init__(self,
                 theta_calm: float = 0.08, sigma_calm: float = 0.15,
                 theta_crisis: float = 0.02, sigma_crisis: float = 0.6,
                 p_calm_to_crisis: float = 0.01,
                 p_crisis_to_calm: float = 0.05,
                 clamp_lo: float = 0.0, clamp_hi: float = 30.0):
        self.theta = [theta_calm, theta_crisis]
        self.sigma = [sigma_calm, sigma_crisis]
        self.p_transition = [p_calm_to_crisis, p_crisis_to_calm]
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self.regime: int = 0  # Start in calm regime

    def step(self, x, mu, rng, dt=1.0, stress_level=0.0):
        # Regime transition (stress_level increases crisis probability)
        if self.regime == 0:
            p_switch = self.p_transition[0] * (1.0 + 3.0 * stress_level)
            if rng.random() < p_switch:
                self.regime = 1
        else:
            p_switch = self.p_transition[1] * (1.0 - 0.5 * stress_level)
            if rng.random() < max(0.001, p_switch):
                self.regime = 0

        theta = self.theta[self.regime]
        sigma = self.sigma[self.regime]
        dx = theta * (mu - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        return float(np.clip(x + dx, self.clamp_lo, self.clamp_hi))

    def get_params(self):
        return {"theta_calm": self.theta[0], "theta_crisis": self.theta[1],
                "sigma_calm": self.sigma[0], "sigma_crisis": self.sigma[1],
                "p_calm_to_crisis": self.p_transition[0],
                "p_crisis_to_calm": self.p_transition[1],
                "current_regime": self.regime,
                "clamp_lo": self.clamp_lo, "clamp_hi": self.clamp_hi}


class StateDependentOUDynamics(RatioDynamics):
    """OU with state-dependent mean reversion speed.

    Distressed banks (low CET1) revert more slowly — reflecting the empirical
    observation that capital rebuilding takes longer than capital depletion.

    theta_effective = theta_base * recovery_factor(x, threshold)
    where recovery_factor < 1 when x < threshold (stressed state)
    """

    def __init__(self, theta_base: float = 0.05, sigma: float = 0.25,
                 stress_threshold: float = 8.0, recovery_slowdown: float = 0.3,
                 clamp_lo: float = 0.0, clamp_hi: float = 30.0):
        self.theta_base = theta_base
        self.sigma = sigma
        self.stress_threshold = stress_threshold
        self.recovery_slowdown = recovery_slowdown  # Theta multiplier when stressed
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi

    def step(self, x, mu, rng, dt=1.0, stress_level=0.0):
        # State-dependent mean reversion
        if x < self.stress_threshold:
            # Below stress threshold: slow recovery
            gap = (self.stress_threshold - x) / self.stress_threshold
            theta = self.theta_base * (self.recovery_slowdown + (1 - self.recovery_slowdown) * (1 - gap))
        else:
            theta = self.theta_base

        # Additional system-wide stress effect
        theta *= (1.0 - 0.5 * stress_level)

        dx = theta * (mu - x) * dt + self.sigma * np.sqrt(dt) * rng.standard_normal()
        return float(np.clip(x + dx, self.clamp_lo, self.clamp_hi))

    def get_params(self):
        return {"theta_base": self.theta_base, "sigma": self.sigma,
                "stress_threshold": self.stress_threshold,
                "recovery_slowdown": self.recovery_slowdown,
                "clamp_lo": self.clamp_lo, "clamp_hi": self.clamp_hi}


# ── Default dynamics configurations ──────────────────────────────

DEFAULT_OU = {
    "CET1_ratio": OUDynamics(theta=0.05, sigma=0.25, clamp_lo=0.0, clamp_hi=30.0),
    "LCR": OUDynamics(theta=0.04, sigma=1.5, clamp_lo=20.0, clamp_hi=300.0),
    "NSFR": OUDynamics(theta=0.04, sigma=0.8, clamp_lo=30.0, clamp_hi=250.0),
}

DEFAULT_JUMP_DIFFUSION = {
    "CET1_ratio": JumpDiffusionDynamics(theta=0.05, sigma=0.25, lambda_j=0.02,
                                         jump_mean=-2.0, jump_std=1.5,
                                         clamp_lo=0.0, clamp_hi=30.0),
    "LCR": JumpDiffusionDynamics(theta=0.04, sigma=1.5, lambda_j=0.03,
                                  jump_mean=-15.0, jump_std=10.0,
                                  clamp_lo=20.0, clamp_hi=300.0),
    "NSFR": JumpDiffusionDynamics(theta=0.04, sigma=0.8, lambda_j=0.01,
                                   jump_mean=-5.0, jump_std=3.0,
                                   clamp_lo=30.0, clamp_hi=250.0),
}

DEFAULT_REGIME_SWITCHING = {
    "CET1_ratio": RegimeSwitchingDynamics(
        theta_calm=0.08, sigma_calm=0.15,
        theta_crisis=0.02, sigma_crisis=0.6,
        clamp_lo=0.0, clamp_hi=30.0),
    "LCR": RegimeSwitchingDynamics(
        theta_calm=0.06, sigma_calm=1.0,
        theta_crisis=0.01, sigma_crisis=4.0,
        clamp_lo=20.0, clamp_hi=300.0),
    "NSFR": RegimeSwitchingDynamics(
        theta_calm=0.06, sigma_calm=0.5,
        theta_crisis=0.015, sigma_crisis=2.0,
        clamp_lo=30.0, clamp_hi=250.0),
}

DEFAULT_STATE_DEPENDENT = {
    "CET1_ratio": StateDependentOUDynamics(theta_base=0.05, sigma=0.25,
                                            stress_threshold=8.0, recovery_slowdown=0.3,
                                            clamp_lo=0.0, clamp_hi=30.0),
    "LCR": StateDependentOUDynamics(theta_base=0.04, sigma=1.5,
                                     stress_threshold=100.0, recovery_slowdown=0.4,
                                     clamp_lo=20.0, clamp_hi=300.0),
    "NSFR": StateDependentOUDynamics(theta_base=0.04, sigma=0.8,
                                      stress_threshold=100.0, recovery_slowdown=0.4,
                                      clamp_lo=30.0, clamp_hi=250.0),
}
