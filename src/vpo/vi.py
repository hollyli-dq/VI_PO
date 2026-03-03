from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .hpo import log_joint


Array = np.ndarray
LOG_2PI = float(np.log(2.0 * np.pi))


def _logmeanexp(x: Array) -> float:
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x - m))))


def _log_softmax(x: Array) -> Array:
    """Numerically stable log-softmax."""
    m = np.max(x)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted)))


class GradientEstimator(str, Enum):
    VIMCO = "vimco"
    VIMCO_STAR = "vimco_star"


@dataclass
class MeanFieldGaussian:
    """Mean-field Gaussian q(U) over latent action embeddings U."""

    mu: Array
    log_sigma: Array
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @classmethod
    def initialize(
        cls,
        num_actions: int,
        latent_dim: int,
        init_scale: float = 0.1,
        seed: int = 0,
    ) -> "MeanFieldGaussian":
        rng = np.random.default_rng(seed)
        mu = rng.normal(scale=init_scale, size=(num_actions, latent_dim))
        log_sigma = np.full((num_actions, latent_dim), -0.5)
        return cls(mu=mu, log_sigma=log_sigma, rng=rng)

    @property
    def sigma(self) -> Array:
        return np.exp(self.log_sigma)

    def sample(self, num_particles: int) -> Array:
        eps = self.rng.normal(size=(num_particles,) + self.mu.shape)
        return self.mu[None, :, :] + self.sigma[None, :, :] * eps

    def log_prob(self, U: Array) -> Array:
        sigma2 = np.exp(2.0 * self.log_sigma)
        centered2 = (U - self.mu[None, :, :]) ** 2
        terms = centered2 / sigma2[None, :, :] + 2.0 * self.log_sigma[None, :, :] + LOG_2PI
        return -0.5 * np.sum(terms, axis=(1, 2))

    def grad_log_prob(self, U: Array) -> tuple[Array, Array]:
        """Gradient of log q(U) w.r.t. variational parameters (mu, log_sigma).

        Returns per-particle gradients with shape (N, m, K) for each parameter.
        """
        sigma2 = np.exp(2.0 * self.log_sigma)
        centered = U - self.mu[None, :, :]
        grad_mu = centered / sigma2[None, :, :]
        grad_log_sigma = centered**2 / sigma2[None, :, :] - 1.0
        return grad_mu, grad_log_sigma


@dataclass
class IWVIConfig:
    num_particles: int = 8
    alpha: float = 0.0
    lr: float = 1e-2
    beta: float = 4.0
    epsilon: float = 0.05
    grad_clip: float = 10.0
    estimator: GradientEstimator = GradientEstimator.VIMCO_STAR


class IWVITrainer:
    """Importance-weighted black-box VI with VIMCO/VIMCO-star gradient estimators.

    Implements the VR-IWAE objective (parameterized by alpha in [0,1)):

        L^(alpha)_N = (1/(1-alpha)) * E[log((1/N) sum_n w_n^(1-alpha))]

    with alpha=0 giving standard IWAE.

    Two gradient estimators are supported:
      - VIMCO: leave-one-out geometric-mean baseline (Mnih & Rezende 2016)
      - VIMCO_STAR: the simplified VIMCO-star estimator (Daudel et al. 2026)
        which uses per-sample signal -(W_n + log(1-W_n)) and achieves sqrt(N) SNR.
    """

    def __init__(self, traces: list[list[int]], config: IWVIConfig):
        self.traces = traces
        self.cfg = config
        if not (0.0 <= self.cfg.alpha < 1.0):
            raise ValueError("alpha must be in [0, 1)")
        self.baseline = 0.0

    def _vimco_signal(self, transformed_log_w: Array) -> tuple[float, Array]:
        """Original VIMCO leave-one-out baseline (Mnih & Rezende 2016)."""
        k = transformed_log_w.shape[0]
        base = _logmeanexp(transformed_log_w)
        if k == 1:
            signal = transformed_log_w - self.baseline
            return base, signal

        signals = np.zeros_like(transformed_log_w)
        for i in range(k):
            others = np.delete(transformed_log_w, i)
            replacement = float(np.mean(others))
            control = transformed_log_w.copy()
            control[i] = replacement
            control_base = _logmeanexp(control)
            signals[i] = base - control_base
        return base, signals

    def _vimco_star_signal(self, log_w: Array) -> tuple[float, Array]:
        r"""VIMCO-star estimator (Daudel et al. 2026, Eq. 15 specialized to alpha=0).

        For alpha=0 (IWAE), the per-sample signal is:
            signal_n = -(W_n + log(1 - W_n))
        where W_n = w_n / sum_j w_j (self-normalized importance weights).

        The gradient estimator is then:
            nabla L = sum_n signal_n * nabla_phi log q(z_n)
        """
        log_W = _log_softmax(log_w)
        W = np.exp(log_W)

        W_clipped = np.clip(W, 0.0, 1.0 - 1e-12)
        signals = -(W_clipped + np.log(1.0 - W_clipped))

        objective = _logmeanexp(log_w)
        return objective, signals

    def objective_and_grad(self, q: MeanFieldGaussian) -> tuple[float, Array, Array]:
        U_particles = q.sample(self.cfg.num_particles)
        log_q = q.log_prob(U_particles)

        log_p = np.array(
            [
                log_joint(self.traces, U_particles[n], beta=self.cfg.beta, epsilon=self.cfg.epsilon)
                for n in range(self.cfg.num_particles)
            ],
            dtype=float,
        )

        log_w = log_p - log_q

        if self.cfg.estimator == GradientEstimator.VIMCO_STAR and self.cfg.alpha == 0.0:
            objective, signals = self._vimco_star_signal(log_w)

            grad_logq_mu, grad_logq_log_sigma = q.grad_log_prob(U_particles)

            weighted = signals[:, None, None]
            grad_mu = np.sum(weighted * grad_logq_mu, axis=0)
            grad_log_sigma = np.sum(weighted * grad_logq_log_sigma, axis=0)
        else:
            transformed = (1.0 - self.cfg.alpha) * log_w
            base, signals = self._vimco_signal(transformed)
            objective = base / (1.0 - self.cfg.alpha)

            if self.cfg.num_particles > 1:
                signals = signals + (transformed - np.mean(transformed))

            grad_logq_mu, grad_logq_log_sigma = q.grad_log_prob(U_particles)

            scale = 1.0 / (1.0 - self.cfg.alpha)
            weighted = (signals * scale)[:, None, None]
            grad_mu = np.mean(weighted * grad_logq_mu, axis=0)
            grad_log_sigma = np.mean(weighted * grad_logq_log_sigma, axis=0)

        grad_mu = np.clip(grad_mu, -self.cfg.grad_clip, self.cfg.grad_clip)
        grad_log_sigma = np.clip(grad_log_sigma, -self.cfg.grad_clip, self.cfg.grad_clip)
        return objective, grad_mu, grad_log_sigma

    def fit(self, q: MeanFieldGaussian, num_steps: int = 500) -> list[float]:
        history: list[float] = []
        for _ in range(num_steps):
            objective, grad_mu, grad_log_sigma = self.objective_and_grad(q)
            q.mu += self.cfg.lr * grad_mu
            q.log_sigma += self.cfg.lr * grad_log_sigma
            q.log_sigma = np.clip(q.log_sigma, -5.0, 2.0)
            self.baseline = 0.9 * self.baseline + 0.1 * ((1.0 - self.cfg.alpha) * objective)
            history.append(objective)
        return history
