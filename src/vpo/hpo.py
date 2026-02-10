from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .poset import dominance_relation, frontier


Array = np.ndarray
UtilityFn = Callable[[int, int, Array], float]


def _logsumexp(values: Array) -> float:
    m = np.max(values)
    return float(m + np.log(np.sum(np.exp(values - m))))


@dataclass
class HPOModel:
    """Frontier-restricted Plackett-Luce trace likelihood with trembling-hand noise."""

    beta: float = 1.0
    epsilon: float = 0.05
    utility_fn: UtilityFn | None = None

    def utility(self, action: int, step: int, rel: Array) -> float:
        if self.utility_fn is not None:
            return float(self.utility_fn(action, step, rel))
        return 0.0

    def step_probability(self, action: int, prefix: Sequence[int], rel: Array) -> float:
        m = rel.shape[0]
        prefix_set = set(prefix)
        remaining = [a for a in range(m) if a not in prefix_set]
        if action not in remaining:
            return 0.0

        front = frontier(rel, remaining)
        tremble = self.epsilon / max(len(remaining), 1)

        if action not in front:
            return tremble

        logits = np.array([self.beta * self.utility(a, len(prefix), rel) for a in front], dtype=float)
        log_denom = _logsumexp(logits)
        probs = np.exp(logits - log_denom)
        action_idx = front.index(action)
        return (1.0 - self.epsilon) * float(probs[action_idx]) + tremble

    def trace_log_likelihood(self, trace: Sequence[int], rel: Array) -> float:
        logp = 0.0
        prefix: list[int] = []
        for action in trace:
            p = self.step_probability(action, prefix, rel)
            if p <= 0.0:
                return -np.inf
            logp += float(np.log(p))
            prefix.append(action)
        return logp

    def dataset_log_likelihood(self, traces: Sequence[Sequence[int]], rel: Array) -> float:
        return float(sum(self.trace_log_likelihood(t, rel) for t in traces))

    def sample_trace(self, rel: Array, rng: np.random.Generator) -> list[int]:
        m = rel.shape[0]
        trace: list[int] = []
        remaining = list(range(m))
        while remaining:
            front = frontier(rel, remaining)
            if rng.random() < self.epsilon:
                action = int(rng.choice(remaining))
            else:
                logits = np.array([self.beta * self.utility(a, len(trace), rel) for a in front], dtype=float)
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / probs.sum()
                action = int(rng.choice(front, p=probs))
            trace.append(action)
            remaining.remove(action)
        return trace


LOG_2PI = float(np.log(2.0 * np.pi))


def log_standard_normal(U: Array) -> float:
    return float(-0.5 * np.sum(U * U + LOG_2PI))


def log_joint(
    traces: Sequence[Sequence[int]],
    U: Array,
    beta: float = 1.0,
    epsilon: float = 0.05,
    utility_fn: UtilityFn | None = None,
) -> float:
    rel = dominance_relation(U)
    model = HPOModel(beta=beta, epsilon=epsilon, utility_fn=utility_fn)
    return log_standard_normal(U) + model.dataset_log_likelihood(traces, rel)
