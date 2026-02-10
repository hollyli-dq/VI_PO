from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import expm

from .order import Edge, closure_edges, maximal_paths, path_to_edges, transitive_reduction


Array = np.ndarray


def acyclicity_expm(W: Array) -> float:
    """NOTEARS-style smooth acyclicity function h(W) = tr(exp(W∘W)) - d."""
    d = W.shape[0]
    return float(np.trace(expm(W * W)) - d)


def path_penalty_naive(W: Array, closure_order_edges: list[Edge], max_power: int | None = None) -> float:
    """Naive O(|O^+|) penalty based on all transitive-closure pairs."""
    d = W.shape[0]
    power = d if max_power is None else max_power
    A = W * W
    S = np.zeros_like(W)
    Ak = A.copy()
    for _ in range(power):
        S += Ak
        Ak = Ak @ A

    penalty = 0.0
    for i, j in closure_order_edges:
        penalty += float(S[j, i])
    return penalty


def augment_matrix(W: Array, path_edges: list[Edge], tau: float = 1.0) -> Array:
    Wo = np.zeros_like(W)
    for i, j in path_edges:
        Wo[i, j] = 1.0
    return W + tau * Wo - W * Wo


def h_prime(
    W: Array,
    maximal_order_paths: list[list[int]],
    tau: float = 1.0,
    h_fn: Callable[[Array], float] = acyclicity_expm,
) -> float:
    score = 0.0
    for path in maximal_order_paths:
        A = augment_matrix(W, path_to_edges(path), tau=tau)
        score += h_fn(A)
    return score


def build_maximal_paths_from_order(num_nodes: int, order_edges: list[Edge]) -> list[list[int]]:
    reduced = transitive_reduction(num_nodes, order_edges)
    return maximal_paths(num_nodes, reduced)


def order_stats(num_nodes: int, order_edges: list[Edge]) -> dict[str, int]:
    closure = closure_edges(num_nodes, order_edges)
    reduction = transitive_reduction(num_nodes, order_edges)
    paths = maximal_paths(num_nodes, reduction)
    return {
        "num_order_edges": len(order_edges),
        "num_closure_edges": len(closure),
        "num_reduction_edges": len(reduction),
        "num_maximal_paths": len(paths),
    }
