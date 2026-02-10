from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


def dominance_relation(U: Array) -> Array:
    """Return relation matrix R where R[i, j] means i \\succ j under coordinate dominance."""
    if U.ndim != 2:
        raise ValueError(f"U must be 2D, got shape {U.shape}")
    rel = np.all(U[:, None, :] > U[None, :, :], axis=-1)
    np.fill_diagonal(rel, False)
    return rel


def is_irreflexive(rel: Array) -> bool:
    return not np.any(np.diag(rel))


def is_transitive(rel: Array) -> bool:
    rel_int = rel.astype(np.int32)
    two_hop = (rel_int @ rel_int) > 0
    violation = two_hop & ~rel
    np.fill_diagonal(violation, False)
    return not np.any(violation)


def is_strict_partial_order(rel: Array) -> bool:
    if rel.ndim != 2 or rel.shape[0] != rel.shape[1]:
        return False
    return is_irreflexive(rel) and is_transitive(rel)


def frontier(rel: Array, remaining: Sequence[int]) -> list[int]:
    """Return minimal elements of `remaining` under relation rel (i \\succ j)."""
    if len(remaining) == 0:
        return []
    idx = np.array(remaining, dtype=np.int64)
    sub_rel = rel[np.ix_(idx, idx)]
    has_predecessor = np.any(sub_rel, axis=0)
    return [remaining[i] for i in range(len(remaining)) if not has_predecessor[i]]


def relation_edges(rel: Array) -> list[tuple[int, int]]:
    """Convert relation matrix into a list of directed edges (i, j) for i \\succ j."""
    src, dst = np.nonzero(rel)
    return list(zip(src.tolist(), dst.tolist()))


def relation_from_edges(num_nodes: int, edges: Iterable[tuple[int, int]]) -> Array:
    rel = np.zeros((num_nodes, num_nodes), dtype=bool)
    for u, v in edges:
        rel[u, v] = True
    return rel
