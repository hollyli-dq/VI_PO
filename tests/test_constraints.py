from __future__ import annotations

import numpy as np
import pytest

from vpo.order import closure_edges, maximal_paths, transitive_reduction
from vpo.po_dsl import build_maximal_paths_from_order, h_prime


def test_transitive_reduction_chain():
    m = 6
    chain = [(i, i + 1) for i in range(m - 1)]
    closure = closure_edges(m, chain)
    reduction = transitive_reduction(m, closure)
    assert len(closure) == m * (m - 1) // 2
    assert len(reduction) == m - 1


def test_maximal_paths_branching_graph():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    paths = maximal_paths(4, edges)
    assert sorted(paths) == [[0, 1, 3], [0, 2, 3]]


def test_chain_has_single_maximal_path():
    m = 10
    chain = [(i, i + 1) for i in range(m - 1)]
    paths = build_maximal_paths_from_order(m, chain)
    assert len(paths) == 1
    assert paths[0] == list(range(m))


def test_h_prime_zero_when_order_satisfied_by_dag():
    W = np.array(
        [
            [0.0, 0.8, 0.3],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ]
    )
    order = [(0, 1), (1, 2)]
    paths = build_maximal_paths_from_order(3, order)
    score = h_prime(W, paths)
    assert abs(score) < 1e-8


def test_maximal_paths_reject_cycle():
    with pytest.raises(ValueError):
        maximal_paths(3, [(0, 1), (1, 2), (2, 0)])
