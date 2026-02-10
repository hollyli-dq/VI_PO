from __future__ import annotations

import numpy as np

from vpo.hpo import HPOModel
from vpo.poset import dominance_relation, frontier, is_strict_partial_order, relation_from_edges


def test_dominance_is_strict_partial_order():
    rng = np.random.default_rng(42)
    U = rng.normal(size=(7, 4))
    rel = dominance_relation(U)
    assert is_strict_partial_order(rel)


def test_frontier_for_chain():
    rel = relation_from_edges(3, [(0, 1), (1, 2), (0, 2)])
    assert frontier(rel, [0, 1, 2]) == [0]
    assert frontier(rel, [1, 2]) == [1]
    assert frontier(rel, [2]) == [2]


def test_trace_likelihood_is_finite():
    rel = relation_from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    model = HPOModel(beta=2.0, epsilon=0.1)
    trace = [0, 1, 2, 3]
    ll = model.trace_log_likelihood(trace, rel)
    assert np.isfinite(ll)


def test_sample_trace_is_permutation():
    rel = relation_from_edges(5, [(0, 1), (0, 2), (1, 3), (2, 4)])
    model = HPOModel(beta=2.0, epsilon=0.05)
    rng = np.random.default_rng(0)
    trace = model.sample_trace(rel, rng)
    assert sorted(trace) == [0, 1, 2, 3, 4]
