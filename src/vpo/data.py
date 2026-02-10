from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .hpo import HPOModel
from .poset import dominance_relation


@dataclass
class SyntheticTraceData:
    embeddings: np.ndarray
    relation: np.ndarray
    traces: list[list[int]]


def make_synthetic_trace_data(
    num_actions: int,
    latent_dim: int,
    num_traces: int,
    beta: float = 4.0,
    epsilon: float = 0.05,
    structured_order: bool = True,
    seed: int = 0,
) -> SyntheticTraceData:
    rng = np.random.default_rng(seed)
    if structured_order:
        # Construct near-monotonic embeddings so default demos have identifiable order.
        base = np.linspace(num_actions, 1, num_actions)[:, None]
        embeddings = np.repeat(base, latent_dim, axis=1) + rng.normal(scale=0.05, size=(num_actions, latent_dim))
    else:
        embeddings = rng.normal(size=(num_actions, latent_dim))
    relation = dominance_relation(embeddings)
    model = HPOModel(beta=beta, epsilon=epsilon)
    traces = [model.sample_trace(relation, rng) for _ in range(num_traces)]
    return SyntheticTraceData(embeddings=embeddings, relation=relation, traces=traces)


def relation_f1(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    pred_edges = set(zip(*np.nonzero(pred)))
    true_edges = set(zip(*np.nonzero(truth)))
    if not pred_edges and not true_edges:
        return 1.0, 1.0, 1.0
    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
