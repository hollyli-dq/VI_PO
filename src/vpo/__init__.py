"""Variational Partial Orders (VPO) research scaffold."""

from .hpo import HPOModel, log_joint
from .order import maximal_paths, transitive_closure, transitive_reduction
from .po_dsl import acyclicity_expm, h_prime, path_penalty_naive
from .poset import dominance_relation, frontier, is_strict_partial_order

__all__ = [
    "HPOModel",
    "log_joint",
    "dominance_relation",
    "frontier",
    "is_strict_partial_order",
    "transitive_closure",
    "transitive_reduction",
    "maximal_paths",
    "acyclicity_expm",
    "h_prime",
    "path_penalty_naive",
]
