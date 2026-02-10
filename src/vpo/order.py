from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


Edge = tuple[int, int]


def relation_matrix(num_nodes: int, edges: Iterable[Edge]) -> np.ndarray:
    rel = np.zeros((num_nodes, num_nodes), dtype=bool)
    for u, v in edges:
        rel[u, v] = True
    return rel


def transitive_closure(
    num_nodes: int, edges: Iterable[Edge], drop_diagonal: bool = True
) -> np.ndarray:
    closure = relation_matrix(num_nodes, edges)
    for k in range(num_nodes):
        closure = closure | (closure[:, [k]] & closure[[k], :])
    if drop_diagonal:
        np.fill_diagonal(closure, False)
    return closure


def closure_edges(num_nodes: int, edges: Iterable[Edge]) -> list[Edge]:
    cls = transitive_closure(num_nodes, edges)
    src, dst = np.nonzero(cls)
    return list(zip(src.tolist(), dst.tolist()))


def transitive_reduction(num_nodes: int, edges: Iterable[Edge]) -> list[Edge]:
    edges = list(dict.fromkeys(edges))
    cls = transitive_closure(num_nodes, edges)
    reduced: list[Edge] = []
    for u, v in edges:
        removable = False
        for w in range(num_nodes):
            if w == u or w == v:
                continue
            if cls[u, w] and cls[w, v]:
                removable = True
                break
        if not removable:
            reduced.append((u, v))
    reduced.sort()
    return reduced


def _ensure_dag(num_nodes: int, edges: Iterable[Edge]) -> None:
    cls = transitive_closure(num_nodes, edges, drop_diagonal=False)
    if np.any(np.diag(cls)):
        raise ValueError("Order edges contain a cycle; expected a DAG/poset relation")


def maximal_paths(num_nodes: int, edges: Iterable[Edge]) -> list[list[int]]:
    edges = list(dict.fromkeys(edges))
    _ensure_dag(num_nodes, edges)

    out_neighbors: dict[int, list[int]] = defaultdict(list)
    indeg = np.zeros(num_nodes, dtype=np.int32)
    outdeg = np.zeros(num_nodes, dtype=np.int32)
    for u, v in edges:
        out_neighbors[u].append(v)
        indeg[v] += 1
        outdeg[u] += 1

    starts = [i for i in range(num_nodes) if outdeg[i] > 0 and indeg[i] == 0]
    if not starts:
        return []

    paths: list[list[int]] = []

    def dfs(path: list[int]) -> None:
        u = path[-1]
        nxt = out_neighbors.get(u, [])
        if not nxt:
            paths.append(path.copy())
            return
        for v in nxt:
            path.append(v)
            dfs(path)
            path.pop()

    for s in starts:
        dfs([s])

    return paths


def path_to_edges(path: list[int]) -> list[Edge]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]
