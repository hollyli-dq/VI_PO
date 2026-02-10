#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vpo.order import closure_edges, transitive_reduction
from vpo.po_dsl import build_maximal_paths_from_order


def chain_edges(length: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(length - 1)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress test: closure-size scaling vs maximal-path factoring")
    p.add_argument("--max-chain", type=int, default=120)
    p.add_argument("--step", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("m\t|O+|\t|O-|\tmax_paths\tnaive_terms\tfactored_terms")
    for m in range(10, args.max_chain + 1, args.step):
        order = chain_edges(m)
        closure = closure_edges(m, order)
        reduction = transitive_reduction(m, order)
        max_paths = build_maximal_paths_from_order(m, order)
        naive_terms = len(closure)
        factored_terms = len(max_paths)
        print(
            f"{m}\t{len(closure)}\t{len(reduction)}\t{len(max_paths)}\t"
            f"{naive_terms}\t{factored_terms}"
        )


if __name__ == "__main__":
    main()
