#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from vpo.data import make_synthetic_trace_data, relation_f1
from vpo.poset import dominance_relation
from vpo.vi import IWVIConfig, IWVITrainer, MeanFieldGaussian


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HPO-VI on synthetic traces")
    p.add_argument("--num-actions", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--num-traces", type=int, default=300)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--num-particles", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--unstructured", action="store_true", help="Use fully random embeddings instead of near-chain synthetic order")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = make_synthetic_trace_data(
        num_actions=args.num_actions,
        latent_dim=args.latent_dim,
        num_traces=args.num_traces,
        beta=args.beta,
        epsilon=args.epsilon,
        structured_order=not args.unstructured,
        seed=args.seed,
    )

    q = MeanFieldGaussian.initialize(
        num_actions=args.num_actions,
        latent_dim=args.latent_dim,
        seed=args.seed + 1,
    )

    cfg = IWVIConfig(
        num_particles=args.num_particles,
        alpha=args.alpha,
        lr=args.lr,
        beta=args.beta,
        epsilon=args.epsilon,
    )
    trainer = IWVITrainer(traces=data.traces, config=cfg)
    history = trainer.fit(q, num_steps=args.steps)

    pred_rel = dominance_relation(q.mu)
    precision, recall, f1 = relation_f1(pred_rel, data.relation)

    print("=== HPO-VI Synthetic Run ===")
    print(f"actions={args.num_actions} latent_dim={args.latent_dim} traces={args.num_traces}")
    print(f"steps={args.steps} particles={args.num_particles} alpha={args.alpha}")
    print(f"final objective={history[-1]:.4f}")
    print(f"relation precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    main()
