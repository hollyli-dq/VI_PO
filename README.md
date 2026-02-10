# Variational Partial Orders Project

This repository is a runnable scaffold for the paper **"Variational Partial Orders for Traces and Structure Learning"**.

## What is included

- `paper/main.tex`: your NeurIPS-style paper draft.
- `src/vpo/`: Python implementation skeleton for:
  - dominance-induced latent posets and frontier trace likelihood (HPO model),
  - importance-weighted score-function VI with VIMCO-style control variates,
  - partial-order constraints via transitive reduction + maximal-path factoring (PO-DSL).
- `scripts/run_hpo_synthetic.py`: synthetic HPO-VI demo.
- `scripts/run_po_dsl_stress.py`: chain-scaling stress test for naive closure penalties vs maximal paths.
- `tests/`: unit tests for key algorithmic pieces.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run demos

```bash
python scripts/run_hpo_synthetic.py --steps 400 --num-actions 8 --latent-dim 3
python scripts/run_po_dsl_stress.py --max-chain 120 --step 20
```

## Run tests

```bash
pytest -q
```

## Build paper PDF

`paper/main.tex` will use `neurips_2024.sty` if available; otherwise it falls back to plain article formatting so it can still compile on minimal TeX installs.

```bash
cd paper
latexmk -pdf main.tex
```

## Notes on scope

This is a strong baseline scaffold (not a final reproduction package). It prioritizes:

- clear mapping from equations to code,
- small, testable core modules,
- easy extension to full experiments and ablations.
