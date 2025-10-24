# TRM Integration for The Beer Game

This repository hosts the glue code that lets *The Beer Game* use the
[Tiny Recursive Model (TRM)](https://github.com/lucidrains/tiny-recursive-model)
as a supply-policy brain.  It provides:

- `TRMSupplyPolicy`: a drop-in policy that consumes Beer Game observations and
  emits order quantities.
- Encoding helpers that transform Beer Game node state into TRM-friendly
  integer sequences.
- A small factory for constructing TRM instances with sensible defaults for
  the Beer Game DAG.

The project is designed to live as a git submodule inside
`MilesAheadToo/The_Beer_Game` under `external/trm_integration`.

## Quick Start

```python
from trm_policy.dag_trm_policy import TRMSupplyPolicy

policy = TRMSupplyPolicy(
    checkpoint_path="checkpoints/trm-supply.pt",  # optional
    history_length=32,
)

order = policy.order(
    {
        "inventory": 12,
        "backlog": 3,
        "pipeline_on_order": 9,
        "last_incoming_order": 7,
        "base_stock": 18,
        "inventory_position": 18,
    }
)
print(order)
```

The policy keeps a rolling history of observations, encodes them to integer
tokens, and lets TRM predict the next demand signal.  The predicted demand is
blended with backlog and base-stock corrections to produce the order quantity.

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development Scripts

Run the linters/tests:

```
ruff check trm_policy
```
