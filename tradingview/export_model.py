"""Regenerate the win-probability constants embedded in atlas_alpha.pine.

The Pine indicator can't read data_store/prob_model.json, so the trained
logistic-regression coefficients are inlined in the script. After retraining
(`python -m engine.prob_model`) run this to rewrite that block in place:

    python -m tradingview.export_model
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config.settings import DATA_STORE

PINE = Path(__file__).with_name("atlas_alpha.pine")


def _arr(name: str, vals: list[float], indent: int) -> str:
    """Format a float list as a Pine `array.from(...)`, 4 values per line."""
    pad = " " * indent
    nums = [f"{v:.10f}" for v in vals]
    rows = [", ".join(nums[i:i + 4]) for i in range(0, len(nums), 4)]
    joined = (",\n" + pad + " " * (len(name) + 14)).join(rows)
    return f"{pad}{name} = array.from({joined})"


def main() -> None:
    model = json.loads((DATA_STORE / "prob_model.json").read_text())
    src = PINE.read_text()

    block = "\n".join([
        _arr("mean", model["mean"], 4),
        _arr("sd", model["std"], 4),
        _arr("w", model["weights"], 4),
        f'    z = {model["bias"]:.10f}',
    ])
    new, n = re.subn(r"    mean = array\.from\(.*?\n    z = -?[\d.]+",
                     block, src, count=1, flags=re.S)
    if not n:
        raise SystemExit("Could not find the constant block in atlas_alpha.pine")
    PINE.write_text(new)
    print(f"Updated {PINE.name} — {model['n_samples']} samples, "
          f"base_rate {model['base_rate']}, trained {model['trained_at'][:10]}")


if __name__ == "__main__":
    main()
