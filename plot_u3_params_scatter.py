#!/usr/bin/env python3
"""Read data/u3_params.csv, group by gate_index and qubits, and 3D scatter theta/phi/lambda."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import ast
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/plots/u3_params_scatter.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv("data/u3_params.csv")

    if "diagram" not in df.columns:
        raise SystemExit("Missing 'diagram' column in CSV.")

    def _parse_partition(val):
        if isinstance(val, (list, tuple)):
            return list(val)
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            return None
        return None

    def _conjugate(partition):
        if not partition:
            return None
        max_len = max(partition)
        return [sum(1 for x in partition if x >= k) for k in range(1, max_len + 1)]

    def _is_self_conjugate(partition):
        if not partition:
            return False
        return partition == _conjugate(partition)

    df["partition"] = df["diagram"].apply(_parse_partition)
    df["self_conjugate"] = df["partition"].apply(_is_self_conjugate)

    grouped = (
        df.groupby(["diagram", "gate_index", "qubits", "self_conjugate"], as_index=False)
        .agg({"theta": "mean", "phi": "mean", "lambda": "mean"})
    )

    grouped["theta"] = grouped["theta"] % np.pi
    grouped["phi"] = grouped["phi"] % (2 * np.pi)
    grouped["lambda"] = grouped["lambda"] % (2 * np.pi)

    fig = px.scatter_3d(
        grouped,
        x="theta",
        y="phi",
        z="lambda",
        color="gate_index",
        hover_data=["gate_index", "diagram"],
        title="U3 parameters grouped by gate_index and qubits",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)
    fig.show()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
