#!/usr/bin/env python3
"""Evaluate aggregate errors without access to the private split key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROBLEMS = (
    "diffusion_reaction",
    "burgers",
    "advection",
    "poisson",
    "cavity_flow",
)


def summarize(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = target.reshape(len(target), -1)
    prediction = prediction.reshape(target.shape)
    per_sample = np.linalg.norm(target - prediction, axis=1) / np.linalg.norm(
        target, axis=1
    )
    return {
        "global_relative_l2_percent": float(
            np.linalg.norm(target - prediction) / np.linalg.norm(target) * 100
        ),
        "per_sample_mean_percent": float(per_sample.mean() * 100),
        "per_sample_std_percent": float(per_sample.std() * 100),
    }


def evaluate(problem: str, prediction_dir: Path) -> dict:
    with np.load(ROOT / "problems" / problem / "test_data.npz") as truth, np.load(
        prediction_dir / f"{problem}.npz"
    ) as prediction:
        if not np.array_equal(truth["sample_id"], prediction["sample_id"]):
            raise ValueError(f"sample_id mismatch for {problem}")
        if problem == "cavity_flow":
            return {
                component: summarize(truth[component], prediction[component])
                for component in ("velocity_u", "velocity_v")
            }
        return summarize(truth["solution"], prediction["prediction"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        problem: evaluate(problem, args.prediction_dir.resolve())
        for problem in PROBLEMS
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
