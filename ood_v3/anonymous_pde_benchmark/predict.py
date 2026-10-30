#!/usr/bin/env python3
"""Restore a released DeepONet checkpoint and predict from an NPZ input file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "tensorflow.compat.v1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/deeponet-model-mpl")

import deepxde as dde
import numpy as np
from deepxde.backend import tf


ROOT = Path(__file__).resolve().parent
PROBLEMS = (
    "diffusion_reaction",
    "burgers",
    "advection",
    "poisson",
    "cavity_flow",
)


def gelu(value):
    return 0.5 * value * (
        1
        + tf.math.tanh(
            tf.math.sqrt(2 / np.pi) * (value + 0.044715 * value**3)
        )
    )


def build_model(problem: str, checkpoint: Path):
    """Build the original network, then restore its trained weights."""
    tf.reset_default_graph()
    branch = np.zeros((1, 101), dtype=np.float32)
    trunk = np.zeros((1, 2), dtype=np.float32)
    target = np.zeros((1, 1), dtype=np.float32)
    data = dde.data.Triple(
        X_train=(branch, trunk),
        y_train=target,
        X_test=(branch, trunk),
        y_test=target,
    )
    kwargs = {"regularization": ["l2", 1e-5]} if problem == "burgers" else {}
    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        **kwargs,
    )

    if problem == "diffusion_reaction":

        def output_transform(inputs, outputs):
            x, t = inputs[1][:, 0:1], inputs[1][:, 1:2]
            return 10 * x * (1 - x) * t * (outputs + 1)

        net.apply_output_transform(output_transform)
    elif problem == "advection":

        def output_transform(inputs, outputs):
            x, t = inputs[1][:, 0:1], inputs[1][:, 1:2]
            return 4 * x * t * outputs + tf.sin(np.pi * x) + tf.sin(0.5 * np.pi * t)

        net.apply_output_transform(output_transform)
    elif problem == "burgers":

        def periodic(value):
            x = value[:, 0:1] * 2 * np.pi
            return tf.concat(
                [
                    tf.math.cos(x),
                    tf.math.sin(x),
                    tf.math.cos(2 * x),
                    tf.math.sin(2 * x),
                    tf.math.cos(3 * x),
                    tf.math.sin(3 * x),
                    tf.math.cos(4 * x),
                    tf.math.sin(4 * x),
                    value[:, 1:2],
                ],
                axis=1,
            )

        net.apply_feature_transform(periodic)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.restore(str(checkpoint), verbose=0)
    return model


def load_inputs(path: Path, problem: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Read original training arrays or a custom branch/trunk NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"branch", "trunk"} <= keys:
            branch = data["branch"]
            trunk = data["trunk"]
        elif problem == "cavity_flow" and {"branch_train", "trunk_train"} <= keys:
            branch = data["branch_train"]
            trunk = data["trunk_train"]
        elif problem != "cavity_flow" and {"X_train0", "X_train1"} <= keys:
            branch = data["X_train0"]
            trunk = data["X_train1"]
        else:
            raise ValueError(f"Unsupported input fields in {path}: {sorted(keys)}")

    if branch.ndim != 2 or branch.shape[1] != 101 or len(branch) == 0:
        raise ValueError("branch must have shape (N, 101) with N > 0")
    if count:
        branch = branch[:count]
        if problem == "cavity_flow":
            trunk = trunk[:count]
        elif trunk.ndim == 3:
            trunk = trunk[:count]

    if (
        trunk.ndim not in (2, 3)
        or trunk.shape[-1] != 2
        or len(trunk) == 0
        or trunk.shape[-2] == 0
    ):
        raise ValueError("trunk must have shape (M, 2) or (N, M, 2)")
    if problem == "cavity_flow" and trunk.ndim == 2:
        if len(trunk) != len(branch):
            raise ValueError("cavity_flow requires one trunk row per branch row")
        trunk = trunk[:, None, :]
    elif trunk.ndim == 3 and len(trunk) != len(branch):
        raise ValueError("per-sample trunk coordinates must match branch rows")

    return np.asarray(branch, dtype=np.float32), np.asarray(trunk, dtype=np.float32)


def predict_values(model, branch: np.ndarray, trunk: np.ndarray, batch: int):
    points = trunk.shape[-2]
    result = np.empty((len(branch), points), dtype=np.float32)
    for start in range(0, len(branch), batch):
        stop = min(start + batch, len(branch))
        batch_branch = np.repeat(branch[start:stop], points, axis=0)
        if trunk.ndim == 2:
            batch_trunk = np.tile(trunk, (stop - start, 1))
        else:
            batch_trunk = trunk[start:stop].reshape(-1, 2)
        result[start:stop] = model.predict((batch_branch, batch_trunk)).reshape(
            stop - start, points
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=PROBLEMS)
    parser.add_argument(
        "--input", type=Path, help="NPZ input; defaults to the problem's training_data.npz"
    )
    parser.add_argument(
        "--count", type=int, default=1, help="first N branch rows (default: 1; 0: all)"
    )
    parser.add_argument("--batch-samples", type=int, help="branch rows per prediction batch")
    parser.add_argument("--output", type=Path, help="output NPZ file")
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must be nonnegative")
    if args.batch_samples is not None and args.batch_samples <= 0:
        parser.error("--batch-samples must be positive")

    problem_dir = ROOT / "problems" / args.problem
    source = args.input or problem_dir / "training_data.npz"
    branch, trunk = load_inputs(source, args.problem, args.count)
    batch = args.batch_samples or max(1, 16384 // trunk.shape[-2])
    output = args.output or ROOT / "predictions" / f"{args.problem}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.problem == "cavity_flow":
        model_u = build_model(args.problem, problem_dir / "checkpoint_u" / "model")
        velocity_u = predict_values(model_u, branch, trunk, batch)
        del model_u
        model_v = build_model(args.problem, problem_dir / "checkpoint_v" / "model")
        velocity_v = predict_values(model_v, branch, trunk, batch)
        np.savez_compressed(output, velocity_u=velocity_u, velocity_v=velocity_v)
    else:
        model = build_model(args.problem, problem_dir / "checkpoint" / "model")
        prediction = predict_values(model, branch, trunk, batch)
        np.savez_compressed(output, prediction=prediction)
    print(f"Saved {args.problem} predictions: {output}")


if __name__ == "__main__":
    main()
