#!/usr/bin/env python3
"""Run the bundled DeepONet checkpoints on the anonymous test sets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "tensorflow.compat.v1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/anonymous-pde-benchmark-mpl")

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
    kwargs = {}
    if problem == "burgers":
        kwargs["regularization"] = ["l2", 1e-5]
    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        **kwargs,
    )

    if problem == "diffusion_reaction":
        def output_transform(inputs, outputs):
            x_trunk = inputs[1]
            x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
            return 10 * x * (1 - x) * t * (outputs + 1)

        net.apply_output_transform(output_transform)
    elif problem == "advection":
        def output_transform(inputs, outputs):
            x_trunk = inputs[1]
            x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
            return (
                4 * x * t * outputs
                + tf.sin(np.pi * x)
                + tf.sin(0.5 * np.pi * t)
            )

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


def predict_values(
    model,
    branch: np.ndarray,
    trunk: np.ndarray,
    batch_samples: int,
) -> np.ndarray:
    samples = len(branch)
    points = trunk.shape[-2]
    prediction = np.empty((samples, points), dtype=np.float32)
    for start in range(0, samples, batch_samples):
        stop = min(start + batch_samples, samples)
        batch_branch = np.repeat(branch[start:stop], points, axis=0)
        if trunk.ndim == 2:
            batch_trunk = np.tile(trunk, (stop - start, 1))
        else:
            batch_trunk = trunk[start:stop].reshape(-1, 2)
        prediction[start:stop] = model.predict(
            (batch_branch, batch_trunk)
        ).reshape(stop - start, points)
    return prediction


def predict_problem(problem: str, output_dir: Path, batch_samples: int) -> Path:
    problem_dir = ROOT / "problems" / problem
    with np.load(problem_dir / "test_data.npz") as data:
        sample_id = data["sample_id"]
        branch = data["branch"]
        trunk = data["trunk"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{problem}.npz"
    if problem == "cavity_flow":
        model_u = build_model(problem, problem_dir / "checkpoint_u" / "model")
        velocity_u = predict_values(model_u, branch, trunk, batch_samples)
        del model_u
        model_v = build_model(problem, problem_dir / "checkpoint_v" / "model")
        velocity_v = predict_values(model_v, branch, trunk, batch_samples)
        np.savez_compressed(
            output,
            sample_id=sample_id,
            velocity_u=velocity_u,
            velocity_v=velocity_v,
        )
    else:
        model = build_model(problem, problem_dir / "checkpoint" / "model")
        prediction = predict_values(model, branch, trunk, batch_samples)
        np.savez_compressed(output, sample_id=sample_id, prediction=prediction)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=(*PROBLEMS, "all"), default="all")
    parser.add_argument("--batch-samples", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "predictions")
    args = parser.parse_args()
    if args.batch_samples <= 0:
        raise ValueError("--batch-samples must be positive")
    problems = PROBLEMS if args.problem == "all" else (args.problem,)
    for problem in problems:
        output = predict_problem(problem, args.output_dir.resolve(), args.batch_samples)
        print(f"Saved {problem}: {output}")


if __name__ == "__main__":
    main()

