#!/usr/bin/env python3
"""OOD / error-magnitude score for the released DeepONet models.

For every test case the policy returns ``ood_score`` = an estimate of the released model's RMSE on
that case (absolute error, in output units).  Larger means the model is expected to be further off.
It combines three label-free signals (see README.md):

  * ensemble disagreement  - RMS gap between the released model and an ensemble of independently
                             trained replica DeepONets (same architecture, same training arrays);
  * physics consistency    - PDE residual and boundary/initial-condition mismatch of the released
                             model's own prediction (PDEs identified from the training arrays);
  * input novelty          - distance of the branch input to the training inputs (PCA Mahalanobis,
                             PCA reconstruction residual, k-NN in input and in branch-embedding space).

Input NPZ format is the one accepted by ``predict.py``: ``branch`` (N,101) and ``trunk``
((M,2) shared, (N,M,2) per sample, or (N,2) paired rows for cavity_flow).  A test case is one branch
row together with its trunk points; for cavity_flow paired rows each row is its own test case.

Usage (from the bundle root; works in the `mindquan` env (PyTorch, GPU if available) or in the
bundled `.conda/repro` env (JAX on CPU); set OOD_BACKEND=torch|jax to force one):
    conda activate mindquan
    python ood_policy/ood_score.py --problem burgers --input test.npz --output burgers_ood.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_BACKEND = os.environ.get("OOD_BACKEND", "").lower()
if _BACKEND != "jax":
    try:  # PyTorch backend (e.g. the `mindquan` env, GPU when available)
        import deeponet_torch as dj  # noqa: E402
    except ImportError:
        if _BACKEND == "torch":
            raise
        import deeponet_jax as dj  # noqa: E402
else:  # JAX backend (e.g. the bundled `.conda/repro` env)
    import deeponet_jax as dj  # noqa: E402

ROOT = HERE.parent
PROBLEMS = ("diffusion_reaction", "burgers", "advection", "poisson", "cavity_flow")


# --------------------------------------------------------------------------- novelty
class Novelty:
    """Distances of branch inputs to the training inputs, normalised so that ~1 is typical in-distribution."""

    def __init__(self, ref: dict):
        self.r = ref

    def __call__(self, G: np.ndarray, w_released: dict) -> dict:
        r = self.r
        D = G - r["mu"]
        z = (D @ r["V"].T) / np.sqrt(r["ev"])
        res = np.linalg.norm(D - (D @ r["V"].T) @ r["V"], axis=1)
        out = {"maha": np.sqrt((z ** 2).sum(1) / len(r["ev"])), "pca_res": res / r["res_scale"]}
        out["knn"] = _knn(G, r["G"], 5) / r["nn_scale"]
        h = np.maximum(G @ w_released["W0"] + w_released["b0"], 0)
        h = np.maximum(h @ w_released["W1"] + w_released["b1"], 0)
        L = h @ w_released["W2"] + w_released["b2"]
        out["knn_lat"] = _knn(L, r["L"], 5) / r["lat_scale"]
        Dh = h - r["h_mu"]
        out["maha_hid"] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Dh, r["h_P"], Dh), 0)) / r["h_scale"]
        return out


def _knn(X, R, k):
    out = np.empty(len(X))
    rn = (R ** 2).sum(1)
    for s in range(0, len(X), 512):
        x = X[s:s + 512]
        d2 = (x ** 2).sum(1)[:, None] + rn[None] - 2 * x @ R.T
        d = np.sqrt(np.maximum(d2, 0))
        out[s:s + 512] = np.sort(d, 1)[:, :k].mean(1)
    return out


# --------------------------------------------------------------------------- policy
class OODPolicy:
    def __init__(self, problem: str, artifacts: Path | None = None, root: Path = ROOT):
        if problem not in PROBLEMS:
            raise ValueError(f"unknown problem {problem}")
        self.problem = problem
        art = Path(artifacts or HERE / "artifacts" / problem)
        self.calib = json.loads((art / "calib.json").read_text())
        with np.load(art / "reference.npz") as d:
            self.nov = Novelty({k: d[k] for k in d.files})
        pdir = root / "problems" / problem
        if problem == "cavity_flow":
            self.w = {c: _released(pdir / f"checkpoint_{c}" / "model", art / f"released_{c}.npz") for c in "uv"}
            self.ens = {c: [dj.load_npz_weights(str(p)) for p in sorted(art.glob(f"ens_{c}_*.npz"))] for c in "uv"}
        else:
            self.w = _released(pdir / "checkpoint" / "model", art / "released.npz")
            self.ens = [dj.load_npz_weights(str(p)) for p in sorted(art.glob("ens_*.npz"))]

    # ---------------------------------------------------------------- features
    def features(self, branch, trunk, max_points=4096, seed=0):
        G = np.asarray(branch, np.float64)
        P = np.asarray(trunk, np.float64)
        if self.problem == "cavity_flow":
            return self._features_cavity(G, P, max_points, seed)
        return self._features_scalar(G, P, max_points, seed)

    def _features_scalar(self, G, P, max_points, seed):
        p, w = self.problem, self.w
        rng = np.random.default_rng(seed)
        M = P.shape[-2]
        if M > max_points:  # estimate per-case RMS statistics from a fixed random subset of points
            idx = np.sort(rng.choice(M, max_points, replace=False))
            P = P[..., idx, :]
        F = {"ens_dev": [], "ens_std": [], "phys_res": [], "phys_bc": [], "pred_rms": []}
        for s in range(0, len(G), 64):
            g = G[s:s + 64]; q = P if P.ndim == 2 else P[s:s + 64]
            f0 = dj.predict(p, w, g, q)
            E = np.stack([dj.predict(p, we, g, q) for we in self.ens])
            mu = E.mean(0)
            F["ens_dev"].append(np.sqrt(((f0 - mu) ** 2).mean(1)))
            F["ens_std"].append(np.sqrt(E.var(0).mean(1)))
            F["pred_rms"].append(np.sqrt((f0 ** 2).mean(1)))
            r, bc = dj.physics_scalar(p, w, g, q)
            if p == "poisson":
                m = dj.poisson_interior_mask(q)
                m = np.broadcast_to(m if m.ndim == 2 else m[None], r.shape)
                F["phys_res"].append(np.sqrt((r ** 2).sum(1) / np.maximum(m.sum(1), 1)))
            else:
                F["phys_res"].append(np.sqrt((r ** 2).mean(1)))
            F["phys_bc"].append(np.sqrt((bc ** 2).mean(1)) if bc is not None else np.zeros(len(g)))
        F = {k: np.concatenate(v) for k, v in F.items()}
        F.update(self.nov(G, w))
        return F

    def _features_cavity(self, G, P, max_points, seed):
        wu, wv = self.w["u"], self.w["v"]
        paired = P.ndim == 2
        if paired:
            if len(P) != len(G):
                raise ValueError("cavity_flow paired trunk must have one row per branch row")
            # group rows by geometry so geometry-level features are computed once
            uniq, inv = np.unique(G, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
        else:
            uniq, inv = G, np.arange(len(G))
        geo = self.nov(uniq, wu)
        _, _, bc = dj.physics_cavity(wu, wv, uniq, np.full((len(uniq), 1, 2), 0.5))
        geo["phys_bc"] = np.sqrt((bc ** 2).mean(1))
        if paired:
            Q = P[:, None, :]
        else:
            Q = P
            M = P.shape[1]
            if M > max_points:
                idx = np.sort(np.random.default_rng(seed).choice(M, max_points, replace=False))
                Q = P[:, idx]
        pt = {"ens_dev": [], "ens_std": [], "phys_res": [], "phys_div": []}
        for s in range(0, len(G), 4096 if paired else 16):
            g = G[s:s + (4096 if paired else 16)]; q = Q[s:s + len(g)]
            fu = dj.predict("cavity_flow", wu, g, q); fv = dj.predict("cavity_flow", wv, g, q)
            Eu = np.stack([dj.predict("cavity_flow", e, g, q) for e in self.ens["u"]])
            Ev = np.stack([dj.predict("cavity_flow", e, g, q) for e in self.ens["v"]])
            pt["ens_dev"].append(np.sqrt((fu - Eu.mean(0)) ** 2 + (fv - Ev.mean(0)) ** 2))
            pt["ens_std"].append(np.sqrt(Eu.var(0) + Ev.var(0)))
            res, div, _ = dj.physics_cavity(wu, wv, g, q)
            pt["phys_res"].append(np.abs(res)); pt["phys_div"].append(np.abs(div))
        pt = {k: np.concatenate(v) for k, v in pt.items()}  # (N, m)
        F = {}
        if paired:
            F.update({k: v[:, 0] for k, v in pt.items()})
        else:  # robust per-case aggregation (corner singularities make the residuals heavy-tailed)
            F["ens_dev"] = np.sqrt((pt["ens_dev"] ** 2).mean(1)); F["ens_std"] = np.sqrt((pt["ens_std"] ** 2).mean(1))
            F["phys_res"] = np.exp(np.log(pt["phys_res"] + 1e-12).mean(1))
            F["phys_div"] = np.exp(np.log(pt["phys_div"] + 1e-12).mean(1))
        for k, v in geo.items():
            F[k] = v[inv]
        F["_pointwise_ens_dev"] = pt["ens_dev"]
        return F

    # ---------------------------------------------------------------- score
    def score(self, branch, trunk, max_points=4096, seed=0) -> dict:
        F = self.features(branch, trunk, max_points, seed)
        c = self.calib
        logs = c["intercept"] + sum(wj * np.log(F[k] + c["eps"][k]) for k, wj in c["weights"].items())
        out = {"ood_score": np.exp(logs)}
        if "pred_rms" in F:
            out["est_rel_error"] = out["ood_score"] / np.maximum(F["pred_rms"], 1e-12)
        out.update({f"feat_{k}": v for k, v in F.items() if not k.startswith("_")})
        return out


def _released(prefix: Path, fallback: Path) -> dict:
    if fallback.exists():
        return dj.load_npz_weights(str(fallback))
    return dj.load_checkpoint_weights(str(prefix))


def load_inputs(path: Path, problem: str):
    with np.load(path, allow_pickle=False) as d:
        keys = set(d.files)
        if {"branch", "trunk"} <= keys:
            return d["branch"], d["trunk"]
        if problem == "cavity_flow" and {"branch_train", "trunk_train"} <= keys:
            return d["branch_train"], d["trunk_train"]
        if {"X_train0", "X_train1"} <= keys:
            return d["X_train0"], d["X_train1"]
    raise ValueError(f"unsupported input fields in {path}: {sorted(keys)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True, choices=PROBLEMS)
    ap.add_argument("--input", type=Path, required=True, help="NPZ with branch (N,101) and trunk")
    ap.add_argument("--output", type=Path, help="output NPZ (default: ood_scores/<problem>.npz)")
    ap.add_argument("--max-points", type=int, default=4096,
                    help="points per test case used to estimate per-case statistics (default 4096)")
    args = ap.parse_args()
    branch, trunk = load_inputs(args.input, args.problem)
    out = OODPolicy(args.problem).score(branch, trunk, args.max_points)
    path = args.output or ROOT / "ood_scores" / f"{args.problem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    s = out["ood_score"]
    print(f"Saved {len(s)} OOD scores for {args.problem} to {path} "
          f"(min {s.min():.3g}, median {np.median(s):.3g}, max {s.max():.3g})")


if __name__ == "__main__":
    main()
