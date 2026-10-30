"""PyTorch backend with the same numpy-in / numpy-out API as deeponet_jax.py (used when torch is
installed, e.g. the `mindquan` env; runs on GPU when available).  See deeponet_jax.py for the PDEs."""
from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch.func import jacfwd, vmap

from common import (D_DR, K_DR, NU_BURGERS, NU_CAV, POIS_SRC, SENSORS, cavity_walls,  # noqa: F401
                                 interp_branch, poisson_boundary, poisson_interior_mask)

DEV = torch.device(os.environ.get("OOD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DT = torch.float64


def load_npz_weights(path: str) -> dict:
    with np.load(path) as d:
        return {k: d[k].astype(np.float64) for k in d.files}


def load_checkpoint_weights(prefix: str) -> dict:
    from common import read_tf_checkpoint
    return read_tf_checkpoint(prefix)


def _t(a):
    return torch.as_tensor(np.asarray(a, np.float64), dtype=DT, device=DEV)


def gelu(z):
    return 0.5 * z * (1 + torch.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))


class _Net:
    """Torch tensors for one weight dict (cached per dict id)."""

    def __init__(self, problem, w):
        self.problem = problem
        self.W = [_t(w[f"W{i}"]) for i in range(6)]
        self.b = [_t(w[f"b{i}"]) for i in range(6)]
        self.b_out = float(np.asarray(w["b_out"]).reshape(-1)[0])

    def branch(self, u):
        h = torch.relu(u @ self.W[0] + self.b[0])
        h = torch.relu(h @ self.W[1] + self.b[1])
        return h @ self.W[2] + self.b[2]

    def trunk(self, p):
        if self.problem == "burgers":
            x = p[..., 0:1] * 2 * math.pi
            p = torch.cat([torch.cos(x), torch.sin(x), torch.cos(2 * x), torch.sin(2 * x), torch.cos(3 * x),
                           torch.sin(3 * x), torch.cos(4 * x), torch.sin(4 * x), p[..., 1:2]], -1)
        h = p
        for i in (3, 4, 5):
            h = gelu(h @ self.W[i] + self.b[i])
        return h


_NETS: dict = {}


def _net(problem, w):
    key = (problem, id(w))
    if key not in _NETS:
        _NETS[key] = (_Net(problem, w), w)  # keep w alive so its id is not reused
    return _NETS[key][0]


def branch_np(w, u):
    return _net("any", w).branch(_t(u)).cpu().numpy()


def trunk_jets(problem, w, P, order, chunk=8192):
    net = _net(problem, w)
    f = lambda p: net.trunk(p[None])[0]
    keys3 = [("xxx", (0, 0, 0)), ("xxt", (0, 0, 1)), ("xtt", (0, 1, 1)), ("ttt", (1, 1, 1))]
    outs = {}
    Pt = _t(P)
    with torch.no_grad():
        for s in range(0, len(Pt), chunk):
            p = Pt[s:s + chunk]
            part = {"": net.trunk(p)}
            if order >= 1:
                J = vmap(jacfwd(f))(p); part["x"], part["t"] = J[..., 0], J[..., 1]
            if order >= 2:
                H = vmap(jacfwd(jacfwd(f)))(p)
                part["xx"], part["xt"], part["tt"] = H[..., 0, 0], H[..., 0, 1], H[..., 1, 1]
            if order >= 3:
                K = vmap(jacfwd(jacfwd(jacfwd(f))))(p)
                for k, ix in keys3:
                    part[k] = K[(...,) + ix]
            for k, v in part.items():
                outs.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in outs.items()}


def u_jets(problem, w, g, P, order):
    net = _net(problem, w)
    with torch.no_grad():
        B = net.branch(_t(g))
        if P.ndim == 2:
            J = trunk_jets(problem, w, P, order)
            Z = {k: B @ v.T for k, v in J.items()}
        else:
            N, M = P.shape[:2]
            J = trunk_jets(problem, w, P.reshape(-1, 2), order)
            Z = {k: torch.einsum("np,nmp->nm", B, v.reshape(N, M, -1)) for k, v in J.items()}
    Z = {k: v.cpu().numpy() for k, v in Z.items()}
    Z[""] = Z[""] + net.b_out
    x = P[..., 0] if P.ndim == 3 else P[None, :, 0]
    t = P[..., 1] if P.ndim == 3 else P[None, :, 1]
    if problem == "diffusion_reaction":
        phi, phx, pht, phxx = 10 * x * (1 - x) * t, 10 * (1 - 2 * x) * t, 10 * x * (1 - x), -20 * t
        z1 = Z[""] + 1
        U = {"": phi * z1}
        if order >= 1:
            U["x"] = phx * z1 + phi * Z["x"]; U["t"] = pht * z1 + phi * Z["t"]
        if order >= 2:
            U["xx"] = phxx * z1 + 2 * phx * Z["x"] + phi * Z["xx"]
        return U
    if problem == "advection":
        U = {"": 4 * x * t * Z[""] + np.sin(np.pi * x) + np.sin(0.5 * np.pi * t)}
        if order >= 1:
            U["x"] = 4 * t * Z[""] + 4 * x * t * Z["x"] + np.pi * np.cos(np.pi * x)
            U["t"] = 4 * x * Z[""] + 4 * x * t * Z["t"] + 0.5 * np.pi * np.cos(0.5 * np.pi * t)
        return U
    return Z


def predict(problem, w, g, P):
    return u_jets(problem, w, np.asarray(g, np.float64), np.asarray(P, np.float64), 0)[""]


def physics_scalar(problem, w, g, P):
    if problem == "diffusion_reaction":
        U = u_jets(problem, w, g, P, 2)
        x = P[..., 0] if P.ndim == 3 else P[None, :, 0]
        return U["t"] - D_DR * U["xx"] - K_DR * U[""] ** 2 - interp_branch(g, x), None
    if problem == "burgers":
        U = u_jets(problem, w, g, P, 2)
        P0 = np.stack([SENSORS, np.zeros(101)], 1)
        return U["t"] + U[""] * U["x"] - NU_BURGERS * U["xx"], predict(problem, w, g, P0) - g
    if problem == "advection":
        U = u_jets(problem, w, g, P, 1)
        x = P[..., 0] if P.ndim == 3 else P[None, :, 0]
        return U["t"] + interp_branch(g, x) * U["x"], None
    if problem == "poisson":
        U = u_jets(problem, w, g, P, 2)
        mask = poisson_interior_mask(P)
        mask = mask if mask.ndim == 2 else mask[None]
        r = np.where(mask, U["xx"] + U["tt"] + POIS_SRC, 0.0)
        Pb = poisson_boundary()
        return r, predict(problem, w, g, Pb) - interp_branch(g, Pb[None, :, 0])
    raise ValueError(problem)


def physics_cavity(wu, wv, g, P):
    Uu = u_jets("cavity_flow", wu, g, P, 3)
    Uv = u_jets("cavity_flow", wv, g, P, 3)
    u, v = Uu[""], Uv[""]
    wx = Uv["xx"] - Uu["xt"]; wy = Uv["xt"] - Uu["tt"]
    lapw = Uv["xxx"] + Uv["xtt"] - Uu["xxt"] - Uu["ttt"]
    res = u * wx + v * wy - NU_CAV * lapw
    div = Uu["x"] + Uv["t"]
    W, L = cavity_walls(g)
    bc = np.concatenate([predict("cavity_flow", wu, g, W), predict("cavity_flow", wv, g, W),
                         predict("cavity_flow", wu, g, L) - 1, predict("cavity_flow", wv, g, L)], 1)
    return res, div, bc
