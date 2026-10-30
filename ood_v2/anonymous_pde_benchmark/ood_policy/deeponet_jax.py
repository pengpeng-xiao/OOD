"""JAX backend (numpy-in / numpy-out): re-implementation of the released DeepONets, derivative jets
w.r.t. trunk coordinates, and physics residuals. See common.py for the identified PDEs."""
from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

from common import (D_DR, K_DR, NU_BURGERS, NU_CAV, POIS_SRC, SENSORS, cavity_walls,  # noqa: F401
                    interp_branch, poisson_boundary, poisson_interior_mask, read_tf_checkpoint)

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- weights / forward
def load_checkpoint_weights(prefix: str) -> dict:
    return read_tf_checkpoint(prefix)


def load_npz_weights(path: str) -> dict:
    with np.load(path) as d:
        return {k: d[k].astype(np.float64) for k in d.files}


def gelu(z):
    # The tanh argument is clipped at +-20 (tanh is exactly +-1 there in float64): values are unchanged,
    # but it stops XLA's jitted tanh from leaking tiny non-zero slopes into high-order derivatives of
    # saturated units.
    return 0.5 * z * (1 + jnp.tanh(jnp.clip(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3), -20.0, 20.0)))


def branch_np(w: dict, u: np.ndarray) -> np.ndarray:
    h = np.maximum(u @ w["W0"] + w["b0"], 0)
    h = np.maximum(h @ w["W1"] + w["b1"], 0)
    return h @ w["W2"] + w["b2"]


def _feature(problem, p):
    """p: (2,) single point -> trunk input features."""
    if problem == "burgers":
        x = p[0] * 2 * jnp.pi
        return jnp.stack([jnp.cos(x), jnp.sin(x), jnp.cos(2 * x), jnp.sin(2 * x), jnp.cos(3 * x),
                          jnp.sin(3 * x), jnp.cos(4 * x), jnp.sin(4 * x), p[1]])
    return p


def make_trunk(problem: str, w: dict):
    W3, b3, W4, b4, W5, b5 = (jnp.asarray(w[k]) for k in ("W3", "b3", "W4", "b4", "W5", "b5"))

    def trunk(p):  # (2,) -> (100,)
        h = _feature(problem, p)
        h = gelu(h @ W3 + b3)
        h = gelu(h @ W4 + b4)
        return gelu(h @ W5 + b5)

    return trunk


_JIT_CACHE: dict = {}


def trunk_jets(problem: str, w: dict, P: np.ndarray, order: int, chunk: int = 8192) -> dict:
    """Trunk features and derivatives at points P (M,2). Keys '', x, t, xx, xt, tt, xxx, xxt, xtt, ttt."""
    key = (problem, id(w), order)
    if key not in _JIT_CACHE:
        tr = make_trunk(problem, w)

        def jet(p):
            out = {"": tr(p)}
            if order >= 1:
                J = jax.jacfwd(tr)(p)
                out["x"], out["t"] = J[:, 0], J[:, 1]
            if order >= 2:
                H = jax.jacfwd(jax.jacfwd(tr))(p)
                out["xx"], out["xt"], out["tt"] = H[:, 0, 0], H[:, 0, 1], H[:, 1, 1]
            if order >= 3:
                K = jax.jacfwd(jax.jacfwd(jax.jacfwd(tr)))(p)
                out["xxx"], out["xxt"], out["xtt"], out["ttt"] = K[:, 0, 0, 0], K[:, 0, 0, 1], K[:, 0, 1, 1], K[:, 1, 1, 1]
            return out

        _JIT_CACHE[key] = jax.jit(jax.vmap(jet))
    f = _JIT_CACHE[key]
    P = np.asarray(P, np.float64)
    parts = [f(jnp.asarray(P[s:s + chunk])) for s in range(0, len(P), chunk)]
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in parts[0]}


def z_jets(problem, w, B, P, order):
    """Raw output z = <branch, trunk> + b and derivatives. B (N,100); P (M,2) shared or (N,M,2)."""
    if P.ndim == 2:
        J = trunk_jets(problem, w, P, order)
        Z = {k: B @ v.T for k, v in J.items()}
    else:
        N, M = P.shape[:2]
        J = trunk_jets(problem, w, P.reshape(-1, 2), order)
        Z = {k: np.einsum("np,nmp->nm", B, v.reshape(N, M, -1)) for k, v in J.items()}
    Z[""] = Z[""] + w["b_out"][0]
    return Z


def u_jets(problem, w, g, P, order):
    """Model prediction u (N,M) and its derivatives, output transform included."""
    Z = z_jets(problem, w, branch_np(w, g), P, order)
    x = P[..., 0] if P.ndim == 3 else P[None, :, 0]
    t = P[..., 1] if P.ndim == 3 else P[None, :, 1]
    if problem == "diffusion_reaction":
        phi, phx, pht, phxx = 10 * x * (1 - x) * t, 10 * (1 - 2 * x) * t, 10 * x * (1 - x), -20 * t
        z1 = Z[""] + 1
        U = {"": phi * z1}
        if order >= 1:
            U["x"] = phx * z1 + phi * Z["x"]
            U["t"] = pht * z1 + phi * Z["t"]
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




# --------------------------------------------------------------------------- physics




def physics_scalar(problem, w, g, P):
    """Per-point PDE residual r (N,M) [poisson: zero outside the interior] and per-sample
    boundary/initial-condition mismatch vector bc (N,K) or None."""
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
    """P (N,M,2). Returns vorticity-transport residual (N,M), continuity residual (N,M), wall/lid mismatch (N,K)."""
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
