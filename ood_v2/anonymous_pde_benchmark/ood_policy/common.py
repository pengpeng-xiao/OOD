"""Framework-independent pieces shared by the JAX and PyTorch backends.

PDEs (identified from the released training arrays only; see README.md):
    diffusion_reaction : u_t = 0.01 u_xx + 0.01 u^2 + f(x),  u = 0 at x=0, x=1, t=0   (f = branch)
    burgers            : u_t + u u_x = 0.1 u_xx, periodic in x, u(x,0) = g(x)          (g = branch)
    advection          : u_t + a(x) u_x = 0, u(x,0)=sin(pi x), u(0,t)=sin(pi t/2)       (a = branch)
    poisson            : -lap(u) = 10 on the triangle (0,0),(1,0),(1/2,sqrt3/2) minus the slot
                         [0.49,0.51]x[0,0.4];  u = g(x) on the whole boundary            (g = branch)
    cavity_flow        : steady incompressible Navier-Stokes, Re = 1000, lid u=1 at y=1,
                         no-slip walls x=0, x=1 and bottom wall y = b(x)                  (b = branch)
Branch sensors are always at x_k = k/100, k = 0..100.
"""
from __future__ import annotations

import math

import numpy as np

NU_BURGERS, D_DR, K_DR, POIS_SRC, NU_CAV = 0.1, 0.01, 0.01, 10.0, 0.001
S3 = math.sqrt(3.0)
SENSORS = np.linspace(0.0, 1.0, 101)


def read_tf_checkpoint(prefix: str) -> dict:
    """Read a released TF1 checkpoint into numpy arrays (W0..W5, b0..b5, b_out). Needs tensorflow."""
    import tensorflow as tf

    r = tf.train.load_checkpoint(prefix)
    names = ["dense", "dense_1", "dense_2", "dense_3", "dense_4", "dense_5"]
    w = {}
    for i, n in enumerate(names):
        w[f"W{i}"] = r.get_tensor(n + "/kernel").astype(np.float64)
        w[f"b{i}"] = r.get_tensor(n + "/bias").astype(np.float64)
    w["b_out"] = r.get_tensor("Variable").astype(np.float64)
    return w


def interp_branch(g, x):
    """Linear interpolation of sensor values g (N,101) at x (N,M) or (1,M); clamped to [0,1]."""
    xs = np.clip(x, 0, 1) * 100
    i0 = np.clip(np.floor(xs).astype(int), 0, 99)
    wgt = xs - i0
    if x.shape[0] == 1 and g.shape[0] != 1:
        i0 = np.broadcast_to(i0, (g.shape[0], i0.shape[1])); wgt = np.broadcast_to(wgt, i0.shape)
    return np.take_along_axis(g, i0, 1) * (1 - wgt) + np.take_along_axis(g, i0 + 1, 1) * wgt


def poisson_boundary(n_per_unit=200):
    segs = [((0, 0), (0.49, 0)), ((0.51, 0), (1, 0)), ((0, 0), (0.5, S3 / 2)), ((1, 0), (0.5, S3 / 2)),
            ((0.49, 0), (0.49, 0.4)), ((0.51, 0), (0.51, 0.4)), ((0.49, 0.4), (0.51, 0.4))]
    pts = []
    for a, b in segs:
        a, b = np.array(a, float), np.array(b, float)
        n = max(3, int(np.linalg.norm(b - a) * n_per_unit))
        s = np.linspace(0, 1, n)
        pts.append(a + s[:, None] * (b - a))
    return np.unique(np.concatenate(pts).round(10), axis=0)


def poisson_interior_mask(P):
    x, y = P[..., 0], P[..., 1]
    db = np.minimum(np.minimum(y, (S3 * x - y) / 2), (S3 * (1 - x) - y) / 2)
    inslot = (x > 0.49 - 1e-6) & (x < 0.51 + 1e-6) & (y < 0.4 + 1e-6)
    return (db > 1e-6) & ~inslot


def cavity_walls(g):
    """No-slip wall points (N,303,2) and lid points (N,81,2) implied by bottom-wall profile g (N,101)."""
    N = len(g); s = SENSORS
    bot = np.stack([np.broadcast_to(s, (N, 101)), g], -1)
    lft = np.stack([np.zeros((N, 101)), g[:, :1] + (1 - g[:, :1]) * s], -1)
    rgt = np.stack([np.ones((N, 101)), g[:, -1:] + (1 - g[:, -1:]) * s], -1)
    sl = np.linspace(0.1, 0.9, 81)
    lid = np.stack([np.broadcast_to(sl, (N, 81)), np.ones((N, 81))], -1)
    return np.concatenate([bot, lft, rgt], 1), lid
