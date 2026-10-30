"""Build artifacts/<problem>/reference.npz (novelty statistics) and released*.npz (weights copy)
from the released training arrays and checkpoints. Run in the repro env from the bundle root."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import deeponet_jax as dj
ROOT = HERE.parent


def reference(G, w, var_keep=0.9999, k=5):
    G = np.asarray(G, np.float64)
    mu = G.mean(0)
    U, s, Vt = np.linalg.svd(G - mu, full_matrices=False)
    ev = s ** 2 / (len(G) - 1)
    kp = int(np.searchsorted(np.cumsum(ev) / ev.sum(), var_keep) + 1)
    kp = max(1, min(kp, len(G) - 1, int((ev > ev[0] * 1e-12).sum())))
    V, ev = Vt[:kp], ev[:kp]
    D = G - mu
    res = np.linalg.norm(D - (D @ V.T) @ V, axis=1)
    total = np.sqrt((D ** 2).sum(1).mean())
    res_scale = max(np.sqrt((res ** 2).mean()), 1e-3 * total)

    def knn_self(X):
        d = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * X @ X.T, 0))
        np.fill_diagonal(d, np.inf)
        return np.sort(d, 1)[:, :min(k, len(X) - 1)].mean(1)

    h = np.maximum(G @ w["W0"] + w["b0"], 0); h = np.maximum(h @ w["W1"] + w["b1"], 0)
    L = h @ w["W2"] + w["b2"]
    h_mu = h.mean(0); C = np.cov(h - h_mu, rowvar=False)
    C += 1e-3 * np.trace(C) / len(C) * np.eye(len(C)); h_P = np.linalg.inv(C)
    Dh = h - h_mu; hm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Dh, h_P, Dh), 0))
    return dict(mu=mu, V=V, ev=ev, res_scale=res_scale, G=G, nn_scale=np.median(knn_self(G)),
                L=L, lat_scale=np.median(knn_self(L)), h_mu=h_mu, h_P=h_P, h_scale=np.median(hm))


for p in sys.argv[1:] or ["diffusion_reaction", "burgers", "advection", "poisson", "cavity_flow"]:
    out = HERE / "artifacts" / p; out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "problems" / p / "training_data.npz")
    if p == "cavity_flow":
        G = d["branch_train"][::10201]
        for c in "uv":
            w = dj.load_checkpoint_weights(str(ROOT / "problems" / p / f"checkpoint_{c}" / "model"))
            np.savez(out / f"released_{c}.npz", **w)
        w = dj.load_npz_weights(str(out / "released_u.npz"))
    else:
        G = d["X_train0"]
        w = dj.load_checkpoint_weights(str(ROOT / "problems" / p / "checkpoint" / "model"))
        np.savez(out / "released.npz", **w)
    ref = reference(G, w)
    np.savez(out / "reference.npz", **ref)
    print(p, "n_train", len(G), "pca k", len(ref["ev"]), "res_scale %.3g nn_scale %.3g lat_scale %.3g h_scale %.3g" % (ref["res_scale"], ref["nn_scale"], ref["lat_scale"], ref["h_scale"]))
