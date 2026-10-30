"""Compare raw-input Mahalanobis and W2 OOD scores with the fitted GRF likelihood-ratio test.

Per-sample scores, all computed on the raw branch inputs (no preprocessing) unless noted:
  maha_full   Mahalanobis, full sample covariance, pseudo-inverse (eigenvalues > 1e-13 * max)
  maha_lw     Mahalanobis, Ledoit-Wolf shrinkage covariance
  maha_pcaK   Mahalanobis restricted to the top-K principal directions
  w2_point    W2 between the point mass at x and the training empirical measure
              = sqrt(mean_i ||x - x_i||^2), i.e. Euclidean distance to the training mean
  w2_values   1-D W2 between the 101 values of x and the pooled training values
  w2_gauss    closed-form W2 between N(0, s^2 K_l) fitted to x and the training GRF (uses the fit)
  lrt         fitted GRF likelihood-ratio statistic (the method used for the labels)
Thresholds for the label-free accuracy are the 99th percentile of held-out training scores
(10-fold CV; leave-one-out for cavity flow).
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve

A = Path(__file__).resolve().parent.parent
R, F = A / "results", A / "feat"
B = Path("/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems")
KEY = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/mnt/ssd/pengpengxiao/deeponet-extrapolation/distribution/private/legacy_with_labels_20261013/anonymous_pde_benchmark_test_key.json")  # matches this bundle (manifest 0605dffb); the top-level key is for the re-shuffled release
key = json.load(open(KEY))["problems"]
PROBS = ["diffusion_reaction", "burgers", "advection", "poisson", "cavity_flow"]
PCA_K = [3, 5, 10, 15, 20]
rng = np.random.default_rng(0)


def truth(prob):
    sid = np.load(B / prob / "test_data.npz")["sample_id"]
    m = {e["sample_id"]: e for e in key[prob]}
    return (np.array([m[s]["split"] == "ood" for s in sid]),
            np.array([m[s]["source"].split("/")[-1] for s in sid]))


def load_inputs(prob):
    te = np.load(B / prob / "test_data.npz")["branch"].astype(np.float64)
    if prob == "cavity_flow":
        tr = np.load(B / prob / "training_data.npz")["branch_train"].reshape(21, 10201, 101)[:, 0]
    else:
        tr = np.load(B / prob / "training_data.npz")["X_train0"]
    return tr.astype(np.float64), te


# ------------------------------------------------------------------ score families
def fit_maha(X, mode, k=None):
    mu = X.mean(0)
    C = LedoitWolf().fit(X).covariance_ if mode == "lw" else np.cov(X.T)
    w, U = np.linalg.eigh(C)
    w, U = w[::-1], U[:, ::-1]
    if mode == "pca":
        w, U = w[:k], U[:, :k]
    elif mode == "full":
        keep = w > 1e-13 * w.max()
        w, U = w[keep], U[:, keep]
    return lambda Z: (((Z - mu) @ U) ** 2 / w).sum(1)


def fit_w2_point(X):
    mu = X.mean(0); tr = ((X - mu) ** 2).sum(1).mean()
    return lambda Z: np.sqrt((((Z - mu) ** 2).sum(1) + tr) / X.shape[1])


def fit_w2_values(X):
    p = (np.arange(X.shape[1]) + 0.5) / X.shape[1]
    q = np.quantile(X.ravel(), p)
    return lambda Z: np.sqrt(((np.sort(Z, 1) - q) ** 2).mean(1))


def cv_null(X, fitter, folds):
    idx = rng.permutation(len(X)); out = np.empty(len(X))
    for f in np.array_split(idx, folds):
        m = np.ones(len(X), bool); m[f] = False
        out[f] = fitter(X[m])(X[f])
    return out


# ------------------------------------------------------------------ Gaussian W2 between GRFs
x101 = np.linspace(0, 1, 101)
def kern(l, kind, n):
    D = x101[:n, None] - x101[None, :n]
    return np.exp(-D ** 2 / (2 * l * l)) if kind == "rbf" else np.exp(-2 * np.sin(np.pi * D) ** 2 / (l * l))
def psd_sqrt(C):
    w, U = np.linalg.eigh(C); return (U * np.sqrt(np.clip(w, 0, None))) @ U.T
GRF = {"diffusion_reaction": ("rbf", 101, 0.5), "advection": ("rbf", 101, 0.5),
       "poisson": ("rbf", 101, 0.5), "burgers": ("per", 100, 1.0)}
def w2_gauss_scores(prob):
    kind, n, l0 = GRF[prob]
    g = np.load(F / f"{prob}_gp2.npz")
    s0 = float(np.median(g["s_tr"]))
    K0h = psd_sqrt(kern(l0, kind, n)); cache = {}
    def T(l):                                   # tr((K0^1/2 K_l K0^1/2)^1/2) / n
        if l not in cache:
            cache[l] = np.trace(psd_sqrt(K0h @ kern(l, kind, n) @ K0h)) / n
        return cache[l]
    def w2(l, s):                               # tr K / n = 1 for both kernels
        return np.sqrt(max(s0 ** 2 + s ** 2 - 2 * s0 * s * T(float(l)), 0.0))
    tr_ = np.array([w2(l, s) for l, s in zip(g["l_tr"], g["s_tr"])])
    te_ = np.array([w2(l, s) for l, s in zip(g["l_te"], g["s_te"])])
    return tr_, te_


# ------------------------------------------------------------------ metrics
def metrics(y, s, null):
    fpr, tpr, _ = roc_curve(y, s)
    thr = np.quantile(null, 0.99)
    return dict(auroc=roc_auc_score(y, s), fpr95=float(fpr[np.searchsorted(tpr, 0.95)]),
                acc=float(np.mean((s > thr) == y)), tpr=float(np.mean(s[y] > thr)), fpr=float(np.mean(s[~y] > thr)))


results = {}
for prob in PROBS:
    y, src = truth(prob)
    Xtr, Xte = load_inputs(prob)
    folds = len(Xtr) if prob == "cavity_flow" else 10
    fam = {"maha_full": lambda X: fit_maha(X, "full"), "maha_lw": lambda X: fit_maha(X, "lw")}
    for k in PCA_K:
        if k <= (1 if prob == "cavity_flow" else 101):
            fam[f"maha_pca{k}"] = (lambda kk: (lambda X: fit_maha(X, "pca", kk)))(k)
    if prob == "cavity_flow":
        fam["maha_pca1"] = lambda X: fit_maha(X, "pca", 1)
    fam["w2_point"] = fit_w2_point
    fam["w2_values"] = fit_w2_values
    res = {}
    for name, fitter in fam.items():
        res[name] = metrics(y, fitter(Xtr)(Xte), cv_null(Xtr, fitter, folds))
    if prob != "cavity_flow":
        g_tr, g_te = w2_gauss_scores(prob)
        res["w2_gauss"] = metrics(y, g_te, g_tr)
        L = np.load(R / f"{prob}_labels.npz")
        res["lrt"] = metrics(y, L["lr_te"], L["lr_tr"])
        res["lrt"]["acc"] = float(np.mean(L["ood"] == y))          # the actual p < 1e-4 rule
        res["lrt"]["tpr"] = float(L["ood"][y].mean()); res["lrt"]["fpr"] = float(L["ood"][~y].mean())
        # per-regime AUROC for the raw-input scores (advection has four OOD regimes)
        if prob == "advection":
            for name, fitter in [("w2_point", fit_w2_point), ("maha_pca5", lambda X: fit_maha(X, "pca", 5)),
                                 ("maha_lw", lambda X: fit_maha(X, "lw"))]:
                s = fitter(Xtr)(Xte)
                res[name]["per_regime"] = {r: roc_auc_score(y[(src == r) | ~y], s[(src == r) | ~y])
                                           for r in sorted(set(src[y]))}
    else:
        U = Xte[:, 0]
        res["range_check"] = metrics(y, U, Xtr[:, 0])
        res["range_check"]["acc"] = float(np.mean((U > Xtr[:, 0].max()) == y))
    results[prob] = res

# ------------------------------------------------------------------ set-level W2 (Gaussian/Bures) between groups
def bures(X1, X2):
    m1, m2 = X1.mean(0), X2.mean(0); C1, C2 = np.cov(X1.T), np.cov(X2.T)
    h = psd_sqrt(C1)
    v = ((m1 - m2) ** 2).sum() + np.trace(C1) + np.trace(C2) - 2 * np.trace(psd_sqrt(h @ C2 @ h))
    return float(np.sqrt(max(v, 0) / X1.shape[1]))
setlevel = {}
for prob in PROBS[:4]:
    y, src = truth(prob); Xtr, Xte = load_inputs(prob)
    base = [bures(Xtr, Xtr[rng.choice(len(Xtr), 100, replace=False)]) for _ in range(20)]
    setlevel[prob] = {"train_vs_random_100_train": float(np.mean(base))}
    for r in sorted(set(src)):
        setlevel[prob][r] = bures(Xtr, Xte[src == r])

out = {"per_sample": results, "set_level_w2": setlevel}
(R / "distance_comparison.json").write_text(json.dumps(out, indent=2))

# ------------------------------------------------------------------ print
for prob, res in results.items():
    print(f"===== {prob}")
    print(f"  {'score':12s} {'AUROC':>7s} {'FPR@95':>7s} {'acc':>6s} {'TPR':>6s} {'FPR':>6s}   (threshold = 99th pct of held-out train)")
    for name, m in res.items():
        print(f"  {name:12s} {m['auroc']:7.3f} {m['fpr95']:7.3f} {m['acc']:6.3f} {m['tpr']:6.3f} {m['fpr']:6.3f}")
        if "per_regime" in m:
            print("      per regime AUROC:", {k.split('_ls_')[1].split('_')[0]: round(v, 3) for k, v in m["per_regime"].items()})
print("===== set-level Gaussian W2 (L2[0,1] units) between training set and each test group")
for prob, d in setlevel.items():
    print(f"  {prob}: " + ", ".join(f"{k.split('_ls_')[1].split('_')[0] if '_ls_' in k else k}={v:.3f}" for k, v in d.items()))
