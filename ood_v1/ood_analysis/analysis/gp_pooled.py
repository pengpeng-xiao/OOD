"""Pooled ML over training set: find (l, sigma, nugget) of the training GRF; compare kernels."""
import numpy as np
from pathlib import Path
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems')
x=np.linspace(0,1,101); D=x[:,None]-x[None,:]
def kern(l, kind, nu=None):
    if kind=='rbf': return np.exp(-D**2/(2*l*l))
    if kind=='per': return np.exp(-2*np.sin(np.pi*D)**2/(l*l))
    if kind=='m52':
        r=np.abs(D)/l; return (1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)
def contrast(n):
    Q,_=np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n-1]]); return Q[:,1:]
def pooled(X, kind, l, nug, const_mean=False, drop_last=False):
    Xw=X[:,:-1] if drop_last else X; n=Xw.shape[1]
    K=kern(l,kind); K=K[:n,:n]
    if const_mean:
        A=contrast(n); K=A.T@K@A; Xw=Xw@A; n=n-1
    w,U=np.linalg.eigh(K); w=np.maximum(w,0)+nug
    z=Xw@U; s2=(z**2/w).sum()/ (n*len(X))   # common sigma^2
    ll=-0.5*(len(X)*(n*np.log(2*np.pi*s2)+np.log(w).sum())+n*len(X))
    return ll/len(X), np.sqrt(s2)
cfg={'diffusion_reaction':dict(kinds=['rbf','m52']),'poisson':dict(kinds=['rbf','m52']),
     'advection':dict(kinds=['rbf','m52'],const_mean=True),'burgers':dict(kinds=['per','rbf'],drop_last=True)}
for prob,c in cfg.items():
    X=np.load(B/prob/'training_data.npz')['X_train0'].astype(np.float64)
    kw={k:v for k,v in c.items() if k!='kinds'}
    for kind in c['kinds']:
        best=None
        for nug in [1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6]:
            for l in np.round(np.arange(0.05,1.51,0.01),3):
                ll,s=pooled(X,kind,l,nug,**kw)
                if best is None or ll>best[0]: best=(ll,l,s,nug)
        print(f'{prob:20s} {kind}: best ll/sample={best[0]:.1f}  l={best[1]}  sigma={best[2]:.3f} nug={best[3]:g}')
