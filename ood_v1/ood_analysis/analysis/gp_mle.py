"""Per-sample profile-likelihood MLE of GRF length scale (and amplitude) for branch inputs."""
import numpy as np
from pathlib import Path
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems')
OUT=Path('/mnt/ssd/pengpengxiao/ood/ood_analysis/feat'); OUT.mkdir(exist_ok=True)
x=np.linspace(0,1,101)
D=x[:,None]-x[None,:]
LS=np.exp(np.linspace(np.log(0.02),np.log(3.0),300))
def kern(l, kind):
    if kind=='rbf': return np.exp(-D**2/(2*l*l))
    if kind=='per': return np.exp(-2*np.sin(np.pi*D)**2/(l*l))   # ExpSineSquared, period 1
def profile_ll(X, kind, nug=1e-6, const_mean=False, drop_last=False):
    """returns loglik[n_samples, n_ls] with amplitude profiled out, and sigma2_hat."""
    Xw=X[:, :-1] if drop_last else X
    n=Xw.shape[1]
    LL=np.empty((len(X),len(LS))); S2=np.empty_like(LL)
    for j,l in enumerate(LS):
        K=kern(l,kind)
        if drop_last: K=K[:-1,:-1]
        K=K+nug*np.eye(n)
        if const_mean:   # REML: project out constant via contrast matrix
            Q,_=np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n-1]])
            A=Q[:,1:]                       # n x (n-1), orthogonal to ones
            K=A.T@K@A; Z=Xw@A; m=n-1
        else:
            Z=Xw; m=n
        w,U=np.linalg.eigh(K)
        w=np.maximum(w, 1e-300)
        z=Z@U
        quad=(z**2/w).sum(1)
        s2=quad/m
        LL[:,j]=-0.5*(m*np.log(2*np.pi*s2)+np.log(w).sum()+m)
        S2[:,j]=s2
    return LL,S2
cfg={'diffusion_reaction':dict(kind='rbf'),'poisson':dict(kind='rbf'),
     'advection':dict(kind='rbf',const_mean=True),'burgers':dict(kind='per',drop_last=True)}
q=[1,5,25,50,75,95,99]
for prob,c in cfg.items():
    tr=np.load(B/prob/'training_data.npz')['X_train0'].astype(np.float64)
    te=np.load(B/prob/'test_data.npz')['branch'].astype(np.float64)
    for nug in [1e-6,1e-4]:
        res={}
        for name,X in [('train',tr),('test',te)]:
            LL,S2=profile_ll(X, nug=nug, **c)
            j=LL.argmax(1); res[name]=(LS[j], np.sqrt(S2[np.arange(len(X)),j]), LL)
        print(f'== {prob} nug={nug}')
        print('  l_hat train pct', np.round(np.percentile(res['train'][0],q),3))
        print('  l_hat test  pct', np.round(np.percentile(res['test'][0],q),3))
        print('  sig_hat train pct', np.round(np.percentile(res['train'][1],q),3))
        print('  sig_hat test  pct', np.round(np.percentile(res['test'][1],q),3))
        if nug==1e-6: np.savez(OUT/f'{prob}_gp.npz', l_tr=res['train'][0], s_tr=res['train'][1], l_te=res['test'][0], s_te=res['test'][1], LL_tr=res['train'][2], LL_te=res['test'][2], LS=LS)
