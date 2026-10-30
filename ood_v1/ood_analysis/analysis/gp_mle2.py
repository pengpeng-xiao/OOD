import numpy as np, sys
from pathlib import Path
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems')
OUT=Path('/mnt/ssd/pengpengxiao/ood/ood_analysis/feat'); OUT.mkdir(exist_ok=True)
x=np.linspace(0,1,101); D=x[:,None]-x[None,:]
LS=np.round(np.arange(0.02,2.0001,0.005),4)
def kern(l, kind):
    if kind=='rbf': return np.exp(-D**2/(2*l*l))
    if kind=='per': return np.exp(-2*np.sin(np.pi*D)**2/(l*l))
def contrast(n):
    Q,_=np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :n-1]]); return Q[:,1:]
def profile(X, kind, nug, const_mean=False, drop_last=False, fix_sigma=None):
    Xw=X[:,:-1] if drop_last else X; n=Xw.shape[1]
    A=contrast(n) if const_mean else None
    if const_mean: Xw=Xw@A; m=n-1
    else: m=n
    LL=np.empty((len(X),len(LS))); S2=np.empty_like(LL)
    for j,l in enumerate(LS):
        K=kern(l,kind)[:n,:n]
        if const_mean: K=A.T@K@A
        w,U=np.linalg.eigh(K); w=np.maximum(w,0)+nug
        z=Xw@U; quad=(z**2/w).sum(1)
        s2=quad/m if fix_sigma is None else np.full(len(X),fix_sigma**2)
        LL[:,j]=-0.5*(m*np.log(2*np.pi*s2)+np.log(w).sum()+quad/s2); S2[:,j]=s2
    return LL,S2
cfg={'diffusion_reaction':dict(kind='rbf',nug=1e-13),'poisson':dict(kind='rbf',nug=1e-14),
     'advection':dict(kind='rbf',nug=1e-10,const_mean=True),'burgers':dict(kind='per',nug=1e-13,drop_last=True)}
q=[0,1,5,25,50,75,95,99,100]
for prob,c in cfg.items():
    tr=np.load(B/prob/'training_data.npz')['X_train0'].astype(np.float64)
    te=np.load(B/prob/'test_data.npz')['branch'].astype(np.float64)
    out={}
    for name,X in [('tr',tr),('te',te)]:
        LL,S2=profile(X,**c); j=LL.argmax(1)
        out['l_'+name]=LS[j]; out['s_'+name]=np.sqrt(S2[np.arange(len(X)),j]); out['LL_'+name]=LL
    np.savez(OUT/f'{prob}_gp2.npz', LS=LS, **out)
    print(f'== {prob}')
    print('  l_hat train', np.percentile(out['l_tr'],q)); print('  l_hat test ', np.percentile(out['l_te'],q))
    print('  s_hat train', np.round(np.percentile(out['s_tr'],q),3)); print('  s_hat test ', np.round(np.percentile(out['s_te'],q),3))
    # histogram of test l_hat
    bins=np.r_[0.02,0.06,0.085,0.115,0.15,0.175,0.225,0.275,0.325,0.375,0.45,0.55,0.7,0.9,1.1,1.4,2.0]
    ht,_=np.histogram(out['l_tr'],bins); he,_=np.histogram(out['l_te'],bins)
    print('  bins ', bins); print('  train', ht); print('  test ', he)
