import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems'); R=Path('/mnt/ssd/pengpengxiao/ood/ood_analysis/results')
def maha_fit(X,k):
    mu=X.mean(0); U,s,Vt=np.linalg.svd(X-mu,full_matrices=False); lam=s**2/(len(X)-1)
    return mu,Vt[:k],lam[:k]
def maha(X,m): mu,V,lam=m; z=(X-mu)@V.T; return (z**2/lam).sum(1)
rng=np.random.default_rng(0)
for prob in ['diffusion_reaction','burgers','advection','poisson']:
    Xtr=np.load(B/prob/'training_data.npz')['X_train0'].astype(np.float64); Xte=np.load(B/prob/'test_data.npz')['branch'].astype(np.float64)
    if prob=='advection':   # remove per-sample offset (a = f - min f + 1)
        Xtr=Xtr-Xtr.mean(1,keepdims=True); Xte=Xte-Xte.mean(1,keepdims=True)
    L=np.load(R/f'{prob}_labels.npz'); ood=L['ood']
    for k in [6,8,10]:
        # 10-fold CV null distribution on training
        idx=rng.permutation(len(Xtr)); null=np.empty(len(Xtr))
        for f in np.array_split(idx,10):
            msk=np.ones(len(Xtr),bool); msk[f]=False; null[f]=maha(Xtr[f],maha_fit(Xtr[msk],k))
        d=maha(Xte,maha_fit(Xtr,k)); thr=np.quantile(null,0.999)
        flag=d>thr
        print(f'{prob:18s} k={k:2d} AUROC(maha vs GRF-label)={roc_auc_score(ood,d):.4f}  agree@CV-99.9% thr={np.mean(flag==ood):.3f}  ID max d={d[~ood].max():.1f} OOD min d={d[ood].min():.1f} null99.9%={thr:.1f}')
    if prob=='poisson':
        K0=np.exp(-(np.linspace(0,1,101)[:,None]-np.linspace(0,1,101)[None,:])**2/0.5); w,U=np.linalg.eigh(K0); U=U[:,::-1]
        t_tr=((Xtr@U[:,11:])**2).sum(1); t_te=((Xte@U[:,11:])**2).sum(1)
        print(f'  poisson tail energy: train max {t_tr.max():.2e} | ID-test max {t_te[~ood].max():.2e} | OOD-test min {t_te[ood].min():.2e} | agree(tail>train max)={np.mean((t_te>t_tr.max())==ood):.3f}')
    if prob=='advection':
        Xr=np.load(B/prob/'test_data.npz')['branch']; Xtr_r=np.load(B/prob/'training_data.npz')['X_train0']
        print(f'  advection: ID-test max a(x)={Xr[~ood].max():.3f} (train max {Xtr_r.max():.3f}); OOD samples with max a > train max: {(Xr[ood].max(1)>Xtr_r.max()).sum()}/{ood.sum()}')
