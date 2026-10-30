import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems'); R=Path('/mnt/ssd/pengpengxiao/ood/ood_analysis/results')
Xtr=np.load(B/'burgers/training_data.npz')['X_train0'].astype(np.float64)[:,:-1]; Xte=np.load(B/'burgers/test_data.npz')['branch'].astype(np.float64)[:,:-1]
ood=np.load(R/'burgers_labels.npz')['ood']
mu=Xtr.mean(0); U,s,Vt=np.linalg.svd(Xtr-mu,full_matrices=False); lam=s**2/(len(Xtr)-1)
print('train PCA eig log10', np.round(np.log10(lam[:24]),2))
rng=np.random.default_rng(0)
for k in [12,14,16,18,20]:
    idx=rng.permutation(len(Xtr)); null=np.empty(len(Xtr))
    for f in np.array_split(idx,10):
        m=np.ones(len(Xtr),bool); m[f]=False; mu_=Xtr[m].mean(0); _,s_,V_=np.linalg.svd(Xtr[m]-mu_,full_matrices=False); l_=s_**2/(m.sum()-1)
        null[f]=((((Xtr[f]-mu_)@V_[:k].T)**2)/l_[:k]).sum(1)
    d=((((Xte-mu)@Vt[:k].T)**2)/lam[:k]).sum(1); thr=np.quantile(null,0.999)
    print(f'k={k} AUROC={roc_auc_score(ood,d):.4f} agree={np.mean((d>thr)==ood):.3f} ID max d={d[~ood].max():.1f} OOD min d={d[ood].min():.1f} null99.9%={thr:.1f}')
