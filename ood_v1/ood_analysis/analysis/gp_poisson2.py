"""Poisson: likelihood restricted to the leading-k KL subspace of the training kernel (l=0.5)."""
import numpy as np
from pathlib import Path
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems')
x=np.linspace(0,1,101); D=x[:,None]-x[None,:]
LS=np.round(np.arange(0.02,2.0001,0.005),4)
K0=np.exp(-D**2/(2*0.25)); w0,U0=np.linalg.eigh(K0); U0=U0[:,::-1]
tr=np.load(B/'poisson/training_data.npz')['X_train0'].astype(np.float64)
te=np.load(B/'poisson/test_data.npz')['branch'].astype(np.float64)
q=[0,1,5,25,50,75,95,99,100]
bins=np.r_[0.02,0.06,0.085,0.115,0.15,0.175,0.225,0.275,0.325,0.375,0.45,0.55,0.7,0.9,1.1,1.4,2.0]
def fit(X,k,nug=1e-13):
    P=U0[:,:k]; Z=X@P
    LL=np.empty((len(X),len(LS))); S2=np.empty_like(LL)
    for j,l in enumerate(LS):
        C=P.T@np.exp(-D**2/(2*l*l))@P + nug*np.eye(k)
        w,V=np.linalg.eigh(C); w=np.maximum(w,1e-300); z=Z@V; quad=(z**2/w).sum(1); s2=quad/k
        LL[:,j]=-0.5*(k*np.log(2*np.pi*s2)+np.log(w).sum()+k); S2[:,j]=s2
    return LL,S2
for k in [9,10,11]:
    o={}
    for n,X in [('tr',tr),('te',te)]:
        LL,S2=fit(X,k); j=LL.argmax(1); o['l_'+n]=LS[j]; o['s_'+n]=np.sqrt(S2[np.arange(len(X)),j]); o['LL_'+n]=LL
    print('k',k,'| l train',np.percentile(o['l_tr'],q),'\n   l test ',np.percentile(o['l_te'],q),'\n   s train',np.round(np.percentile(o['s_tr'],[1,50,99]),3),'s test',np.round(np.percentile(o['s_te'],[1,50,99]),3))
    print('   hist train',np.histogram(o['l_tr'],bins)[0],'\n   hist test ',np.histogram(o['l_te'],bins)[0])
    if k==11: np.savez('/mnt/ssd/pengpengxiao/ood/ood_analysis/feat/poisson_gp2.npz', LS=LS, **o)
# tail energy beyond 11 modes
for n,X in [('train',tr),('test',te)]:
    t=((X@U0[:,11:])**2).sum(1); print('tail energy log10',n,np.round(np.percentile(np.log10(t),q),2))
