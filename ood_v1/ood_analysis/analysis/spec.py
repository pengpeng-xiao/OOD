import numpy as np
from pathlib import Path
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems')
np.set_printoptions(precision=2, linewidth=200)
x=np.linspace(0,1,101); D=x[:,None]-x[None,:]
for prob,l in [('diffusion_reaction',0.5),('poisson',0.5),('poisson',0.45)]:
    X=np.load(B/prob/'training_data.npz')['X_train0'].astype(np.float64)
    K=np.exp(-D**2/(2*l*l)); w,U=np.linalg.eigh(K); w=w[::-1]; U=U[:,::-1]
    z=X@U; emp=(z**2).mean(0)
    print(prob,l); print('  model eig  log10', np.log10(np.maximum(w[:16],1e-30)))
    print('  data  proj log10', np.log10(emp[:16]))
X=np.load(B/'poisson/training_data.npz')['X_train0']
print('poisson dtype check: values representable float16?', np.mean(X.astype(np.float16).astype(np.float32)==X))
# check whether the poisson branch is linear combination of few functions: SVD of training
s=np.linalg.svd(X.astype(np.float64),compute_uv=False); print('poisson train sv log10', np.log10(s[:20]/s[0]))
s=np.linalg.svd(np.load(B/'diffusion_reaction/training_data.npz')['X_train0'].astype(np.float64),compute_uv=False); print('dr      train sv log10', np.log10(s[:20]/s[0]))
